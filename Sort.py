from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import numpy as np

# ================================================================
# SORT Helper Functions
# ================================================================

def iou(bb_test, bb_gt):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Parameters
    ----------
    bb_test : [x1, y1, x2, y2]
    bb_gt   : [x1, y1, x2, y2]

    Returns
    -------
    float : IoU value between 0 and 1
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h

    area1 = max(0.0, (bb_test[2] - bb_test[0])) * max(0.0, (bb_test[3] - bb_test[1]))
    area2 = max(0.0, (bb_gt[2] - bb_gt[0])) * max(0.0, (bb_gt[3] - bb_gt[1]))

    union = area1 + area2 - inter + 1e-6
    return inter / union


def convert_bbox_to_z(bbox):
    """
    Convert bounding box from [x1,y1,x2,y2] to Kalman state format [x,y,s,r]^T.

    x, y : center of the box
    s    : scale (area = width * height)
    r    : aspect ratio (width / height)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / (h + 1e-6)

    return np.array([[x], [y], [s], [r]], dtype=np.float32)


def convert_x_to_bbox(x, score=None):
    """
    Convert Kalman state vector [x,y,s,r] back to bounding box [x1,y1,x2,y2].

    NOTE:
    The Kalman state may be shaped (4,1), so we flatten it first.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    x0, y0, s, r = float(x[0]), float(x[1]), float(x[2]), float(x[3])

    # Recover width and height from scale and aspect ratio
    w = np.sqrt(abs(s) * abs(r))
    h = abs(s) / (w + 1e-6)

    x1 = x0 - w / 2.0
    y1 = y0 - h / 2.0
    x2 = x0 + w / 2.0
    y2 = y0 + h / 2.0

    if score is None:
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    return np.array([x1, y1, x2, y2, float(score)], dtype=np.float32)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assign detections to trackers using IoU + Hungarian algorithm.

    Parameters
    ----------
    detections : Nx5 array [x1,y1,x2,y2,score]
    trackers   : Mx4 array [x1,y1,x2,y2]

    Returns
    -------
    matches         : Kx2 (det_idx, trk_idx)
    unmatched_dets  : list of unmatched detection indices
    unmatched_trks  : list of unmatched tracker indices
    """

    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det[:4], trk)

    # Hungarian matching (maximize IoU → minimize negative IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matches = []
    unmatched_dets = []
    unmatched_trks = []

    for d in range(len(detections)):
        if d not in row_ind:
            unmatched_dets.append(d)

    for t in range(len(trackers)):
        if t not in col_ind:
            unmatched_trks.append(t)

    # Filter matches by IoU threshold
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] < iou_threshold:
            unmatched_dets.append(r)
            unmatched_trks.append(c)
        else:
            matches.append([r, c])

    return np.array(matches, dtype=int), np.array(unmatched_dets, dtype=int), np.array(unmatched_trks, dtype=int)


# ================================================================
# Kalman Box Tracker
# ================================================================

class KalmanBoxTracker:
    """
    Represents the internal state of an individual tracked object using Kalman Filter.

    State vector:
    [x, y, s, r, x_velocity, y_velocity, s_velocity]
    """

    count = 0  # Global counter for assigning unique track IDs

    def __init__(self, bbox):
        # 7 state variables, 4 measurement variables
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix (we measure x,y,s,r)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0
        self.kf.R *= 1.0

        # Initial covariance (high uncertainty in velocity)
        self.kf.P *= 10.0
        self.kf.P[4:, 4:] *= 1000.0

        # Process noise
        self.kf.Q *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        # Tracking statistics
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

        # Assign unique ID
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def update(self, bbox):
        """Update tracker with new detected bounding box."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Predict the next state of the object.
        """
        # Prevent negative scale
        if (self.kf.x[2] + self.kf.x[6]) <= 0:
            self.kf.x[6] = 0.0

        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        return convert_x_to_bbox(self.kf.x[:4])

    def get_state(self):
        """Return current bounding box estimate."""
        return convert_x_to_bbox(self.kf.x[:4])


# ================================================================
# SORT Tracker
# ================================================================

class Sort:
    """
    SORT (Simple Online Realtime Tracking)

    Combines:
    - Kalman Filter (motion prediction)
    - IoU + Hungarian Algorithm (data association)
    """

    def __init__(self, max_age=10, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        Update tracker with current frame detections.

        Parameters
        ----------
        detections : Nx5 [x1,y1,x2,y2,score]

        Returns
        -------
        Mx5 : [x1,y1,x2,y2,track_id]
        """
        self.frame_count += 1

        # -------------------- Predict step --------------------
        trks = np.zeros((len(self.trackers), 4), dtype=np.float32)
        to_del = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = pos

            # Remove invalid trackers
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)

        trks = trks[:len(self.trackers), :] if len(self.trackers) else np.empty((0, 4), dtype=np.float32)

        # -------------------- Data association --------------------
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )

        # -------------------- Update matched trackers --------------------
        for m in matches:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(detections[det_idx, :4])

        # -------------------- Create new trackers --------------------
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i, :4]))

        # -------------------- Prepare output & remove dead trackers --------------------
        ret = []
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.get_state()

            # Only return valid confirmed tracks
            if (trk.time_since_update < 1) and (
                trk.hits >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append([d[0], d[1], d[2], d[3], trk.id])

            i -= 1

            # Remove dead track
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.array(ret, dtype=np.float32) if len(ret) else np.empty((0, 5), dtype=np.float32)
