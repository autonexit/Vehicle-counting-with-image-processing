"""
============================================================
Project: Vehicle Detection + SORT Tracking (YOLOv11 Large)
============================================================

Author      : Mohammad Siahtiri
Website     : https://autonexit.com or https://poweren.ir
Email       : siahtirim@gmail.com
Phone       : +989123874216

Description
-----------
This script performs:
1) Vehicle detection using Ultralytics YOLOv11 Large model (yolo11l.pt)
2) Multi-object tracking using SORT (Kalman Filter + Hungarian assignment)

Input Source
------------
- By default, it reads from a video file: "Car.mp4"
- You can easily switch to a webcam or an IP camera by changing cv2.VideoCapture(...)

Examples:
- Webcam (default camera):
    cap = cv2.VideoCapture(0)

- Another webcam index:
    cap = cv2.VideoCapture(1)

- IP Camera / RTSP stream:
    cap = cv2.VideoCapture("rtsp://username:password@ip_address:554/stream")

Notes
-----
- The tracker assigns a stable ID to each detected car when possible.
- If a car is not detected for several frames (occlusion / detection miss),
  SORT may create a new ID when it reappears. You can reduce this by tuning:
  max_age, min_hits, iou_threshold, and YOLO confidence threshold.
============================================================
"""

# Import the SORT tracker implementation (your Sort.py file/module)
# IMPORTANT: Sort.py must be in the same folder as this script or in your PYTHONPATH.
from Sort import *  # provides Sort class + numpy (np) dependency used below

import cv2
from ultralytics import YOLO


# -------------------------
# 1) Load YOLO model
# -------------------------
# Using YOLOv11 Large model for better detection quality (heavier than nano/small).
# You can replace "yolo11l.pt" with another YOLO model if needed (e.g., yolo11n.pt, yolo11s.pt, ...)
model = YOLO("yolo11l.pt")


# -------------------------
# 2) Open input source
# -------------------------
# Video file input:
cap = cv2.VideoCapture("Car.mp4")

# If you want webcam instead, comment the line above and uncomment one of these:
# cap = cv2.VideoCapture(0)  # Webcam
# cap = cv2.VideoCapture("rtsp://username:password@ip:554/stream")  # IP camera / RTSP


# -------------------------
# 3) Initialize SORT tracker
# -------------------------
# max_age      : How many frames to keep "lost" tracks before deleting them.
# min_hits     : How many consecutive matches required before a track is considered "confirmed".
# iou_threshold: Minimum IoU required to match a detection to an existing track.
tracker = Sort(max_age=10, min_hits=2, iou_threshold=0.3)


# -------------------------
# 4) Main processing loop
# -------------------------
while True:
    # Read next frame from the video/camera
    ret, frame = cap.read()

    # If reading fails, we are at the end of the video (or camera disconnected)
    if not ret:
        break

    # -------------------------
    # 4.1) Run YOLO detection
    # -------------------------
    # conf=0.25 means only detections with confidence >= 0.25 are returned.
    # Lower conf => more detections (and more false positives).
    # Higher conf => fewer detections (but may miss objects).
    results = model.predict(frame, conf=0.25, verbose=False)[0]

    # -------------------------
    # 4.2) Build detection list for SORT
    # -------------------------
    # SORT expects detections as: [x1, y1, x2, y2, score]
    dets = []

    for b in results.boxes:
        # Map class index to class name (e.g., "car", "person", ...)
        class_name = results.names[int(b.cls[0])]

        # Only keep cars (you can add "truck" or "bus" if desired)
        if class_name == "car":
            # Bounding box coordinates (float)
            x1, y1, x2, y2 = b.xyxy[0].tolist()

            # Detection confidence score
            conf = float(b.conf[0])

            # Append detection in SORT format
            dets.append([x1, y1, x2, y2, conf])

    # Convert to numpy array (Nx5). If no detections, pass an empty Nx5 array.
    dets = np.array(dets, dtype=np.float32) if len(dets) else np.empty((0, 5), dtype=np.float32)

    # -------------------------
    # 4.3) Update SORT tracker
    # -------------------------
    # tracks is an array of shape (M,5): [x1, y1, x2, y2, track_id]
    tracks = tracker.update(dets)

    # -------------------------
    # 4.4) Draw results
    # -------------------------
    for trk in tracks:
        x1, y1, x2, y2, tid = trk

        # Convert coordinates and ID to integers for OpenCV drawing
        x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)

        # Draw bounding box around the tracked car
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put the tracking ID above the bounding box
        cv2.putText(
            frame,
            f"ID {tid}",
            (x1, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        # Draw the center point of the bounding box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # -------------------------
    # 4.5) Show output frame
    # -------------------------
    frame_resized = cv2.resize(frame, (1280, 720))
    cv2.imshow("frame", frame_resized)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# -------------------------
# 5) Cleanup
# -------------------------
cap.release()
cv2.destroyAllWindows()
