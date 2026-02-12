# Vehicle counting with image processing
# شمارش خودرو با پردازش تصویر
The vehicle detection and tracking system using YOLOv11 and the SORT algorithm can identify cars in video streams in real time and assign a unique ID to each one. By combining accurate object detection with motion prediction, it is widely used in applications such as traffic monitoring, vehicle counting, and smart surveillance systems.


<h1 align="center">Vehicle counting</h1>
<a href="https://autonexit.com" target="_blank">
    <img alt="Vehicle-counting" src="assets/banners/bg.png"/>
</a>

# 🚗 Vehicle Detection & Tracking using YOLOv11 + SORT  
# 🚗 سیستم تشخیص و ردیابی خودرو با YOLOv11 و SORT

---

## 👤 Author | نویسنده
**Name / نام:** Siahtiri  
**Website / وب‌سایت:** https://poweren.ir
**Email / ایمیل:** siahtirim@gmail.com 
**Phone / تلفن:** +989123874216 

---
## 🎥 Demo Video

[▶ Watch Demo](assets/demo.mp4)



## 📌 Overview | معرفی پروژه

**EN:**  
This project implements a real-time vehicle detection and tracking system using **YOLOv11 Large** for object detection and **SORT (Simple Online Realtime Tracking)** for multi-object tracking. The system detects vehicles in video streams and assigns a stable unique ID to each one.

**FA:**  
این پروژه یک سیستم تشخیص و ردیابی خودرو در زمان واقعی است که از **YOLOv11 Large** برای تشخیص اشیاء و از الگوریتم **SORT** برای ردیابی چندگانه استفاده می‌کند. سیستم قادر است خودروها را در ویدیو شناسایی کرده و برای هر خودرو یک شناسه یکتا و پایدار اختصاص دهد.

---

## 🧠 How It Works | نحوه عملکرد

**EN**
1. Capture frames from video / webcam / IP camera  
2. Detect vehicles using YOLOv11  
3. Send detections to SORT tracker  
4. Assign unique ID to each vehicle  
5. Draw bounding box + ID  
6. Continue in real-time  

**FA**
1. دریافت فریم از ویدیو، وب‌کم یا دوربین IP  
2. تشخیص خودروها با YOLOv11  
3. ارسال مختصات به الگوریتم SORT  
4. اختصاص شناسه یکتا به هر خودرو  
5. نمایش کادر و ID روی تصویر  
6. ادامه پردازش در زمان واقعی  

---

## 🎯 Features | ویژگی‌ها

**EN**
- High accuracy vehicle detection (YOLOv11 Large)  
- Stable multi-object tracking with unique IDs  
- Real-time performance  
- Works with video / webcam / IP camera  
- Lightweight Kalman-based tracking  
- Easily extendable (vehicle counting, traffic analysis)

**FA**
- تشخیص دقیق خودرو با YOLOv11 Large  
- ردیابی همزمان چند خودرو با شناسه ثابت  
- عملکرد Real-time  
- قابلیت اتصال به فایل ویدیو، وب‌کم یا دوربین IP  
- ردیابی سبک مبتنی بر فیلتر کالمن  
- قابل توسعه برای شمارش خودرو و تحلیل ترافیک  

---

## 📷 Input Sources | منابع ورودی

### Video File | فایل ویدیو
```python
cap = cv2.VideoCapture("Car.mp4")
Webcam | وب‌کم
cap = cv2.VideoCapture(0)

📊 Applications | کاربردها

EN

Traffic monitoring & control

Vehicle counting systems

Smart parking

Urban & security surveillance

Traffic behavior analysis

Computer vision research

FA

مدیریت و کنترل ترافیک

سیستم شمارش خودرو

پارکینگ هوشمند

نظارت شهری و امنیتی

تحلیل رفتار ترافیکی

پروژه‌های تحقیقاتی بینایی ماشین

🔧 Optimization Tips | نکات بهینه‌سازی

EN

Increase max_age

Adjust iou_threshold

Tune YOLO confidence

Use higher resolution video

FA

افزایش max_age

تنظیم iou_threshold

تنظیم Confidence مدل YOLO

استفاده از ویدیوی با کیفیت بهتر

