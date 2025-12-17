# 🐔 Bird Counting & Weight Estimation from CCTV Video

## 📌 Overview
This project implements an end-to-end Computer Vision and Machine Learning system
that analyzes a fixed-camera poultry CCTV video to detect, track, and count birds
over time and estimate bird weight using a relative visual proxy. The system also
exposes results via a REST API and generates an annotated output video.

The solution is designed for real-world conditions including occlusions,
overlapping birds, and missing ground-truth labels.

---

## 🧠 System Pipeline
Video  
→ Frame Sampling  
→ YOLO Detection  
→ DeepSORT Tracking  
→ Unique ID Counting  
→ Weight Estimation  
→ FastAPI Response + Annotated Video

---

## 1️⃣ Bird Detection
- Model: YOLOv8 (pretrained)
- Detects birds in sampled frames
- Outputs bounding boxes and confidence scores

---

## 2️⃣ Bird Tracking & Counting
- Tracker: DeepSORT
- Assigns stable unique IDs to birds across frames
- Prevents double counting
- Handles short-term occlusions and re-identification

Counting Logic:
- Each unique tracking ID corresponds to one bird
- Output is cumulative over time: timestamp → total unique birds seen

---

## ⚖️ Weight Estimation

Important Note:
Public datasets do not provide true per-bird weight labels from video.
Therefore, a relative weight proxy is used.

Weight Proxy:
- Derived from average bounding box area per bird over time
- Larger pixel area implies larger relative weight

weight_index ∝ average bounding box area

Confidence:
- Represents temporal stability of bounding box area
- Lower confidence indicates higher variation due to movement,
  occlusion, and perspective distortion

Unit: relative_weight_index (not grams)

To convert this index into grams, camera calibration or labeled
weight samples are required.

---

## 🎥 Annotated Output Video
The output video includes:
- Bounding boxes
- Tracking IDs
- Bird count overlay

Weight values are returned via the API response and are not required
to be displayed on the annotated video.

---

## 🌐 API Service

### POST /analyze_video
Accepts a video file and returns analysis results.

Response Includes:
- Bird count over time
- Per-bird weight estimates
- Confidence values
- Annotated output video path

---

## 📦 Sample API Response
```json
{
  "counts": [
    { "timestamp": 0, "count": 0 },
    { "timestamp": 80, "count": 3 },
    { "timestamp": 190, "count": 8 }
  ],
  "weight_estimates": {
    "1": { "weight_index": 10.40, "confidence": 0.000008 },
    "9": { "weight_index": 115.75, "confidence": 0.000001 }
  },
  "unit": "relative_weight_index",
  "artifacts": ["outputs/annotated_video.mp4"]
}
