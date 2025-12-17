# 🐦 Bird Counting and Weight Estimation from CCTV Video

## Candidate Task: ML Prototype (Detection, Tracking & Weight Proxy)

This project implements a **machine learning–based prototype** that processes fixed-camera poultry CCTV videos to:

1. **Count birds over time** using object detection and multi-object tracking
2. **Estimate bird weight** from video using a **relative weight proxy**, with clear assumptions and calibration requirements

The solution is designed to demonstrate **ML depth**, system design, and practical trade-offs under limited ground-truth data.

---

## 🎯 Problem Statement

Given a fixed CCTV video of a poultry environment:

* Detect birds with bounding boxes and confidence scores
* Assign **stable tracking IDs** to avoid double counting
* Produce a **time series of bird counts**
* Estimate **per-bird and/or aggregate weight** from video
* Generate visual and JSON artifacts

---

## 🧠 Approach Overview

### 1️⃣ Bird Detection

* Model: **YOLOv8 (Ultralytics)**
* Strategy: **Transfer learning + fine-tuning** on a custom bird dataset
* Output: Bounding boxes with confidence scores per frame

### 2️⃣ Bird Tracking & Counting

* Tracker: **DeepSORT**
* Each detected bird is assigned a **persistent ID** across frames
* Bird count at time *t* = number of **unique active track IDs** seen so far

#### Handling Occlusions & ID Switches

* DeepSORT uses motion + appearance embeddings
* `max_age` allows temporary occlusions without losing IDs
* Frame sampling (fps_sample) reduces jitter and duplicate detections

---

## ⚖️ Weight Estimation Method (Mandatory)

Since **true weight ground truth (grams)** is not available in the video, the system outputs a **Relative Weight Index**.

### Weight Proxy Logic

* For each tracked bird:

  * Compute bounding box area per frame
  * Aggregate statistics across time

```text
Weight Index = mean(bounding_box_area) / normalization_factor
Confidence   = 1 / (1 + std_dev_of_area)
```

### Interpretation

* Larger birds → larger bounding box area → higher weight index
* Confidence reflects stability of size across frames

### What is required to convert to grams?

To estimate **absolute weight (grams)**, one of the following is required:

1. Camera calibration (pixel-to-real-world mapping)
2. Known reference object dimensions in the scene
3. Labeled dataset with true bird weights for regression

---

## 🏗️ System Architecture

```
CCTV Video
    ↓
YOLOv8 Detector (fine-tuned)
    ↓
DeepSORT Tracker (stable IDs)
    ↓
Counting + Weight Proxy Logic
    ↓
Annotated Video + JSON Output
```

---

## 🌐 API Specification (FastAPI)

### 1️⃣ Health Check

```
GET /health
```

Response:

```json
{"status": "OK"}
```

---

### 2️⃣ Video Analysis Endpoint

```
POST /analyze_video
```

**Request:** `multipart/form-data`

* `video` (required): CCTV video file
* `fps_sample` (optional): frame sampling rate (default = 5)
* `conf_thresh` (optional): detection confidence threshold

**Response JSON includes:**

* `counts`: timestamp → bird count time series
* `tracks_sample`: sample tracking IDs with bounding boxes
* `weight_estimates`: per-bird weight index with confidence
* `artifacts`: generated output files

---

## 📄 Sample API Response

A real sample response is provided in:

```
sample_response.json
```

Example structure:

```json
{
  "counts": [{"timestamp": 0, "count": 3}],
  "weight_estimates": {
    "12": {"weight_index": 4.2, "confidence": 0.92}
  },
  "unit": "relative_weight_index",
  "artifacts": ["annotated_video.mp4"]
}
```

---

## 📽️ Annotated Output Video

The system generates at least one annotated video containing:

* Bounding boxes
* Tracking IDs
* Live bird count overlay

🔗 **Annotated video link:** (provided in submission ZIP / Drive link)

---

## 📁 Project Structure

```
bird_ml_system/
├── app/
│   ├── main.py              # FastAPI app
│   ├── detector.py          # YOLOv8 detector
│   ├── tracker.py           # DeepSORT tracker
│   ├── video_analyzer.py    # Core pipeline
│   └── weight.py            # Weight proxy logic
│
├── requirements.txt
├── bird_dataset.yaml
├── split_dataset.py
├── README.md
├── sample_response.json
└── .gitignore
```

---

## ⚙️ Setup & Execution

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### Example API Call

```bash
curl -X POST "http://127.0.0.1:8000/analyze_video" \
  -F "video=@sample_video.mp4" \
  -F "fps_sample=5"
```

---

## 🧪 Frame Sampling Justification

* CCTV videos are high FPS with limited motion
* Sampling every N frames:

  * Reduces compute cost
  * Improves ID stability
  * Avoids duplicate counting

---

## 🎓 Conclusion

This prototype demonstrates an **end-to-end ML system** combining:

* Fine-tuned object detection
* Robust multi-object tracking
* Practical weight estimation under real-world constraints
* Clean API design with reproducible artifacts

The solution balances **accuracy, efficiency, and explainability**, meeting all task requirements.

---

## 👤 Author

**Vivek**
Machine Learning & Computer Vision

---


