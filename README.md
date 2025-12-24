# 🐦 Bird Counting and Weight Estimation from CCTV Video

## Candidate Task: ML Prototype (Detection, Tracking & Weight Proxy)

This project implements a **machine learning–based prototype** that processes fixed-camera poultry CCTV videos to:

1. **Count birds over time** using object detection and multi-object tracking
2. **Estimate bird weight** from video using a **relative weight proxy**, with clear assumptions and calibration requirements

The solution demonstrates **ML depth**, system design, and practical trade-offs under limited ground-truth data.

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
* Input size: **768 × 768**
* Output: Bounding boxes with confidence scores per frame

> The model was fine-tuned to improve detection of **small and distant birds**, which are common in fixed CCTV views.

---

### 2️⃣ Bird Tracking & Counting

* Tracker: **DeepSORT**
* Each detected bird is assigned a **persistent ID** across frames
* Bird count at time *t* = number of **unique track IDs observed so far**

#### Handling Occlusions & ID Switches

* DeepSORT combines:

  * Motion prediction (Kalman Filter)
  * Appearance embeddings
* `max_age` allows temporary occlusions without losing IDs
* Frame sampling reduces flickering detections and duplicate counts

---

## 🚀 Model Improvements & Training Strategy

This section summarizes the **practical improvements applied during development**.

### 🔹 Higher Input Resolution (imgsz = 768)

* The model was trained with **imgsz = 768** instead of the default 640
* Benefit:

  * Better detection of **small birds far from the camera**
  * Improved localization in dense scenes
* Trade-off:

  * Slightly slower inference
  * Higher GPU memory usage (handled using batch size = 4)

---

### 🔹 Multi-Frame Processing (FPS Sampling)

* Instead of processing every frame, the pipeline samples frames:

```text
Process frame if frame_index % fps_sample == 0
```

* Benefits:

  * Reduces computational load
  * Improves tracking stability
  * Prevents repeated counting of the same bird
* Default value: `fps_sample = 5`

---

### 🔹 Bounding Box Tightening (Post-Training)

* Bounding box size is influenced by:

  * Training annotations
  * Detection confidence threshold (`conf_thresh`)
* Improvements applied:

  * Higher input resolution
  * Better anchor learning through fine-tuning
* Remaining limitation:

  * Boxes may still appear slightly larger in crowded or occluded scenes

---

### 🔹 Fine-Tuning Summary

* Base model: `yolov8n.pt`
* Fine-tuned on: custom poultry bird dataset
* Epochs: 60–80 (with early stopping)
* Result:

  * Improved recall for distant birds
  * More stable detections across frames

---

## ⚖️ Weight Estimation Method (Mandatory)

Since **true weight ground truth (grams)** is not available, the system outputs a **Relative Weight Index**.

### Weight Proxy Logic

For each tracked bird:

* Compute bounding box area per frame
* Aggregate statistics across time

```text
Weight Index = mean(bounding_box_area) / normalization_factor
Confidence   = 1 / (1 + std_dev_of_area)
```

### Interpretation

* Larger birds → larger bounding box area → higher weight index
* Confidence reflects **temporal stability**, not prediction certainty

> ⚠️ Note: Confidence values are small because they are derived from variance, not model probability.

---

### What Is Required to Convert to Grams?

To estimate **absolute weight (grams)**, one of the following is required:

1. Camera calibration (pixel → real-world scale)
2. Known reference object dimensions in the scene
3. Labeled dataset with bird weights for regression

---

## 🏗️ System Architecture

```
CCTV Video
    ↓
YOLOv8 Detector (fine-tuned, imgsz=768)
    ↓
DeepSORT Tracker (stable IDs)
    ↓
Multi-frame aggregation
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

**Request (multipart/form-data):**

* `video` (required): CCTV video file
* `fps_sample` (optional): frame sampling rate (default = 5)
* `conf_thresh` (optional): detection confidence threshold

**Response JSON includes:**

* `counts`: timestamp → bird count time series
* `weight_estimates`: per-bird weight index with confidence
* `artifacts`: generated output files

---

## 📄 Sample API Response

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

The system generates an annotated video with:

* Bounding boxes
* Tracking IDs
* Live bird count overlay

📁 Included in submission (`outputs/annotated_video.mp4`)

---

## 📁 Project Structure

```
bird_ml_system/
├── app/
│   ├── main.py
│   ├── detector.py
│   ├── tracker.py
│   ├── video_analyzer.py
│   └── weight.py
├── requirements.txt
├── bird_dataset.yaml
├── split_dataset.py
├── README.md
└── sample_response.json
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

## 🎓 Conclusion

This project demonstrates a **realistic, end-to-end ML system** for poultry analytics:

* Fine-tuned object detection (YOLOv8)
* Robust multi-object tracking (DeepSORT)
* Multi-frame reasoning for stability
* Interpretable weight proxy under real-world constraints

The system prioritizes **engineering clarity, explainability, and honest assumptions**, making it suitable for both academic evaluation and real-world prototyping.

---

## 👤 Author

**Vivek**
Machine Learning & Computer Vision

