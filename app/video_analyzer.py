import cv2
from collections import defaultdict
from app.detector import YOLOBirdDetector
from app.tracker import BirdTracker
from app.weight import compute_weight_with_confidence

def draw_annotations(frame, tracks, count):
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        tid = t["id"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    return frame


def analyze_video(video_path, fps_sample=5):
    cap = cv2.VideoCapture(video_path)

    detector = YOLOBirdDetector()
    tracker = BirdTracker()

    unique_ids = set()
    bird_boxes = defaultdict(list)
    counts = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "outputs/annotated_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (width, height)
    )

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % fps_sample != 0:
            frame_idx += 1
            continue

        detections = detector.detect(frame)
        tracks = tracker.track(detections, frame)

        for t in tracks:
            unique_ids.add(t["id"])
            bird_boxes[t["id"]].append(t["bbox"])

        count = len(unique_ids)
        counts.append({"timestamp": frame_idx, "count": count})

        frame = draw_annotations(frame, tracks, count)
        out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()

    weights = {}
    for tid, boxes in bird_boxes.items():
        w, conf = compute_weight_with_confidence(boxes)
        weights[tid] = {"weight_index": w, "confidence": conf}

    return counts, weights
