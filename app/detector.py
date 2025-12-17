from ultralytics import YOLO

class YOLOBirdDetector:
    def __init__(
        self,
        model_path="runs/detect/train/weights/best.pt",
        conf_thresh=0.4
    ):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        results = self.model(frame, conf=self.conf_thresh)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])

        return detections
