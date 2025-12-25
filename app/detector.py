from ultralytics import YOLO
import cv2

class YOLOBirdDetector:
    def __init__(
        self,
        model_path="models/best.pt",
        conf_thresh=0.4, #
        iou=0.6,
        imgsz=896
    ):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou = iou
        self.imgsz = imgsz

    def detect(self, frame):
        # 🔹 Light contrast enhancement (helps low-contrast & far birds)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)

        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou,
            imgsz=self.imgsz,
            augment=True,     #  Test Time Augmentation
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])

        return detections
