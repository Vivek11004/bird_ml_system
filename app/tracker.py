from deep_sort_realtime.deepsort_tracker import DeepSort

class BirdTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def track(self, detections, frame):
        # Convert YOLO detections ([x1, y1, x2, y2, conf]) to DeepSort format
        formatted_dets = [([x1, y1, x2, y2], conf) for x1, y1, x2, y2, conf in detections]

        tracks = self.tracker.update_tracks(formatted_dets, frame=frame)
        active_tracks = []

        for t in tracks:
            if not t.is_confirmed():
                continue

            l, t_, r, b = map(int, t.to_ltrb())
            active_tracks.append({
                "id": t.track_id,
                "bbox": [l, t_, r, b]
            })

        return active_tracks
