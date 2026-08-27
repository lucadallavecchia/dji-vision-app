import threading
import time

import cv2
from ultralytics import YOLO

RTSP_URL = "rtsp://127.0.0.1:8554/drone"
DEFAULT_TARGET_CLASSES = {"person", "dog", "cat", "bird", "horse", "sheep", "cow"}
MODEL_NAME = "yolo11n.pt"


class VideoProcessor:
    def __init__(self):
        self.model = YOLO(MODEL_NAME)
        self.class_names = self.model.names

        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._running = False
        self._thread = None

        self._config_lock = threading.Lock()
        self.target_ids = [i for i, n in self.class_names.items() if n in DEFAULT_TARGET_CLASSES]

    def get_available_classes(self):
        """Tutte le classi riconosciute dal modello, in ordine alfabetico."""
        return sorted(self.class_names.values())

    def get_target_classes(self):
        with self._config_lock:
            ids = list(self.target_ids)
        return sorted(self.class_names[i] for i in ids)

    def set_target_classes(self, names):
        valid = set(self.class_names.values())
        ids = [i for i, n in self.class_names.items() if n in names and n in valid]
        with self._config_lock:
            self.target_ids = ids

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        cap = None
        while self._running:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(RTSP_URL)
                if not cap.isOpened():
                    time.sleep(1)
                    continue

            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                time.sleep(0.5)
                continue

            with self._config_lock:
                target_ids = list(self.target_ids)

            results = self.model.predict(frame, classes=target_ids, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{self.class_names[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._lock:
                    self._latest_jpeg = jpeg.tobytes()

        if cap:
            cap.release()

    def get_jpeg(self):
        with self._lock:
            return self._latest_jpeg
