import threading
import time

import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

RTSP_URL = "rtsp://127.0.0.1:8554/drone"
DEFAULT_TARGET_CLASSES = {"person", "dog", "cat", "bird", "horse", "sheep", "cow"}
MODEL_NAME = "yolo11n.pt"
SAHI_SLICE_SIZE = 768
SAHI_OVERLAP_RATIO = 0.2
SAHI_STANDARD_PRED = False
SAHI_SLICE_SIZE_RANGE = (128, 1536)
SAHI_OVERLAP_RATIO_RANGE = (0.0, 0.9)

CPU_BATCH_SIZE = 1
GPU_BATCH_SIZE = 8


def _detect_gpu_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


class VideoProcessor:
    def __init__(self):
        self.gpu_torch_device = _detect_gpu_torch_device()

        self._model_lock = threading.Lock()
        self.model, self.sahi_model = self._build_models("cpu")
        self.class_names = self.model.names

        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._running = False
        self._thread = None
        self._fps = 0.0

        self._config_lock = threading.Lock()
        self.target_ids = [i for i, n in self.class_names.items() if n in DEFAULT_TARGET_CLASSES]
        self.sahi_enabled = False
        self.sahi_slice_size = SAHI_SLICE_SIZE
        self.sahi_overlap_ratio = SAHI_OVERLAP_RATIO
        self.sahi_standard_pred = SAHI_STANDARD_PRED

        self.processing_device = "cpu"
        self.batch_size = CPU_BATCH_SIZE
        self.reloading = False

    def _build_models(self, torch_device):
        model = YOLO(MODEL_NAME)
        model.to(torch_device)
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=MODEL_NAME,
            confidence_threshold=0.25,
            device=torch_device,
        )
        return model, sahi_model

    def get_device_status(self):
        with self._config_lock:
            return {
                "device": self.processing_device,
                "reloading": self.reloading,
                "gpu_available": self.gpu_torch_device is not None,
            }

    def set_processing_device(self, device):
        device = "gpu" if device == "gpu" else "cpu"
        if device == "gpu" and self.gpu_torch_device is None:
            return
        with self._config_lock:
            if device == self.processing_device or self.reloading:
                return
            self.reloading = True
        threading.Thread(target=self._reload_models, args=(device,), daemon=True).start()

    def _reload_models(self, device):
        torch_device = self.gpu_torch_device if device == "gpu" else "cpu"
        batch_size = GPU_BATCH_SIZE if device == "gpu" else CPU_BATCH_SIZE
        model, sahi_model = self._build_models(torch_device)
        with self._model_lock:
            self.model = model
            self.sahi_model = sahi_model
        with self._config_lock:
            self.processing_device = device
            self.batch_size = batch_size
            self.reloading = False

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

    def get_sahi_enabled(self):
        with self._config_lock:
            return self.sahi_enabled

    def set_sahi_enabled(self, enabled):
        with self._config_lock:
            self.sahi_enabled = bool(enabled)

    def get_sahi_config(self):
        with self._config_lock:
            return {
                "slice_size": self.sahi_slice_size,
                "overlap_ratio": self.sahi_overlap_ratio,
                "standard_pred": self.sahi_standard_pred,
            }

    def set_sahi_config(self, slice_size=None, overlap_ratio=None, standard_pred=None):
        with self._config_lock:
            if slice_size is not None:
                lo, hi = SAHI_SLICE_SIZE_RANGE
                self.sahi_slice_size = max(lo, min(hi, int(slice_size)))
            if overlap_ratio is not None:
                lo, hi = SAHI_OVERLAP_RATIO_RANGE
                self.sahi_overlap_ratio = max(lo, min(hi, float(overlap_ratio)))
            if standard_pred is not None:
                self.sahi_standard_pred = bool(standard_pred)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def get_fps(self):
        with self._lock:
            return self._fps

    def _run(self):
        cap = None
        last_frame_time = None
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
                sahi_enabled = self.sahi_enabled
                slice_size = self.sahi_slice_size
                overlap_ratio = self.sahi_overlap_ratio
                standard_pred = self.sahi_standard_pred
                batch_size = self.batch_size

            with self._model_lock:
                model = self.model
                sahi_model = self.sahi_model

            if sahi_enabled:
                detections = self._detect_sahi(
                    frame, sahi_model, target_ids, slice_size, overlap_ratio, standard_pred, batch_size
                )
            else:
                detections = self._detect_yolo(frame, model, target_ids)

            for x1, y1, x2, y2, cls_id, conf in detections:
                label = f"{self.class_names[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            now = time.monotonic()
            instant_fps = 1 / (now - last_frame_time) if last_frame_time else 0.0
            last_frame_time = now

            with self._lock:
                if ok:
                    self._latest_jpeg = jpeg.tobytes()
                self._fps = self._fps * 0.9 + instant_fps * 0.1 if self._fps else instant_fps

        if cap:
            cap.release()

    def _detect_yolo(self, frame, model, target_ids):
        results = model.predict(frame, classes=target_ids, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append((x1, y1, x2, y2, cls_id, conf))
        return detections

    def _detect_sahi(self, frame, sahi_model, target_ids, slice_size, overlap_ratio, standard_pred, batch_size):
        target_set = set(target_ids)
        result = get_sliced_prediction(
            frame,
            sahi_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            perform_standard_pred=standard_pred,
            batch_size=batch_size,
            verbose=0,
        )
        detections = []
        for pred in result.object_prediction_list:
            cls_id = pred.category.id
            if cls_id not in target_set:
                continue
            x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
            detections.append((x1, y1, x2, y2, cls_id, pred.score.value))
        return detections

    def get_jpeg(self):
        with self._lock:
            return self._latest_jpeg
