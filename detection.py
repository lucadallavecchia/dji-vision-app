import threading
import time
from pathlib import Path

import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

RTSP_URL = "rtsp://127.0.0.1:8554/drone"
DEFAULT_TARGET_CLASSES = {"person", "dog", "cat", "bird", "horse", "sheep", "cow"}
MODEL_VARIANTS = {
    "general": "yolo11n.pt",
    "aerial-person": "yolo11n-aerial-person.pt",
}
DEFAULT_MODEL_VARIANT = "aerial-person"
SAHI_SLICE_SIZE = 750
SAHI_OVERLAP_RATIO = 0.2
SAHI_STANDARD_PRED = False
SAHI_SLICE_SIZE_RANGE = (128, 1536)
SAHI_OVERLAP_RATIO_RANGE = (0.0, 0.9)
DETECT_EVERY_DEFAULT = 2
DETECT_EVERY_RANGE = (1, 30)

CPU_BATCH_SIZE = 1
GPU_BATCH_SIZE = 8


def _detect_gpu_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def _variant_for_model_path(model_path):
    """Nome della variante nota (MODEL_VARIANTS) corrispondente a un path di modello,
    o None se è un .pt arbitrario non tra le varianti (es. passato da tools/evaluate.py)."""
    return next((k for k, v in MODEL_VARIANTS.items() if v == model_path), None)


class VideoProcessor:
    def __init__(self, model_path=None):
        """model_path: .pt da caricare all'avvio invece della variante di default. Pensato
        per tools/evaluate.py, che vuole valutare modelli arbitrari (es. yolo11s.pt) senza
        passare dalle due varianti fisse esposte in UI (general/aerial-person)."""
        self.gpu_torch_device = _detect_gpu_torch_device()

        self._model_lock = threading.Lock()
        self.model_path = model_path or self._resolve_default_model_path()
        self.model, self.sahi_model = self._build_models("cpu", self.model_path)
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
        self.detect_every = DETECT_EVERY_DEFAULT

        self.processing_device = "cpu"
        self.model_variant = _variant_for_model_path(self.model_path)
        self.batch_size = CPU_BATCH_SIZE
        self.reloading = False
        self.reload_error = None
        self._last_detections = []

    @staticmethod
    def _resolve_default_model_path():
        """Modello con cui parte l'app. DEFAULT_MODEL_VARIANT può puntare a un checkpoint
        custom (es. yolo11n-aerial-person.pt, fine-tuned in casa, non su Ultralytics Hub):
        se manca sul disco — tipicamente al primo avvio su una macchina diversa da quella
        di training/sviluppo — niente auto-download è possibile per quel file, quindi
        ripieghiamo sul modello "general", che Ultralytics scarica da sé in automatico."""
        default_path = MODEL_VARIANTS[DEFAULT_MODEL_VARIANT]
        if DEFAULT_MODEL_VARIANT != "general" and not Path(default_path).exists():
            fallback = MODEL_VARIANTS["general"]
            print(
                f"ATTENZIONE: modello di default '{default_path}' non trovato sul disco, "
                f"uso '{fallback}' al suo posto (passa a '{DEFAULT_MODEL_VARIANT}' dalla UI "
                "una volta procurato il file)."
            )
            return fallback
        return default_path

    def _build_models(self, torch_device, model_name):
        model = YOLO(model_name)
        model.to(torch_device)
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_name,
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
                "gpu_backend": self.gpu_torch_device,  # "cuda" | "mps" | None
                "error": self.reload_error,
            }

    def set_processing_device(self, device):
        device = "gpu" if device == "gpu" else "cpu"
        if device == "gpu" and self.gpu_torch_device is None:
            return
        with self._config_lock:
            if device == self.processing_device or self.reloading:
                return
            self.reloading = True
        threading.Thread(target=self._reload_models, args=(device, self.model_path), daemon=True).start()

    def get_model_status(self):
        with self._config_lock:
            return {
                "variant": self.model_variant,
                "reloading": self.reloading,
                "available": list(MODEL_VARIANTS.keys()),
                "error": self.reload_error,
            }

    def set_model_variant(self, variant):
        if variant not in MODEL_VARIANTS:
            return
        with self._config_lock:
            if variant == self.model_variant or self.reloading:
                return
            self.reloading = True
        threading.Thread(target=self._reload_models, args=(self.processing_device, MODEL_VARIANTS[variant]), daemon=True).start()

    def _reload_models(self, device, model_path):
        torch_device = self.gpu_torch_device if device == "gpu" else "cpu"
        batch_size = GPU_BATCH_SIZE if device == "gpu" else CPU_BATCH_SIZE
        try:
            model, sahi_model = self._build_models(torch_device, model_path)
        except Exception as e:
            # Es. model_path punta a un checkpoint custom mancante sul disco (vedi
            # _resolve_default_model_path). Senza questo except, un'eccezione qui lascia
            # reloading=True per sempre: lo spinner nella UI gira all'infinito e i radio
            # restano disabilitati finché non si riavvia l'app.
            print(f"Cambio modello/device fallito ({model_path}, {device}): {e}")
            with self._config_lock:
                self.reloading = False
                self.reload_error = str(e)
            return
        class_names = model.names
        with self._model_lock:
            self.model = model
            self.sahi_model = sahi_model
        with self._config_lock:
            path_changed = model_path != self.model_path
            # Svuotata PRIMA di sostituire class_names: _run() legge entrambi senza lock
            # (per non bloccare lo streaming durante un reload), quindi se l'ordine fosse
            # invertito un frame potrebbe leggere class_names già nuovo ma detection
            # ancora vecchie, con cls_id che non esistono più nel nuovo modello (es. da
            # 80 classi a 1 sola) -> KeyError e thread di detection morto (video
            # congelato). Vedi anche il fallback in _run() più sotto, seconda rete di
            # sicurezza sulla stessa razza di problema.
            if path_changed:
                self._last_detections = []
            self.class_names = class_names
            self.processing_device = device
            self.model_path = model_path
            self.model_variant = _variant_for_model_path(model_path)
            self.batch_size = batch_size
            self.reload_error = None
            if path_changed:
                self.target_ids = [i for i, n in class_names.items() if n in DEFAULT_TARGET_CLASSES]
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

    def get_detect_every(self):
        with self._config_lock:
            return self.detect_every

    def set_detect_every(self, n):
        lo, hi = DETECT_EVERY_RANGE
        with self._config_lock:
            self.detect_every = max(lo, min(hi, int(n)))

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
        frame_counter = 0
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

            try:
                with self._config_lock:
                    target_ids = list(self.target_ids)
                    sahi_enabled = self.sahi_enabled
                    slice_size = self.sahi_slice_size
                    overlap_ratio = self.sahi_overlap_ratio
                    standard_pred = self.sahi_standard_pred
                    batch_size = self.batch_size
                    detect_every = self.detect_every

                frame_counter += 1
                if frame_counter % detect_every == 0:
                    with self._model_lock:
                        model = self.model
                        sahi_model = self.sahi_model

                    if sahi_enabled:
                        self._last_detections = self._detect_sahi(
                            frame, sahi_model, target_ids, slice_size, overlap_ratio, standard_pred, batch_size
                        )
                    else:
                        self._last_detections = self._detect_yolo(frame, model, target_ids)

                class_names = self.class_names
                for x1, y1, x2, y2, cls_id, conf in self._last_detections:
                    name = class_names.get(cls_id)
                    if name is None:
                        # Detection residua di un modello appena sostituito (cambio
                        # variante a metà giro) i cui id classe non esistono più nel
                        # nuovo class_names: la scartiamo invece di far esplodere il
                        # thread (vedi commento in _reload_models).
                        continue
                    label = f"{name} {conf:.2f}"
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
            except Exception as e:
                # Rete di sicurezza generale: qualunque errore imprevisto nell'elaborazione
                # di un frame non deve congelare per sempre lo streaming (il thread è
                # daemon e non viene mai riavviato da solo se muore). Si salta il frame e
                # si continua con il prossimo invece di uccidere il loop.
                print(f"Errore nell'elaborazione del frame, salto: {e}")

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
