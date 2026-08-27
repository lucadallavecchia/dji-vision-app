import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from detection import VideoProcessor
from utils import get_lan_ip

STATIC_DIR = Path(__file__).parent / "static"
processor = VideoProcessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    processor.start()
    yield
    processor.stop()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/info")
def info():
    return {"rtmp_url": f"rtmp://{get_lan_ip()}:1935/drone"}


class ClassesPayload(BaseModel):
    classes: list[str]


@app.get("/classes")
def get_classes():
    return {
        "available": processor.get_available_classes(),
        "selected": processor.get_target_classes(),
    }


@app.post("/classes")
def set_classes(payload: ClassesPayload):
    processor.set_target_classes(payload.classes)
    return {"selected": processor.get_target_classes()}


class SahiPayload(BaseModel):
    enabled: bool


@app.get("/sahi")
def get_sahi():
    return {"enabled": processor.get_sahi_enabled()}


@app.post("/sahi")
def set_sahi(payload: SahiPayload):
    processor.set_sahi_enabled(payload.enabled)
    return {"enabled": processor.get_sahi_enabled()}


@app.get("/fps")
def get_fps():
    return {"fps": round(processor.get_fps(), 1)}


class SahiConfigPayload(BaseModel):
    slice_size: int | None = None
    overlap_ratio: float | None = None
    standard_pred: bool | None = None


@app.get("/sahi-config")
def get_sahi_config():
    return processor.get_sahi_config()


@app.post("/sahi-config")
def set_sahi_config(payload: SahiConfigPayload):
    processor.set_sahi_config(
        slice_size=payload.slice_size,
        overlap_ratio=payload.overlap_ratio,
        standard_pred=payload.standard_pred,
    )
    return processor.get_sahi_config()


class DevicePayload(BaseModel):
    device: str


@app.get("/device")
def get_device():
    return processor.get_device_status()


@app.post("/device")
def set_device(payload: DevicePayload):
    processor.set_processing_device(payload.device)
    return processor.get_device_status()


def mjpeg_generator():
    boundary = b"--frame"
    while True:
        jpeg = processor.get_jpeg()
        if jpeg is not None:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.05)


@app.get("/stream")
def stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
