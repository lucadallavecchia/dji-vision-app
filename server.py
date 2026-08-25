import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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
