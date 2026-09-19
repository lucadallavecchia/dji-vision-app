import argparse
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn
import webview

from utils import get_lan_ip

BASE_DIR = Path(__file__).parent
MEDIAMTX_DIR = BASE_DIR / "mediamtx"
CONFIG_PATH = MEDIAMTX_DIR / "mediamtx.yml"
DEFAULT_TEST_VIDEO = BASE_DIR / "test-data" / "aereo_test.mp4"


def mediamtx_binary_path() -> Path:
    name = "mediamtx.exe" if platform.system() == "Windows" else "mediamtx"
    return MEDIAMTX_DIR / name


def start_mediamtx() -> subprocess.Popen:
    binary = mediamtx_binary_path()
    if not binary.exists():
        raise FileNotFoundError(
            f"Binario mediamtx non trovato in {binary}.\n"
            "Scaricalo da https://github.com/bluenviron/mediamtx/releases "
            "e mettilo nella cartella mediamtx/."
        )
    return subprocess.Popen([str(binary), str(CONFIG_PATH)], cwd=MEDIAMTX_DIR)


def start_server():
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="warning")


def start_test_publisher(video_path: Path) -> subprocess.Popen:
    """Pubblica un video locale in loop sull'endpoint RTMP, al posto del drone reale."""
    return subprocess.Popen(
        [
            "ffmpeg", "-re", "-stream_loop", "-1", "-i", str(video_path),
            "-c", "copy", "-f", "flv", "rtmp://127.0.0.1:1935/drone",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_server(url: str, timeout: float = 60) -> None:
    """Attende che il server risponda, invece di una sleep fissa: il caricamento
    dei modelli YOLO/SAHI in server.py può richiedere qualche secondo, e pywebview
    non ritenta la navigazione se apre la finestra troppo presto (pagina bianca)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.3)
    raise TimeoutError(f"Server non raggiungibile su {url} dopo {timeout}s")


def parse_args():
    parser = argparse.ArgumentParser(description="DJI Vision")
    parser.add_argument(
        "--test",
        nargs="?",
        const=str(DEFAULT_TEST_VIDEO),
        default=None,
        metavar="VIDEO_PATH",
        help="Modalità test: pubblica un video locale al posto del drone reale "
        f"(default: {DEFAULT_TEST_VIDEO})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    test_video_path = None
    if args.test is not None:
        test_video_path = Path(args.test)
        if not test_video_path.exists():
            sys.exit(f"Errore: video di test non trovato: {test_video_path}")
        if shutil.which("ffmpeg") is None:
            sys.exit(
                "Errore: ffmpeg non trovato, necessario per la modalità --test.\n"
                "Installalo con `brew install ffmpeg` (Mac) o vedi https://ffmpeg.org/download.html."
            )

    ip = get_lan_ip()
    if test_video_path:
        print(f"Modalità TEST: pubblico '{test_video_path}' al posto del drone reale")
    else:
        print(f"DJI Fly deve pubblicare su: rtmp://{ip}:1935/drone")
    print(f"Altri dispositivi possono guardare su: http://{ip}:8000")

    mediamtx_proc = start_mediamtx()
    time.sleep(1)

    test_publisher_proc = None
    try:
        test_publisher_proc = start_test_publisher(test_video_path) if test_video_path else None

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        wait_for_server("http://127.0.0.1:8000")

        webview.create_window("DJI Vision", "http://127.0.0.1:8000", width=1000, height=700)
        webview.start()
    finally:
        if test_publisher_proc:
            test_publisher_proc.terminate()
        mediamtx_proc.terminate()

    sys.exit(0)


if __name__ == "__main__":
    main()
