import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn
import webview

from utils import get_lan_ip

BASE_DIR = Path(__file__).parent
MEDIAMTX_DIR = BASE_DIR / "mediamtx"
CONFIG_PATH = MEDIAMTX_DIR / "mediamtx.yml"


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


def main():
    ip = get_lan_ip()
    print(f"DJI Fly deve pubblicare su: rtmp://{ip}:1935/drone")
    print(f"Altri dispositivi possono guardare su: http://{ip}:8000")

    mediamtx_proc = start_mediamtx()
    time.sleep(1)

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1)

    webview.create_window("DJI Vision", "http://127.0.0.1:8000", width=1000, height=700)
    webview.start()

    mediamtx_proc.terminate()
    sys.exit(0)


if __name__ == "__main__":
    main()
