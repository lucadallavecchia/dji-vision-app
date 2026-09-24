import argparse
import platform
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import certifi
import uvicorn
import webview

from utils import get_lan_ip

BASE_DIR = Path(__file__).parent
MEDIAMTX_DIR = BASE_DIR / "mediamtx"
CONFIG_PATH = MEDIAMTX_DIR / "mediamtx.yml"
DEFAULT_TEST_VIDEO = BASE_DIR / "test-data" / "aereo_test.mp4"

# Versione di MediaMTX scaricata automaticamente al primo avvio (vedi ensure_mediamtx).
# Va aggiornata a mano di tanto in tanto controllando l'ultima release su:
# https://github.com/bluenviron/mediamtx/releases
MEDIAMTX_VERSION = "v1.21.1"
MEDIAMTX_RELEASE_BASE = f"https://github.com/bluenviron/mediamtx/releases/download/{MEDIAMTX_VERSION}"


def mediamtx_binary_path() -> Path:
    name = "mediamtx.exe" if platform.system() == "Windows" else "mediamtx"
    return MEDIAMTX_DIR / name


def _mediamtx_asset_name() -> str | None:
    """Nome dell'asset da scaricare dalle release di MediaMTX per la piattaforma
    corrente, o None se non riconosciuta (l'utente dovrà scaricarlo a mano)."""
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin":
        arch = "arm64" if machine == "arm64" else "amd64"
        return f"mediamtx_{MEDIAMTX_VERSION}_darwin_{arch}.tar.gz"

    if system == "Windows":
        return f"mediamtx_{MEDIAMTX_VERSION}_windows_amd64.zip"

    if system == "Linux":
        arch_map = {
            "x86_64": "amd64",
            "aarch64": "arm64",
            "arm64": "arm64",
            "armv7l": "armv7",
            "armv6l": "armv6",
        }
        arch = arch_map.get(machine)
        return f"mediamtx_{MEDIAMTX_VERSION}_linux_{arch}.tar.gz" if arch else None

    return None


def ensure_mediamtx() -> Path:
    """Scarica automaticamente il binario di MediaMTX per la piattaforma corrente al
    primo avvio, così l'utente non deve procurarselo a mano: finisce in mediamtx/
    (escluso da git, vedi .gitignore) e viene riusato agli avvii successivi. Estrae
    solo il binario, senza toccare mediamtx.yml (che è la nostra config, non quella
    di default contenuta nell'archivio)."""
    binary = mediamtx_binary_path()
    if binary.exists():
        return binary

    asset_name = _mediamtx_asset_name()
    if asset_name is None:
        raise FileNotFoundError(
            "Piattaforma non riconosciuta automaticamente per il download di MediaMTX "
            f"({platform.system()} {platform.machine()}).\n"
            "Scaricalo manualmente da https://github.com/bluenviron/mediamtx/releases "
            f"e mettilo in {binary}."
        )

    url = f"{MEDIAMTX_RELEASE_BASE}/{asset_name}"
    print(f"Primo avvio: scarico MediaMTX ({asset_name})...")
    MEDIAMTX_DIR.mkdir(exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            archive_path = tmp_dir / asset_name
            # SSL context esplicito con la CA bundle di certifi: le installazioni
            # Python "vanilla" (es. python.org su Mac) spesso non hanno i certificati
            # di sistema configurati e urlretrieve fallirebbe con CERTIFICATE_VERIFY_FAILED.
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(url, context=ssl_context) as response, open(archive_path, "wb") as f:
                shutil.copyfileobj(response, f)

            extract_dir = tmp_dir / "extracted"
            if asset_name.endswith(".zip"):
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall(extract_dir)
            else:
                with tarfile.open(archive_path) as tf:
                    tf.extractall(extract_dir)

            extracted_binary = extract_dir / binary.name
            if not extracted_binary.exists():
                raise FileNotFoundError(f"{binary.name} non trovato nell'archivio {asset_name}.")
            shutil.move(str(extracted_binary), str(binary))
    except (urllib.error.URLError, OSError) as e:
        raise FileNotFoundError(
            f"Download automatico di MediaMTX fallito ({e}).\n"
            "Verifica la connessione a internet, oppure scaricalo manualmente da "
            f"https://github.com/bluenviron/mediamtx/releases e mettilo in {binary}."
        ) from e

    if platform.system() != "Windows":
        binary.chmod(binary.stat().st_mode | 0o111)
    if platform.system() == "Darwin":
        # Rimuove l'attributo di quarantena, altrimenti Gatekeeper blocca il binario
        # al primo avvio ("Apple could not verify...").
        subprocess.run(["xattr", "-d", "com.apple.quarantine", str(binary)], stderr=subprocess.DEVNULL)

    return binary


def start_mediamtx() -> subprocess.Popen:
    binary = ensure_mediamtx()
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

    def cleanup(*_args):
        # Chiamata sia dall'evento "closed" della finestra sia dal finally sotto: guardata
        # con poll() perché può girare due volte (rischio, se non guardata, di mandare
        # SIGTERM a un PID nel frattempo riciclato dall'OS per un processo estraneo).
        if test_publisher_proc and test_publisher_proc.poll() is None:
            test_publisher_proc.terminate()
        if mediamtx_proc.poll() is None:
            mediamtx_proc.terminate()

    try:
        test_publisher_proc = start_test_publisher(test_video_path) if test_video_path else None

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        wait_for_server("http://127.0.0.1:8000")

        window = webview.create_window("DJI Vision", "http://127.0.0.1:8000", width=1000, height=700)
        # Rete di sicurezza: su macOS abbiamo osservato webview.start() terminare il
        # processo (es. su SIGINT mentre gira il run loop nativo di Cocoa) senza lasciar
        # girare il finally sottostante, orfanando MediaMTX. L'evento "closed" della
        # finestra passa invece dal normale event loop di pywebview e scatta in modo
        # affidabile alla chiusura, quindi facciamo la cleanup anche lì.
        window.events.closed += cleanup
        webview.start()
    finally:
        cleanup()

    sys.exit(0)


if __name__ == "__main__":
    main()
