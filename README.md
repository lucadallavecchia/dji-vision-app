# DJI Vision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

App desktop (Mac, Windows, Linux) che riceve lo streaming RTMP da DJI Fly (drone DJI Mini 3
Pro), esegue detection di persone/animali (YOLO, CPU o GPU a seconda di cosa trova sulla
macchina) e mostra il video con i bounding box, accessibile anche da altri dispositivi
(iOS, Android, altri computer) via browser sulla stessa rete.

## Setup

1. Crea un virtualenv e installa le dipendenze:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # su Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Avvia l'app:

   ```bash
   python main.py
   ```

   Al primo avvio l'app scarica automaticamente (serve connessione internet):
   - il binario di [MediaMTX](https://github.com/bluenviron/mediamtx) giusto per la tua
     piattaforma (Mac Apple Silicon/Intel, Windows, Linux x86_64/arm64), niente da
     scaricare a mano;
   - il modello YOLO (`yolo11n.pt`).

   Se la piattaforma non viene riconosciuta (caso raro), l'app stampa un errore con le
   istruzioni per scaricare MediaMTX a mano da
   [releases](https://github.com/bluenviron/mediamtx/releases) e metterlo in `mediamtx/`.

   **Linux**: `pywebview` richiede un backend di sistema per la finestra (es.
   `webkit2gtk`), non installabile col solo `pip` — vedi la
   [documentazione pywebview](https://pywebview.flowrl.com/guide/installation.html) per il
   pacchetto giusto per la tua distro.

3. Nel terminale vedrai l'URL RTMP da configurare in DJI Fly, es:

   ```
   DJI Fly deve pubblicare su: rtmp://192.168.1.23:1935/drone
   Altri dispositivi possono guardare su: http://192.168.1.23:8000
   ```

   Su DJI Fly (smartphone/tablet collegato al drone), attiva lo streaming live
   RTMP personalizzato e inserisci quell'URL.

4. Si apre una finestra desktop con il video annotato. Per guardarlo da un altro
   dispositivo (telefono, tablet, altro PC) sulla stessa rete Wi-Fi, apri
   `http://<ip-stampato>:8000` nel browser.

## GPU

Nel menu "Modello di rilevamento" della UI puoi passare da CPU a GPU: l'app rileva da
sola il backend disponibile (mostrato tra parentesi, es. "GPU (Metal)" su Apple Silicon o
"GPU (CUDA)" su NVIDIA) e disabilita l'opzione se non c'è nessuna GPU utilizzabile.

Su **Mac (Apple Silicon)** funziona subito: `pip install -r requirements.txt` installa una
build di PyTorch con supporto Metal (MPS) già inclusa.

Su **Windows/Linux con GPU NVIDIA**, invece, `pip install -r requirements.txt` installa di
default una build di PyTorch **CPU-only** (è il comportamento standard di PyPI), quindi la
GPU non verrebbe usata anche se fisicamente presente. Per abilitarla, installa PyTorch con
supporto CUDA *prima* di installare le altre dipendenze — scegli il comando giusto per la
tua versione CUDA dalla [guida ufficiale PyTorch](https://pytorch.org/get-started/locally/),
ad es. per CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Senza GPU dedicata (o su CPU-only), l'app funziona comunque su CPU, semplicemente più lenta.

## Test senza drone reale

Per provare l'app (detection, SAHI, switch CPU/GPU) senza avere il drone a disposizione,
puoi avviarla in **modalità test**: pubblica in automatico un video locale sull'endpoint
RTMP al posto di DJI Fly. Richiede [ffmpeg](https://ffmpeg.org/) installato
(`brew install ffmpeg` su Mac).

```bash
python main.py --test                        # usa test-data/aereo_test.mp4 di default
python main.py --test path/al/tuo/video.mp4   # oppure un video a scelta
```

Il comportamento normale (`python main.py`, in attesa del drone reale) resta invariato.

**Procurarsi un video aereo di test**: un buon punto di partenza è il dataset
[VisDrone](https://github.com/VisDrone/VisDrone-Dataset) (task "Object Detection in
Videos", valset), che ha soggetti in movimento (persone, veicoli...) ripresi da drone.
Fornisce però sequenze di frame JPEG numerati invece di file video, quindi vanno prima
ricomposti in un `.mp4`:

```bash
ffmpeg -r 25 -i sequences/<nome_sequenza>/%07d.jpg \
  -c:v libx264 -pix_fmt yuv420p test-data/aereo_test.mp4
```

`%07d` presuppone nomi file a 7 cifre con zero padding (es. `0000001.jpg`) — verifica il
pattern nella cartella scaricata e correggilo se necessario. La cartella `test-data/` è
già esclusa da `.gitignore`, comoda per tenerci dataset/video di prova senza rischiare di
committarli.

## Note

- Tutti i dispositivi (host + spettatori) devono essere sulla stessa rete locale.
- Le classi rilevate di default sono: persona, cane, gatto, uccello, cavallo,
  pecora, mucca (modificabile in `detection.py`, `TARGET_CLASSES`).
- Per ridurre il carico CPU puoi processare 1 frame ogni N in `detection.py`.

## Per sviluppatori

- [`docs/NOTES.md`](docs/NOTES.md) — decisioni architetturali e stato di avanzamento del progetto.
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — risultati dei test di accuratezza (recall/precision) su riprese aeree.
- [`tools/evaluate.py`](tools/evaluate.py) — valutazione offline contro ground truth, indipendente dall'app live.
- [`tools/prepare_finetune_dataset.py`](tools/prepare_finetune_dataset.py) e [`tools/train_aerial_person.ipynb`](tools/train_aerial_person.ipynb) — pipeline di fine-tuning del modello aereo (dataset prep locale + training notebook per Kaggle).
