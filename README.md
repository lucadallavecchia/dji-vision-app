# DJI Vision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

App desktop (Mac/Windows) che riceve lo streaming RTMP da DJI Fly (drone DJI Mini 3 Pro),
esegue detection di persone/animali (YOLO) e mostra il video con i bounding box, accessibile
anche da altri dispositivi (iOS, Android, altri computer) via browser sulla stessa rete.

## Setup

1. Crea un virtualenv e installa le dipendenze:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # su Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Scarica il binario di [MediaMTX](https://github.com/bluenviron/mediamtx/releases) per il
   tuo sistema operativo (non è incluso nel repo, va scaricato manualmente — vedi `.gitignore`).

   **Mac (Apple Silicon)**, dalla root del progetto:

   ```bash
   curl -L -o mediamtx.tar.gz \
     https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_v1.20.1_darwin_arm64.tar.gz
   tar -xzf mediamtx.tar.gz mediamtx -C mediamtx/
   chmod +x mediamtx/mediamtx
   rm mediamtx.tar.gz
   ```

   Su Mac, al primo avvio Gatekeeper potrebbe bloccare il binario ("Apple could not
   verify..."); in tal caso rimuovi l'attributo di quarantena:

   ```bash
   xattr -d com.apple.quarantine mediamtx/mediamtx
   ```

   **Mac (Intel)**: usa l'asset `mediamtx_v1.20.1_darwin_amd64.tar.gz` nello stesso modo.

   **Windows**: scarica l'asset `mediamtx_v1.20.1_windows_amd64.zip` dalla pagina
   [releases](https://github.com/bluenviron/mediamtx/releases), estrai `mediamtx.exe` e
   mettilo in `mediamtx/mediamtx.exe`.

   > Verifica sempre l'ultima versione disponibile nella pagina releases: i comandi sopra
   > puntano alla v1.20.1, che potrebbe non essere più la più recente.

3. Avvia l'app:

   ```bash
   python main.py
   ```

   Al primo avvio il modello YOLO (`yolo11n.pt`) viene scaricato automaticamente
   (serve connessione internet).

4. Nel terminale vedrai l'URL RTMP da configurare in DJI Fly, es:

   ```
   DJI Fly deve pubblicare su: rtmp://192.168.1.23:1935/drone
   Altri dispositivi possono guardare su: http://192.168.1.23:8000
   ```

   Su DJI Fly (smartphone/tablet collegato al drone), attiva lo streaming live
   RTMP personalizzato e inserisci quell'URL.

5. Si apre una finestra desktop con il video annotato. Per guardarlo da un altro
   dispositivo (telefono, tablet, altro PC) sulla stessa rete Wi-Fi, apri
   `http://<ip-stampato>:8000` nel browser.

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
