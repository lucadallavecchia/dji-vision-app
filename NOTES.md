# Note di progetto / decisioni prese

Contesto per riprendere il lavoro senza dover rispiegare tutto da capo.

## Obiettivo
App che riceve lo streaming RTMP dal DJI Mini 3 Pro (via DJI Fly su smartphone/tablet),
esegue detection di persone/animali con YOLO, mostra il video con i bounding box.
Deve avere l'aspetto di un'app desktop (Mac/Windows) e essere visibile anche da altri
dispositivi (iOS, Android, altri computer) via browser sulla stessa rete.

## Decisioni architetturali

- **MediaMTX bundled, non installato a parte**: `main.py` avvia MediaMTX come subprocess
  gestito internamente (binario in `mediamtx/`), così l'utente finale non deve configurarlo
  manualmente. Config in `mediamtx/mediamtx.yml`, path fisso `drone`.
- **pywebview per l'aspetto "app desktop"**: la UI è in realtà una pagina web (FastAPI +
  HTML), ma pywebview la incapsula in una finestra nativa. Stessa pagina raggiungibile da
  altri dispositivi via browser su `http://<ip-mac>:8000`.
- **Un solo thread di detection, N spettatori**: `detection.py` gira in un thread unico che
  legge da RTSP (`rtsp://127.0.0.1:8554/drone`, esposto da MediaMTX), fa inference YOLO una
  volta sola, salva l'ultimo frame annotato in memoria. Ogni client HTTP connesso a `/stream`
  riceve solo una copia dell'ultimo frame (MJPEG) — la detection NON viene rifatta per ogni
  spettatore. Il collo di bottiglia con più dispositivi collegati è la banda di rete in
  uscita dal Mac (ogni client = una copia extra dello stream), non la CPU.
- **Classi rilevate**: person, dog, cat, bird, horse, sheep, cow (modificabile in
  `detection.py`, `TARGET_CLASSES`). Modello: `yolo11n.pt` (nano, leggero, scaricato
  automaticamente al primo avvio da ultralytics).

## Setup ambiente (fatto)

- Progetto aperto in **PyCharm** (non IntelliJ Ultimate — l'utente ha PyCharm Community).
- venv creato in `.venv/`, Base Python: sistema (3.14, Apple Silicon M3 Pro).
- Dipendenze installate da `requirements.txt` via il bottone "Pip Update from requirements.txt".
- MediaMTX: scaricato `mediamtx_v1.20.1_darwin_arm64.tar.gz` da GitHub releases (bluenviron/mediamtx),
  estratto, l'eseguibile `mediamtx` spostato in `mediamtx/mediamtx`, reso eseguibile con `chmod +x`.

## Problemi incontrati e risolti

1. **macOS Gatekeeper bloccava `mediamtx`** ("Apple could not verify..."). Risolto con:
   ```bash
   xattr -d com.apple.quarantine mediamtx/mediamtx
   ```
2. **"Whitelabel Error Page" nel browser**: non è un bug dell'app — era un typo, l'utente
   aveva digitato la porta **8080** (dove probabilmente gira un servizio Java/Spring Boot
   locale) invece della porta corretta **8000** usata da questa app.

## Stato attuale

- App funzionante end-to-end sul Mac: MediaMTX + server FastAPI + finestra desktop pywebview.
- Accesso da altri dispositivi (iPad) sulla stessa Wi-Fi confermato funzionante su
  `http://<ip-lan-del-mac>:8000`.
- **Non ancora testato con il drone reale** — prossimo passo: configurare DJI Fly con
  l'URL RTMP mostrato in console/app all'avvio (`rtmp://<ip-lan-del-mac>:1935/drone`) e
  verificare che i bounding box compaiano davvero sul video live.

## Possibili prossimi passi (non ancora fatti)

- Tuning latenza (buffering OpenCV/RTSP) se il video arriva troppo in ritardo.
- Processare 1 frame ogni N invece di tutti, se la CPU va sotto sforzo.
- Packaging con PyInstaller in un eseguibile singolo per distribuzione senza Python installato.
- Eventuale QR code nella UI per collegarsi più comodamente da mobile.
- Il progetto non è ancora un repository git.
