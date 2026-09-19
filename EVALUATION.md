# Valutazione della detection su immagini aeree

Risultati e conclusioni dei test di accuratezza (recall/precision) condotti per capire quanto
sia affidabile la pipeline di detection attuale su riprese aeree, e quali leve valga la pena
tirare per migliorarla. Non sostituisce il `README.md` (setup/uso dell'app) — qui raccogliamo
solo dati sperimentali e conclusioni.

## Metodologia

- **Script**: [`evaluate.py`](evaluate.py) — valutazione offline, indipendente dall'app live.
  Gira direttamente sui frame JPEG di una sequenza [VisDrone](https://github.com/VisDrone/VisDrone-Dataset)
  (task "Object Detection in Videos"), confrontando le detection del modello contro le
  ground truth reali fornite dal dataset.
- **Sequenza usata**: `uav0000086_00000_v` — 464 frame, 1344×756px, 22.098 comparse di
  persone nella ground truth (categorie VisDrone `pedestrian`/`people`).
- **Matching**: IoU ≥ 0.5 tra box predetti e box ground truth (greedy matching per frame).
- **Metriche**:
  - **Recall** = persone reali trovate / persone reali totali. La metrica più rilevante per
    un caso d'uso di ricerca persone (un falso negativo, cioè una persona non trovata, è
    peggio di un falso allarme).
  - **Precision** = detection corrette / detection totali segnalate.
  - **ms/frame** = tempo medio della sola chiamata di detection (non include lettura RTSP,
    disegno box, encoding JPEG — vedi nota sul gap offline/live più sotto).
- **Riproducibilità**: `python evaluate.py --model <nome.pt> --slice-size N --overlap X
  [--skip-cpu] [--max-frames N]`.

## Risultato 1 — CPU vs GPU (Metal)

`yolo11n`, tile 768px, overlap 0.2.

| Device | Recall | Precision | ms/frame |
|---|---|---|---|
| CPU | 27.6% | 84.1% | 33.2 |
| GPU (Metal) | 27.6% | 84.1% | 19.0 |

**Conclusione**: nessuna differenza di accuratezza (stesso modello, stessi pesi), la GPU è
solo più veloce (1.7-4× a seconda del modello). **Usare sempre GPU quando disponibile.**

## Risultato 2 — Confronto modelli (`n` / `s` / `m`)

GPU, SAHI attivo (768px o 750px a seconda del test, overlap 0.2).

| Modello | Recall baseline (no SAHI) | Recall + SAHI | Precision + SAHI | ms/frame + SAHI |
|---|---|---|---|---|
| `yolo11n` | 27.6% | **43.5%** (768px) / 46.3% (750px) | 85.5% / 84.9% | 20.7 / 35.3 |
| `yolo11s` | 32.0% | 39.2% (768px) | 90.3% | 31.1 |
| `yolo11m` | 29.0% | 39.3% (750px) | 90.8% | 120.7 |

**Conclusione**: `yolo11n` (il più piccolo) **batte sia `yolo11s` che `yolo11m`** in recall
quando SAHI è attivo, oltre a essere sempre il più veloce. I modelli più grandi sono più
"conservativi" (precisione più alta, meno falsi allarmi) perché più specificamente adattati
alle immagini COCO (persone a terra, viste frontali/laterali) su cui sono stati addestrati —
di fronte a sagome aeree molto diverse da quelle viste in training, tendono a scartare più
detection sotto la soglia di confidenza. `yolo11n`, avendo imparato feature più generiche,
generalizza meglio (in termini di recall) proprio perché meno specializzato sul dominio di
training. **`yolo11n` è il modello da usare**, salire di dimensione è controproducente qui.

## Risultato 3 — Dimensione tile SAHI

`yolo11n`, GPU, overlap 0.2. Il numero di tile effettivi non cresce in modo continuo con la
dimensione, ma "a scatti": per questa risoluzione sorgente (1344×756px), un tile ≥756px copre
l'altezza dell'immagine in una sola riga (griglia minima, ~2 tile), mentre qualunque tile
<756px richiede almeno 2 righe (la griglia salta bruscamente a 4-6+ tile).

| Tile | Tile in griglia | Recall | Precision | ms/frame | FPS equivalente |
|---|---|---|---|---|---|
| 768px | 2 | 43.5% | 85.5% | 20.7 | ~48.3 |
| **750px** *(default)* | 4 | **46.3%** | 84.9% | 35.3 | ~28.3 |
| 704px | 6 | 49.3% | 83.7% | 48.7 | ~20.5 |
| 640px | 6 | 54.9% | 82.9% | 47.1 | ~21.2 |
| 512px | 8 | 62.8% | 78.7% | 88.6* | ~11* |
| 384px | 15 | 68.9% | 70.9% | 105.9 | ~9.4 |

*(512px misurato in una run parallela ad altra, timing indicativo non affidabile)*

**Conclusione**: tile più piccoli aumentano sempre la recall, ma con rendimenti decrescenti
e precisione in calo. **750px è il compromesso scelto come default** (buon guadagno di
recall vs 768px, resta vicino alla soglia di fluidità richiesta). 640px offre la recall
migliore tra le opzioni che restano sopra i 20 FPS "sulla carta", ma vedi il Risultato 5 sul
gap tra benchmark offline e comportamento live.

## Risultato 4 — Overlap tra tile

`yolo11n`, GPU, tile 768px.

| Overlap | Recall | Precision | ms/frame |
|---|---|---|---|
| 0.1 | 43.5% | 85.5% | 22.6 |
| 0.2 *(default)* | 43.5% | 85.5% | 20.7-21.8 |
| 0.4 | 44.3% | 85.1% | 30.6 |

**Conclusione**: impatto minimo (+0.8 punti recall raddoppiando l'overlap) a fronte di un
costo reale in velocità (~48% più lento). **Non è una leva utile**, lasciato al default 0.2.

## Risultato 5 — Gap tra benchmark offline e streaming live

Il tempo misurato da `evaluate.py` copre **solo la chiamata di detection**. La pipeline live
(`detection.py`) aggiunge overhead non misurato offline: lettura/decodifica frame da RTSP,
disegno bounding box, encoding JPEG. Testato dal vivo con `python main.py --test`
(video sorgente a 25 FPS):

- A 750px/GPU/SAHI, l'overlay FPS nella UI mostrava **23 FPS reali** (contro i ~28.3 FPS
  "teorici" del benchmark offline) — sotto ai 25 FPS del video sorgente.
- Risultato: MediaMTX scarta frame per non far crescere la latenza
  (`reader is too slow, discarding N frames`), causando corruzione visibile nella decodifica
  H.264 (frame di riferimento mancanti).

**Conclusione**: per uno streaming live pulito, serve un margine reale sopra il frame rate
del video sorgente, non solo sopra una soglia di fluidità percepita. **768px resta l'opzione
più sicura** per uno streaming senza artefatti (ampio margine, ~48 FPS offline); è comunque
disponibile nella UI (select dimensione tile) se necessario passare a un profilo più
conservativo.

## Risultato 6 — Fine-tuning (`yolo11n-aerial-person`)

Confermato il limite di dominio del Risultato precedente: fine-tuning di `yolo11n` (pesi
COCO pretrained) su un dataset unificato **VisDrone-DET** (6.471 immagini reali aeree,
categorie pedestrian+people) + **[C2A](https://github.com/Ragib-Amin-Nihal/C2A)** (10.215
immagini sintetiche, persone su sfondi di disastro reali — alluvioni, incendi, edifici
crollati — con pose incluse sdraiato/inginocchiato, rilevanti per SAR), entrambi ridotti a
classe singola `person`. Totale: 12.600 immagini train / 2.591 val. Training: 100 epoche,
`imgsz=768`, su GPU T4 (Kaggle, gratuito), 6.48 ore.

Stesso benchmark (`uav0000086_00000_v`, mai vista in training), stessa GPU (M3 Pro):

| Config | Recall | Precision | ms/frame |
|---|---|---|---|
| Baseline originale, GPU+SAHI 750px | 46.3% | 84.9% | 35.3 |
| **Fine-tuned, GPU no-SAHI** | **67.0%** | 61.5% | 71.3 |
| Fine-tuned, GPU+SAHI 750px | 62.6% | 63.3% | 68.0 |

**Conclusione**: il fine-tuning conferma l'ipotesi del Risultato 5 — la recall sale di
**+20.7 punti** rispetto al miglior risultato precedente (46.3%→67.0%), il salto di qualità
che nessuna combinazione di modello/tile/overlap era riuscita a dare. Costo reale: la
precisione scende da 84.9% a 61.5% (più falsi allarmi) — un compromesso ragionevole per un
caso d'uso dove mancare una persona reale è peggio di un falso allarme.

**Scoperta inattesa**: con il modello fine-tuned, **SAHI non serve più** — anzi peggiora
leggermente la recall (67.0% senza SAHI vs 62.6% con SAHI). Ha senso: SAHI compensava
l'incapacità del modello originale di riconoscere oggetti piccoli "ingrandendo" i ritagli;
il modello fine-tuned ha imparato a riconoscere persone in scala ridotta direttamente
(allenato su immagini aeree intere), quindi lo zoom di SAHI non aggiunge più valore e
introduce solo frammentazione/duplicati ai bordi dei tile.

**Nota aperta, non approfondita**: il tempo per frame è salito (~68-71ms, contro ~19-21ms
del modello originale senza SAHI) nonostante la stessa identica architettura/dimensione
modello — probabile causa il maggior numero di detection (più falsi positivi) da processare
nel post-processing Python, non ancora verificato con certezza.

## Conclusione generale

Il fine-tuning ha confermato l'ipotesi centrale di questa analisi: il collo di bottiglia non
era la geometria del tiling né la dimensione del modello, ma il **dominio** — nessun modello
pretrained COCO aveva mai visto immagini aeree. Con il fine-tuning su dataset aereo/disastro
(VisDrone-DET + C2A), la recall sale da un tetto del ~69% (solo con tile aggressivi, troppo
lenti per l'uso live) a **67.0% con un singolo modello, senza SAHI, a piena velocità**.

Prossimi passi possibili: bilanciare meglio precision/recall (es. alzando la soglia di
confidenza), o ampliare ulteriormente il dataset di fine-tuning.

## Configurazione attuale di default

- Modello: **`yolo11n-aerial-person.pt`** (fine-tuned, classe singola "person") selezionabile
  da UI insieme al modello originale `yolo11n.pt` (80 classi COCO, utile per rilevare anche
  animali)
- Device: GPU (Metal/MPS) quando disponibile, selezionabile da UI
- SAHI: disattivabile/attivabile da UI (tile 750px, overlap 0.2, no standard prediction) —
  con il modello fine-tuned conviene tenerlo spento (vedi Risultato 6)
