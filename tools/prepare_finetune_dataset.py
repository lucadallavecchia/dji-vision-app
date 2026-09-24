"""
Prepara il dataset unificato per il fine-tuning di yolo11n sulla detection di
persone in immagini aeree/da disastro: unisce VisDrone-DET (persone reali,
riprese aeree) e C2A (persone sintetiche su sfondi di disastro), entrambi
ridotti a classe singola "person" (0), in un'unica struttura YOLO pronta per
`yolo detect train`.

Prerequisiti: VisDrone2019-DET-{train,val}.zip e c2a-dataset.zip estratti sotto
test-data/ (lo script li scompatta automaticamente se trova gli zip).

Uso (dalla root del progetto):
    python tools/prepare_finetune_dataset.py
"""
import shutil
import zipfile
from pathlib import Path

from PIL import Image

TEST_DATA_DIR = Path(__file__).parent.parent / "test-data"
OUTPUT_DIR = TEST_DATA_DIR / "finetune-dataset"

VISDRONE_PERSON_CATEGORIES = {1, 2}  # VisDrone: 1=pedestrian, 2=people


def maybe_unzip(zip_name, extracted_check):
    if extracted_check.exists():
        return
    zip_path = TEST_DATA_DIR / zip_name
    if not zip_path.exists():
        return
    print(f"Estraggo {zip_name}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(TEST_DATA_DIR)


def convert_visdrone_det(split, source_dir, images_out, labels_out):
    """Converte le annotazioni VisDrone-DET in YOLO classe singola 'person', prefisso 'visdrone_'."""
    images_dir = source_dir / "images"
    annotations_dir = source_dir / "annotations"
    if not images_dir.exists() or not annotations_dir.exists():
        print(f"  VisDrone-DET {split}: cartella non trovata in {source_dir}, salto")
        return 0

    count = 0
    for ann_path in sorted(annotations_dir.glob("*.txt")):
        img_path = images_dir / ann_path.with_suffix(".jpg").name
        if not img_path.exists():
            continue
        width, height = Image.open(img_path).size
        lines = []
        for row in ann_path.read_text().strip().splitlines():
            parts = row.split(",")
            if len(parts) < 8:
                continue
            x, y, w, h, score, category = (int(p) for p in parts[:6])
            if score == 0 or category not in VISDRONE_PERSON_CATEGORIES:
                continue
            x_center, y_center = (x + w / 2) / width, (y + h / 2) / height
            w_norm, h_norm = w / width, h / height
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        new_name = f"visdrone_{ann_path.stem}"
        shutil.copy(img_path, images_out / f"{new_name}.jpg")
        (labels_out / f"{new_name}.txt").write_text("".join(lines))
        count += 1
    print(f"  VisDrone-DET {split}: {count} immagini convertite")
    return count


def convert_c2a(split, source_dir, images_out, labels_out):
    """Copia immagini/label di C2A (già YOLO, classe singola 'human'), prefisso 'c2a_'."""
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        print(f"  C2A {split}: cartella non trovata in {source_dir}, salto")
        return 0

    count = 0
    for img_path in sorted(images_dir.iterdir()):
        label_path = labels_dir / img_path.with_suffix(".txt").name
        if not label_path.exists():
            continue
        new_name = f"c2a_{img_path.stem}"
        shutil.copy(img_path, images_out / f"{new_name}{img_path.suffix}")
        shutil.copy(label_path, labels_out / f"{new_name}.txt")
        count += 1
    print(f"  C2A {split}: {count} immagini copiate")
    return count


def find_c2a_root():
    """Cerca la cartella che contiene train/val con images/+labels/ dentro test-data/."""
    for labels_dir in TEST_DATA_DIR.rglob("labels"):
        split_dir = labels_dir.parent
        if (split_dir / "images").exists() and split_dir.name in ("train", "val", "test"):
            return split_dir.parent
    return None


def main():
    maybe_unzip("VisDrone2019-DET-train.zip", TEST_DATA_DIR / "VisDrone2019-DET-train")
    maybe_unzip("VisDrone2019-DET-val.zip", TEST_DATA_DIR / "VisDrone2019-DET-val")
    maybe_unzip("c2a-dataset.zip", TEST_DATA_DIR / "c2a-dataset")

    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    c2a_root = find_c2a_root()
    if c2a_root:
        print(f"C2A trovato in: {c2a_root}")
    else:
        print("ATTENZIONE: struttura C2A (train/val con images/+labels/) non trovata sotto test-data/.")

    totals = {"train": 0, "val": 0}
    for split in ("train", "val"):
        images_out = OUTPUT_DIR / "images" / split
        labels_out = OUTPUT_DIR / "labels" / split

        visdrone_source = TEST_DATA_DIR / f"VisDrone2019-DET-{split}"
        totals[split] += convert_visdrone_det(split, visdrone_source, images_out, labels_out)

        if c2a_root:
            totals[split] += convert_c2a(split, c2a_root / split, images_out, labels_out)

    data_yaml = OUTPUT_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {OUTPUT_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['person']\n"
    )
    print(f"\nTotale train: {totals['train']} immagini, val: {totals['val']} immagini")
    print(f"data.yaml scritto in {data_yaml}")


if __name__ == "__main__":
    main()
