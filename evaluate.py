"""
Valutazione offline della detection: confronta le predizioni del modello contro le
ground truth di una sequenza VisDrone, per misurare quante persone reali vengono
trovate (recall) nelle configurazioni CPU, GPU e GPU+SAHI.

Uso:
    python evaluate.py [--sequence uav0000086_00000_v] [--iou 0.5] [--max-frames N]
"""
import argparse
import time
from pathlib import Path

import cv2

import detection
from detection import VideoProcessor

VISDRONE_DIR = Path(__file__).parent / "test-data" / "VisDrone2019-VID-val"
PERSON_CATEGORIES = {1, 2}  # VisDrone: 1=pedestrian, 2=people


def load_ground_truth(annotation_path):
    """Ritorna {frame_index: [(x1, y1, x2, y2), ...]} per le categorie persona."""
    gt_by_frame = {}
    for line in annotation_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        frame_idx, _, x, y, w, h, score, category = (int(p) for p in parts[:8])
        if category not in PERSON_CATEGORIES or score == 0:
            continue
        gt_by_frame.setdefault(frame_idx, []).append((x, y, x + w, y + h))
    return gt_by_frame


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def match(pred_boxes, gt_boxes, iou_threshold):
    """Greedy matching per-frame; ritorna (true_positive, false_positive, false_negative)."""
    unmatched_gt = list(gt_boxes)
    tp = 0
    for pb in pred_boxes:
        best_idx, best_iou = None, 0.0
        for i, gb in enumerate(unmatched_gt):
            score = iou(pb, gb)
            if score > best_iou:
                best_idx, best_iou = i, score
        if best_idx is not None and best_iou >= iou_threshold:
            tp += 1
            unmatched_gt.pop(best_idx)
    fp = len(pred_boxes) - tp
    fn = len(unmatched_gt)
    return tp, fp, fn


def evaluate_config(processor, sahi_enabled, target_ids, frame_paths, gt_by_frame, iou_threshold, label):
    model = processor.model
    sahi_model = processor.sahi_model
    sahi_cfg = processor.get_sahi_config()
    batch_size = processor.batch_size

    total_tp = total_fp = total_fn = 0
    elapsed_total = 0.0
    for frame_path in frame_paths:
        frame_idx = int(frame_path.stem)
        gt_boxes = gt_by_frame.get(frame_idx, [])
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        t0 = time.monotonic()
        if sahi_enabled:
            detections = processor._detect_sahi(
                frame, sahi_model, target_ids,
                sahi_cfg["slice_size"], sahi_cfg["overlap_ratio"], sahi_cfg["standard_pred"],
                batch_size,
            )
        else:
            detections = processor._detect_yolo(frame, model, target_ids)
        elapsed_total += time.monotonic() - t0

        pred_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, cls_id, conf in detections]
        tp, fp, fn = match(pred_boxes, gt_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    avg_ms = (elapsed_total / len(frame_paths)) * 1000 if frame_paths else 0.0

    print(f"\n=== {label} ===")
    print(f"  Persone reali totali (GT): {total_tp + total_fn}")
    print(f"  True positive:  {total_tp}")
    print(f"  False negative: {total_fn}  (persone mancate)")
    print(f"  False positive: {total_fp}  (falsi allarmi)")
    print(f"  Recall:    {recall:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Tempo medio/frame: {avg_ms:.1f} ms")

    return {"label": label, "recall": recall, "precision": precision, "avg_ms": avg_ms}


def main():
    parser = argparse.ArgumentParser(description="Valuta la detection su una sequenza VisDrone")
    parser.add_argument("--sequence", default="uav0000086_00000_v")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None, help="limita il numero di frame (debug rapido)")
    parser.add_argument("--model", default=detection.MODEL_NAME, help="modello YOLO da valutare (es. yolo11s.pt)")
    parser.add_argument("--slice-size", type=int, default=detection.SAHI_SLICE_SIZE, help="dimensione tile SAHI (px)")
    parser.add_argument("--overlap", type=float, default=detection.SAHI_OVERLAP_RATIO, help="overlap tra tile SAHI (0-0.9)")
    parser.add_argument("--skip-cpu", action="store_true", help="salta la configurazione CPU baseline")
    args = parser.parse_args()

    detection.MODEL_NAME = args.model

    seq_dir = VISDRONE_DIR / "sequences" / args.sequence
    ann_path = VISDRONE_DIR / "annotations" / f"{args.sequence}.txt"
    if not seq_dir.exists() or not ann_path.exists():
        raise SystemExit(f"Sequenza non trovata: {seq_dir} / {ann_path}")

    frame_paths = sorted(seq_dir.glob("*.jpg"))
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    gt_by_frame = load_ground_truth(ann_path)

    total_gt_boxes = sum(len(v) for v in gt_by_frame.values())
    print(f"Modello: {args.model}")
    print(f"SAHI slice size: {args.slice_size}px, overlap: {args.overlap}")
    print(f"Sequenza: {args.sequence} ({len(frame_paths)} frame, {total_gt_boxes} bounding box persona nella GT)")

    processor = VideoProcessor()
    processor.set_sahi_config(slice_size=args.slice_size, overlap_ratio=args.overlap)
    target_ids = [i for i, n in processor.class_names.items() if n == "person"]

    results = []

    if not args.skip_cpu:
        results.append(evaluate_config(
            processor, False, target_ids, frame_paths, gt_by_frame, args.iou,
            "1) CPU, baseline (no SAHI)",
        ))

    if processor.gpu_torch_device:
        processor.set_processing_device("gpu")
        while processor.get_device_status()["reloading"]:
            time.sleep(0.2)

        results.append(evaluate_config(
            processor, False, target_ids, frame_paths, gt_by_frame, args.iou,
            "2) GPU (Metal), no SAHI",
        ))
        results.append(evaluate_config(
            processor, True, target_ids, frame_paths, gt_by_frame, args.iou,
            f"3) GPU (Metal) + SAHI (tile {args.slice_size}px, overlap {args.overlap})",
        ))
    else:
        print("\nGPU non disponibile su questa macchina: salto i test 2 e 3.")

    print("\n=== Riepilogo ===")
    print(f"{'Config':32s} {'Recall':>8s} {'Precision':>10s} {'ms/frame':>10s}")
    for r in results:
        print(f"{r['label']:32s} {r['recall']:>8.1%} {r['precision']:>10.1%} {r['avg_ms']:>10.1f}")


if __name__ == "__main__":
    main()
