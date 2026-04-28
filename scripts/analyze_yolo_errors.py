from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from pathlib import Path


FILE_PATH = Path(__file__).resolve()
ROOT_PATH = FILE_PATH.parents[1]


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# BGR colors for OpenCV visualization
GT_COLOR = (0, 180, 0)
PRED_COLOR = (0, 0, 255)
TEXT_BG = (30, 30, 30)
WHITE = (255, 255, 255)


class Box:
    def __init__(
        self,
        cls_id: int,
        xyxy: Tuple[float, float, float, float],
        conf: Optional[float] = None,
        raw_line: str = "",
    ) -> None:
        self.cls_id = int(cls_id)
        self.xyxy = tuple(float(x) for x in xyxy)
        self.conf = None if conf is None else float(conf)
        self.raw_line = raw_line


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    x1 = max(0.0, min(float(img_w - 1), x1))
    y1 = max(0.0, min(float(img_h - 1), y1))
    x2 = max(0.0, min(float(img_w - 1), x2))
    y2 = max(0.0, min(float(img_h - 1), y2))
    return x1, y1, x2, y2


def read_yolo_label_file(label_path: Path, img_w: int, img_h: int, has_conf: bool, conf_thr: float = 0.0) -> List[Box]:
    boxes: List[Box] = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split()
            min_len = 6 if has_conf else 5
            if len(parts) < min_len:
                print(f"[WARN] Skip invalid label line {label_path}:{line_idx}: {raw}")
                continue
            try:
                cls_id = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if has_conf and len(parts) >= 6 else None
            except ValueError:
                print(f"[WARN] Skip unparsable label line {label_path}:{line_idx}: {raw}")
                continue

            if conf is not None and conf < conf_thr:
                continue
            if not all(math.isfinite(v) for v in [cx, cy, w, h]):
                continue
            if w <= 0 or h <= 0:
                continue
            xyxy = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
            x1, y1, x2, y2 = xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(Box(cls_id=cls_id, xyxy=xyxy, conf=conf, raw_line=raw))
    return boxes


def box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def draw_label(img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, img.shape[1] - 1))
    y = max(th + 4, min(y, img.shape[0] - 1))
    cv2.rectangle(img, (x, y - th - 4), (min(img.shape[1] - 1, x + tw + 4), y + 2), TEXT_BG, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, WHITE, thickness, cv2.LINE_AA)


def draw_boxes(img: np.ndarray, boxes: Sequence[Box], names: Dict[int, str], mode: str) -> np.ndarray:
    out = img.copy()
    color = GT_COLOR if mode == "gt" else PRED_COLOR
    prefix = "GT" if mode == "gt" else "P"
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b.xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cls_name = names.get(b.cls_id, str(b.cls_id))
        if b.conf is None:
            text = f"{prefix}:{cls_name}"
        else:
            text = f"{prefix}:{cls_name} {b.conf:.2f}"
        draw_label(out, text, x1, max(12, y1 - 3), color)
    return out


def make_panel(img: np.ndarray, gt_boxes: Sequence[Box], pred_boxes: Sequence[Box], names: Dict[int, str]) -> np.ndarray:
    gt_vis = draw_boxes(img, gt_boxes, names, "gt")
    pred_vis = draw_boxes(img, pred_boxes, names, "pred")
    overlay = draw_boxes(img, gt_boxes, names, "gt")
    overlay = draw_boxes(overlay, pred_boxes, names, "pred")

    def add_title(im: np.ndarray, title: str) -> np.ndarray:
        out = im.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(out, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        return out

    gt_vis = add_title(gt_vis, "Ground Truth")
    pred_vis = add_title(pred_vis, "Prediction")
    overlay = add_title(overlay, "Overlay: GT=green, Pred=red")
    return np.concatenate([gt_vis, pred_vis, overlay], axis=1)


def image_stem_to_label(label_dir: Path, image_path: Path) -> Path:
    return label_dir / f"{image_path.stem}.txt"


def collect_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def init_stats(names: Dict[int, str]) -> Dict[int, Dict[str, float]]:
    return {
        cid: {
            "gt": 0,
            "pred": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "class_error_as_gt": 0,
            "class_error_as_pred": 0,
            "bad_location": 0,
        }
        for cid in sorted(names.keys())
    }


def add_error_row(
    rows: List[Dict[str, object]],
    image_path: Path,
    error_type: str,
    gt: Optional[Box],
    pred: Optional[Box],
    iou: Optional[float],
    names: Dict[int, str],
    vis_path: Optional[Path],
) -> None:
    gt_cls = gt.cls_id if gt is not None else None
    pred_cls = pred.cls_id if pred is not None else None
    rows.append(
        {
            "image": str(image_path),
            "image_name": image_path.name,
            "error_type": error_type,
            "gt_class_id": "" if gt_cls is None else gt_cls,
            "gt_class": "" if gt_cls is None else names.get(gt_cls, str(gt_cls)),
            "pred_class_id": "" if pred_cls is None else pred_cls,
            "pred_class": "" if pred_cls is None else names.get(pred_cls, str(pred_cls)),
            "conf": "" if pred is None or pred.conf is None else f"{pred.conf:.6f}",
            "iou": "" if iou is None else f"{iou:.6f}",
            "gt_xyxy": "" if gt is None else " ".join(f"{v:.2f}" for v in gt.xyxy),
            "pred_xyxy": "" if pred is None else " ".join(f"{v:.2f}" for v in pred.xyxy),
            "visualization": "" if vis_path is None else str(vis_path),
        }
    )


def analyze_one_image(
    image_path: Path,
    img: np.ndarray,
    gt_boxes: List[Box],
    pred_boxes: List[Box],
    names: Dict[int, str],
    iou_thr: float,
    loc_iou_thr: float,
    stats: Dict[int, Dict[str, float]],
    rows: List[Dict[str, object]],
    out_dir: Path,
    save_error_visuals: bool = True,
) -> None:
    for g in gt_boxes:
        stats.setdefault(g.cls_id, init_stats({g.cls_id: names.get(g.cls_id, str(g.cls_id))})[g.cls_id])
        stats[g.cls_id]["gt"] += 1
    for p in pred_boxes:
        stats.setdefault(p.cls_id, init_stats({p.cls_id: names.get(p.cls_id, str(p.cls_id))})[p.cls_id])
        stats[p.cls_id]["pred"] += 1

    matched_gt = set()
    matched_pred = set()

    # 1) Correct-class TP matching, sorted by IoU desc.
    tp_candidates: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            if g.cls_id == p.cls_id:
                iou = box_iou(g.xyxy, p.xyxy)
                if iou >= iou_thr:
                    tp_candidates.append((iou, gi, pi))
    tp_candidates.sort(reverse=True, key=lambda x: x[0])

    for iou, gi, pi in tp_candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        stats[gt_boxes[gi].cls_id]["tp"] += 1

    # Helper to save a panel for this image once and reuse path.
    panel_path_cache: Optional[Path] = None

    def save_panel_for(error_type: str, class_name: str) -> Path:
        nonlocal panel_path_cache
        class_dir = out_dir / "visual_cases" / error_type / class_name
        ensure_dir(class_dir)
        dst = class_dir / image_path.name
        if not dst.exists():
            panel = make_panel(img, gt_boxes, pred_boxes, names)
            cv2.imwrite(str(dst), panel)
        panel_path_cache = dst
        return dst

    # 2) High-IoU wrong-class matching: class error.
    cls_candidates: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_boxes):
        if gi in matched_gt:
            continue
        for pi, p in enumerate(pred_boxes):
            if pi in matched_pred:
                continue
            if g.cls_id != p.cls_id:
                iou = box_iou(g.xyxy, p.xyxy)
                if iou >= iou_thr:
                    cls_candidates.append((iou, gi, pi))
    cls_candidates.sort(reverse=True, key=lambda x: x[0])

    for iou, gi, pi in cls_candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        g = gt_boxes[gi]
        p = pred_boxes[pi]
        matched_gt.add(gi)
        matched_pred.add(pi)
        stats[g.cls_id]["fn"] += 1
        stats[p.cls_id]["fp"] += 1
        stats[g.cls_id]["class_error_as_gt"] += 1
        stats[p.cls_id]["class_error_as_pred"] += 1
        vis_path = save_panel_for("class_error", names.get(g.cls_id, str(g.cls_id))) if save_error_visuals else None
        add_error_row(rows, image_path, "class_error", g, p, iou, names, vis_path)

    # 3) Same-class but low IoU: bad localization.
    loc_candidates: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_boxes):
        if gi in matched_gt:
            continue
        for pi, p in enumerate(pred_boxes):
            if pi in matched_pred:
                continue
            if g.cls_id == p.cls_id:
                iou = box_iou(g.xyxy, p.xyxy)
                if loc_iou_thr <= iou < iou_thr:
                    loc_candidates.append((iou, gi, pi))
    loc_candidates.sort(reverse=True, key=lambda x: x[0])

    for iou, gi, pi in loc_candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        g = gt_boxes[gi]
        p = pred_boxes[pi]
        matched_gt.add(gi)
        matched_pred.add(pi)
        stats[g.cls_id]["fn"] += 1
        stats[p.cls_id]["fp"] += 1
        stats[g.cls_id]["bad_location"] += 1
        vis_path = save_panel_for("bad_location", names.get(g.cls_id, str(g.cls_id))) if save_error_visuals else None
        add_error_row(rows, image_path, "bad_location", g, p, iou, names, vis_path)

    # 4) Unmatched predictions: FP.
    for pi, p in enumerate(pred_boxes):
        if pi in matched_pred:
            continue
        stats[p.cls_id]["fp"] += 1
        vis_path = save_panel_for("false_positive", names.get(p.cls_id, str(p.cls_id))) if save_error_visuals else None
        add_error_row(rows, image_path, "false_positive", None, p, None, names, vis_path)

    # 5) Unmatched GTs: FN.
    for gi, g in enumerate(gt_boxes):
        if gi in matched_gt:
            continue
        stats[g.cls_id]["fn"] += 1
        vis_path = save_panel_for("false_negative", names.get(g.cls_id, str(g.cls_id))) if save_error_visuals else None
        add_error_row(rows, image_path, "false_negative", g, None, None, names, vis_path)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_summary_rows(stats: Dict[int, Dict[str, float]], names: Dict[int, str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cid in sorted(stats.keys()):
        s = stats[cid]
        tp = s["tp"]
        fp = s["fp"]
        fn = s["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append(
            {
                "class_id": cid,
                "class_name": names.get(cid, str(cid)),
                "gt_instances": int(s["gt"]),
                "pred_instances": int(s["pred"]),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "precision_by_matching": f"{precision:.6f}",
                "recall_by_matching": f"{recall:.6f}",
                "class_error_as_gt": int(s["class_error_as_gt"]),
                "class_error_as_pred": int(s["class_error_as_pred"]),
                "bad_location": int(s["bad_location"]),
            }
        )
    return rows


def write_markdown_report(
    path: Path,
    summary_rows: List[Dict[str, object]],
    error_rows: List[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    ensure_dir(path.parent)
    total_gt = sum(int(r["gt_instances"]) for r in summary_rows)
    total_tp = sum(int(r["TP"]) for r in summary_rows)
    total_fp = sum(int(r["FP"]) for r in summary_rows)
    total_fn = sum(int(r["FN"]) for r in summary_rows)
    total_cls_err = sum(int(r["class_error_as_gt"]) for r in summary_rows)
    total_bad_loc = sum(int(r["bad_location"]) for r in summary_rows)

    def md_table(rows: List[Dict[str, object]]) -> str:
        headers = [
            "class_name",
            "gt_instances",
            "TP",
            "FP",
            "FN",
            "precision_by_matching",
            "recall_by_matching",
            "class_error_as_gt",
            "bad_location",
        ]
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for r in rows:
            lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
        return "\n".join(lines)

    error_type_counts: Dict[str, int] = {}
    for r in error_rows:
        k = str(r["error_type"])
        error_type_counts[k] = error_type_counts.get(k, 0) + 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("# YOLO Class-Level Error Analysis\n\n")
        f.write("## Settings\n\n")
        f.write(f"- images: `{args.images}`\n")
        f.write(f"- ground truth labels: `{args.gt}`\n")
        f.write(f"- prediction labels: `{args.pred}`\n")
        f.write(f"- IoU threshold for TP: `{args.iou}`\n")
        f.write(f"- IoU range for bad localization: `[{args.loc_iou}, {args.iou})`\n")
        f.write(f"- prediction confidence filter: `{args.conf}`\n\n")

        f.write("## Overall Counts\n\n")
        f.write(f"- GT instances: `{total_gt}`\n")
        f.write(f"- TP: `{total_tp}`\n")
        f.write(f"- FP: `{total_fp}`\n")
        f.write(f"- FN: `{total_fn}`\n")
        f.write(f"- Class errors: `{total_cls_err}`\n")
        f.write(f"- Bad localization cases: `{total_bad_loc}`\n\n")

        f.write("## Error Type Counts\n\n")
        for k in sorted(error_type_counts):
            f.write(f"- {k}: `{error_type_counts[k]}`\n")
        f.write("\n")

        f.write("## Class-Level Summary\n\n")
        f.write(md_table(summary_rows))
        f.write("\n\n")

        weak = sorted(summary_rows, key=lambda r: float(r["recall_by_matching"]))[:3]
        f.write("## Weak Classes by Recall\n\n")
        for r in weak:
            f.write(
                f"- `{r['class_name']}`: recall={r['recall_by_matching']}, "
                f"FN={r['FN']}, class_error_as_gt={r['class_error_as_gt']}, bad_location={r['bad_location']}\n"
            )
        f.write("\n")

        f.write("## Output Files\n\n")
        f.write("- `class_error_summary.csv`: class-level TP/FP/FN table.\n")
        f.write("- `error_cases.csv`: detailed FP/FN/class-error/bad-localization cases.\n")
        f.write("- `gt_visualization/`: ground-truth-only visualization images.\n")
        f.write("- `visual_cases/`: side-by-side GT / prediction / overlay panels for error cases.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True, help="测试集图像路径")
    parser.add_argument("--gt", type=Path, required=True, help="测试集标签路径")
    parser.add_argument("--pred", type=Path, required=True, help="模型预测结果路径")
    parser.add_argument("--data", type=Path, default=None, help="dataset.yaml")
    parser.add_argument("--out", type=Path, default=Path("outputs/error_analysis/day1_baseline"), help="输出路径")
    parser.add_argument("--iou", type=float, default=0.5, help="")
    parser.add_argument("--loc-iou", type=float, default=0.1, help="Minimum IoU for bad-localization cases.")
    parser.add_argument("--conf", type=float, default=0.0, help="Filter predictions below this confidence during analysis.")
    parser.add_argument("--no-error-visuals", action="store_true", help="Do not save error-case visual panels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.images = ROOT_PATH / args.images
    args.gt = ROOT_PATH / args.gt
    args.pred = ROOT_PATH / args.pred
    args.data = ROOT_PATH / args.data
    args.out = ROOT_PATH / args.out

    ensure_dir(args.out)
    names = {
        0: "crazing",
        1: "inclusion",
        2: "patches",
        3: "pitted_surface",
        4: "rolled-in_scale",
        5: "scratches",
    }
    stats = init_stats(names)
    error_rows: List[Dict[str, object]] = []

    images = collect_images(args.images)
    if not images:
        raise FileNotFoundError(f"No images found in {args.images}")

    gt_vis_dir = args.out / "gt_visualization"
    ensure_dir(gt_vis_dir)

    processed = 0
    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[WARN] Cannot read image: {image_path}")
            continue
        h, w = img.shape[:2]
        gt_label = image_stem_to_label(args.gt, image_path)
        pred_label = image_stem_to_label(args.pred, image_path)
        gt_boxes = read_yolo_label_file(gt_label, w, h, has_conf=False)
        pred_boxes = read_yolo_label_file(pred_label, w, h, has_conf=True, conf_thr=args.conf)

        gt_vis = draw_boxes(img, gt_boxes, names, mode="gt")
        cv2.imwrite(str(gt_vis_dir / image_path.name), gt_vis)

        analyze_one_image(
            image_path=image_path,
            img=img,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            names=names,
            iou_thr=args.iou,
            loc_iou_thr=args.loc_iou,
            stats=stats,
            rows=error_rows,
            out_dir=args.out,
            save_error_visuals=not args.no_error_visuals,
        )
        processed += 1

    summary_rows = make_summary_rows(stats, names)
    write_csv(args.out / "class_error_summary.csv", summary_rows)
    write_csv(args.out / "error_cases.csv", error_rows)
    write_markdown_report(args.out / "error_analysis_v1.md", summary_rows, error_rows, args)

    print(f"[OK] Processed images: {processed}")
    print(f"[OK] GT visualization saved to: {args.out / 'gt_visualization'}")
    print(f"[OK] Error visual cases saved to: {args.out / 'visual_cases'}")
    print(f"[OK] Summary CSV: {args.out / 'class_error_summary.csv'}")
    print(f"[OK] Error cases CSV: {args.out / 'error_cases.csv'}")
    print(f"[OK] Markdown report: {args.out / 'error_analysis_v1.md'}")


if __name__ == "__main__":
    main()
