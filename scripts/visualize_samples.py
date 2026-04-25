from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2

FILE_PATH = Path(__file__).resolve()
ROOT_PATH = FILE_PATH.parents[1] if len(FILE_PATH.parents) > 1 else Path.cwd()

YOLO_DATA_PATH = ROOT_PATH / "data" / "yolo"
OUTPUT_DIR = ROOT_PATH / "outputs" / "vis" / "day4_samples"

CLASS_NAMES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        return []
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])


def load_yolo_boxes(label_path: Path) -> List[YoloBox]:
    boxes: List[YoloBox] = []
    if not label_path.exists():
        return boxes

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            print(f"[WARN] {label_path.name}:{line_no} 格式错误，已跳过 -> {line}")
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            print(f"[WARN] {label_path.name}:{line_no} 解析失败，已跳过 -> {line}")
            continue

        boxes.append(
            YoloBox(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )
    return boxes


def yolo_to_xyxy(box: YoloBox, image_w: int, image_h: int) -> Tuple[int, int, int, int]:
    x_center = box.x_center * image_w
    y_center = box.y_center * image_h
    width = box.width * image_w
    height = box.height * image_h

    x1 = int(round(x_center - width / 2.0))
    y1 = int(round(y_center - height / 2.0))
    x2 = int(round(x_center + width / 2.0))
    y2 = int(round(y_center + height / 2.0))

    x1 = max(0, min(x1, image_w - 1))
    y1 = max(0, min(y1, image_h - 1))
    x2 = max(0, min(x2, image_w - 1))
    y2 = max(0, min(y2, image_h - 1))
    return x1, y1, x2, y2


def get_class_name(class_id: int) -> str:
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"unknown_{class_id}"


def draw_boxes(image, boxes: Sequence[YoloBox]):
    h, w = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
        class_name = get_class_name(box.class_id)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{box.class_id}:{class_name}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x1 = x1
        text_y1 = max(0, y1 - text_h - baseline - 4)
        text_x2 = min(w - 1, text_x1 + text_w + 6)
        text_y2 = min(h - 1, text_y1 + text_h + baseline + 4)

        cv2.rectangle(image, (text_x1, text_y1), (text_x2, text_y2), (0, 255, 0), -1)
        cv2.putText(
            image,
            label,
            (text_x1 + 3, text_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return image


def collect_pairs(yolo_data_path: Path) -> List[Tuple[str, Path, Path]]:
    pairs: List[Tuple[str, Path, Path]] = []
    for subset in ["train", "val", "test"]:
        image_dir = yolo_data_path / "images" / subset
        label_dir = yolo_data_path / "labels" / subset
        images = find_images(image_dir)
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            pairs.append((subset, image_path, label_path))
    return pairs


def visualize_samples(
    yolo_data_path: Path = YOLO_DATA_PATH,
    output_dir: Path = OUTPUT_DIR,
    sample_num: int = 50,
    seed: int = 666666,
) -> None:
    if not yolo_data_path.exists():
        raise FileNotFoundError(f"YOLO 数据目录不存在：{yolo_data_path}")

    pairs = collect_pairs(yolo_data_path)

    make_dir(output_dir)

    rng = random.Random(seed)
    if len(pairs) > sample_num:
        pairs = rng.sample(pairs, sample_num)

    saved_count = 0
    missing_label_count = 0
    empty_label_count = 0

    for subset, image_path, label_path in pairs:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] 图片读取失败，已跳过：{image_path}")
            continue

        boxes = load_yolo_boxes(label_path)
        if not label_path.exists():
            missing_label_count += 1
        elif len(boxes) == 0:
            empty_label_count += 1

        vis_image = image.copy()
        vis_image = draw_boxes(vis_image, boxes)

        save_name = f"{subset}__{image_path.stem}.jpg"
        save_path = output_dir / save_name
        ok = cv2.imwrite(str(save_path), vis_image)
        if not ok:
            print(f"[WARN] 保存失败：{save_path}")
            continue

        saved_count += 1
        print(f"[INFO] 已保存：{save_path}")

    print("\n" + "=" * 72)
    print("[SUMMARY] 可视化完成")
    print("-" * 72)
    print(f"输出目录:           {output_dir}")
    print(f"总候选样本数:       {len(collect_pairs(yolo_data_path))}")
    print(f"本次抽样数:         {len(pairs)}")
    print(f"实际保存数:         {saved_count}")
    print(f"缺少标签文件数:     {missing_label_count}")
    print(f"空标签文件数:       {empty_label_count}")
    print("=" * 72)


def main() -> None:
    visualize_samples()


if __name__ == "__main__":
    main()
