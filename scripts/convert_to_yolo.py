from __future__ import annotations

import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

FILE_PATH = Path(__file__).resolve()
ROOT_PATH = FILE_PATH.parents[1] if len(FILE_PATH.parents) > 1 else Path.cwd()

# 你也可以把 ROOT_PATH 改成项目根目录，例如：Path("/home/user/your_project")
RAW_DATA_PATH = ROOT_PATH / "data" / "raw" / "NEU-DET"
YOLO_DATA_PATH = ROOT_PATH / "data" / "yolo"
CONFIG_PATH = ROOT_PATH / "configs" / "dataset.yaml"

# NEU-DET 类别顺序
CLASS_NAMES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Box:
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class Record:
    image_path: Path
    xml_path: Path
    width: int
    height: int
    boxes: List[Box]


@dataclass
class ConvertStats:
    total_images: int = 0
    total_xml: int = 0
    total_boxes: int = 0
    dropped_boxes: int = 0
    skipped_images_without_xml: int = 0
    skipped_unknown_class_boxes: int = 0
    empty_label_files: int = 0


def make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_class_name(name: str) -> str:
    raw = (name or "").strip()
    # 额外兜底：统一大小写和连接符
    lowered = raw.lower().replace(" ", "_")
    return raw


def get_all_image_paths(data_path: Path) -> List[Path]:
    image_paths: List[Path] = []
    subsets = ["train", "validation"]

    for subset in subsets:
        images_root = data_path / subset / "images"
        annotation_root = data_path / subset / "annotations"

        for class_dir in sorted(images_root.iterdir()):
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                xml_path = annotation_root / f"{img_path.stem}.xml"
                if not xml_path.exists():
                    print(f"[WARN] 找不到对应标注，跳过：{img_path.name}", file=sys.stderr)
                    continue
                image_paths.append(img_path)

    return image_paths


def grouped_split_dataset(
    img_list: Sequence[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 666666,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    按类别分层划分，尽量保证 train/val/test 的类别分布一致。
    默认假设图片路径的父目录名就是类别名。
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"train/val/test 比例之和必须为 1，当前为 {total}")

    rng = random.Random(seed)
    groups: Dict[str, List[Path]] = defaultdict(list)
    for img_path in img_list:
        class_name = img_path.parent.name
        groups[class_name].append(img_path)

    train, val, test = [], [], []
    for class_name, items in groups.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        # 保证小类也尽量能分到验证/测试，但不能越界
        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_val == 0:
                n_val = 1
                if n_train + n_val > n:
                    n_train = max(1, n_train - 1)
            n_test = n - n_train - n_val
            if n_test == 0:
                if n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
                n_test = n - n_train - n_val

        class_train = items[:n_train]
        class_val = items[n_train:n_train + n_val]
        class_test = items[n_train + n_val:]

        train.extend(class_train)
        val.extend(class_val)
        test.extend(class_test)

        print(
            f"[INFO] 类别 {class_name:<16} | 总数 {n:<4} -> "
            f"train {len(class_train):<4} val {len(class_val):<4} test {len(class_test):<4}"
        )

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def find_xml_path(img_path: Path, raw_data_path: Path) -> Optional[Path]:
    base_name = img_path.stem
    for subset in ["train", "validation"]:
        candidate = raw_data_path / subset / "annotations" / f"{base_name}.xml"
        if candidate.exists():
            return candidate
    return None


def parse_xml_record(img_path: Path, xml_path: Path) -> Record:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width_text = root.findtext("size/width")
    height_text = root.findtext("size/height")
    if width_text is None or height_text is None:
        raise ValueError(f"标注缺少 size 信息：{xml_path}")

    width = int(float(width_text))
    height = int(float(height_text))

    boxes: List[Box] = []
    for obj in root.findall("object"):
        class_name = normalize_class_name(obj.findtext("name", default=""))
        bnd = obj.find("bndbox")
        if bnd is None:
            print(f"[WARN] {xml_path.name} 中 object 缺少 bndbox，已跳过", file=sys.stderr)
            continue

        try:
            xmin = float(bnd.findtext("xmin", default="0"))
            ymin = float(bnd.findtext("ymin", default="0"))
            xmax = float(bnd.findtext("xmax", default="0"))
            ymax = float(bnd.findtext("ymax", default="0"))
        except ValueError as exc:
            print(f"[WARN] 解析框失败：{xml_path.name} | {exc}", file=sys.stderr)
            continue

        boxes.append(Box(class_name=class_name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

    return Record(
        image_path=img_path,
        xml_path=xml_path,
        width=width,
        height=height,
        boxes=boxes,
    )


def clip_box(box: Box, width: int, height: int) -> Optional[Box]:
    xmin = min(max(box.xmin, 0.0), float(width))
    ymin = min(max(box.ymin, 0.0), float(height))
    xmax = min(max(box.xmax, 0.0), float(width))
    ymax = min(max(box.ymax, 0.0), float(height))

    if xmax <= xmin or ymax <= ymin:
        return None

    return Box(
        class_name=box.class_name,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
    )


def make_yolo_line(box: Box, class_id: int, width: int, height: int) -> str:
    x_center = ((box.xmin + box.xmax) / 2.0) / width
    y_center = ((box.ymin + box.ymax) / 2.0) / height
    box_w = (box.xmax - box.xmin) / width
    box_h = (box.ymax - box.ymin) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def save_one_split(
    img_list: Sequence[Path],
    subset: str,
    raw_data_path: Path,
    yolo_data_path: Path,
    stats: ConvertStats,
) -> None:
    image_out_dir = yolo_data_path / "images" / subset
    label_out_dir = yolo_data_path / "labels" / subset
    make_dir(image_out_dir)
    make_dir(label_out_dir)

    for img_path in img_list:
        xml_path = find_xml_path(img_path, raw_data_path)
        if xml_path is None:
            stats.skipped_images_without_xml += 1
            print(f"[WARN] 跳过：{img_path.name} 找不到标注", file=sys.stderr)
            continue

        stats.total_images += 1
        stats.total_xml += 1
        record = parse_xml_record(img_path, xml_path)

        label_lines: List[str] = []
        for raw_box in record.boxes:
            stats.total_boxes += 1
            clipped = clip_box(raw_box, record.width, record.height)
            if clipped is None:
                stats.dropped_boxes += 1
                print(
                    f"[WARN] 丢弃非法框：{record.xml_path.name} -> "
                    f"({raw_box.xmin}, {raw_box.ymin}, {raw_box.xmax}, {raw_box.ymax})",
                    file=sys.stderr,
                )
                continue

            if clipped.class_name not in CLASS_TO_ID:
                stats.skipped_unknown_class_boxes += 1
                print(
                    f"[WARN] 未知类别，已跳过：{record.xml_path.name} -> {clipped.class_name}",
                    file=sys.stderr,
                )
                continue

            class_id = CLASS_TO_ID[clipped.class_name]
            label_lines.append(make_yolo_line(clipped, class_id, record.width, record.height))

        dst_img_path = image_out_dir / img_path.name
        dst_label_path = label_out_dir / f"{img_path.stem}.txt"
        shutil.copy2(img_path, dst_img_path)
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        if not label_lines:
            stats.empty_label_files += 1

    print(f"[INFO] 已完成子集 {subset}: {len(img_list)} 张图片")


def write_dataset_yaml(yolo_data_path: Path, yaml_path: Path) -> None:
    make_dir(yaml_path.parent)
    content = [
        f"path: {str(yolo_data_path.resolve())}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        content.append(f"  {i}: {name}")

    yaml_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    print(f"[INFO] 已生成数据配置文件：{yaml_path}")


def init_yolo_dirs(yolo_data_path: Path) -> None:
    for part in ["images", "labels"]:
        for subset in ["train", "val", "test"]:
            make_dir(yolo_data_path / part / subset)


def print_summary(train_imgs: Sequence[Path], val_imgs: Sequence[Path], test_imgs: Sequence[Path], stats: ConvertStats) -> None:
    print("\n" + "=" * 72)
    print("[SUMMARY] 数据转换完成")
    print("-" * 72)
    print(f"train 图片数:                {len(train_imgs)}")
    print(f"val 图片数:                  {len(val_imgs)}")
    print(f"test 图片数:                 {len(test_imgs)}")
    print(f"总处理图片数:                {stats.total_images}")
    print(f"总处理 XML 数:               {stats.total_xml}")
    print(f"总读取框数:                  {stats.total_boxes}")
    print(f"丢弃非法框数:                {stats.dropped_boxes}")
    print(f"未知类别框数:                {stats.skipped_unknown_class_boxes}")
    print(f"缺少 xml 跳过图片数:         {stats.skipped_images_without_xml}")
    print(f"空标签文件数:                {stats.empty_label_files}")
    print("=" * 72)


def main() -> None:
    init_yolo_dirs(YOLO_DATA_PATH)

    image_paths = get_all_image_paths(RAW_DATA_PATH)

    print(f"图片 {len(image_paths)} 张")
    train_imgs, val_imgs, test_imgs = grouped_split_dataset(image_paths,train_ratio=0.7,val_ratio=0.2,test_ratio=0.1,seed=666666,)

    stats = ConvertStats()
    save_one_split(train_imgs, "train", RAW_DATA_PATH, YOLO_DATA_PATH, stats)
    save_one_split(val_imgs, "val", RAW_DATA_PATH, YOLO_DATA_PATH, stats)
    save_one_split(test_imgs, "test", RAW_DATA_PATH, YOLO_DATA_PATH, stats)

    write_dataset_yaml(YOLO_DATA_PATH, CONFIG_PATH)
    print_summary(train_imgs, val_imgs, test_imgs, stats)


if __name__ == "__main__":
    main()
