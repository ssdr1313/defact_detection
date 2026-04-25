import os
from pathlib import Path
from collections import defaultdict


FILE_PATH = Path(__file__).resolve()
ROOT_PATH = FILE_PATH.parents[1]


def cal_defacts_nums(data_path,mode):
    num_total = 0
    for defact in os.listdir(data_path / "images"):
        nums_defact = len(os.listdir(data_path / "images" / f"{defact}"))
        print(f"    {mode}集中{defact}的数量：{nums_defact}")
        num_total = num_total + nums_defact
    print(f"{mode}集总量：{num_total}")
    return num_total


def main():
    data_path = ROOT_PATH/ "data"/ "raw"/ "NEU-DET"
    train_data_path = data_path / "train"
    valid_data_path = data_path / "validation"

    nums_train_label = len(os.listdir(train_data_path / "annotations"))
    nums_valid_label = len(os.listdir(valid_data_path / "annotations"))
    print(f"    训练标注量: {nums_train_label}")
    print(f"    验证标注量: {nums_valid_label}")
    nums_labels = nums_train_label + nums_valid_label
    print(f"标注总量: {nums_labels}")

    nums_train_data = cal_defacts_nums(train_data_path, "train")
    nums_valid_data = cal_defacts_nums(valid_data_path, "vaild")
    print(f"数据总量： {nums_train_data + nums_valid_data}")

    abnormal_files = []

    subsets = ["train", "validation"]
    for subset in subsets:
        subset_dir = data_path / subset
        images_root = subset_dir / "images"
        annotation_root = subset_dir / "annotations"

        for class_name in os.listdir(images_root):
            class_path = images_root / f"{class_name}"
            for img_name in os.listdir(class_path):
                img_path = class_path / f"{img_name}"
                xml_name = os.path.splitext(img_name)[0] + '.xml'
                xml_path = annotation_root / f"{xml_name}"
                if not os.path.exists(xml_path):
                    abnormal_files.append(f"{img_path}无标注")
                    continue
        for xml_name in os.listdir(annotation_root):
            for i in reversed(range(len(xml_name))):
                if xml_name[i] == "_":
                    class_name = xml_name[:i]
                    break
            img_path = images_root / class_name / (os.path.splitext(xml_name)[0] + '.jpg')
            if not os.path.exists(img_path):
                abnormal_files.append(f"{xml_name}无对应图片")
                continue

    print(abnormal_files)


if __name__=="__main__":
    main()