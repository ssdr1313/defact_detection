import os
from pathlib import Path


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


if __name__=="__main__":
    data_path = ROOT_PATH/ "data"/ "raw"/ "NEU-DET"
    train_data_path = data_path/ "train"
    valid_data_path = data_path/ "validation"

    nums_train_label = len(os.listdir(train_data_path/"annotations"))
    nums_valid_label = len(os.listdir(valid_data_path/"annotations"))
    print(f"    训练标注量: {nums_train_label}")
    print(f"    验证标注量: {nums_valid_label}")
    nums_labels = nums_train_label + nums_train_label
    print(f"标注总量: {nums_labels}")

    nums_train_data = cal_defacts_nums(train_data_path,"train")
    nums_valid_data = cal_defacts_nums(valid_data_path,"vaild")
    print(f"数据总量： {nums_train_data + nums_valid_data}")






