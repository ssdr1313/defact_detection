import numpy
import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

class IMAGE_Dataset(Dataset):
    def __init__(self,img_paths,label_paths):
        super().__init__()
        self.img_paths = img_paths
        self.label_path = label_paths

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def get_dataloader():
    pass