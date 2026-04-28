import argparse
import numpy
import cv2
import os
from pathlib import  Path
from scripts.dataset import IMAGE_Dataset
from ultralytics import YOLO
FILE_PATH = Path(__file__).resolve()
ROOT_PATH = FILE_PATH.parents[1]


def main():

    # # Create a new YOLO model from scratch
    # model = YOLO("yolo26n.yaml")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolo11n.pt")

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data="configs/dataset.yaml", epochs=3,imgsz = 640,project="runs",name="neu_baseline",device = "cuda")

    # # Evaluate the model's performance on the validation set
    # results = model.val()

    # # Perform object detection on an image using the model
    # results = model("https://ultralytics.com/images/bus.jpg")
    #
    # # Export the model to ONNX format
    # success = model.export(format="onnx")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dath_path",type = Path,default=ROOT_PATH/ "data" / "yolo")
    parser.add_argument("--epoch",type = int,default=1,help="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
