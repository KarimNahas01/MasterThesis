import json
from ultralytics import YOLO


CONSTANTS = json.load(open('constants.json'))

DIRECTORIES = CONSTANTS["directories"]
PARENT_FOLDER_NAME = DIRECTORIES["parent"]["folder_name"]
DATASET_STRUCTURE = CONSTANTS["dataset_structure"]

def main(parent_folder=PARENT_FOLDER_NAME):
    dataset_path = f'{PARENT_FOLDER_NAME}/{DATASET_STRUCTURE["name"]}'
    model = YOLO("yolov8n.pt")
    model.train(data=f'{dataset_path}/data.yaml', epochs=100, imgsz=640, plots=True)