from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
model.train(data='testing_everything/dataset/data.yaml', epochs=25, imgsz=640, plots=True)