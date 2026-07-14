from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="data/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)