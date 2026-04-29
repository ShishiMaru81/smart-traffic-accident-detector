from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="D:\\Real Time Accident Detection\\YOLO\\Car Crash.v1i.yolov8\\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)

