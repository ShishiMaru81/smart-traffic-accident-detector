from ultralytics import YOLO

model = YOLO(r"D:\Real Time Accident Detection\runs\detect\train-2\weights\best.pt")
results = model.predict(
    source=r"D:\Real Time Accident Detection\Test_img_YOLO",
    conf=0.25,
    save=True
)