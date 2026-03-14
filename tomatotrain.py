from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(data="C:/Users/35035/Desktop/ultralytics-main/Tomato/data.yaml", imgsz=640, epochs=60, batch=6,  device=0, workers=0)