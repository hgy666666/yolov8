from ultralytics import YOLO
def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data=r"C:\Users\35035\Desktop\ultralytics-main - 副本\Tomato\data.yaml",
        imgsz=640,
        epochs=100,
        batch=16,
        device=0,
        workers=0,
    )


if __name__ == "__main__":
    main()