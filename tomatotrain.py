from ultralytics import YOLO
def main():
    # Use the custom architecture YAML (otherwise you're training stock yolov8n)
    model = YOLO("my_yolov8n_eca_ghost_fast_decoupled.yaml", task="detect")
    # Optional: load pretrained weights to speed up convergence (may partially load if shapes differ)
    model.load("yolov8n.pt")
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