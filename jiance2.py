import cv2
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO("best.pt")  # 确保模型文件路径正确

# 输入图片路径
image_path = "Tomato/train/images/Leaf_Blight-11-_jpg.rf.49018eccf0a647b7966297aa30ffd1aa.jpg"  # 替换为你的图片路径

# 读取图片
image = cv2.imread(image_path)

if image is None:
    print(f"无法读取图片，请检查路径: {image_path}")
else:
    # 对图片进行目标检测
    results = model.predict(source=image, show=True)  # show=True 会显示检测结果

    # 等待用户按键后关闭窗口
    cv2.waitKey(0)
    # 关闭所有窗口
    cv2.destroyAllWindows()
