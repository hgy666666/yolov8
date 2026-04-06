import cv2
import time
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO("best.pt")

# 打开摄像头
cap = cv2.VideoCapture(0)

# 帧率计算变量
prev_time = 0
current_time = 0
total_fps = 0
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLO 推理
    results = model.predict(frame, verbose=False)
    annotated_frame = results[0].plot()

    # 计算 FPS（修复了未定义报错）
    current_time = time.time()
    fps = 0  # 先默认赋值，避免第一帧报错
    if prev_time > 0:
        fps = 1 / (current_time - prev_time)
        total_fps += fps
        frame_count += 1
    prev_time = current_time

    # 显示实时帧率
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示画面
    cv2.imshow("YOLOv8 实时检测", annotated_frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

# 退出后输出平均帧率
if frame_count > 0:
    avg_fps = total_fps / frame_count
    print(f"\n✅ 程序已退出")
    print(f"📊 平均帧率：{avg_fps:.2f} FPS")
else:
    print("\n❌ 未读取到摄像头画面")