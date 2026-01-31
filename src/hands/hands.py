import cv2
import mediapipe as mp
import time

# 打开默认摄像头（设备索引0）
cap = cv2.VideoCapture(0)

# 初始化MediaPipe Hands模块
mp_hands = mp.solutions.hands  # ty:ignore[possibly-missing-attribute]
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5
)  # 使用默认参数：静态图模式关闭、最多检测2只手、置信度阈值0.5
mp_draw = mp.solutions.drawing_utils  # ty:ignore[possibly-missing-attribute]

# 设置手部关键点与连接线的绘制样式
hand_landmarks_style = mp_draw.DrawingSpec(
    color=(255, 177, 78), thickness=5
)  # 关键点样式：橙色、线宽5
hand_connections_style = mp_draw.DrawingSpec(
    color=(177, 78, 255), thickness=10
)  # 连接线样式：紫色、线宽10

# 初始化时间变量，用于计算FPS
p_time = 0
current_time = 0

# 主循环：逐帧读取并处理图像
while cap.isOpened():
    success, img = cap.read()  # ret为读取成功标志，img为当前帧
    if not success:
        continue
    # 将BGR图像转为RGB，以满足MediaPipe输入要求
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)  # 进行手部关键点检测
    img_height, img_width, _ = img.shape  # 获取图像高宽（后续用于坐标转换）

    # 若检测到至少一只手
    if results.multi_hand_landmarks:
        # 遍历每只手的21个关键点
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制关键点及连接线
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                hand_landmarks_style,
                hand_connections_style,
            )
            # 在每个关键点旁边标注其索引（0~20）
            for i, lm in enumerate(hand_landmarks.landmark):
                # 将归一化坐标转换为像素坐标
                x_pos = int(lm.x * img_width)
                y_pos = int(lm.y * img_height)
                cv2.putText(
                    img,
                    str(i),
                    (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
    current_time = time.time()
    fps = 1 / (current_time - p_time)
    p_time = current_time
    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    # 显示处理后的图像
    cv2.imshow("img", img)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
