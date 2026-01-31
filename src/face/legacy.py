import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils  # ty:ignore[possibly-missing-attribute]
mp_face_mesh = mp.solutions.face_mesh  # ty:ignore[possibly-missing-attribute]

# 初始化人脸网格模型
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue
    
    # 将BGR图像转为RGB，以满足MediaPipe输入要求
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)  # 进行人脸网格检测
    img_height, img_width, _ = img.shape  # 获取图像高宽（后续用于坐标转换）

    # 若检测到至少一个人脸
    if results.multi_face_landmarks:
        # 遍历每个检测到的人脸
        for face_landmarks in results.multi_face_landmarks:
            # 绘制人脸网格
            mp_drawing.draw_landmarks(
                img,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=1, circle_radius=1
                ),
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                ),
            )
    cv2.imshow("Face Mesh", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()