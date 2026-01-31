import cv2
import mediapipe as mp
import time
import queue

mp_drawing = mp.solutions.drawing_utils  # ty:ignore[possibly-missing-attribute]
mp_face_mesh = mp.solutions.face_mesh  # ty:ignore[possibly-missing-attribute]

model_path = "models/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a face landmarker instance with the live stream mode:
def handle_result(
    result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    """处理 Face Landmarker 异步检测结果的回调函数。

    Args:
        result (FaceLandmarkerResult): 包含检测到的人脸网格和其他信息的结果对象。
        output_image (mp.Image): 输入图像，用于可视化检测结果。
        timestamp_ms (int): 输入图像的时间戳（毫秒），用于同步处理。
    """
    if not result.face_landmarks:
        return
    face_landmarks = result.face_landmarks[0]
    print(f"face landmarker landmarks length: {len(face_landmarks)}")   # 478, 每个landmark有3个坐标（x, y, z）
    face_blendshapes = result.face_blendshapes[0]
    print(f"face landmarker blendshapes length: {len(face_blendshapes)}")   # 52
    for blendshape in face_blendshapes:
        print(f"blendshape: {blendshape.category_name}, {blendshape.score}")
    
    # 将结果和图像放入队列，以便在主线程中处理
    img = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR)
    result_queue.put((face_landmarks, img))

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_result,
    num_faces=1,
    output_face_blendshapes=True,
)

cap = cv2.VideoCapture(0)
# 创建队列用于在回调函数和主线程之间传递数据
result_queue = queue.Queue()


with FaceLandmarker.create_from_options(options) as landmarker:
    start_time = time.time()  # 记录起始毫秒时间戳
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        landmarker.detect_async(
            mp_image, timestamp_ms
        )  # 向 Face Landmarker 任务提供输入帧的时间戳
        
        # 检查队列中是否有结果需要处理
        try:
            face_landmarks, vis_img = result_queue.get(block=False)
            # 绘制人脸网格
            img_height, img_width, _ = vis_img.shape
            
            # 绘制面部网格连接线
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                start_idx, end_idx = connection
                start_landmark = face_landmarks[start_idx]
                end_landmark = face_landmarks[end_idx]
                
                # 将归一化坐标转换为像素坐标
                start_point = (int(start_landmark.x * img_width), int(start_landmark.y * img_height))
                end_point = (int(end_landmark.x * img_width), int(end_landmark.y * img_height))
                
                # 绘制连接线
                cv2.line(vis_img, start_point, end_point, (0, 255, 0), 1)
            
            # 绘制面部轮廓
            for connection in mp_face_mesh.FACEMESH_CONTOURS:
                start_idx, end_idx = connection
                start_landmark = face_landmarks[start_idx]
                end_landmark = face_landmarks[end_idx]
                
                # 将归一化坐标转换为像素坐标
                start_point = (int(start_landmark.x * img_width), int(start_landmark.y * img_height))
                end_point = (int(end_landmark.x * img_width), int(end_landmark.y * img_height))
                
                # 绘制连接线
                cv2.line(vis_img, start_point, end_point, (255, 255, 255), 1)
            
            # 绘制关键点
            for landmark in face_landmarks:
                x = int(landmark.x * img_width)
                y = int(landmark.y * img_height)
                cv2.circle(vis_img, (x, y), 1, (255, 0, 0), -1)
            
            # 显示图像
            cv2.imshow("Face Landmarker", vis_img)
        except queue.Empty:
            # 队列中没有数据，继续循环
            pass
        
        # 检查是否按下了 'q' 键
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
