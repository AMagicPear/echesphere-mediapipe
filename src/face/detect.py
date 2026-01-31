import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

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
    if cv2.waitKey(1) & 0xFF == ord("q"):
        exit(0)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_result,
    num_faces=1,
    output_face_blendshapes=True,
)

cap = cv2.VideoCapture(0)

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
        break
