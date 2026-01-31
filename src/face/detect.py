import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import time
import queue

mp_drawing = mp.solutions.drawing_utils  # ty:ignore[possibly-missing-attribute]
mp_face_mesh = mp.solutions.face_mesh  # ty:ignore[possibly-missing-attribute]
mp_drawing_styles = mp.solutions.drawing_styles  # ty:ignore[possibly-missing-attribute]

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
    # face_landmarks = result.face_landmarks[0]
    # print(f"face landmarker landmarks length: {len(face_landmarks)}")   # 478, 每个landmark有3个坐标（x, y, z）
    face_landmarks = result.face_landmarks[0]
    face_blendshapes = result.face_blendshapes[0]
    # print(f"face landmarker blendshapes length: {len(face_blendshapes)}")   # 52

    # 将结果和图像放入队列，以便在主线程中处理
    img = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR)
    result_queue.put((face_landmarks, face_blendshapes, img))


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
            face_landmarks, face_blendshapes, vis_img = result_queue.get(block=False)
            # 绘制人脸网格

            # 创建符合 draw_landmarks 函数期望格式的对象
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # ty:ignore[unresolved-attribute]
            face_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(  # ty:ignore[unresolved-attribute]
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z,
                    )
                    for landmark in face_landmarks
                ]
            )

            # 绘制面部网格
            mp_drawing.draw_landmarks(
                image=vis_img,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            # 绘制面部轮廓
            mp_drawing.draw_landmarks(
                image=vis_img,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
            # 绘制眼部轮廓
            mp_drawing.draw_landmarks(
                image=vis_img,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

            # 在图像上显示blendshape值
            y_offset = 30
            for i, blendshape in enumerate(
                face_blendshapes
            ):  # 只显示前10个，避免图像过于拥挤
                text = f"{blendshape.category_name}: {blendshape.score:.2f}"
                cv2.putText(
                    vis_img,
                    text,
                    (10, y_offset + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

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
