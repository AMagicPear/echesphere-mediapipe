from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass
import threading
import time

import cv2
import mediapipe
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np


@dataclass
class FaceResult:
    """面部识别结果"""

    blendshapes: dict[str, float]  # {category_name: score}，仅 score > 0.1
    face_center: tuple[float, float]  # 鼻尖坐标 (x, y)，归一化坐标
    timestamp_ms: int


class FaceRecognizer:
    """面部表情识别器"""

    BLENDSHAPE_THRESHOLD = 0.1
    CHANGE_THRESHOLD = 0.15  # blendshapes 均值变化阈值
    MIN_CALLBACK_INTERVAL_MS = 500  # 最小发送间隔（毫秒）

    def __init__(
        self,
        model: Path,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        preview: bool = False,
    ):
        self.model = model
        self.num_faces = num_faces
        self.min_face_detection_confidence = min_face_detection_confidence
        self.min_face_presence_confidence = min_face_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.preview = preview

        self._detector: Optional[vision.FaceLandmarker] = None
        self._result_callbacks: list[Callable[[FaceResult], None]] = []

        # 线程安全的结果缓存
        self._lock = threading.Lock()
        self._cached_result: Optional[FaceResult] = None

        # FPS 追踪
        self._fps_counter = 0
        self._fps_start_time = time.time()
        self._fps = 0.0
        self._fps_avg_frame_count = 10

        # 回调节流
        self._last_blendshape_mean: float = 0.0
        self._last_callback_time: float = 0.0  # 上次触发时间（秒）

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """初始化识别器"""
        base_options = mp_python.BaseOptions(model_asset_path=str(self.model))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=self.num_faces,
            min_face_detection_confidence=self.min_face_detection_confidence,
            min_face_presence_confidence=self.min_face_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_face_blendshapes=True,
            result_callback=self._on_result,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def stop(self) -> None:
        """停止识别器并释放资源"""
        if self._detector:
            self._detector.close()
            self._detector = None

    def recognize_async(self, rgb_image, timestamp_ms: int) -> None:
        """异步识别面部（外部送帧模式使用）"""
        if self._detector:
            mp_image = mediapipe.Image(
                image_format=mediapipe.ImageFormat.SRGB, data=rgb_image
            )
            self._detector.detect_async(mp_image, timestamp_ms)

    def on_result(self, callback: Callable[[FaceResult], None]) -> None:
        """注册结果回调"""
        self._result_callbacks.append(callback)

    def close(self) -> None:
        """别名，兼容 close() 调用"""
        self.stop()

    def get_cached_result(self) -> Optional[FaceResult]:
        """获取缓存的识别结果（用于预览）"""
        with self._lock:
            return self._cached_result

    # ------------------------------------------------------------------
    # MediaPipe 回调
    # ------------------------------------------------------------------

    def _on_result(
        self,
        result: vision.FaceLandmarkerResult,
        unused_output_image: mediapipe.Image,
        timestamp_ms: int,
    ) -> None:
        # FPS
        self._fps_counter += 1
        if self._fps_counter % self._fps_avg_frame_count == 0:
            self._fps = self._fps_avg_frame_count / (time.time() - self._fps_start_time)
            self._fps_start_time = time.time()

        # 提取 blendshapes（仅 score > 0.1）
        blendshapes: dict[str, float] = {}
        face_center: tuple[float, float] = (0.0, 0.0)

        if result.face_blendshapes and result.face_blendshapes[0]:
            for cat in result.face_blendshapes[0]:
                if cat.score > self.BLENDSHAPE_THRESHOLD:
                    blendshapes[cat.category_name] = round(cat.score, 3)

        # 鼻尖坐标（landmark 1）
        if result.face_landmarks and result.face_landmarks[0]:
            nose = result.face_landmarks[0][1]
            face_center = (round(nose.x, 3), round(nose.y, 3))

        face_result = FaceResult(
            blendshapes=blendshapes,
            face_center=face_center,
            timestamp_ms=timestamp_ms,
        )

        # 缓存
        with self._lock:
            self._cached_result = face_result

        # 节流检测：变化阈值 + 最小间隔双重约束
        if blendshapes:
            current_mean = sum(blendshapes.values()) / len(blendshapes)
        else:
            current_mean = 0.0

        now = time.time()
        if abs(current_mean - self._last_blendshape_mean) > self.CHANGE_THRESHOLD:
            elapsed_ms = (now - self._last_callback_time) * 1000
            if elapsed_ms >= self.MIN_CALLBACK_INTERVAL_MS:
                self._last_blendshape_mean = current_mean
                self._last_callback_time = now
                for callback in self._result_callbacks:
                    callback(face_result)

    # ------------------------------------------------------------------
    # 预览渲染（不画 FPS，避免重叠）
    # ------------------------------------------------------------------

    def render_preview(self, image: np.ndarray) -> np.ndarray:
        """在图像上渲染面部标记（鼻尖绿点 + blendshapes 文字）"""
        with self._lock:
            result = self._cached_result

        if result is not None and result.face_center != (0.0, 0.0):
            h, w = image.shape[:2]
            cx, cy = int(result.face_center[0] * w), int(result.face_center[1] * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

            # blendshapes 文字（最多显示 5 个）
            if result.blendshapes:
                y = 90
                for name, score in list(result.blendshapes.items())[:5]:
                    text = f"{name}: {score:.2f}"
                    cv2.putText(
                        image,
                        text,
                        (24, y),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        (0, 200, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    y += 22

        return image
