from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass
import logging
import threading
import time

import cv2
import mediapipe
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import (
    hands as mp_hands,
    drawing_utils as mp_drawing,
    drawing_styles as mp_drawing_styles,
)

logger = logging.getLogger("HandsRecognizer")


@dataclass
class HandResult:
    """手部识别结果"""

    hand_landmarks: list
    gestures: list
    hand_centers: list[tuple[float, float]]  # 每只手的腕部坐标 (x, y)，归一化坐标
    timestamp_ms: int
    left_direction: dict | None = None  # {"x": float, "y": float}，检测到左手指示时非空


class HandsRecognizer:
    """手势识别器"""

    def __init__(
        self,
        model: Path,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.7,
        min_hand_presence_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        preview: bool = False,
    ):
        self.model = model
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.preview = preview

        self._recognizer: Optional[vision.GestureRecognizer] = None
        self._result_callbacks: list[Callable[[HandResult], None]] = []
        self._direction_callbacks: list[Callable[[dict], None]] = []
        self._current_gestures: dict[int, str] = {}

        # 左手方向指示功能开关
        self._left_hand_direction_enabled = False

        # 帧尺寸（从实际图像更新，用于方向向量计算）
        self._frame_width = 1280
        self._frame_height = 720

        # 线程安全的结果缓存
        self._lock = threading.Lock()
        self._cached_result: Optional[HandResult] = None

        # FPS 追踪
        self._fps_counter = 0
        self._fps_start_time = time.time()
        self._fps = 0.0
        self._fps_avg_frame_count = 10

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """初始化识别器"""
        base_options = mp_python.BaseOptions(model_asset_path=str(self.model))
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            result_callback=self._on_result,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

    def stop(self) -> None:
        """停止识别器并释放资源"""
        if self._recognizer:
            self._recognizer.close()
            self._recognizer = None

    def recognize_async(self, rgb_image, timestamp_ms: int) -> None:
        """异步识别手势（外部送帧模式使用）"""
        if self._recognizer is None:
            logger.warning("recognize_async called before start() — frame discarded")
            return
        h, w = rgb_image.shape[:2]
        self._frame_width = w
        self._frame_height = h
        mp_image = mediapipe.Image(
            image_format=mediapipe.ImageFormat.SRGB, data=rgb_image
        )
        self._recognizer.recognize_async(mp_image, timestamp_ms)

    def on_result(self, callback: Callable[[HandResult], None]) -> None:
        """注册结果回调"""
        self._result_callbacks.append(callback)

    def on_direction(self, callback: Callable[[dict], None]) -> None:
        """注册方向结果回调（接收 {"x": float, "y": float}）"""
        self._direction_callbacks.append(callback)

    def set_left_hand_direction(self, enabled: bool) -> None:
        """开关左手方向指示功能"""
        self._left_hand_direction_enabled = enabled

    def close(self) -> None:
        """别名，兼容 close() 调用"""
        self.stop()

    def get_cached_result(self) -> Optional[HandResult]:
        """获取缓存的识别结果（用于预览）"""
        with self._lock:
            return self._cached_result

    # ------------------------------------------------------------------
    # MediaPipe 回调
    # ------------------------------------------------------------------

    def _is_finger_extended(self, hand: list, tip_idx: int, base_idx: int) -> bool:
        """判断手指是否大致伸直（指尖到掌骨基部的距离大于阈值）"""
        tip = hand[tip_idx]
        base = hand[base_idx]
        dist = ((tip.x - base.x) ** 2 + (tip.y - base.y) ** 2) ** 0.5
        return dist > 0.07

    def _on_result(
        self,
        result: vision.GestureRecognizerResult,
        unused_output_image: mediapipe.Image,
        timestamp_ms: int,
    ) -> None:
        # FPS
        self._fps_counter += 1
        if self._fps_counter % self._fps_avg_frame_count == 0:
            self._fps = self._fps_avg_frame_count / (time.time() - self._fps_start_time)
            self._fps_start_time = time.time()

        # 左手方向指示（每帧检测，在创建 HandResult 前计算）
        left_dir = None
        if self._left_hand_direction_enabled:
            left_dir = self._compute_left_direction(result)

        hand_result = HandResult(
            hand_landmarks=result.hand_landmarks,
            gestures=result.gestures,
            hand_centers=[(hand[0].x, hand[0].y) for hand in result.hand_landmarks],
            timestamp_ms=timestamp_ms,
            left_direction=left_dir,
        )

        # 缓存
        with self._lock:
            self._cached_result = hand_result

        # 检测手势变化并触发回调
        changed = False
        for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
            gesture = ""
            if result.gestures and hand_index < len(result.gestures):
                gesture = result.gestures[hand_index][0].category_name

            if self._current_gestures.get(hand_index, "") != gesture:
                self._current_gestures[hand_index] = gesture
                changed = True

        if changed:
            for callback in self._result_callbacks:
                callback(hand_result)

        # 触发方向回调
        if left_dir and self._direction_callbacks:
            for callback in self._direction_callbacks:
                callback(left_dir)

    def _compute_left_direction(
        self, result: vision.GestureRecognizerResult
    ) -> dict | None:
        """计算左手食指方向向量，返回归一化的 {"x": float, "y": float} 或 None"""
        # 选取左手（两只手时取left，一只时直接用）
        target_hand = None

        if len(result.hand_landmarks) == 1:
            target_hand = result.hand_landmarks[0]
        elif len(result.hand_landmarks) == 2 and result.handedness:
            for i, handedness in enumerate(result.handedness):
                if handedness[0].category_name == "Left":
                    target_hand = result.hand_landmarks[i]
                    break
            # 如果没找到left手（可能right手在前面），用第一只
            if target_hand is None:
                target_hand = result.hand_landmarks[0]

        if target_hand is None:
            return None

        # 只判定食指是否伸直（食指尖=8, 食指掌骨基部=5）
        if not self._is_finger_extended(target_hand, 8, 5):
            return None

        # 方向向量 = 食指尖 - 食指掌骨基部，按实际帧尺寸转像素后归一化
        # y 在像素坐标系向下为正，取反使其向上为正
        dx = (target_hand[8].x - target_hand[5].x) * self._frame_width
        dy = -(target_hand[8].y - target_hand[5].y) * self._frame_height
        mag = (dx * dx + dy * dy) ** 0.5
        if mag > 0:
            dx, dy = dx / mag, dy / mag
        return {
            "x": round(dx, 3),
            "y": round(dy, 3),
        }

    # ------------------------------------------------------------------
    # 预览渲染（CameraCapture 调用）
    # ------------------------------------------------------------------

    def render_preview(self, image: np.ndarray) -> np.ndarray:
        """在图像上渲染识别结果（返回绘制后的图像）"""
        with self._lock:
            result = self._cached_result

        handedness_color = (88, 205, 54)
        direction_color = (255, 255, 0)
        font_size, font_thickness = 1, 1

        # FPS 覆盖层（左上角）
        fps_text = f"HANDS FPS = {self._fps:.1f}"
        cv2.putText(
            image,
            fps_text,
            (24, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            font_size,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

        # 左手方向指示（已开启且检测到姿势时显示）
        if result is not None and result.left_direction:
            dir_text = f"INDEX x={result.left_direction['x']:.3f} y={result.left_direction['y']:.3f}"
            cv2.putText(
                image,
                dir_text,
                (24, 80),
                cv2.FONT_HERSHEY_DUPLEX,
                font_size,
                direction_color,
                font_thickness,
                cv2.LINE_AA,
            )

        if result is not None:
            frame_h, frame_w = image.shape[:2]
            for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
                # 绘制骨架
                hand_proto = landmark_pb2.NormalizedLandmarkList()  # ty:ignore[unresolved-attribute]
                hand_proto.landmark.extend(
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)  # ty:ignore[unresolved-attribute]
                    for lm in hand_landmarks
                )
                mp_drawing.draw_landmarks(
                    image,
                    hand_proto,
                    mp_hands.HAND_CONNECTIONS,  # ty:ignore[invalid-argument-type]
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # 手势文字
                if result.gestures and hand_index < len(result.gestures):
                    x_min_px = int(min(lm.x for lm in hand_landmarks) * frame_w)
                    y_min_px = int(min(lm.y for lm in hand_landmarks) * frame_h) - 10
                    gesture = result.gestures[hand_index][0]
                    text = f"{gesture.category_name} ({round(gesture.score, 2)})"
                    cv2.putText(
                        image,
                        text,
                        (x_min_px, y_min_px),
                        cv2.FONT_HERSHEY_DUPLEX,
                        font_size,
                        handedness_color,
                        font_thickness,
                        cv2.LINE_AA,
                    )

        return image
