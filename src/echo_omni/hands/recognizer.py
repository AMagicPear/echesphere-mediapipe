from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass
import threading
import time
import sys

import cv2
import mediapipe
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import (
    hands as mp_hands,
    drawing_utils as mp_drawing,
    drawing_styles as mp_drawing_styles,
)


@dataclass
class HandResult:
    """手部识别结果"""

    hand_landmarks: list
    gestures: list
    timestamp_ms: int


class HandsRecognizer:
    """手势识别器"""

    def __init__(
        self,
        model: Path,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.7,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        camera_id: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        preview: bool = True,
    ):
        self.model = model
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.preview = preview

        self._recognizer: Optional[vision.GestureRecognizer] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._result_callbacks: list[Callable[[HandResult], None]] = []
        self._current_gestures: dict[int, str] = {}

        # 线程安全的结果缓存（回调线程 → 主显示循环）
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
        """启动识别器（打开相机并运行检测循环）"""
        self._cap = cv2.VideoCapture(self.camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

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

        if self.preview:
            self._run_with_preview()
        else:
            self._run_headless()

        self.stop()

    def stop(self) -> None:
        """停止识别器并释放资源"""
        if self._recognizer:
            self._recognizer.close()
            self._recognizer = None
        if self._cap:
            self._cap.release()
            self._cap = None
        if self.preview:
            cv2.destroyAllWindows()

    def recognize_async(self, rgb_image, timestamp_ms: int) -> None:
        """异步识别手势（外部送帧模式使用）"""
        if self._recognizer:
            mp_image = mediapipe.Image(
                image_format=mediapipe.ImageFormat.SRGB, data=rgb_image
            )
            self._recognizer.recognize_async(mp_image, timestamp_ms)

    def on_result(self, callback: Callable[[HandResult], None]) -> None:
        """注册结果回调"""
        self._result_callbacks.append(callback)

    def close(self) -> None:
        """别名，兼容 close() 调用"""
        self.stop()

    # ------------------------------------------------------------------
    # 内部循环
    # ------------------------------------------------------------------

    def _run_headless(self) -> None:
        """无预览模式：只做检测，不显示画面"""
        assert self._cap is not None
        while self._cap.isOpened():
            success, image = self._cap.read()
            if not success:
                sys.exit("ERROR: Unable to read from webcam.")
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.recognize_async(rgb_image, int(time.time() * 1_000))
            time.sleep(0.01)

    def _run_with_preview(self) -> None:
        """预览模式：检测 + 可视化"""
        assert self._cap is not None
        handedness_color = (88, 205, 54)
        font_size, font_thickness = 1, 1

        while self._cap.isOpened():
            success, image = self._cap.read()
            if not success:
                sys.exit("ERROR: Unable to read from webcam.")

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.recognize_async(rgb_image, int(time.time() * 1_000))

            # FPS 覆盖层
            fps_text = f"FPS = {self._fps:.1f}"
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

            # 绘制缓存的骨架
            with self._lock:
                result = self._cached_result

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
                        y_min_px = (
                            int(min(lm.y for lm in hand_landmarks) * frame_h) - 10
                        )
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

            cv2.imshow("gesture_recognition", image)
            if cv2.waitKey(1) == 27:
                break

    # ------------------------------------------------------------------
    # MediaPipe 回调
    # ------------------------------------------------------------------

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

        hand_result = HandResult(
            hand_landmarks=result.hand_landmarks,
            gestures=result.gestures,
            timestamp_ms=timestamp_ms,
        )

        # 缓存（可视化线程）
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
