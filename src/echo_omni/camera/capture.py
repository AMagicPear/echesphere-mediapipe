import logging
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("CameraCapture")

# 帧回调类型
FrameCallback = Callable[[np.ndarray, int], None]


class CameraCapture:
    """共享相机捕获器，支持多个 recognizer 共享同一路视频流

    可作为上下文管理器使用::

        with CameraCapture() as camera:
            camera.start()
            ...
    """

    MAX_CONSECUTIVE_FAILURES = 30

    def __init__(
        self,
        camera_id: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
    ):
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height

        self._cap: Optional[cv2.VideoCapture] = None
        self._callbacks: list[FrameCallback] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None

    def __enter__(self) -> "CameraCapture":
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def start(self) -> None:
        """启动相机捕获（非阻塞）"""
        self._cap = cv2.VideoCapture(self.camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开相机 {self.camera_id}")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止相机捕获"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None

    def on_frame(self, callback: FrameCallback) -> None:
        """注册帧回调"""
        self._callbacks.append(callback)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新帧（主线程调用，用于预览）"""
        with self._lock:
            return self._latest_rgb.copy() if self._latest_rgb is not None else None

    def _capture_loop(self) -> None:
        """捕获循环（后台线程）"""
        failures = 0
        while self._running and self._cap.isOpened():
            success, image = self._cap.read()
            if not success:
                failures += 1
                if failures >= self.MAX_CONSECUTIVE_FAILURES:
                    logger.warning(
                        "Camera read failed %d consecutive times — camera may be disconnected",
                        failures,
                    )
                time.sleep(0.01)
                continue
            failures = 0

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.time() * 1_000)

            with self._lock:
                self._latest_rgb = rgb_image.copy()

            for callback in self._callbacks:
                callback(rgb_image, timestamp_ms)

            time.sleep(0.001)

    @property
    def is_running(self) -> bool:
        return self._running
