"""Hand landmark detection using MediaPipe, emitting ``GestureEvent`` objects."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from echoesphere_omni.hands.gesture_emitter import GestureEmitter
from echoesphere_omni.hands.gestures import GestureEvent, GestureType


# Index of the index-finger tip landmark in MediaPipe's hand landmark model
_INDEX_TIP = 8


class HandDetector:
    """Captures camera frames, runs MediaPipe hand landmark detection,
    and emits structured ``GestureEvent`` objects through a ``GestureEmitter``.

    All TCP / network I/O is delegated to the ``GestureEmitter`` → asyncio
    pipeline. This class is entirely synchronous and must be called from
    the detection thread (e.g. the OpenCV display loop).
    """

    def __init__(
        self,
        emitter: GestureEmitter,
        *,
        model: str = "models/hand_landmarker.task",
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        camera_id: int = 0,
        frame_width: int = 1280,
        frame_height: int = 960,
    ) -> None:
        self._emitter = emitter
        self._model = model
        self._num_hands = num_hands
        self._min_hand_detection_confidence = min_hand_detection_confidence
        self._min_hand_presence_confidence = min_hand_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._camera_id = camera_id
        self._frame_width = frame_width
        self._frame_height = frame_height

        self._cap: cv2.VideoCapture | None = None
        self._detector: vision.HandLandmarker | None = None
        self._last_pos: tuple[float, float] = (0.5, 0.5)

        # FPS tracking
        self._fps_counter = 0
        self._fps_start_time = time.time()
        self._fps = 0.0
        self._fps_avg_frame_count = 10

        # Cached landmarks for visualisation (written by callback thread,
        # read by display loop thread — protected by a lock).
        self._lock = threading.Lock()
        self._cached_landmarks: list[Any] = []
        self._cached_handedness: list[Any] = []

    def _make_result_callback(self) -> callable:
        """Build the MediaPipe result callback for ``LIVE_STREAM`` mode."""

        def callback(
            result: vision.HandLandmarkerResult,
            unused_output_image: mp.Image,
            timestamp_ms: int,
        ) -> None:
            # FPS calculation
            self._fps_counter += 1
            if self._fps_counter % self._fps_avg_frame_count == 0:
                self._fps = self._fps_avg_frame_count / (
                    time.time() - self._fps_start_time
                )
                self._fps_start_time = time.time()

            # Cache landmarks for the display loop (under lock)
            with self._lock:
                self._cached_landmarks = list(result.hand_landmarks or [])
                self._cached_handedness = list(result.handedness or [])

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                pos_x = hand[_INDEX_TIP].x
                pos_y = hand[_INDEX_TIP].y

                velocity = (
                    (pos_x - self._last_pos[0]) ** 2
                    + (pos_y - self._last_pos[1]) ** 2
                ) ** 0.5
                self._last_pos = (pos_x, pos_y)

                self._emitter.emit(
                    GestureEvent(
                        gesture_type=GestureType.INDEX_TIP_MOVED,
                        data={"x": pos_x, "y": pos_y, "velocity": velocity},
                        timestamp_ms=timestamp_ms,
                    )
                )
            else:
                self._emitter.emit(
                    GestureEvent(
                        gesture_type=GestureType.HAND_LOST,
                        data={},
                        timestamp_ms=timestamp_ms,
                    )
                )

        return callback

    def start(self) -> None:
        """Open the camera, initialise the detector, and run the display loop."""
        self._cap = cv2.VideoCapture(self._camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)

        base_options = python.BaseOptions(model_asset_path=self._model)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=self._num_hands,
            min_hand_detection_confidence=self._min_hand_detection_confidence,
            min_hand_presence_confidence=self._min_hand_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            result_callback=self._make_result_callback(),
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        # Visualisation constants
        row_size = 50
        left_margin = 24
        text_color = (0, 0, 0)
        font_size = 1
        font_thickness = 1
        margin = 10
        handedness_color = (88, 205, 54)  # vibrant green

        while self._cap.isOpened():
            success, image = self._cap.read()
            if not success:
                sys.exit(
                    "ERROR: Unable to read from webcam. "
                    "Please verify your webcam settings."
                )

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb_image
            )

            self._detector.detect_async(
                mp_image, int(time.time() * 1_000)
            )

            # Draw FPS
            fps_text = f"FPS = {self._fps:.1f}"
            cv2.putText(
                image,
                fps_text,
                (left_margin, row_size),
                cv2.FONT_HERSHEY_DUPLEX,
                font_size,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )

            # Draw hand landmarks from the cached result (written by callback)
            with self._lock:
                landmarks = list(self._cached_landmarks)
                handedness = list(self._cached_handedness)

            if landmarks:
                for hand_landmarks, handedness_item in zip(landmarks, handedness):
                    proto = landmark_pb2.NormalizedLandmarkList()
                    proto.landmark.extend(
                        landmark_pb2.NormalizedLandmark(
                            x=lm.x, y=lm.y, z=lm.z
                        )
                        for lm in hand_landmarks
                    )
                    mp_drawing.draw_landmarks(
                        image,
                        proto,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    h, w = image.shape[:2]
                    x_coords = [lm.x for lm in hand_landmarks]
                    y_coords = [lm.y for lm in hand_landmarks]
                    text_x = int(min(x_coords) * w)
                    text_y = int(min(y_coords) * h) - margin
                    cv2.putText(
                        image,
                        handedness_item[0].category_name,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX,
                        font_size,
                        handedness_color,
                        font_thickness,
                        cv2.LINE_AA,
                    )

            cv2.imshow("hand_landmarker", image)
            if cv2.waitKey(1) == 27:
                break

        self.stop()

    def stop(self) -> None:
        if self._detector:
            self._detector.close()
            self._detector = None
        if self._cap:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()
