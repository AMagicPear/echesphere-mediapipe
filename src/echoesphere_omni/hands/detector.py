"""Hand landmark detection using MediaPipe with a state-machine gesture emitter."""

from __future__ import annotations

import sys
import collections.abc
import threading
import time
from enum import Enum, auto
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from echoesphere_omni.event_bus import EventBus
from echoesphere_omni.events import UnifiedEvent
from echoesphere_omni.hands.gestures import (
    GestureThresholds,
    HandGestureType,
    HandLandmark,
)


class _HandState(Enum):
    NO_HAND = auto()
    HAND_PRESENT = auto()
    PINCH = auto()
    PINCH_RELEASED = auto()


class HandDetector:
    """MediaPipe hand landmarker with a gesture state machine.

    Publishes ``UnifiedEvent`` objects to an ``EventBus`` whenever a gesture
    transition occurs (hand detected / lost, pinch, swipe, two-hand open).
    Does NOT emit every frame — only state changes.
    """

    def __init__(
        self,
        event_bus: EventBus,
        *,
        model: str = "models/hand_landmarker.task",
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        camera_id: int = 0,
        frame_width: int = 1280,
        frame_height: int = 960,
        preview: bool = True,
    ) -> None:
        self._bus = event_bus
        self._model = model
        self._num_hands = num_hands
        self._min_hand_detection_confidence = min_hand_detection_confidence
        self._min_hand_presence_confidence = min_hand_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._camera_id = camera_id
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._preview = preview

        self._cap: cv2.VideoCapture | None = None
        self._detector: vision.HandLandmarker | None = None

        # Cached landmarks for visualisation (callback thread → display loop)
        self._lock = threading.Lock()
        self._cached_landmarks: list[Any] = []
        self._cached_handedness: list[Any] = []

        # FPS tracking
        self._fps_counter = 0
        self._fps_start_time = time.time()
        self._fps = 0.0
        self._fps_avg_frame_count = 10

        # --- Gesture state machine ---
        self._state = _HandState.NO_HAND
        # Track which hand (by index) is left/right
        self._left_hand: dict[str, Any] | None = None
        self._right_hand: dict[str, Any] | None = None
        # Swipe tracking: {hand_idx: (last_x, last_time_ms)}
        self._swipe_track: dict[int, tuple[float, int]] = {}
        self._last_swipe: dict[int, str] = {}  # hand_idx -> last emitted direction

    # ------------------------------------------------------------------
    # Gesture helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _landmark_dist(a: Any, b: Any) -> float:
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    def _is_pinching(self, hand: Any) -> bool:
        thumb = hand[HandLandmark.THUMB_TIP]
        index = hand[HandLandmark.INDEX_TIP]
        return self._landmark_dist(thumb, index) < GestureThresholds.PINCH_THRESHOLD

    def _is_hand_open(self, hand: Any) -> bool:
        """All four fingertips far enough from wrist → hand is open."""
        wrist = hand[HandLandmark.WRIST]
        for tip_idx in [
            HandLandmark.INDEX_TIP,
            HandLandmark.MIDDLE_TIP,
            HandLandmark.RING_TIP,
            HandLandmark.PINKY_TIP,
        ]:
            if (
                self._landmark_dist(hand[tip_idx], wrist)
                < GestureThresholds.OPEN_FINGER_DISTANCE
            ):
                return False
        return True

    def _check_swipe(self, hand_idx: int, hand: Any, timestamp_ms: int) -> str | None:
        """Detect swipe left/right. Returns direction string or None."""
        wrist = hand[HandLandmark.WRIST]
        if hand_idx not in self._swipe_track:
            self._swipe_track[hand_idx] = (wrist.x, timestamp_ms)
            return None

        last_x, last_t = self._swipe_track[hand_idx]
        dt = max(timestamp_ms - last_t, 1)
        velocity = (wrist.x - last_x) / dt

        self._swipe_track[hand_idx] = (wrist.x, timestamp_ms)

        if abs(velocity) < GestureThresholds.SWIPE_VELOCITY_THRESHOLD:
            return None

        direction = "swipe_right" if velocity > 0 else "swipe_left"
        last = self._last_swipe.get(hand_idx)
        if last == direction:
            return None  # Already emitted this direction recently

        self._last_swipe[hand_idx] = direction
        return direction

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _publish(self, gesture: HandGestureType, data: dict[str, Any], ts: int) -> None:
        self._bus.publish(
            UnifiedEvent(source="hand", event=gesture.value, data=data, timestamp_ms=ts)
        )

    def _on_hands_detected(
        self, hands: list[tuple[Any, Any]], timestamp_ms: int
    ) -> None:
        """Process all detected hands through the state machine."""
        prev_state = self._state

        if len(hands) == 0:
            if prev_state != _HandState.NO_HAND:
                self._state = _HandState.NO_HAND
                self._publish(HandGestureType.HAND_LOST, {}, timestamp_ms)
                self._left_hand = None
                self._right_hand = None
                self._swipe_track.clear()
            return

        # Identify left / right hands
        left_hand = None
        right_hand = None
        for hand_landmarks, handedness in hands:
            label = handedness[0].category_name.lower()
            if label == "left":
                left_hand = hand_landmarks
            else:
                right_hand = hand_landmarks

        self._left_hand = left_hand
        self._right_hand = right_hand

        if prev_state == _HandState.NO_HAND:
            self._state = _HandState.HAND_PRESENT
            self._publish(HandGestureType.HAND_DETECTED, {}, timestamp_ms)
            # Reset swipe state
            self._swipe_track.clear()
            self._last_swipe.clear()

        # Gesture checks on top of HAND_PRESENT / PINCH_RELEASED
        both_open = (
            left_hand is not None
            and right_hand is not None
            and self._is_hand_open(left_hand)
            and self._is_hand_open(right_hand)
        )
        if both_open:
            self._publish(HandGestureType.OPEN_BOTH_HANDS, {}, timestamp_ms)

        # Per-hand swipe detection
        for hand_landmarks, handedness in hands:
            hand_idx = handedness[0].index
            swipe_dir = self._check_swipe(hand_idx, hand_landmarks, timestamp_ms)
            if swipe_dir:
                self._publish(
                    HandGestureType.SWIPE_RIGHT
                    if swipe_dir == "swipe_right"
                    else HandGestureType.SWIPE_LEFT,
                    {},
                    timestamp_ms,
                )

        # Pinch state machine
        any_pinching = any(self._is_pinching(h) for h, _ in hands if h is not None)

        if prev_state == _HandState.HAND_PRESENT and any_pinching:
            self._state = _HandState.PINCH
            # Emit pinch with average position of all pinching hands
            avg_x = sum(
                h[HandLandmark.INDEX_TIP].x for h, _ in hands if h is not None
            ) / len(hands)
            avg_y = sum(
                h[HandLandmark.INDEX_TIP].y for h, _ in hands if h is not None
            ) / len(hands)
            self._publish(HandGestureType.PINCH, {"x": avg_x, "y": avg_y}, timestamp_ms)

        elif prev_state == _HandState.PINCH and not any_pinching:
            self._state = _HandState.PINCH_RELEASED
            self._publish(HandGestureType.PINCH_RELEASED, {}, timestamp_ms)

        elif prev_state == _HandState.PINCH_RELEASED:
            # Stay in PINCH_RELEASED until hands go away or we re-enter PINCH
            if any_pinching:
                self._state = _HandState.PINCH
                avg_x = sum(
                    h[HandLandmark.INDEX_TIP].x for h, _ in hands if h is not None
                ) / len(hands)
                avg_y = sum(
                    h[HandLandmark.INDEX_TIP].y for h, _ in hands if h is not None
                ) / len(hands)
                self._publish(
                    HandGestureType.PINCH, {"x": avg_x, "y": avg_y}, timestamp_ms
                )
            elif not any_pinching:
                # No hands at all
                self._state = _HandState.NO_HAND
                self._publish(HandGestureType.HAND_LOST, {}, timestamp_ms)

        # If already HAND_PRESENT, just stay there (don't re-emit HAND_DETECTED)

    # ------------------------------------------------------------------
    # MediaPipe callback
    # ------------------------------------------------------------------

    def _make_result_callback(self) -> "collections.abc.Callable":
        def callback(
            result: vision.HandLandmarkerResult,
            unused_output_image: mp.Image,
            timestamp_ms: int,
        ) -> None:
            # FPS
            self._fps_counter += 1
            if self._fps_counter % self._fps_avg_frame_count == 0:
                self._fps = self._fps_avg_frame_count / (
                    time.time() - self._fps_start_time
                )
                self._fps_start_time = time.time()

            # Cache for visualisation
            with self._lock:
                self._cached_landmarks = list(result.hand_landmarks or [])
                self._cached_handedness = list(result.handedness or [])

            # State machine
            hands = list(zip(result.hand_landmarks or [], result.handedness or []))
            self._on_hands_detected(hands, timestamp_ms)

        return callback

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the camera, init detector, run the display loop (or detection only)."""
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

        if not self._preview:
            # Headless: just run detection without any window
            while self._cap.isOpened():
                success, image = self._cap.read()
                if not success:
                    sys.exit("ERROR: Unable to read from webcam.")
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                self._detector.detect_async(mp_image, int(time.time() * 1_000))
                time.sleep(0.01)  # ~100 fps max
        else:
            self._run_with_preview()

        self.stop()

    def _run_with_preview(self) -> None:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        row_size, left_margin = 50, 24
        text_color, font_size, font_thickness = (0, 0, 0), 1, 1
        handedness_color = (88, 205, 54)

        while self._cap.isOpened():
            success, image = self._cap.read()
            if not success:
                sys.exit("ERROR: Unable to read from webcam.")

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            self._detector.detect_async(mp_image, int(time.time() * 1_000))

            # FPS overlay
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

            # Draw cached landmarks
            with self._lock:
                landmarks = list(self._cached_landmarks)
                handedness = list(self._cached_handedness)

            if landmarks:
                for hand_landmarks, handedness_item in zip(landmarks, handedness):
                    proto = landmark_pb2.NormalizedLandmarkList()  # ty:ignore[unresolved-attribute]
                    proto.landmark.extend(
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)  # ty:ignore[unresolved-attribute]
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
                    text_x = int(min(x_coords) * w)
                    text_y = int(min([lm.y for lm in hand_landmarks]) * h) - 10
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

    def stop(self) -> None:
        if self._detector:
            self._detector.close()
            self._detector = None
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._preview:
            cv2.destroyAllWindows()
