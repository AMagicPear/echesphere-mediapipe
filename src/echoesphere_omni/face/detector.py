"""Face landmark detection using MediaPipe, publishing events to an EventBus."""

from __future__ import annotations

import collections.abc
import sys
import threading
import time
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from echoesphere_omni.event_bus import EventBus
from echoesphere_omni.events import UnifiedEvent


class FaceDetector:
    """MediaPipe face-landmark detector.

    Publishes ``UnifiedEvent`` objects to an ``EventBus`` when a face is
    detected or lost. Does NOT emit every frame — only on state transitions.
    """

    def __init__(
        self,
        event_bus: EventBus,
        *,
        model: str = "models/face_landmarker.task",
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        camera_id: int = 0,
        frame_width: int = 1280,
        frame_height: int = 960,
        preview: bool = True,
    ) -> None:
        self._bus = event_bus
        self._model = model
        self._num_faces = num_faces
        self._min_face_detection_confidence = min_face_detection_confidence
        self._min_face_presence_confidence = min_face_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._camera_id = camera_id
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._preview = preview

        self._cap: cv2.VideoCapture | None = None
        self._detector: vision.FaceLandmarker | None = None

        # Cached result for visualisation
        self._lock = threading.Lock()
        self._cached_result: vision.FaceLandmarkerResult | None = None

        # FPS tracking
        self._fps_counter = 0
        self._fps_start_time = time.time()
        self._fps = 0.0
        self._fps_avg_frame_count = 10

        # State: whether a face was present in the last processed frame
        self._face_present = False

    def _publish(self, event: str, data: dict[str, Any], ts: int) -> None:
        self._bus.publish(UnifiedEvent(source="face", event=event, data=data, timestamp_ms=ts))

    def _make_result_callback(self) -> "collections.abc.Callable":
        def callback(
            result: vision.FaceLandmarkerResult,
            unused_output_image: mp.Image,
            timestamp_ms: int,
        ) -> None:
            self._fps_counter += 1
            if self._fps_counter % self._fps_avg_frame_count == 0:
                self._fps = self._fps_avg_frame_count / (time.time() - self._fps_start_time)
                self._fps_start_time = time.time()

            with self._lock:
                self._cached_result = result

            face_now = len(result.face_landmarks) > 0 if result.face_landmarks else False

            if face_now and not self._face_present:
                self._face_present = True
                # Emit face centre position (nose tip ≈ landmark 1)
                landmarks = result.face_landmarks[0]
                nose = landmarks[1]  # nose tip
                self._publish("face_detected", {"x": nose.x, "y": nose.y}, timestamp_ms)
            elif not face_now and self._face_present:
                self._face_present = False
                self._publish("face_lost", {}, timestamp_ms)

        return callback

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)

        base_options = python.BaseOptions(model_asset_path=self._model)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=self._num_faces,
            min_face_detection_confidence=self._min_face_detection_confidence,
            min_face_presence_confidence=self._min_face_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            output_face_blendshapes=False,
            result_callback=self._make_result_callback(),
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

        if not self._preview:
            while self._cap.isOpened():
                success, image = self._cap.read()
                if not success:
                    sys.exit("ERROR: Unable to read from webcam.")
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                self._detector.detect_async(mp_image, int(time.time() * 1_000))
                time.sleep(0.01)
        else:
            self._run_with_preview()

        self.stop()

    def _run_with_preview(self) -> None:
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        row_size, left_margin = 50, 24
        text_color, font_size, font_thickness = (0, 0, 0), 1, 1

        while self._cap.isOpened():
            success, image = self._cap.read()
            if not success:
                sys.exit("ERROR: Unable to read from webcam.")

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            self._detector.detect_async(mp_image, int(time.time() * 1_000))

            fps_text = f"FPS = {self._fps:.1f}"
            cv2.putText(
                image, fps_text, (left_margin, row_size),
                cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness, cv2.LINE_AA,
            )

            with self._lock:
                result = self._cached_result

            if result and result.face_landmarks:
                for face_landmarks in result.face_landmarks:
                    proto = landmark_pb2.NormalizedLandmarkList()
                    proto.landmark.extend(
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                        for lm in face_landmarks
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=proto,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=proto,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=proto,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )

            cv2.imshow("face_landmarker", image)
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
