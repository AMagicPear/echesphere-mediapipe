"""Main entry point for the unified tracker.

Orchestrates hand detector, face detector, event bus, and TCP sender.
All detectors run in parallel threads; the TCP sender runs in its own
background thread.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from echoesphere_omni.event_bus import EventBus
from echoesphere_omni.face import FaceDetector
from echoesphere_omni.hands.detector import HandDetector
from echoesphere_omni.sender import TcpSender


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preview", action="store_true", help="Enable OpenCV preview windows")
    parser.add_argument("--no-face", action="store_true", help="Disable face detector")
    parser.add_argument("--no-hand", action="store_true", help="Disable hand detector")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # TCP
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)

    # Camera
    parser.add_argument("--cameraId", type=int, default=0)
    parser.add_argument("--frameWidth", type=int, default=1280)
    parser.add_argument("--frameHeight", type=int, default=960)

    # Hand detector
    parser.add_argument("--handModel", type=str, default="models/hand_landmarker.task")
    parser.add_argument("--numHands", type=int, default=2)
    parser.add_argument("--minHandDetectionConfidence", type=float, default=0.5)
    parser.add_argument("--minHandPresenceConfidence", type=float, default=0.5)
    parser.add_argument("--minTrackingConfidence", type=float, default=0.5)

    # Face detector
    parser.add_argument("--faceModel", type=str, default="models/face_landmarker.task")
    parser.add_argument("--numFaces", type=int, default=1)
    parser.add_argument("--minFaceDetectionConfidence", type=float, default=0.5)
    parser.add_argument("--minFacePresenceConfidence", type=float, default=0.5)
    parser.add_argument("--minFaceTrackingConfidence", type=float, default=0.5)

    return parser


def _run_hand_detector(bus: EventBus, args: argparse.Namespace) -> None:
    detector = HandDetector(
        bus,
        model=args.handModel,
        num_hands=args.numHands,
        min_hand_detection_confidence=args.minHandDetectionConfidence,
        min_hand_presence_confidence=args.minHandPresenceConfidence,
        min_tracking_confidence=args.minTrackingConfidence,
        camera_id=args.cameraId,
        frame_width=args.frameWidth,
        frame_height=args.frameHeight,
        preview=args.preview,
    )
    detector.start()


def _run_face_detector(bus: EventBus, args: argparse.Namespace) -> None:
    detector = FaceDetector(
        bus,
        model=args.faceModel,
        num_faces=args.numFaces,
        min_face_detection_confidence=args.minFaceDetectionConfidence,
        min_face_presence_confidence=args.minFacePresenceConfidence,
        min_tracking_confidence=args.minFaceTrackingConfidence,
        camera_id=args.cameraId,
        frame_width=args.frameWidth,
        frame_height=args.frameHeight,
        preview=args.preview,
    )
    detector.start()


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.no_hand and args.no_face:
        logging.error("Error: at least one detector must be enabled (--no-hand and --no-face both set)")
        sys.exit(1)

    bus = EventBus()

    # Start TCP sender first
    sender = TcpSender(bus, host=args.host, port=args.port)
    sender.start()

    threads: list[threading.Thread] = []

    if not args.no_hand:
        t = threading.Thread(
            target=_run_hand_detector,
            args=(bus, args),
            name="hand-detector",
            daemon=True,
        )
        t.start()
        threads.append(t)

    if not args.no_face:
        t = threading.Thread(
            target=_run_face_detector,
            args=(bus, args),
            name="face-detector",
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
