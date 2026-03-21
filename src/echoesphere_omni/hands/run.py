"""Entry point for the hand-tracking + TCP pipeline.

Architecture
------------
Two parallel execution paths share data through a thread-safe ``queue.Queue``:

1. **Main thread** — runs the OpenCV capture loop (``HandDetector.start()``).
   It repeatedly calls ``detector.detect_async()`` which fires
   ``result_callback`` in MediaPipe's internal thread.
   The callback converts each detection into a ``GestureEvent`` and puts
   it into the shared queue (never blocking the detection pipeline).

2. **Background sender thread** — owns its own asyncio event loop.
   It runs ``GestureTCPHandler`` which continuously reads events from the
   same shared queue and ``await``s transmission through ``TcpClient``.

This keeps all asyncio I/O completely off the cv2/OpenCV thread and avoids
any coupling between the two subsystems.
"""

from __future__ import annotations

import argparse
import asyncio
import queue
import threading
from typing import NoReturn

from echoesphere_omni.hands.detector import HandDetector
from echoesphere_omni.hands.gesture_emitter import GestureEmitter
from echoesphere_omni.hands.gestures import GestureEvent
from echoesphere_omni.net.client import TcpClient
from echoesphere_omni.net.event_handler import GestureTCPHandler


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/hand_landmarker.task",
        help="Path to the MediaPipe hand landmarker model file.",
    )
    parser.add_argument(
        "--numHands",
        type=int,
        default=2,
        help="Maximum number of hands to detect.",
    )
    parser.add_argument(
        "--minHandDetectionConfidence",
        type=float,
        default=0.5,
        help="Minimum confidence for hand detection.",
    )
    parser.add_argument(
        "--minHandPresenceConfidence",
        type=float,
        default=0.5,
        help="Minimum confidence for hand presence.",
    )
    parser.add_argument(
        "--minTrackingConfidence",
        type=float,
        default=0.5,
        help="Minimum confidence for hand tracking.",
    )
    parser.add_argument(
        "--cameraId",
        type=int,
        default=0,
        help="Camera device index.",
    )
    parser.add_argument(
        "--frameWidth",
        type=int,
        default=1280,
        help="Camera capture frame width.",
    )
    parser.add_argument(
        "--frameHeight",
        type=int,
        default=960,
        help="Camera capture frame height.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="TCP server host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=65432,
        help="TCP server port.",
    )
    return parser


async def _sender_loop(
    event_queue: queue.Queue[GestureEvent],
    host: str,
    port: int,
) -> NoReturn:
    """Async consumer: reads events from a thread-safe queue and sends them."""
    client = TcpClient(host, port)
    await client.connect()
    await client.send_text("hand_tracker connected")

    handler = GestureTCPHandler(client)
    handler.attach_queue(event_queue)

    # Pump events directly rather than going through start()/stop()
    while True:
        try:
            event = event_queue.get()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
        try:
            await handler._dispatch(event)
        except Exception:
            import traceback
            traceback.print_exc()


def _run_detector(
    event_queue: queue.Queue[GestureEvent],
    args: argparse.Namespace,
) -> None:
    emitter = GestureEmitter(event_queue)
    detector = HandDetector(
        emitter,
        model=args.model,
        num_hands=args.numHands,
        min_hand_detection_confidence=args.minHandDetectionConfidence,
        min_hand_presence_confidence=args.minHandPresenceConfidence,
        min_tracking_confidence=args.minTrackingConfidence,
        camera_id=args.cameraId,
        frame_width=args.frameWidth,
        frame_height=args.frameHeight,
    )
    detector.start()


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    # Thread-safe queue bridging detection → async sender
    event_queue: queue.Queue[GestureEvent] = queue.Queue(maxsize=100)

    # Start the asyncio sender thread
    sender_thread = threading.Thread(
        target=lambda: asyncio.run(
            _sender_loop(event_queue, args.host, args.port)
        ),
        name="tcp-sender",
        daemon=True,
    )
    sender_thread.start()

    # Run the detector (blocking — owns the OpenCV main loop)
    _run_detector(event_queue, args)


if __name__ == "__main__":
    main()
