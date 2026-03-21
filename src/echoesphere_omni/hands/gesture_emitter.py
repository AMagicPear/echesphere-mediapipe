import queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from echoesphere_omni.hands.gestures import GestureEvent


class GestureEmitter:
    """Thread-safe emitter that bridges synchronous detection callbacks
    to a background asyncio sender thread via a ``queue.Queue``.

    The cv2/ OpenCV thread runs the detector and calls ``emit()``.
    A dedicated sender thread owns an asyncio event loop and consumes
    events from the same queue to perform non-blocking TCP I/O.

    This class intentionally avoids any asyncio imports so it remains
    usable from any thread without coordinating event-loop state.
    """

    __slots__ = ("_queue",)

    def __init__(self, event_queue: queue.Queue["GestureEvent"]) -> None:
        self._queue = event_queue

    def emit(self, event: "GestureEvent") -> None:
        """Put an event into the queue. Safe to call from any thread.

        Uses ``put_nowait`` so this never blocks the detection thread.
        If the queue is full the event is dropped, preventing a slow
        network consumer from stalling hand detection.
        """
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Back-pressured; drop the event rather than stall detection
            pass
