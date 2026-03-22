"""Thread-safe event bus shared by all detectors and the TCP sender."""

from __future__ import annotations

import queue
import threading
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from echoesphere_omni.events import UnifiedEvent


class EventBus:
    """A single-threaded, thread-safe queue that acts as a shared event
    conduit between detectors (producers) and the TCP sender (consumer).

    Detectors call ``publish(event)`` from their own threads.
    The TCP sender calls ``get_event()`` to retrieve events.

    The bus intentionally uses ``queue.Queue`` (not ``asyncio.Queue``) so
    that it stays agnostic to any event-loop state and works across
    arbitrary ``threading`` boundaries.
    """

    def __init__(self, maxsize: int = 100) -> None:
        self._queue: queue.Queue[UnifiedEvent] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._subscribers: list[Callable] = []

    def publish(self, event: UnifiedEvent) -> None:
        """Publish an event from a detector. Never blocks the caller."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Back-pressure: drop the oldest event and try again
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                # Queue is completely stuck; silently drop
                pass

        # Notify subscribers (e.g. for debug logging)
        with self._lock:
            for sub in self._subscribers:
                try:
                    sub(event)
                except Exception:
                    pass

    def get_event(self, timeout: float = 0.05) -> UnifiedEvent | None:
        """Blocking read for the consumer. Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def subscribe(self, callback: Callable) -> None:
        """Register a callback to be called on every published event."""
        with self._lock:
            self._subscribers.append(callback)
