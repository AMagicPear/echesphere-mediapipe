"""Async consumer that reads ``GestureEvent`` objects from a thread-safe
``queue.Queue`` and forwards serialised representations through a ``TcpClient``.
"""

import asyncio
import queue
import traceback

from echoesphere_omni.hands.gestures import GestureEvent
from echoesphere_omni.net.client import TcpClient


class GestureTCPHandler:
    """Consumes ``GestureEvent`` objects from a ``queue.Queue`` and forwards
    them over TCP via a ``TcpClient``.

    This class is entirely async and must run inside an asyncio event loop.
    It polls the shared ``queue.Queue`` with ``asyncio.sleep``, ensuring
    non-blocking reads without mixing asyncio primitives with ``threading``
    primitives across thread boundaries.
    """

    def __init__(self, client: TcpClient) -> None:
        self._client = client
        self._queue: queue.Queue[GestureEvent] | None = None
        self._running = False

    def attach_queue(self, event_queue: queue.Queue[GestureEvent]) -> None:
        """Attach the shared thread-safe queue to this handler."""
        self._queue = event_queue

    async def start(self) -> None:
        """Run the consume loop. Blocks until ``stop()`` is called."""
        if self._queue is None:
            raise RuntimeError("Queue not attached; call attach_queue() first.")
        self._running = True
        while self._running:
            try:
                event = self._queue.get(timeout=0.05)
            except queue.Empty:
                await asyncio.sleep(0.005)
                continue
            try:
                await self._dispatch(event)
            except Exception:
                traceback.print_exc()
            finally:
                self._queue.task_done()

    async def _dispatch(self, event: GestureEvent) -> None:
        """Map a ``GestureEvent`` to a TCP message and send it."""
        payload = event.to_tcp_payload()
        if payload:
            await self._client.send_text(payload)

    def stop(self) -> None:
        """Gracefully stop the consume loop after the current event."""
        self._running = False
