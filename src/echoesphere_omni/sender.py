"""Background thread that owns an asyncio event loop and sends events over TCP."""

from __future__ import annotations

import asyncio
import logging
import threading
import traceback

from echoesphere_omni.event_bus import EventBus
from echoesphere_omni.events import UnifiedEvent
from echoesphere_omni.net.client import TcpClient

logger = logging.getLogger("TcpSender")


class TcpSender:
    """Owns a background thread running an asyncio event loop.

    Reads ``UnifiedEvent`` objects from an ``EventBus`` and sends them
    over TCP via ``TcpClient``. Any detector can publish to the bus;
    this class is the sole consumer.
    """

    def __init__(
        self,
        event_bus: EventBus,
        host: str = "127.0.0.1",
        port: int = 65432,
    ) -> None:
        self._bus = event_bus
        self._host = host
        self._port = port
        self._thread: threading.Thread | None = None
        self._client: TcpClient | None = None
        self._running = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="tcp-sender", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.run(self._async_main())

    async def _async_main(self) -> None:
        self._client = TcpClient(self._host, self._port)
        try:
            await self._client.connect()
            logger.info(f"Connected to TCP server {self._host}:{self._port}")
            # 发送注册消息（同时包含 hand 和 face 检测）
            await self._client.send_register("mediapipe")
            logger.debug("Registration message sent")
        except Exception:
            traceback.print_exc()
            return

        while self._running:
            event = self._bus.get_event(timeout=0.05)
            if event is None:
                await asyncio.sleep(0.005)
                continue
            try:
                await self._dispatch(event)
            except Exception:
                traceback.print_exc()

    async def _dispatch(self, event: UnifiedEvent) -> None:
        if self._client is None:
            return
        payload = event.to_json()
        if payload:
            logger.debug(f"Sending event: {payload}")
            await self._client.send_text(payload)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
