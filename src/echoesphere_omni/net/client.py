"""TCP client with a simple length-prefixed binary protocol.

Protocol
--------
Each message is encoded as:
  - 4 bytes (big-endian, signed int): total byte length of the payload
  - 1 byte: message type (0x00 = TEXT, 0x01 = IMAGE)
  - N bytes: payload data

All integers use network byte order (big-endian).
"""

import asyncio
import struct
import traceback
from typing import Callable, Optional

import io
from PIL import Image


class MessageType:
    TEXT = 0x00
    IMAGE = 0x01


class TcpClient:
    """Async TCP client that connects to a single server.

    Supports text and image messages with a length-prefixed framing protocol.
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_text: Optional[Callable[[str], None]] = None,
        on_image: Optional[Callable[[bytes], None]] = None,
    ):
        self.host = host
        self.port = port
        self._on_text = on_text or (lambda msg: None)
        self._on_image = on_image or (lambda img: None)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._receive_task: Optional[asyncio.Task[None]] = None

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_connection(
            self.host, self.port
        )
        self._receive_task = asyncio.create_task(self._receive_loop())
        print(f"[TcpClient] Connected to {self.host}:{self.port}")

    async def _receive_loop(self) -> None:
        try:
            while True:
                length_data = await self._reader.readexactly(4)
                total_length = struct.unpack("!i", length_data)[0]
                data_with_type = await self._reader.readexactly(total_length)
                msg_type = data_with_type[0]
                payload = data_with_type[1:]

                if msg_type == MessageType.TEXT:
                    self._on_text(payload.decode("utf-8"))
                elif msg_type == MessageType.IMAGE:
                    self._on_image(payload)
                else:
                    print(f"[TcpClient] Unknown message type: {msg_type}")
        except asyncio.IncompleteReadError:
            print("[TcpClient] Connection closed by peer")
        except ConnectionResetError:
            print("[TcpClient] Connection reset")
        except Exception:
            traceback.print_exc()
        finally:
            await self.close()

    async def send_text(self, text: str) -> None:
        """Send a UTF-8 text message."""
        if not self._writer:
            print("[TcpClient] Not connected")
            return
        data = text.encode("utf-8")
        total_length = 1 + len(data)
        self._writer.write(struct.pack("!i", total_length))
        self._writer.write(bytes([MessageType.TEXT]) + data)
        await self._writer.drain()

    async def send_image(self, image_bytes: bytes) -> None:
        """Send raw image bytes."""
        if not self._writer:
            print("[TcpClient] Not connected")
            return
        total_length = 1 + len(image_bytes)
        self._writer.write(struct.pack("!i", total_length))
        self._writer.write(bytes([MessageType.IMAGE]) + image_bytes)
        await self._writer.drain()

    async def close(self) -> None:
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
        if self._receive_task:
            self._receive_task.cancel()
            self._receive_task = None


# ----------------------------------------------------------------------
# Standalone test
# ----------------------------------------------------------------------
async def _test_main() -> None:
    client = TcpClient("127.0.0.1", 65432)
    await client.connect()

    async def periodic_send() -> None:
        for i in range(5):
            await asyncio.sleep(1)
            await client.send_text(f"Ping {i}")

    send_task = asyncio.create_task(periodic_send())
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        send_task.cancel()
        await client.close()


if __name__ == "__main__":
    asyncio.run(_test_main())
