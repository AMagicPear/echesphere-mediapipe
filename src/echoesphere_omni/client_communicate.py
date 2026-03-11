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
    def __init__(
        self,
        host: str,
        port: int,
        on_message: Optional[Callable[[str], None]] = None,
        on_image: Optional[Callable[[bytes], None]] = None,
    ):
        self.host = host
        self.port = port
        self.on_message = on_message or (lambda msg: print(f"[消息] {msg}"))
        self.on_image = on_image or (lambda img: print(f"[图像] 收到 {len(img)} 字节"))
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self._receive_task: Optional[asyncio.Task] = None

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        self._receive_task = asyncio.create_task(self._receive_loop())
        print(f"已连接到 {self.host}:{self.port}")

    async def _receive_loop(self):
        try:
            while True:
                # 读取4字节长度（网络字节序）
                length_data = await self.reader.readexactly(4)  # ty:ignore[possibly-missing-attribute]
                total_length = struct.unpack("!i", length_data)[0]
                # 读取类型+数据
                data_with_type = await self.reader.readexactly(total_length)  # ty:ignore[possibly-missing-attribute]
                msg_type = data_with_type[0]
                payload = data_with_type[1:]

                if msg_type == MessageType.TEXT:
                    message = payload.decode("utf-8")
                    self.on_message(message)
                elif msg_type == MessageType.IMAGE:
                    self.on_image(payload)
                else:
                    print(f"[警告] 未知消息类型: {msg_type}")
        except (asyncio.IncompleteReadError, ConnectionResetError) as e:
            print(f"[错误] 连接中断: {e}")
            traceback.print_exc()
        except Exception as e:
            print(f"[错误] 接收循环异常: {e}")
            traceback.print_exc()
        finally:
            await self.close()

    async def send_text(self, text: str):
        if not self.writer:
            print("[警告] 未连接，无法发送")
            return
        data = text.encode("utf-8")
        total_length = 1 + len(data)
        self.writer.write(struct.pack("!i", total_length))
        self.writer.write(bytes([MessageType.TEXT]) + data)
        await self.writer.drain()
        print(f"[发送] {text}")

    async def send_image(self, image_bytes: bytes):
        if not self.writer:
            print("[警告] 未连接，无法发送")
            return
        total_length = 1 + len(image_bytes)
        self.writer.write(struct.pack("!i", total_length))
        self.writer.write(bytes([MessageType.IMAGE]) + image_bytes)
        await self.writer.drain()
        print(f"[发送] 图像 ({len(image_bytes)} 字节)")

    async def close(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            self.writer = None
        if self._receive_task:
            self._receive_task.cancel()
            self._receive_task = None


def on_message(msg: str):
    print(f"收到消息: {msg}")


def on_image(data: bytes):
    print(f"收到图像, 大小: {len(data)} 字节")
    try:
        image = Image.open(io.BytesIO(data))
        image.show()
    except Exception as e:
        print(f"图像显示失败: {e}")


async def main():
    client = TcpClient("127.0.0.1", 65432, on_message, on_image)
    await client.connect()

    async def periodic_send():
        counter = 0
        while True:
            await asyncio.sleep(2)
            await client.send_text(f"来自Python的消息 {counter}")
            counter += 1

    send_task = asyncio.create_task(periodic_send())
    try:
        await asyncio.Future()  # 运行直到中断
    except KeyboardInterrupt:
        print("用户中断")
    finally:
        send_task.cancel()
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
