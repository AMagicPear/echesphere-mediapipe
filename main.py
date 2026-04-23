import argparse
import asyncio
import threading
from pathlib import Path

from echo_omni.hands.recognizer import HandsRecognizer
from echoesphere_omni.net.client import TcpClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/gesture_recognizer.task",
        help="手势识别模型路径",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="摄像头 ID",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=1,
        help="最大检测手数",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="显示预览窗口",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="TCP 服务器地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=65432,
        help="TCP 服务器端口",
    )
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    client = TcpClient(args.host, args.port)

    async def on_gesture(r):
        if r.gestures:
            for gesture in r.gestures:
                await client.send_text(f"gesture:{gesture[0].category_name}")
        else:
            await client.send_text("gesture:none")

    def bridge(r):
        asyncio.run_coroutine_threadsafe(on_gesture(r), loop)

    recognizer = HandsRecognizer(
        model=Path(args.model),
        num_hands=args.num_hands,
        camera_id=args.camera_id,
        preview=args.preview,
    )
    recognizer.on_result(bridge)

    if args.preview:
        # 预览模式：识别器在主线程（cv2.imshow 必须在主线程），
        # TCP 协程在后台线程
        async def tcp_loop():
            await client.connect()
            await client.send_register("mediapipe", "hands")
            try:
                await asyncio.Future()
            except KeyboardInterrupt:
                pass
            finally:
                await client.close()

        tcp_thread = threading.Thread(target=lambda: loop.run_until_complete(tcp_loop()), daemon=True)
        tcp_thread.start()

        recognizer.start()
    else:
        # 无头模式：识别器在后台线程，TCP 协程在主线程
        recognizer_thread = threading.Thread(target=recognizer.start, daemon=True)
        recognizer_thread.start()

        async def run():
            await client.connect()
            await client.send_register("mediapipe", "hands")
            try:
                await asyncio.Future()
            except KeyboardInterrupt:
                pass
            finally:
                await client.close()

        try:
            loop.run_until_complete(run())
        finally:
            loop.close()


if __name__ == "__main__":
    main()
