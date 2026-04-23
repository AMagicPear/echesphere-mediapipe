import argparse
import asyncio
import threading
from pathlib import Path

from echo_omni.hands.recognizer import HandsRecognizer
from echoesphere_omni.net.client import TcpClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gesture_recognizer.task")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--num-hands", type=int, default=1)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    client = TcpClient(args.host, args.port)

    async def on_gesture(r):
        import json

        if r.gestures:
            hands = [
                {
                    "gesture": g[0].category_name,
                    "x": round(c[0], 3),
                    "y": round(c[1], 3),
                }
                for g, c in zip(r.gestures, r.hand_centers)
            ]
            msg = json.dumps({"omni_type": "hand_gesture", "data": hands})
        else:
            msg = json.dumps({"omni_type": "hand_gesture", "data": []})
        await client.send_text(msg)

    def handle_result(r):
        asyncio.run_coroutine_threadsafe(on_gesture(r), loop)

    recognizer = HandsRecognizer(
        model=Path(args.model),
        num_hands=args.num_hands,
        camera_id=args.camera_id,
        preview=args.preview,
        frame_width=1280,
        frame_height=720,
    )
    recognizer.on_result(handle_result)

    # TCP 在后台线程运行事件循环
    async def tcp_loop():
        await client.connect()
        await client.send_register("mediapipe", "hands")
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            await client.close()

    tcp_thread = threading.Thread(
        target=lambda: loop.run_until_complete(tcp_loop()), daemon=True
    )
    tcp_thread.start()

    # 识别器在主线程（阻塞），ESC 退出后关闭事件循环
    recognizer.start()
    loop.call_soon_threadsafe(loop.stop)
    tcp_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
