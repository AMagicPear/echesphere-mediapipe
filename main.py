import argparse
import asyncio
import threading
import json
from pathlib import Path

import cv2

from echo_omni.hands.recognizer import HandsRecognizer
from echo_omni.face.recognizer import FaceRecognizer
from echo_omni.camera.capture import CameraCapture
from echoesphere_omni.net.client import TcpClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gesture_recognizer.task")
    parser.add_argument("--face-model", default="models/face_landmarker.task")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--num-hands", type=int, default=1)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    client = TcpClient(args.host, args.port)

    async def on_gesture(r):
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

    async def on_face(r):
        if r.blendshapes:
            blends = [{"category": k, "score": v} for k, v in r.blendshapes.items()]
            msg = json.dumps({"omni_type": "face_blendshape", "data": blends})
        else:
            msg = json.dumps({"omni_type": "face_blendshape", "data": []})
        await client.send_text(msg)

    def handle_gesture(r):
        asyncio.run_coroutine_threadsafe(on_gesture(r), loop)

    def handle_face(r):
        asyncio.run_coroutine_threadsafe(on_face(r), loop)

    recognizer = HandsRecognizer(
        model=Path(args.model),
        num_hands=args.num_hands,
        preview=args.preview,
    )
    recognizer.on_result(handle_gesture)
    recognizer.start()

    face_recognizer = FaceRecognizer(
        model=Path(args.face_model),
        preview=args.preview,
    )
    face_recognizer.on_result(handle_face)
    face_recognizer.start()

    camera = CameraCapture(
        camera_id=args.camera_id,
        frame_width=1280,
        frame_height=720,
    )
    camera.on_frame(recognizer.recognize_async)
    camera.on_frame(face_recognizer.recognize_async)

    async def tcp_loop():
        await client.connect()
        await client.send_register("mediapipe", "hands")
        await client.send_register("mediapipe", "face")
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

    camera.start()

    while True:
        if args.preview:
            rgb = camera.get_latest_frame()
            if rgb is not None:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                frame = recognizer.render_preview(bgr)
                frame = face_recognizer.render_preview(frame)
                cv2.imshow("recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    camera.stop()
    recognizer.stop()
    face_recognizer.stop()
    loop.call_soon_threadsafe(loop.stop)
    tcp_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
