# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run hand landmarker."""

from echoesphere_omni.client_communicate import TcpClient

import argparse
import sys
import time
import struct

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import asyncio
import queue


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None
LAST_POS = (0.5, 0.5)
TCP_CLIENT = None
MESSAGE_QUEUE = None
import queue

def run(
    model: str,
    num_hands: int,
    min_hand_detection_confidence: float,
    min_hand_presence_confidence: float,
    min_tracking_confidence: float,
    camera_id: int,
    width: int,
    height: int,
) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
        model: Name of the hand landmarker model bundle.
        num_hands: Max number of hands that can be detected by the landmarker.
        min_hand_detection_confidence: The minimum confidence score for hand
          detection to be considered successful.
        min_hand_presence_confidence: The minimum confidence score of hand
          presence score in the hand landmark detection.
        min_tracking_confidence: The minimum confidence score for the hand
          tracking to be considered successful.
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
    """

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    def save_result(
        result: vision.HandLandmarkerResult,
        unused_output_image: mp.Image,
        timestamp_ms: int,
    ):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT, LAST_POS, TCP_CLIENT

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1
        if result.hand_landmarks:
            # 取第一只手的食指指尖 (Landmark 8)
            hand = result.hand_landmarks[0]
            pos_x = hand[8].x
            pos_y = hand[8].y

            # 计算瞬时速度 (Euclidean distance)
            vel = ((pos_x - LAST_POS[0]) ** 2 + (pos_y - LAST_POS[1]) ** 2) ** 0.5
            LAST_POS = (pos_x, pos_y)

            if TCP_CLIENT and MESSAGE_QUEUE:
                data = '{"h":1, "x":' + f'{pos_x:.3f}' + ', "y":' + f'{pos_y:.3f}' + ', "v":' + f'{vel:.3f}' + '}'
                MESSAGE_QUEUE.put(data)
        else:
            if TCP_CLIENT and MESSAGE_QUEUE:
                MESSAGE_QUEUE.put('{"h":0}')  # 告诉 Unity 手丢了

    # Initialize the hand landmarker model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=save_result,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                "ERROR: Unable to read from webcam. Please verify your webcam settings."
            )

        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run hand landmarker using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show the FPS
        fps_text = "FPS = {:.1f}".format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(
            current_frame,
            fps_text,
            text_location,
            cv2.FONT_HERSHEY_DUPLEX,
            font_size,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        # 处理消息队列
        if MESSAGE_QUEUE and TCP_CLIENT:
            try:
                while not MESSAGE_QUEUE.empty():
                    message = MESSAGE_QUEUE.get(block=False)
                    # 直接发送消息，不使用异步
                    if hasattr(TCP_CLIENT, 'writer') and TCP_CLIENT.writer:
                        # 手动构建消息并发送
                        data = message.encode("utf-8")
                        total_length = 1 + len(data)
                        TCP_CLIENT.writer.write(struct.pack("!i", total_length))
                        TCP_CLIENT.writer.write(bytes([0]) + data)
                        TCP_CLIENT.writer.drain()
                        print(f"[发送] {message}")
            except Exception as e:
                print(f"[错误] 处理消息队列时出错: {e}")

        # Landmark visualization parameters.
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

        if DETECTION_RESULT:
            # Draw landmarks and indicate handedness.
            for idx in range(len(DETECTION_RESULT.hand_landmarks)):
                hand_landmarks = DETECTION_RESULT.hand_landmarks[idx]
                handedness = DETECTION_RESULT.handedness[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x, y=landmark.y, z=landmark.z
                        )
                        for landmark in hand_landmarks
                    ]
                )
                mp_drawing.draw_landmarks(
                    current_frame,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Get the top left corner of the detected hand's bounding box.
                height, width, _ = current_frame.shape
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height) - MARGIN

                # Draw handedness (left or right hand) on the image.
                cv2.putText(
                    current_frame,
                    f"{handedness[0].category_name}",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE,
                    HANDEDNESS_TEXT_COLOR,
                    FONT_THICKNESS,
                    cv2.LINE_AA,
                )

        cv2.imshow("hand_landmarker", current_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


async def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        help="Name of the hand landmarker model bundle.",
        required=False,
        type=str,
        default="models/hand_landmarker.task",
    )
    parser.add_argument(
        "--numHands",
        help="Max number of hands that can be detected by the landmarker.",
        required=False,
        type=int,
        default=2,
    )
    parser.add_argument(
        "--minHandDetectionConfidence",
        help="The minimum confidence score for hand detection to be considered "
        "successful.",
        required=False,
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--minHandPresenceConfidence",
        help="The minimum confidence score of hand presence score in the hand "
        "landmark detection.",
        required=False,
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--minTrackingConfidence",
        help="The minimum confidence score for the hand tracking to be "
        "considered successful.",
        required=False,
        type=float,
        default=0.5,
    )
    # Finding the camera ID can be very reliant on platform-dependent methods.
    # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
    # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
    # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
    parser.add_argument(
        "--cameraId", help="Id of camera.", required=False, type=int, default=0
    )
    parser.add_argument(
        "--frameWidth",
        help="Width of frame to capture from camera.",
        required=False,
        type=int,
        default=1280,
    )
    parser.add_argument(
        "--frameHeight",
        help="Height of frame to capture from camera.",
        required=False,
        type=int,
        default=960,
    )
    args = parser.parse_args()
    global TCP_CLIENT, MESSAGE_QUEUE
    MESSAGE_QUEUE = queue.Queue()
    TCP_CLIENT = TcpClient("127.0.0.1", 65432)
    await TCP_CLIENT.connect()
    await TCP_CLIENT.send_text('detect_pi connected')

    async def process_message_queue():
        while True:
            try:
                message = MESSAGE_QUEUE.get(block=False)
                print(f"[消息队列] 处理消息: {message}")
                await TCP_CLIENT.send_text(message)
            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"[错误] 处理消息队列时出错: {e}")

    # 在主线程中运行 run 函数，这样 OpenCV 的 GUI 操作能正常工作
    run(
        args.model,
        args.numHands,
        args.minHandDetectionConfidence,
        args.minHandPresenceConfidence,
        args.minTrackingConfidence,
        args.cameraId,
        args.frameWidth,
        args.frameHeight,
    )
    await TCP_CLIENT.close()


if __name__ == "__main__":
    asyncio.run(main())
