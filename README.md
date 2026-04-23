# EchoSphere Omni

手势与面部 landmark 检测，统一通过 TCP Socket 向外发送结构化事件。

## 架构

```
CameraCapture (共享相机捕获，后台线程)
        │
        ├── HandsRecognizer.recognize_async(frame)
        └── FaceRecognizer.recognize_async(frame)
                    │
                    ▼
         ┌──────────────────┐
         │  on_result 回调   │  (手势/表情变化时触发)
         └────────┬─────────┘
                  │ asyncio.run_coroutine_threadsafe
                  ▼
         ┌──────────────────┐
         │   TcpClient      │ ──────────────▶ Server
         │  (daemon thread) │
         └──────────────────┘
```

**设计原则：**

- **共享相机**：`CameraCapture` 运行独立捕获线程，多个 recognizer 共用同一路视频流
- **异步识别**：各 recognizer 通过 `recognize_async` 接收帧，结果通过回调返回
- **节流发送**：仅在检测结果显著变化时触发回调，避免刷屏
- **预览分离**：渲染逻辑由各 recognizer 的 `render_preview` 负责，可在主线程调用

## TCP 协议

长度前缀 JSON（4 bytes big-endian length + UTF-8 JSON）：

```json
{"omni_type": "hand_gesture", "data": [{"gesture": "Thumb_Up", "x": 0.234, "y": 0.567}]}
{"omni_type": "face_blendshape", "data": [{"category": "BROW_DOWN_LEFT", "score": 0.123}]}
```

## 运行

```bash
# 完整运行（手势 + 人脸 + 预览窗口）
PYTHONPATH=src python main.py --preview

# 无预览（纯 headless）
PYTHONPATH=src python main.py

# 指定 TCP 目标
PYTHONPATH=src python main.py --host 192.168.1.100 --port 65432

# 调整摄像头参数
PYTHONPATH=src python main.py --camera-id 0 --preview
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--preview` | False | 打开 OpenCV 预览窗口 |
| `--host` | 127.0.0.1 | TCP 服务器地址 |
| `--port` | 65432 | TCP 服务器端口 |
| `--camera-id` | 0 | 摄像头设备索引 |
| `--num-hands` | 1 | 最大检测手部数量 |
| `--model` | models/gesture_recognizer.task | 手势识别模型路径 |
| `--face-model` | models/face_landmarker.task | 人脸 landmark 模型路径 |

## 项目结构

```
src/echo_omni/                  # 新架构（推荐）
├── camera/
│   └── capture.py              # CameraCapture 共享相机捕获
├── hands/
│   └── recognizer.py           # HandsRecognizer 手势识别器
└── face/
    └── recognizer.py           # FaceRecognizer 面部表情识别器

src/echoesphere_omni/           # ⚠️ 已弃用（旧架构）
├── run.py                      # 入口
├── event_bus.py               # 事件总线
├── face/detector.py           # FaceDetector（旧）
└── hands/detector.py          # HandDetector（旧）

main.py                         # 入口脚本（使用 echo_omni）
```

## 手势识别输出

```json
{"omni_type": "hand_gesture", "data": [
  {"gesture": "Thumb_Up", "x": 0.234, "y": 0.567}
]}
```

- `gesture`：手势类别名（如 `Thumb_Up`, `Open_Palm`, `Pointing_Up`）
- `x`, `y`：腕部归一化坐标

## 面部表情输出

```json
{"omni_type": "face_blendshape", "data": [
  {"category": "BROW_DOWN_LEFT", "score": 0.123},
  {"category": "MOUTH_SMILE_LEFT", "score": 0.456}
]}
```

- `category`：表情类别名（52 个 MediaPipe blendshape 之一，仅输出 score > 0.1 的）
- `score`：置信度（0.0 ~ 1.0）

## 依赖

- Python 3.12
- mediapipe >= 0.10.21
- opencv-python >= 4.11.0.86
- asyncio

模型文件（放在 `models/` 目录）：

- `gesture_recognizer.task` — MediaPipe 手势识别模型
- `face_landmarker.task` — MediaPipe 人脸 landmark 模型
