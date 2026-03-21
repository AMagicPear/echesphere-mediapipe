# EchoSphere Omni

手势与面部 landmark 检测，统一通过 TCP Socket 向外发送结构化事件。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                       run.py                            │
│                                                         │
│   ┌──────────────┐   ┌──────────────┐                 │
│   │  HandDetector │   │  FaceDetector │  (并行线程)     │
│   └──────┬───────┘   └──────┬───────┘                 │
│          │                  │                         │
│          └────────┬─────────┘                         │
│                   ▼                                    │
│          ┌──────────────┐                             │
│          │   EventBus   │  (queue.Queue, 线程安全)     │
│          └──────┬───────┘                             │
│                 │ bus.publish(event)                   │
│                 ▼                                      │
│          ┌──────────────┐                             │
│          │  TcpSender   │  (后台线程, asyncio)         │
│          └──────┬───────┘                             │
│                 │ await send_text()                    │
│                 ▼                                      │
│          ┌──────────────┐                             │
│          │  TcpClient   │ ───────────────────────────▶ Server
│          └──────────────┘                             │
└─────────────────────────────────────────────────────────┘
```

**设计原则：**

- **解耦**：检测器（Hand/Face）只负责检测，TCP 发送逻辑完全不感知
- **线程安全**：EventBus 使用 `queue.Queue`，跨线程通信无需锁
- **静默丢弃**：队列满时丢弃最旧事件，防止慢消费者阻塞检测管线
- **仅状态变化发消息**：检测器内部维护状态机，只在手势状态跳转时 publish，避免刷屏

## 手势类型

| 事件 | 触发条件 |
|------|---------|
| `hand_detected` | 首次检测到手 |
| `hand_lost` | 手从画面消失 |
| `pinch` | 拇指尖 + 食指尖距离 < 阈值（捏合），携带 `{x, y}` 位置 |
| `pinch_released` | 从捏合状态放开 |
| `open_both_hands` | 左右手同时张开（你伸开双手） |
| `swipe_left` / `swipe_right` | 手腕水平速度超过阈值 |

## TCP 协议

基于长度前缀的二进制协议（与 Unity/服务器约定）：

```
4 bytes (big-endian int) : total payload length
1 byte                   : message type (0x00 = TEXT)
N bytes                  : UTF-8 JSON payload
```

发送的 JSON 格式：

```json
{"source": "hand", "event": "open_both_hands", "data": {}, "timestamp_ms": 1234567890}
{"source": "hand", "event": "pinch", "data": {"x": 0.234, "y": 0.567}, "timestamp_ms": 1234567891}
{"source": "face", "event": "face_detected", "data": {"x": 0.5, "y": 0.3}, "timestamp_ms": 1234567892}
```

## 运行

```bash
# 完整运行（手势 + 人脸 + 预览窗口）
python -m echoesphere_omni.run

# 无预览（纯 headless，服务端调试）
python -m echoesphere_omni.run --no-preview

# 仅手势，无预览
python -m echoesphere_omni.run --no-preview --no-face

# 指定 TCP 目标
python -m echoesphere_omni.run --host 192.168.1.100 --port 65432

# 调整摄像头参数
python -m echoesphere_omni.run --cameraId 0 --frameWidth 1280 --frameHeight 960
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--no-preview` | False | 关闭 OpenCV 预览窗口 |
| `--no-face` | False | 禁用面部检测 |
| `--no-hand` | False | 禁用手势检测 |
| `--host` | 127.0.0.1 | TCP 服务器地址 |
| `--port` | 65432 | TCP 服务器端口 |
| `--cameraId` | 0 | 摄像头设备索引 |
| `--frameWidth` | 1280 | 摄像头采集帧宽 |
| `--frameHeight` | 960 | 摄像头采集帧高 |
| `--handModel` | models/hand_landmarker.task | 手部模型路径 |
| `--numHands` | 2 | 最大检测手部数量 |
| `--faceModel` | models/face_landmarker.task | 人脸模型路径 |
| `--numFaces` | 1 | 最大检测人脸数量 |

## 项目结构

```
src/echoesphere_omni/
├── run.py              # 入口：组装所有组件
├── event_bus.py        # 线程安全事件总线
├── events.py           # UnifiedEvent 统一事件格式
├── sender.py           # TcpSender 后台线程
├── face/
│   └── detector.py     # FaceDetector
├── hands/
│   ├── gestures.py     # 手势类型、地标索引、阈值常量
│   └── detector.py      # HandDetector + 状态机
└── net/
    └── client.py        # TcpClient 异步 TCP 客户端
```

## 依赖

- Python 3.12
- mediapipe >= 0.10.21
- opencv-python >= 4.11.0.86
- asyncio >= 4.0.0

模型文件（放在 `models/` 目录）：

- `hand_landmarker.task` — MediaPipe 手部 landmark 模型
- `face_landmarker.task` — MediaPipe 人脸 landmark 模型