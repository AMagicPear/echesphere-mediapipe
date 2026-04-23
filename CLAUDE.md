# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目状态

**新架构**：`src/echo_omni/` — 基于 `CameraCapture` 的共享视频流模式
**旧架构**：`src/echoesphere_omni/` — ⚠️ 已弃用，基于 EventBus 的独立线程模式

## 运行

```bash
# 完整运行（手势 + 人脸 + 预览窗口）
PYTHONPATH=src python main.py --preview

# 无预览（纯 headless）
PYTHONPATH=src python main.py

# 指定 TCP 目标
PYTHONPATH=src python main.py --host 192.168.1.100 --port 65432
```

## 架构

```
CameraCapture (独立后台线程，共享视频流)
        │
        ├── HandsRecognizer.recognize_async(frame)  → on_result 回调
        └── FaceRecognizer.recognize_async(frame)    → on_result 回调
                    │
                    │ asyncio.run_coroutine_threadsafe
                    ▼
         ┌──────────────────┐
         │   TcpClient      │ ──────────────▶ Server (65432)
         │  (daemon thread) │
         └──────────────────┘
```

### 核心模块 (echo_omni)

| 文件 | 职责 |
|------|------|
| `camera/capture.py` | `CameraCapture` — 独立捕获线程，`get_latest_frame()` 供主线程预览，`on_frame()` 分发帧 |
| `hands/recognizer.py` | `HandsRecognizer` — 手势识别，`render_preview()` 渲染骨架和 FPS（左上角） |
| `face/recognizer.py` | `FaceRecognizer` — 面部表情识别，输出 blendshapes，`render_preview()` 渲染鼻尖点 |

### TCP 协议

长度前缀 JSON（4 bytes big-endian length + UTF-8 JSON）：

```json
{"omni_type": "hand_gesture", "data": [{"gesture": "Thumb_Up", "x": 0.234, "y": 0.567}]}
{"omni_type": "face_blendshape", "data": [{"category": "BROW_DOWN_LEFT", "score": 0.123}]}
```

### 面部表情节流策略

`FaceRecognizer` 使用双重过滤：
- 变化阈值：`CHANGE_THRESHOLD = 0.15`（blendshapes 均值变化超过 0.15 才可能触发）
- 最小间隔：`MIN_CALLBACK_INTERVAL_MS = 500`（两次回调至少间隔 500ms）
- 仅输出：`BLENDSHAPE_THRESHOLD = 0.1`（只输出 score > 0.1 的 blendshape）

### 预览窗口布局

- `HandsRecognizer` FPS：左上角 `(24, 50)`
- `FaceRecognizer` FPS + 鼻尖标注：右上角 `(w - 220, 50)`，鼻尖绿点 + blendshape 文字

## 已弃用模块 (echoesphere_omni)

| 文件 | 状态 | 说明 |
|------|------|------|
| `run.py` | ⚠️ deprecated | 旧入口，基于 EventBus + 独立检测线程 |
| `event_bus.py` | ⚠️ deprecated | 线程安全队列，已被 CameraCapture 替代 |
| `face/detector.py` | ⚠️ deprecated | 旧 FaceDetector，输出 face_detected/face_lost 事件 |
| `hands/detector.py` | ⚠️ deprecated | 旧 HandDetector，输出 pinch/swipe 等事件 |
| `sender.py` | ⚠️ deprecated | TcpSender 后台线程，已被 daemon thread + TcpClient 替代 |

## 依赖

- Python 3.12
- mediapipe >= 0.10.21
- opencv-python >= 4.11.0.86
- asyncio

模型文件（`models/` 目录）：
- `gesture_recognizer.task` — MediaPipe 手势识别模型
- `face_landmarker.task` — MediaPipe 人脸 landmark 模型
