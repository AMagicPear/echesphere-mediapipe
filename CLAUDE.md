# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 运行

```bash
PYTHONPATH=src python main.py --preview                    # 完整运行（手势 + 人脸 + 预览）
PYTHONPATH=src python main.py                              # headless 模式
PYTHONPATH=src python main.py --left-hand-direction        # 启用左手方向指示
PYTHONPATH=src python main.py --host 192.168.1.100 --port 65432
```

## 线程模型

macOS 要求 `cv2.imshow`/`cv2.waitKey` 在主线程，因此必须三条线程：

```
主线程                    CameraCapture 线程           TCP 线程 (asyncio)
cv2.imshow/waitKey        cap.read() → 分发帧           event loop
  get_latest_frame()        recognize_async()             send_text / send_command
  render_preview()              │                              ↑
                                │    MediaPipe 回调            │
                                └── _on_result() ──── asyncio.run_coroutine_threadsafe
```

- 主线程**只**做预览渲染，不参与识别或网络
- 所有 TCP 操作通过 `asyncio.run_coroutine_threadsafe` 从 MediaPipe 回调线程桥接到 TCP 事件循环

## 架构关键点

### 手势变化检测

`HandsRecognizer` **仅在手势类别名变化时**触发回调（`_current_gestures` 字典做前后对比）。维持同一手势不产生 TCP 流量。方向向量（`on_direction`）例外——每帧触发，用于 Unity 实时输入。

### 表情双重节流

`FaceRecognizer` 的双重约束：
- blendshape 均值变化 > `CHANGE_THRESHOLD`（0.15）
- 距上次发送 >= `MIN_CALLBACK_INTERVAL_MS`（500ms）
- 只输出 score > `BLENDSHAPE_THRESHOLD`（0.1）的 blendshape

### TCP 自动重连

`TcpClient` 断开后指数退避重连（1s → 2s → 4s → ... → 上限 30s）。`on_connected` 事件在每次（重）连接时触发，`main.py` 通过此事件自动重新注册。`close()` 停止重连循环。

### 左手方向向量计算

`_compute_left_direction`：归一化 landmark 坐标 → 乘以实际帧尺寸（`recognize_async` 中从图像 shape 提取）获得像素空间向量 → L2 归一化。食指伸直判定：指尖到掌骨基部归一化距离 > 0.07。通过 CLI `--left-hand-direction` 或 TCP command `hand_direction:on/off` 开关。

### TCP 协议

长度前缀 JSON（4 bytes big-endian length + UTF-8 JSON）。发送方向：

| type | 用途 |
|------|------|
| `text` | `omni_type: "hand_gesture"` / `"face_blendshape"` 数据 |
| `command` | `relay_to: "unity"` 方向输入；服务端也可下发 command 控制开关 |
| `register` | 客户端注册（`client_type` + `modules`） |

### 预览布局

- HandsRecognizer FPS + 方向向量：左上角 `(24, 50)` 和 `(24, 80)`
- FaceRecognizer 鼻尖绿点 + blendshape 文字（最多 5 个）：从 `(24, 90)` 开始（不绘制 FPS 避免重叠）

## 依赖

- Python 3.12
- mediapipe >= 0.10.21
- opencv-python >= 4.11.0.86
- 模型文件：`models/gesture_recognizer.task`、`models/face_landmarker.task`

## 旧架构

`src/echoesphere_omni/` 已从磁盘移除。仅保留 `src/echoesphere_omni.egg-info/`（构建元数据）和 `docs/` 中的旧文档。
