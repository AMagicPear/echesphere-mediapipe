# 多模态视觉感知模块（Echo-Omni）技术方案

**版本**：v2.0
**日期**：2026-04-23
**架构级别**：边缘设备端视觉感知 + TCP 网络传输

---

## 1. 系统概述

本系统部署于边缘计算设备，负责从摄像头实时采集视频流并完成手势识别与面部表情分析两类视觉感知任务。感知结果通过 TCP Socket 以长度前缀 JSON 协议传输至上位机，供后续多模态融合与智能决策使用。

系统采用**共享相机捕获 + 异步事件驱动**的解耦架构，多个感知模块共享同一路视频流输入，各自独立完成推理，通过回调机制在检测结果发生显著变化时触发网络传输。

---

## 2. 系统架构

### 2.1 整体数据流

```
┌──────────────────────────────────────────────────────────────┐
│                     CameraCapture                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ cv2.VideoCapture (后台 daemon thread)                   │ │
│  │   └─→ BGR 帧 → cv2.flip(水平镜像) → BGR2RGB 转换        │ │
│  │   └─→ _latest_rgb (线程安全缓存)                        │ │
│  │   └─→ on_frame(rgb_image, timestamp_ms) 分发给各 callback │ │
│  └─────────────────────────────────────────────────────────┘ │
└────────────────────────────┬─────────────────────────────────┘
                             │ rgb_image (numpy.ndarray, H×W×3)
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
┌─────────────────────┐             ┌─────────────────────────┐
│   HandsRecognizer   │             │    FaceRecognizer       │
│  recognize_async()  │             │   recognize_async()     │
│         ↓           │             │          ↓              │
│  MediaPipe Gesture  │             │  MediaPipe FaceLandmarker│
│   Recognizer        │             │        (blendshapes)     │
│         ↓           │             │          ↓              │
│  _on_result callback│             │   _on_result callback    │
│  (状态变化检测)      │             │  (双重节流过滤)          │
└──────────┬──────────┘             └────────────┬────────────┘
           │ HandResult                           │ FaceResult
           │ on_result 回调链                     │ on_result 回调链
           │                                      │
           ▼                                      ▼
    ┌──────────────────────────────────────────────────┐
    │         asyncio.run_coroutine_threadsafe()        │
    │  (跨线程调度至主事件循环，避免 event loop 冲突)     │
    └──────────────────────┬───────────────────────────┘
                           │ await client.send_text()
                           ▼
                    ┌──────────────┐
                    │   TcpClient  │───────────▶ 上位机 (65432)
                    │ (daemon thr) │
                    └──────────────┘
```

### 2.2 线程模型

| 线程 | 类型 | 职责 |
|------|------|------|
| `CameraCapture._capture_loop` | `threading.Thread` (daemon) | 视频采集、后处理、帧分发 |
| `tcp_thread` | `threading.Thread` (daemon) | 运行 asyncio 事件循环，处理 TCP I/O |
| 主线程 | `threading.main_thread` | OpenCV 窗口事件循环 (`cv2.waitKey`) |

**线程安全策略**：
- `CameraCapture._latest_rgb`：由 `threading.Lock` 保护，主线程读取时加锁复制
- `HandsRecognizer._cached_result` / `FaceRecognizer._cached_result`：各自独立的 `threading.Lock`
- OpenCV (`cv2.imshow` / `cv2.waitKey`) 必须在主线程调用，macOS 约束

---

## 3. 核心模块

### 3.1 CameraCapture（共享视频捕获器）

**文件**：`src/echo_omni/camera/capture.py`

CameraCapture 是系统的视频采集中枢，以单例形式运行一个后台捕获线程，负责：
1. 打开指定摄像头并配置分辨率
2. 以无限循环从 `cv2.VideoCapture` 读取帧
3. 对每帧执行水平镜像（`cv2.flip`）和色彩空间转换（BGR→RGB）
4. 将最新帧缓存至线程安全变量 `_latest_rgb`，供主线程预览使用
5. 将帧以 `(rgb_image, timestamp_ms)` 格式分发给所有已注册的 `on_frame` 回调

**关键设计决策**：
- **一次采集，多次消费**：视频帧只需从摄像头读取一次，即可同时分发给手势和面部两个识别器，避免重复采集的性能开销
- **时间戳注入**：捕获时记录毫秒级时间戳，经由回调传递至各识别器，保证多模态数据的时间一致性
- **非阻塞启动**：相机启动后立即返回，捕获线程在后台运行，主线程可继续执行预览循环

### 3.2 HandsRecognizer（手势识别器）

**文件**：`src/echo_omni/hands/recognizer.py`

基于 MediaPipe Gesture Recognizer 的实时手势识别模块。

#### 3.2.1 MediaPipe 配置

```python
running_mode = vision.RunningMode.LIVE_STREAM  # 异步流模式
num_hands = 1  # 同时检测的手数量
min_hand_detection_confidence = 0.7
min_hand_presence_confidence = 0.7
min_tracking_confidence = 0.5
output_face_blendshapes = False  # 手势识别无需 blendshapes
```

#### 3.2.2 识别结果数据结构

```python
@dataclass
class HandResult:
    hand_landmarks: list              # List[NormalizedLandmarkList]，每手21个关键点
    gestures: list                    # List[List[Category]]，每手一个 top-1 手势
    hand_centers: list[tuple[float, float]]  # 每手腕部（landmark[0]）归一化坐标
    timestamp_ms: int                 # 捕获时注入的时间戳
```

#### 3.2.3 手势变化检测策略

回调触发条件：任一手的 `category_name` 发生变化。

维护 `_current_gestures: dict[int, str]`，键为手序号（0或1），值为当前手势名。每次 `_on_result` 回调时比较新旧手势名，任一手指手势名变化即触发注册的回调函数。

优势：仅在手势状态跳转时通知上层，避免每帧都传输，适用于交互式应用场景。

#### 3.2.4 预览渲染

`render_preview(image: np.ndarray)` 在输入图像上叠加：
- **FPS 覆盖层**：左上角 `(24, 50)`，标签 `HANDS FPS = X.X`
- **手部骨架**：MediaPipe `HAND_CONNECTIONS` 标准化连线
- **手势标签**：每只手的边界框左上角，手势名 + 置信度

### 3.3 FaceRecognizer（面部表情识别器）

**文件**：`src/echo_omni/face/recognizer.py`

基于 MediaPipe FaceLandmarker 的实时面部表情分析模块。

#### 3.3.1 MediaPipe 配置

```python
running_mode = vision.RunningMode.LIVE_STREAM
num_faces = 1
output_face_blendshapes = True  # 启用融合变形系数输出
min_face_detection_confidence = 0.5
min_face_presence_confidence = 0.5
min_tracking_confidence = 0.5
```

#### 3.3.2 识别结果数据结构

```python
@dataclass
class FaceResult:
    blendshapes: dict[str, float]           # {category_name: score}，仅 score > 0.1
    face_center: tuple[float, float]        # 鼻尖（landmark[1]）归一化坐标
    timestamp_ms: int
```

#### 3.3.3 Blendshape 数据说明

MediaPipe FaceLandmarker 输出 52 维 blendshape 向量，涵盖眉毛、眼睛、嘴巴、下颌等面部肌群动作：

| 类别 | 示例 | 阈值 |
|------|------|------|
| 眉毛 | `BROW_DOWN_LEFT`, `BROW_INNER_UP` | > 0.1 |
| 眼睛 | `EYE_BLINK_LEFT`, `EYE_LOOK_UP_RIGHT` | > 0.1 |
| 嘴巴 | `MOUTH_SMILE_LEFT`, `JAW_OPEN`, `MOUTH_FUNNEL` | > 0.1 |
| 面部 | `CHEEK_PUFF`, `NOSE_SNEER_LEFT` | > 0.1 |

本系统仅输出 `score > 0.1` 的非零表情，典型帧输出 3~8 个维度，大幅降低网络负载。

#### 3.3.4 双重节流策略

面部表情数据变化频繁，直接发送每帧结果会造成网络泛洪。本模块采用**双重过滤**策略：

**第一层：变化幅度阈值**
```
if |current_mean - last_mean| > CHANGE_THRESHOLD(0.15):
    # 进入候选
```

其中 `current_mean = sum(scores) / len(scores)`，反映当前表情的总体激活程度。

**第二层：时间间隔闸门**
```
elapsed_ms = (now - last_callback_time) * 1000
if elapsed_ms >= MIN_CALLBACK_INTERVAL_MS(500ms):
    # 确认触发
```

两层条件同时满足才触发回调，确保：
- 表情大幅变化时能够及时上报
- 短时间内连续变化被抑制
- 上报频率上限为 2 次/秒

#### 3.3.5 预览渲染

`render_preview(image: np.ndarray)` 在输入图像上叠加：
- **FPS 覆盖层**：右上角 `(w - 220, 50)`，标签 `FACE FPS = X.X`
- **鼻尖标记**：绿色圆圈 `(cx, cy, 8px)` + 文字 `表情 N 个`

---

## 4. 网络通信协议

### 4.1 传输层

基于 TCP 长连接，使用 `echoesphere_omni.net.client.TcpClient`（异步 I/O）。

**长度前缀帧格式**：
```
[4 bytes big-endian uint32] payload_length
[4 × payload_length bytes]   UTF-8 JSON payload
```

### 4.2 消息类型

| type | data 格式 | 说明 |
|------|-----------|------|
| `text` | 任意 UTF-8 字符串 | 应用层数据 |
| `image` | base64 编码字节流 | 图像帧（当前未使用） |
| `register` | `{"client_type": "...", "subtype": "..."}` | 客户端注册 |

### 4.3 应用层消息格式

**手势消息**：
```json
{"omni_type": "hand_gesture", "data": [
  {"gesture": "Thumb_Up", "x": 0.234, "y": 0.567}
]}
```
- 出现多手时 `data` 数组包含多个对象
- 无手时 `data` 为空数组 `[]`

**面部表情消息**：
```json
{"omni_type": "face_blendshape", "data": [
  {"category": "BROW_DOWN_LEFT", "score": 0.123},
  {"category": "MOUTH_SMILE_LEFT", "score": 0.456}
]}
```
- 仅包含 `score > 0.1` 的非零表情
- 无面部时 `data` 为空数组 `[]`

### 4.4 注册协议

客户端启动时向服务器注册身份：
```json
{"type": "register", "client_type": "mediapipe"}
{"type": "register", "client_type": "mediapipe"}
```
（两次注册分别标识 hands 和 face 两个感知通道，上位机可通过 `subtype` 区分）

---

## 5. 跨线程调度机制

### 5.1 问题背景

MediaPipe 的 `result_callback` 在其内部推理线程执行（不同于相机采集线程）。当回调需要向 TCP 发送数据时，涉及从 MediaPipe 推理线程跨到 asyncio 主事件循环的调度问题。

直接调用 `asyncio.create_task()` 会因「当前线程没有 running event loop」而失败。

### 5.2 解决方案

```python
def handle_result(r):
    asyncio.run_coroutine_threadsafe(on_gesture(r), loop)
```

`asyncio.run_coroutine_threadsafe(coro, loop)` 将协程提交至指定事件循环，在那个线程中执行。本系统中 `loop` 是 TCP 线程的事件循环，因此：
1. 推理线程调用 `run_coroutine_threadsafe` → 安全，不阻塞
2. 协程被投递至 TCP 线程的事件循环 → 由该线程执行 `await client.send_text()`
3. TCP 发送与推理完全并行，无相互阻塞

---

## 6. 技术指标

| 指标 | 数值 |
|------|------|
| 手势识别模型 | MediaPipe Gesture Recognizer (`gesture_recognizer.task`) |
| 人脸 landmark 模型 | MediaPipe FaceLandmarker (`face_landmarker.task`) |
| 手部关键点 | 21 点/手（MediaPipe Hands 标准） |
| 面部关键点 | 478 点（MediaPipe FaceMesh 标准） |
| Blendshape 维度 | 52 维 |
| 手势检测灵敏度 | `min_hand_detection_confidence=0.7` |
| 表情输出阈值 | `score > 0.1` |
| 表情节流间隔 | ≥ 500ms |
| 表情变化检测阈值 | 均值变化 > 0.15 |
| 默认分辨率 | 1280×720 |
| TCP 默认端口 | 65432 |
| 相机镜像 | 水平翻转（`cv2.flip(1)`） |

---

## 7. 模块接口汇总

### CameraCapture

```python
class CameraCapture:
    def __init__(camera_id=0, frame_width=1280, frame_height=720)
    def start()                         # 启动捕获线程（非阻塞）
    def stop()                          # 停止捕获线程
    def on_frame(callback)              # 注册帧回调：Callable[[np.ndarray, int], None]
    def get_latest_frame() -> Optional[np.ndarray]  # 主线程获取最新帧
```

### HandsRecognizer

```python
class HandsRecognizer:
    def __init__(model, num_hands=1, preview=False, ...)
    def start()                         # 初始化 MediaPipe 识别器
    def stop()                         # 释放资源
    def recognize_async(rgb_image, timestamp_ms)  # 外部送帧
    def on_result(callback)            # 注册结果回调：Callable[[HandResult], None]
    def render_preview(image) -> np.ndarray  # 叠加 FPS、骨架、标签
```

### FaceRecognizer

```python
class FaceRecognizer:
    def __init__(model, num_faces=1, preview=False, ...)
    def start()
    def stop()
    def recognize_async(rgb_image, timestamp_ms)
    def on_result(callback)             # Callable[[FaceResult], None]
    def render_preview(image) -> np.ndarray
```

---

## 8. 文件结构

```
src/echo_omni/                    # 当前架构（推荐）
├── camera/
│   └── capture.py               # CameraCapture
├── hands/
│   └── recognizer.py           # HandsRecognizer
└── face/
    └── recognizer.py           # FaceRecognizer

src/echoesphere_omni/            # ⚠️ 已弃用（旧架构）
├── run.py                       # 旧入口
├── event_bus.py                # 旧事件总线
├── face/detector.py            # 旧 FaceDetector
├── hands/detector.py           # 旧 HandDetector
└── net/client.py               # TcpClient（仍在使用）

main.py                          # 当前入口脚本
```
