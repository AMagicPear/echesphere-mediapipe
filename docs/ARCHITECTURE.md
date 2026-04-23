# EchoSphere Omni 技术架构文档

## 项目概述

EchoSphere Omni 是一个基于 MediaPipe 的手部和面部关键点检测系统，能够实时检测手势和面部姿态，并通过 TCP 协议发布结构化事件。系统采用模块化、线程安全的架构，支持多检测器并行运行，并通过统一的事件总线进行解耦。

### 主要特性

- **多检测器支持**：同时支持手部检测（HandDetector）和面部检测（FaceDetector）
- **事件驱动架构**：检测器与网络发送器通过事件总线完全解耦
- **状态机驱动**：仅在手势状态变化时发布事件，避免每帧发送冗余数据
- **线程安全**：使用队列实现生产者-消费者模型，确保跨线程安全通信
- **异步网络**：基于 asyncio 的 TCP 客户端，支持高并发事件发送
- **可配置预览**：支持 OpenCV 实时预览窗口，可关闭以无头模式运行

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                       run.py                            │
│   ┌──────────────┐   ┌──────────────┐                  │
│   │  HandDetector │   │  FaceDetector │  (daemon threads)
│   └──────┬───────┘   └──────┬───────┘                  │
│          │                  │                          │
│          └────────┬─────────┘                          │
│                   ▼                                     │
│          ┌──────────────┐                              │
│          │   EventBus   │  (queue.Queue, thread-safe)  │
│          └──────┬───────┘                              │
│                 │ bus.publish(event)                   │
│                 ▼                                       │
│          ┌──────────────┐                              │
│          │  TcpSender   │  (background thread, asyncio) │
│          └──────┬───────┘                              │
│                 │ await send_text()                    │
│                 ▼                                       │
│          ┌──────────────┐                              │
│          │  TcpClient   │ ───────────────────────────▶ Server
│          └──────────────┘                              │
└─────────────────────────────────────────────────────────┘
```

### 数据流

1. **检测器线程**（HandDetector/FaceDetector）从摄像头捕获帧，使用 MediaPipe 进行关键点检测
2. 检测器内部维护状态机，仅在检测到状态变化时（如手部出现、捏合、面部出现/消失）生成事件
3. 事件通过 `EventBus.publish()` 发布到线程安全队列
4. **TcpSender** 在后台线程中运行 asyncio 事件循环，从队列消费事件
5. **TcpClient** 将事件序列化为 JSON，通过 TCP 协议发送到远程服务器

### 设计原则

- **解耦**：检测器只负责检测，TCP 发送逻辑完全不知道检测细节
- **线程安全**：EventBus 使用 `queue.Queue` 实现无锁跨线程通信
- **静默丢弃**：队列满时丢弃最旧事件，防止慢消费者阻塞检测流水线
- **仅状态变化**：检测器维护内部状态机，仅在手势状态转换时发布事件（非每帧）

## 核心组件

### 1. 运行入口 (run.py)

**文件**: `src/echoesphere_omni/run.py`

主程序入口，负责：
- 解析命令行参数
- 初始化 EventBus
- 启动 TCP 发送器
- 根据配置启动手部和面部检测器线程

**关键参数**：
- `--preview`: 启用 OpenCV 预览窗口
- `--no-face` / `--no-hand`: 禁用特定检测器
- `--host` / `--port`: TCP 服务器地址
- `--cameraId` / `--frameWidth` / `--frameHeight`: 摄像头配置
- `--debug`: 启用调试日志

### 2. 事件总线 (EventBus)

**文件**: `src/echoesphere_omni/event_bus.py`

线程安全的事件队列，连接检测器（生产者）和 TCP 发送器（消费者）。

**主要方法**：
- `publish(event: UnifiedEvent)`: 发布事件（非阻塞）
- `get_event(timeout: float = 0.05) -> UnifiedEvent | None`: 获取事件（可超时）
- `subscribe(callback: Callable)`: 注册事件回调（用于调试）

**实现细节**：
- 使用 `queue.Queue` 而非 `asyncio.Queue`，保持与事件循环无关
- 队列满时采用丢弃最旧事件的背压策略
- 支持订阅者回调（用于日志记录等）

### 3. 统一事件格式 (UnifiedEvent)

**文件**: `src/echoesphere_omni/events.py`

所有检测器共享的事件数据结构。

**属性**：
- `source: str`: 事件来源（"hand" 或 "face"）
- `event: str`: 事件名称（如 "pinch", "open_both_hands"）
- `data: dict[str, Any]`: 事件数据（位置、分数等）
- `timestamp_ms: int`: 事件发生时间戳（毫秒）

**序列化**：
- `to_json() -> str`: 序列化为紧凑 JSON 字符串，用于 TCP 传输

### 4. TCP 发送器 (TcpSender)

**文件**: `src/echoesphere_omni/sender.py`

拥有后台线程和 asyncio 事件循环，负责从 EventBus 读取事件并通过 TCP 发送。

**工作流程**：
1. 在后台线程中启动 asyncio 事件循环
2. 连接到 TCP 服务器并发送注册消息
3. 循环从 EventBus 获取事件（非阻塞，带超时）
4. 将事件序列化为 JSON 并通过 TcpClient 发送

**特点**：
- 与检测器完全解耦，仅依赖 EventBus 接口
- 异常处理完善，网络错误不影响检测器运行
- 支持异步并发发送

### 5. TCP 客户端 (TcpClient)

**文件**: `src/echoesphere_omni/net/client.py`

异步 TCP 客户端，实现长度前缀的二进制协议。

**协议格式**：
```
4 bytes (big-endian unsigned int): 字节长度 N
N bytes: UTF-8 编码的 JSON 数据
```

**JSON 结构**：
```json
{
  "type": "text"|"image"|"command"|"register",
  "data": <内容>
}
```

**支持的消息类型**：
- `text`: 文本消息（data 为字符串）
- `image`: 图像消息（data 为 base64 编码的图片字节）
- `register`: 注册消息（data 包含客户端类型和子类型）

**主要方法**：
- `connect()`: 连接到服务器
- `send_text(text: str)`: 发送文本消息
- `send_image(image_bytes: bytes)`: 发送图片消息
- `send_register(client_type: str, subtype: str = "")`: 发送注册消息

### 6. 手部检测器 (HandDetector)

**文件**: `src/echoesphere_omni/hands/detector.py`

MediaPipe 手部关键点检测器，包含手势状态机。

#### 状态机

状态转换：`NO_HAND → HAND_PRESENT → PINCH → PINCH_RELEASED → NO_HAND`

#### 支持的手势事件

| 事件 | 触发条件 | 数据 |
|------|----------|------|
| `hand_detected` | 手部首次出现 | 无 |
| `hand_lost` | 手部完全消失 | 无 |
| `pinch` | 拇指和食指距离小于阈值 | `{"x": <平均X坐标>, "y": <平均Y坐标>}` |
| `pinch_released` | 捏合状态释放 | 无 |
| `open_both_hands` | 左右手同时张开 | 无 |
| `swipe_left` | 手腕向左快速移动 | 无 |
| `swipe_right` | 手腕向右快速移动 | 无 |

#### 关键算法

1. **捏合检测**：计算拇指尖（landmark 4）和食指尖（landmark 8）的欧氏距离
2. **手部张开检测**：检查所有指尖（8,12,16,20）与手腕（0）的距离
3. **滑动手势检测**：跟踪手腕 X 坐标的速度，超过阈值触发
4. **左右手识别**：基于 MediaPipe 的 handedness 分类结果

#### 阈值配置

见 `hands/gestures.py`：
- `PINCH_THRESHOLD = 0.06`（归一化坐标）
- `OPEN_FINGER_DISTANCE = 0.15`
- `SWIPE_VELOCITY_THRESHOLD = 0.025`

### 7. 面部检测器 (FaceDetector)

**文件**: `src/echoesphere_omni/face/detector.py`

MediaPipe 面部关键点检测器，检测面部出现/消失事件。

#### 状态机

简单布尔状态：`face_present`（面部存在）或 `face_absent`（面部不存在）

#### 支持的事件

| 事件 | 触发条件 | 数据 |
|------|----------|------|
| `face_detected` | 面部首次出现 | `{"x": <鼻尖X坐标>, "y": <鼻尖Y坐标>}` |
| `face_lost` | 面部完全消失 | 无 |

#### 关键算法

- 使用 MediaPipe 面部 landmark 模型（468 个关键点）
- 鼻尖位置作为面部中心参考点（landmark 1）
- 仅在有/无面部状态变化时发布事件，避免每帧发送

### 8. 手势定义 (gestures.py)

**文件**: `src/echoesphere_omni/hands/gestures.py`

定义手部相关常量：
- `HandGestureType`: 手势类型枚举
- `HandLandmark`: 关键点索引常量
- `GestureThresholds`: 手势检测阈值

## TCP 协议详解

### 消息格式

#### 二进制帧格式
```
[ 长度前缀 (4字节) ][ JSON 载荷 (N字节) ]
```

- **长度前缀**: 大端无符号整数，表示 JSON 载荷的字节数
- **JSON 载荷**: UTF-8 编码的 JSON 字符串

#### JSON 消息结构

```json
{
  "type": "text"|"image"|"command"|"register",
  "data": <内容>
}
```

### 事件消息格式

当 `type` 为 "text" 时，`data` 字段包含 UnifiedEvent 的 JSON 序列化：

```json
{
  "source": "hand",
  "event": "pinch",
  "data": {"x": 0.234, "y": 0.567},
  "timestamp_ms": 1234567890
}
```

### 注册消息

客户端连接后自动发送注册消息：

```json
{
  "type": "register",
  "data": {
    "client_type": "mediapipe",
    "subtype": ""  // 可选
  }
}
```

## 配置与运行

### 依赖要求

- Python 3.12.11+
- mediapipe >= 0.10.21
- opencv-python >= 4.11.0.86
- asyncio >= 4.0.0

### 模型文件

模型文件位于 `models/` 目录：
- `hand_landmarker.task`: MediaPipe 手部关键点模型
- `face_landmarker.task`: MediaPipe 面部关键点模型

### 运行命令

```bash
# 完整运行（手部+面部，无预览）
python -m echoesphere_omni.run

# 带预览
python -m echoesphere_omni.run --preview

# 仅手部检测，带预览
python -m echoesphere_omni.run --preview --no-face

# 自定义 TCP 目标
python -m echoesphere_omni.run --host 192.168.1.100 --port 65432

# 摄像头配置
python -m echoesphere_omni.run --cameraId 0 --frameWidth 1280 --frameHeight 960

# 调试模式
python -m echoesphere_omni.run --debug
```

### 配置文件

系统通过命令行参数配置，无外部配置文件。所有参数均可通过 `python -m echoesphere_omni.run --help` 查看。

## 扩展与定制

### 添加新检测器

要添加新的检测器（如姿势检测、物体检测）：

1. 创建新的检测器类，继承类似模式
2. 内部维护状态机，仅在状态变化时发布事件
3. 使用 `EventBus.publish()` 发布 `UnifiedEvent` 事件
4. 在 `run.py` 中添加启动逻辑

### 修改事件格式

如需扩展事件数据：
1. 修改 `UnifiedEvent` 类的 `data` 字段结构
2. 更新检测器的 `_publish` 方法调用
3. 确保 TCP 接收端能解析新格式

### 更换传输协议

要更换传输方式（如 WebSocket、UDP、MQTT）：
1. 创建新的发送器类，替换 `TcpSender`
2. 实现从 EventBus 消费事件的逻辑
3. 实现对应协议的发送逻辑

## 性能考虑

### 线程模型

- **检测器线程**：每个检测器运行在独立线程中，使用 MediaPipe 的异步检测 API
- **TCP 发送器线程**：运行 asyncio 事件循环，处理网络 I/O
- **事件总线**：使用线程安全队列，避免锁竞争

### 内存管理

- 事件队列有大小限制（默认 100），防止内存无限增长
- 图像数据不通过事件总线传输，仅传输坐标和元数据
- MediaPipe 检测器自动管理模型内存

### 网络优化

- 仅发送状态变化事件，减少网络流量
- 异步 I/O 避免阻塞检测流水线
- 自动重连机制（需实现）

## 故障排除

### 常见问题

1. **摄像头无法打开**
   - 检查 `--cameraId` 参数
   - 确保摄像头未被其他程序占用

2. **TCP 连接失败**
   - 检查目标服务器是否运行
   - 验证防火墙设置

3. **检测性能差**
   - 降低摄像头分辨率
   - 关闭预览窗口以节省资源
   - 调整 MediaPipe 置信度阈值

4. **事件丢失**
   - 增加 EventBus 队列大小
   - 检查网络延迟和带宽

### 日志记录

- 默认 INFO 级别日志
- 使用 `--debug` 参数启用详细日志
- 日志格式：`时间 [级别] 组件: 消息`

## 未来扩展方向

1. **更多手势支持**：添加握拳、点赞、滑动等手势
2. **面部表情识别**：基于面部 blendshapes 检测表情
3. **多人支持**：扩展为多人检测和跟踪
4. **Web 界面**：添加实时可视化 Web 界面
5. **录制与回放**：支持录制检测结果并回放
6. **插件系统**：支持动态加载检测器和输出模块

## 参考资料

- [MediaPipe 官方文档](https://developers.google.com/mediapipe)
- [OpenCV Python 文档](https://docs.opencv.org/)
- [Python asyncio 文档](https://docs.python.org/3/library/asyncio.html)

---

*文档最后更新: 2026-04-04*
*项目版本: 基于代码库当前状态*