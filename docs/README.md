# EchoSphere Omni 文档目录

## 概述

EchoSphere Omni 是一个基于 MediaPipe 的手部和面部关键点检测系统，能够实时检测手势和面部姿态，并通过 TCP 协议发布结构化事件。

## 核心文档

### [架构文档](ARCHITECTURE.md)
详细的技术架构说明，包括：
- 系统架构图和数据流
- 所有核心组件的详细说明
- TCP 协议规范
- 配置和运行指南
- 扩展和定制方法

## 快速链接

- [项目根目录 README](../README.md) - 项目基本信息和快速开始指南
- [CLAUDE.md](../CLAUDE.md) - Claude Code 项目说明文件

## 组件文档

| 组件 | 文件 | 说明 |
|------|------|------|
| 运行入口 | `run.py` | 主程序入口，参数解析和线程管理 |
| 事件总线 | `event_bus.py` | 线程安全的事件队列 |
| 统一事件 | `events.py` | 事件数据格式定义 |
| TCP 发送器 | `sender.py` | 事件到网络的桥接 |
| TCP 客户端 | `net/client.py` | TCP 协议实现 |
| 手部检测器 | `hands/detector.py` | 手势检测和状态机 |
| 面部检测器 | `face/detector.py` | 面部检测和事件发布 |
| 手势定义 | `hands/gestures.py` | 手势常量和阈值 |

## 使用指南

### 运行系统
```bash
python -m echoesphere_omni.run --preview --no-face
```

### 查看帮助
```bash
python -m echoesphere_omni.run --help
```

## 开发指南

### 添加新检测器
1. 创建检测器类，维护内部状态机
2. 仅在状态变化时发布 `UnifiedEvent`
3. 在 `run.py` 中添加启动逻辑

### 修改事件格式
1. 更新 `UnifiedEvent` 类的 `data` 字段结构
2. 修改检测器的 `_publish` 方法
3. 确保接收端兼容新格式

## 协议参考

### TCP 消息格式
```
[4字节长度][JSON载荷]
```

### 事件消息结构
```json
{
  "type": "text",
  "data": "{\"source\":\"hand\",\"event\":\"pinch\",\"data\":{\"x\":0.5,\"y\":0.5},\"timestamp_ms\":1234567890}"
}
```

## 故障排除

常见问题请参见 [架构文档](ARCHITECTURE.md#故障排除) 章节。

---

*文档最后更新: 2026-04-04*