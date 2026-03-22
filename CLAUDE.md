# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EchoSphere Omni is a gesture and face landmark detection system that publishes structured events over TCP. It uses MediaPipe for detection, with a thread-safe EventBus connecting detectors to an async TCP sender.

## Running the Application

```bash
# Full run (hand + face + preview)
python -m echoesphere_omni.run

# No preview (headless)
python -m echoesphere_omni.run --no-preview

# Hand only, no preview
python -m echoesphere_omni.run --no-preview --no-face

# Custom TCP target
python -m echoesphere_omni.run --host 192.168.1.100 --port 65432

# Camera settings
python -m echoesphere_omni.run --cameraId 0 --frameWidth 1280 --frameHeight 960
```

## Architecture

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

### Key Design Principles

- **Decoupling**: Detectors only detect; TCP sending logic is completely unaware of detection
- **Thread Safety**: EventBus uses `queue.Queue` for lock-free cross-thread communication
- **Silent Drop**: When queue is full, oldest events are dropped to prevent slow consumers from blocking detection pipeline
- **State-Change Only**: Detectors maintain internal state machines and only publish on gesture state transitions (not every frame)

### Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| `run.py` | Entry point | Wires detectors, EventBus, and TcpSender together in daemon threads |
| `EventBus` | `event_bus.py` | Thread-safe queue between detectors (producers) and TcpSender (consumer) |
| `UnifiedEvent` | `events.py` | Frozen dataclass — single event type for all detectors |
| `HandDetector` | `hands/detector.py` | MediaPipe hand landmarker + state machine (NO_HAND → HAND_PRESENT → PINCH → etc.) |
| `FaceDetector` | `face/detector.py` | MediaPipe face landmarker with face_detected/face_lost state transitions |
| `TcpSender` | `sender.py` | Owns a background thread running an asyncio event loop, reads from EventBus and sends over TCP |
| `TcpClient` | `net/client.py` | Async TCP client with length-prefixed binary protocol (4 bytes length + 1 byte type + N bytes payload) |
| Gesture types | `hands/gestures.py` | `HandGestureType` enum, `HandLandmark` indices, `GestureThresholds` constants |

### TCP Protocol

Binary protocol with length prefix:
```
4 bytes (big-endian int) : total payload length
1 byte                   : message type (0x00 = TEXT, 0x01 = IMAGE)
N bytes                  : UTF-8 JSON payload
```

Event JSON format:
```json
{"source": "hand", "event": "pinch", "data": {"x": 0.234, "y": 0.567}, "timestamp_ms": 1234567890}
```

### Gesture State Machine (HandDetector)

States: `NO_HAND → HAND_PRESENT → PINCH → PINCH_RELEASED → NO_HAND`

Events emitted only on state transitions:
- `hand_detected` / `hand_lost`
- `pinch` (with x,y position) / `pinch_released`
- `open_both_hands` (when both hands open simultaneously)
- `swipe_left` / `swipe_right` (wrist x-velocity threshold)

## Dependencies

- Python 3.12.11
- mediapipe >= 0.10.21
- opencv-python >= 4.11.0.86
- asyncio >= 4.0.0

Model files (in `models/` directory):
- `hand_landmarker.task` — MediaPipe hand landmark model
- `face_landmarker.task` — MediaPipe face landmark model