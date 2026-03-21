"""Hand-specific gesture type definitions."""

from enum import Enum


class HandGestureType(Enum):
    """Events emitted by the hand detector state machine."""

    HAND_DETECTED = "hand_detected"
    HAND_LOST = "hand_lost"
    PINCH = "pinch"
    PINCH_RELEASED = "pinch_released"
    OPEN_BOTH_HANDS = "open_both_hands"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"


# Landmark indices used for gesture recognition
class HandLandmark:
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    THUMB_MCP = 2
    INDEX_MCP = 5
    WRIST = 0


# Distance thresholds (in normalized 0-1 coordinates)
class GestureThresholds:
    # Pinch: distance between thumb tip and index tip
    PINCH_THRESHOLD = 0.06

    # Open hand: each fingertip must be this far from wrist
    OPEN_FINGER_DISTANCE = 0.15

    # Swipe: normalised x-velocity threshold
    SWIPE_VELOCITY_THRESHOLD = 0.025
