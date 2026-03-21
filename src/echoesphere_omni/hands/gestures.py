from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GestureType(Enum):
    HAND_LOST = "hand_lost"
    INDEX_TIP_MOVED = "index_tip_moved"


@dataclass(frozen=True)
class GestureEvent:
    """Immutable event emitted when a gesture is detected.

    Attributes:
        gesture_type: The type of gesture detected.
        data: Arbitrary payload associated with the event.
        timestamp_ms: Timestamp in milliseconds when the event was created.
    """

    gesture_type: GestureType
    data: dict[str, Any]
    timestamp_ms: int

    def to_tcp_payload(self) -> str:
        """Serialize the event into a JSON string for TCP transmission."""
        import json

        if self.gesture_type == GestureType.INDEX_TIP_MOVED:
            return json.dumps(
                {
                    "h": 1,
                    "x": f"{self.data['x']:.3f}",
                    "y": f"{self.data['y']:.3f}",
                    "v": f"{self.data['velocity']:.3f}",
                }
            )
        elif self.gesture_type == GestureType.HAND_LOST:
            return '{"h":0}'
        return "{}"
