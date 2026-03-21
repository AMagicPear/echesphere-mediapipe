"""Unified event format used by all detectors (hand, face, etc.)."""

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class UnifiedEvent:
    """The single event type used across all detectors and the TCP sender.

    Attributes:
        source: Which detector produced the event (e.g. "hand", "face").
        event: The event name (e.g. "pinch", "open_both_hands").
        data: Arbitrary payload dict (positions, scores, etc.).
        timestamp_ms: Wall-clock time in milliseconds when the event fired.
    """

    source: str
    event: str
    data: dict[str, Any]
    timestamp_ms: int

    def to_json(self) -> str:
        """Serialize to a compact JSON string for TCP transmission."""
        import json
        return json.dumps(asdict(self))
