"""
Neural Observatory — Event System
Lets different parts of the framework talk to each other without hard dependencies.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

class EventType(Enum):
    TRAINING_STARTED = auto()
    TRAINING_STOPPED = auto()
    EPOCH_STARTED = auto()
    EPOCH_FINISHED = auto()
    BATCH_STARTED = auto()
    BATCH_FINISHED = auto()
    FORWARD_COMPLETED = auto()
    BACKWARD_COMPLETED = auto()
    ANOMALY_DETECTED = auto()
    OBSERVATION_COLLECTED = auto()
    ANALYSIS_COMPLETED = auto()
    REPORT_GENERATED = auto()


@dataclass
class Event:
    event_type: EventType
    source: str
    step: int = 0
    epoch: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


EventHandler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[EventHandler]] = {}

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, event: Event) -> None:
        for handler in self._subscribers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as exc:
                # Don't let one bad handler crash the whole pipeline
                logger.error(
                    "EventBus handler %s raised %s: %s",
                    handler,
                    type(exc).__name__,
                    exc,
                    exc_info=True
                )

    def clear(self) -> None:
        self._subscribers.clear()