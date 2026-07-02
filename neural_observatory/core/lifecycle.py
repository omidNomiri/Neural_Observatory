"""
Neural Observatory — Lifecycle Manager
"""
from __future__ import annotations

import logging
import threading
from enum import Enum, auto

logger = logging.getLogger(__name__)


class MonitoringState(Enum):
    IDLE = auto()
    ACTIVE = auto()
    STOPPED = auto()
    REPORTING = auto()


class LifecycleError(Exception):
    """Raised on illegal state transitions."""


class LifecycleManager:
    _TRANSITIONS = {
        (MonitoringState.IDLE, MonitoringState.ACTIVE),
        (MonitoringState.ACTIVE, MonitoringState.STOPPED),
        (MonitoringState.ACTIVE, MonitoringState.REPORTING),
        (MonitoringState.STOPPED, MonitoringState.ACTIVE),
        (MonitoringState.STOPPED, MonitoringState.REPORTING),
        (MonitoringState.REPORTING, MonitoringState.ACTIVE),
        (MonitoringState.REPORTING, MonitoringState.STOPPED),
    }

    def __init__(self) -> None:
        self._state = MonitoringState.IDLE
        self._previous = MonitoringState.IDLE
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            self._transition(MonitoringState.ACTIVE)

    def stop(self) -> None:
        with self._lock:
            if self._state == MonitoringState.IDLE:
                logger.warning("stop() called before watch() — nothing to stop.")
                return
            self._transition(MonitoringState.STOPPED)

    def begin_reporting(self) -> None:
        with self._lock:
            self._transition(MonitoringState.REPORTING)

    def end_reporting(self) -> None:
        with self._lock:
            if self._state != MonitoringState.REPORTING:
                raise LifecycleError(
                    f"end_reporting() called but state is {self._state.name}, "
                    f"expected REPORTING"
                )
            target = (
                MonitoringState.STOPPED
                if self._previous == MonitoringState.STOPPED
                else MonitoringState.ACTIVE
            )
            self._transition(target)

    @property
    def state(self) -> MonitoringState:
        with self._lock:
            return self._state

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._state in (MonitoringState.ACTIVE, MonitoringState.REPORTING)

    @property
    def is_idle(self) -> bool:
        with self._lock:
            return self._state == MonitoringState.IDLE

    @property
    def is_stopped(self) -> bool:
        with self._lock:
            return self._state == MonitoringState.STOPPED

    def require_reportable(self, operation: str):
        with self._lock:
            if self._state not in (
                MonitoringState.ACTIVE,
                MonitoringState.STOPPED
            ):
                raise LifecycleError(
                    f"{operation} not allowed in state {self._state}"
                )

    def _transition(self, target: MonitoringState) -> None:
        pair = (self._state, target)
        if pair not in self._TRANSITIONS:
            raise LifecycleError(
                f"Invalid lifecycle transition: {self._state.name} → {target.name}"
            )
        self._previous = self._state
        logger.debug("Lifecycle: %s → %s", self._state.name, target.name)
        self._state = target

    def __repr__(self) -> str:
        return f"LifecycleManager(state={self._state.name})"
