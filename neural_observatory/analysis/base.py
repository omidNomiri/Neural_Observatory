"""
Neural Observatory — Base Analyzer
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..core.configuration import ObservatoryConfig


class AnalysisStatus(str, Enum):
    HEALTHY = "healthy"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


STATUS_SEVERITY = {
    AnalysisStatus.HEALTHY: 0.0,
    AnalysisStatus.INFO: 0.25,
    AnalysisStatus.WARNING: 0.6,
    AnalysisStatus.CRITICAL: 1.0,
}


@dataclass
class AnalysisResult:
    analyzer_name: str
    layer_name: str
    status: AnalysisStatus
    severity: float
    metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    step: int = 0
    timestamp: float = field(default_factory=time.time)

    def is_healthy(self) -> bool:
        return self.status == AnalysisStatus.HEALTHY

    def __repr__(self) -> str:
        return (
            f"AnalysisResult({self.analyzer_name!r}, {self.layer_name!r}, "
            f"status={self.status.value}, severity={self.severity:.2f})"
        )


def _classify_severity(score: float) -> AnalysisStatus:
    """Map a continuous severity score to a status level."""
    # We map a raw severity score to a label here. Keep in mind that 
    # HEALTHY < INFO < WARNING < CRITICAL conceptually maps to the 
    # STATUS_SEVERITY dict, but they don't strictly roundtrip.
    if score < 0.1:
        return AnalysisStatus.HEALTHY
    if score < 0.4:
        return AnalysisStatus.INFO
    if score < 0.75:
        return AnalysisStatus.WARNING
    return AnalysisStatus.CRITICAL


class BaseAnalyzer(ABC):
    def __init__(self, config: Optional[ObservatoryConfig] = None) -> None:
        self._config = config or ObservatoryConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this analyzer."""

    @abstractmethod
    def analyze(self, observations: Dict[str, Any]) -> List[AnalysisResult]:
        """Analyse the collected observations and return a list of results."""

    def is_applicable(self, layer_name: str, layer_type: str) -> bool:
        return True

    @staticmethod
    def _make_result(
        analyzer_name: str,
        layer_name: str,
        severity: float,
        metrics: Optional[Dict[str, float]] = None,
        warnings: Optional[List[str]] = None,
        info: Optional[List[str]] = None,
        step: int = 0,
    ) -> AnalysisResult:
        status = _classify_severity(severity)
        return AnalysisResult(
            analyzer_name=analyzer_name,
            layer_name=layer_name,
            status=status,
            severity=round(severity, 4),
            metrics=metrics or {},
            warnings=warnings or [],
            info=info or [],
            step=step,
        )