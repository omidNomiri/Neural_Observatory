"""
Neural Observatory — Base Reporter
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..analysis.base import AnalysisResult, AnalysisStatus
from ..core.configuration import ObservatoryConfig


@dataclass
class Report:
    timestamp: float
    step: int
    epoch: int
    results: List[AnalysisResult] = field(default_factory=list)

    def by_layer(self) -> Dict[str, List[AnalysisResult]]:
        out: Dict[str, List[AnalysisResult]] = {}
        for r in self.results:
            out.setdefault(r.layer_name, []).append(r)
        return out

    def by_analyzer(self) -> Dict[str, List[AnalysisResult]]:
        out: Dict[str, List[AnalysisResult]] = {}
        for r in self.results:
            out.setdefault(r.analyzer_name, []).append(r)
        return out

    def warnings(self) -> List[AnalysisResult]:
        return [
            r for r in self.results
            if r.status in (AnalysisStatus.WARNING, AnalysisStatus.CRITICAL)
        ]

    def critical(self) -> List[AnalysisResult]:
        return [r for r in self.results if r.status == AnalysisStatus.CRITICAL]

    def healthy(self) -> List[AnalysisResult]:
        return [r for r in self.results if r.status == AnalysisStatus.HEALTHY]

    def overall_status(self) -> AnalysisStatus:
        if any(r.status == AnalysisStatus.CRITICAL for r in self.results):
            return AnalysisStatus.CRITICAL
        if any(r.status == AnalysisStatus.WARNING for r in self.results):
            return AnalysisStatus.WARNING
        if any(r.status == AnalysisStatus.INFO for r in self.results):
            return AnalysisStatus.INFO
        return AnalysisStatus.HEALTHY

    def max_severity(self) -> float:
        if not self.results:
            return 0.0
        return max(r.severity for r in self.results)

    def summary_dict(self) -> Dict[str, Any]:
        status_counts = {s.value: 0 for s in AnalysisStatus}
        for r in self.results:
            status_counts[r.status.value] += 1
        return {
            "step": self.step,
            "epoch": self.epoch,
            "overall_status": self.overall_status().value,
            "max_severity": round(self.max_severity(), 4),
            "total_results": len(self.results),
            "status_counts": status_counts,
        }

    def filtered(
        self,
        min_severity: float = 0.0,
        max_results: int = 1000,
        include_healthy: bool = True,
    ) -> List[AnalysisResult]:
        results = [
            r for r in self.results
            if r.severity >= min_severity
            and (include_healthy or not r.is_healthy())
        ]
        results.sort(key=lambda r: r.severity, reverse=True)
        return results[:max_results]


class BaseReporter(ABC):
    @abstractmethod
    def report(
        self,
        report: Report,
        config: Optional[ObservatoryConfig] = None,
    ) -> None:
        """Render the report."""