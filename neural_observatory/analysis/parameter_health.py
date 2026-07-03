"""
Neural Observatory -- Parameter Health Analyzer
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .base import BaseAnalyzer, AnalysisResult
from ..core.configuration import ObservatoryConfig
from ..collectors.base import Observation

logger = logging.getLogger(__name__)


class ParameterHealthAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "parameter_health"

    def analyze(self, observations: Dict[str, Any]) -> List[AnalysisResult]:
        parameters: Dict[str, List[Observation]] = observations.get("parameters", {})
        results: List[AnalysisResult] = []

        for param_name, obs_list in parameters.items():
            if not obs_list:
                continue
            results.append(self._analyze_param(param_name, obs_list))

        return results

    def _analyze_param(
        self,
        param_name: str,
        obs_list: List[Observation],
    ) -> AnalysisResult:
        cfg = self._config

        latest = obs_list[-1]
        norm = latest.stats.get("norm", 0.0)
        drift = latest.stats.get("drift_from_init", 0.0)
        has_nan = bool(latest.stats.get("has_nan", False))
        has_inf = bool(latest.stats.get("has_inf", False))

        norms = np.array([o.stats.get("norm", 0.0) for o in obs_list], dtype=np.float64)

        mean_update_ratio = _compute_update_ratio(norms)

        metrics: Dict[str, float] = {
            "current_norm": norm,
            "drift_from_init": drift,
            "mean_update_ratio": mean_update_ratio,
            "norm_mean": float(np.mean(norms)),
            "norm_std": float(np.std(norms)),
            "has_nan": int(has_nan),
            "has_inf": int(has_inf),
            "num_observations": len(obs_list),
        }

        severity, warnings, info = self._classify(
            param_name, drift, norm, mean_update_ratio,
            has_nan, has_inf, len(obs_list), cfg,
        )

        return self._make_result(
            analyzer_name=self.name,
            layer_name=param_name,
            severity=severity,
            metrics=metrics,
            warnings=warnings,
            info=info,
        )

    @staticmethod
    def _classify(
        param_name: str,
        drift: float,
        norm: float,
        mean_update_ratio: float,
        has_nan: bool,
        has_inf: bool,
        num_observations: int,
        cfg: ObservatoryConfig,
    ):
        severity = 0.0
        warnings: List[str] = []
        info: List[str] = []

        if has_nan:
            severity = max(severity, 1.0)
            warnings.append(f"NaN in parameter '{param_name}' -- training is broken.")
        if has_inf:
            severity = max(severity, 1.0)
            warnings.append(f"Inf in parameter '{param_name}' -- training is broken.")

        if drift > cfg.parameter_drift_threshold:
            sev = min(drift / (cfg.parameter_drift_threshold * 5), 0.8)
            severity = max(severity, sev)
            warnings.append(
                f"High parameter drift {drift:.1%} from initialization. "
                "Verify learning rate is not too large."
            )

        if mean_update_ratio > cfg.update_to_param_ratio_warning:
            sev = min(mean_update_ratio / (cfg.update_to_param_ratio_warning * 5), 0.7)
            severity = max(severity, sev)
            warnings.append(
                f"Large step-to-step parameter change "
                f"({mean_update_ratio:.1%} of norm per step)."
            )

        if norm < 1e-7 and num_observations > 3:
            severity = max(severity, 0.5)
            warnings.append(f"Parameter norm near zero ({norm:.2e}) -- weight collapse?")

        if not warnings:
            info.append(
                f"Parameters healthy -- norm={norm:.4f}, drift={drift:.1%}."
            )

        return severity, warnings, info


def _compute_update_ratio(norms: np.ndarray) -> float:
    """Mean step-to-step update as a fraction of the current norm."""
    if len(norms) < 2:
        return 0.0

    curr = norms[1:]
    prev = norms[:-1]
    valid = curr > 1e-10
    if not np.any(valid):
        return 0.0

    ratios = np.abs(curr[valid] - prev[valid]) / curr[valid]
    return float(np.mean(ratios))