"""
Neural Observatory — Anomaly Detection Analyzer
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .base import BaseAnalyzer, AnalysisResult
from ..core.configuration import ObservatoryConfig
from ..collectors.base import Observation

logger = logging.getLogger(__name__)


class AnomalyDetectionAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "anomaly_detection"

    def analyze(self, observations: Dict[str, Any]) -> List[AnalysisResult]:
        activations: Dict[str, List[Observation]] = observations.get("activations", {})
        gradients: Dict[str, List[Observation]]   = observations.get("gradients", {})

        all_layers = sorted(set(activations.keys()) | set(gradients.keys()))
        results: List[AnalysisResult] = []

        for layer_name in all_layers:
            act_obs  = activations.get(layer_name, [])
            grad_obs = gradients.get(layer_name, [])
            results.append(self._analyze_layer(layer_name, act_obs, grad_obs))

        return results

    def _analyze_layer(
        self,
        layer_name: str,
        act_obs: List[Observation],
        grad_obs: List[Observation],
    ) -> AnalysisResult:
        cfg = self._config
        severity = 0.0
        warnings: List[str] = []
        info: List[str] = []
        metrics: Dict[str, float] = {}

        if act_obs:
            nan_steps  = [o.step for o in act_obs if o.stats.get("has_nan")]
            inf_steps  = [o.step for o in act_obs if o.stats.get("has_inf")]
            act_norms  = np.array([o.stats.get("norm", 0.0) for o in act_obs])

            metrics["act_nan_count"] = len(nan_steps)
            metrics["act_inf_count"] = len(inf_steps)

            if nan_steps:
                severity = max(severity, 1.0)
                warnings.append(
                    f"NaN detected in activations at steps {nan_steps[:5]}."
                )
            if inf_steps:
                severity = max(severity, 1.0)
                warnings.append(
                    f"Inf detected in activations at steps {inf_steps[:5]}."
                )

            spike_count = self._detect_spikes(act_norms, cfg.spike_std_multiplier)
            metrics["act_spike_count"] = spike_count
            if spike_count > 0:
                sev = min(spike_count / len(act_obs) * 2, 0.8)
                severity = max(severity, sev)
                warnings.append(
                    f"{spike_count} activation norm spike(s) detected "
                    f"({spike_count/len(act_obs):.0%} of steps)."
                )

        if grad_obs:
            nan_g = [o.step for o in grad_obs if o.stats.get("has_nan")]
            inf_g = [o.step for o in grad_obs if o.stats.get("has_inf")]
            grad_norms = np.array([o.stats.get("l2_norm", 0.0) for o in grad_obs])

            metrics["grad_nan_count"] = len(nan_g)
            metrics["grad_inf_count"] = len(inf_g)

            if nan_g:
                severity = max(severity, 1.0)
                warnings.append(
                    f"NaN detected in gradients at steps {nan_g[:5]}."
                )
            if inf_g:
                severity = max(severity, 1.0)
                warnings.append(
                    f"Inf detected in gradients at steps {inf_g[:5]}."
                )

            grad_spike_count = self._detect_spikes(grad_norms, cfg.spike_std_multiplier)
            metrics["grad_spike_count"] = grad_spike_count
            if grad_spike_count > 0:
                sev = min(grad_spike_count / len(grad_obs) * 2, 0.8)
                severity = max(severity, sev)
                warnings.append(
                    f"{grad_spike_count} gradient norm spike(s) detected."
                )

        if not warnings:
            info.append("No numerical anomalies detected.")
            metrics.setdefault("act_nan_count", 0.0)
            metrics.setdefault("grad_nan_count", 0.0)

        metrics["total_anomalies"] = (
            metrics.get("act_nan_count", 0)
            + metrics.get("act_inf_count", 0)
            + metrics.get("grad_nan_count", 0)
            + metrics.get("grad_inf_count", 0)
        )

        return self._make_result(
            analyzer_name=self.name,
            layer_name=layer_name,
            severity=severity,
            metrics=metrics,
            warnings=warnings,
            info=info,
        )

    @staticmethod
    def _detect_spikes(norms: np.ndarray, std_multiplier: float) -> int:
        if len(norms) < 3:
            return 0
        mu = np.mean(norms)
        sigma = np.std(norms)
        if sigma < 1e-10:
            return 0
        return int(np.sum(norms > mu + std_multiplier * sigma))
