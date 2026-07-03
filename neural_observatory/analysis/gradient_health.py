"""
Neural Observatory -- Gradient Health Analyzer
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .base import BaseAnalyzer, AnalysisResult
from ..core.configuration import ObservatoryConfig
from ..collectors.base import Observation

logger = logging.getLogger(__name__)

_VANISH_ORDERS_OF_MAGNITUDE = 6.0
_EXPLODE_ORDERS_OF_MAGNITUDE = 3.0
_TREND_MIN_POINTS = 4


class GradientHealthAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "gradient_health"

    def analyze(self, observations: Dict[str, Any]) -> List[AnalysisResult]:
        gradients: Dict[str, List[Observation]] = observations.get("gradients", {})
        activations: Dict[str, List[Observation]] = observations.get("activations", {})

        reference_norm = _compute_reference_norm(gradients)

        all_layers = set(gradients.keys()) | set(activations.keys())
        results: List[AnalysisResult] = []

        for layer_name in sorted(all_layers):
            obs_list = gradients.get(layer_name, [])
            results.append(self._analyze_layer(layer_name, obs_list, reference_norm))

        return results

    def _analyze_layer(
        self,
        layer_name: str,
        obs_list: List[Observation],
        reference_norm: float,
    ) -> AnalysisResult:
        if not obs_list:
            return self._make_result(
                analyzer_name=self.name,
                layer_name=layer_name,
                severity=0.0,
                metrics={"num_observations": 0},
                info=["No gradient observations recorded for this layer. "
                      "Layer may not be in the backward graph."],
            )

        norms = np.array([o.stats.get("l2_norm", 0.0) for o in obs_list], dtype=np.float64)
        has_nan = any(o.stats.get("has_nan", 0) for o in obs_list)
        has_inf = any(o.stats.get("has_inf", 0) for o in obs_list)

        mean_norm = float(np.mean(norms))
        max_norm = float(np.max(norms))
        min_norm = float(np.min(norms))
        std_norm = float(np.std(norms))

        metrics: Dict[str, float] = {
            "mean_norm": mean_norm,
            "max_norm": max_norm,
            "min_norm": min_norm,
            "std_norm": std_norm,
            "num_observations": len(obs_list),
            "has_nan": int(has_nan),
            "has_inf": int(has_inf),
        }
        if reference_norm > 0:
            metrics["reference_norm"] = reference_norm
            metrics["ratio_to_reference"] = mean_norm / reference_norm

        severity, warnings, info = self._classify(
            norms=norms,
            mean_norm=mean_norm,
            reference_norm=reference_norm,
            has_nan=has_nan,
            has_inf=has_inf,
            cfg=self._config,
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
    def _classify(
        norms: np.ndarray,
        mean_norm: float,
        reference_norm: float,
        has_nan: bool,
        has_inf: bool,
        cfg: ObservatoryConfig,
    ) -> Tuple[float, List[str], List[str]]:
        warnings: List[str] = []
        info: List[str] = []
        severity = 0.0

        if has_nan:
            warnings.append("NaN detected in gradients -- training is broken.")
            severity = max(severity, 1.0)
        if has_inf:
            warnings.append("Inf detected in gradients -- training is broken.")
            severity = max(severity, 1.0)

        vanishing, exploding = _compare_to_reference(
            mean_norm, reference_norm, cfg,
        )

        if vanishing is not None:
            severity = max(severity, vanishing.severity)
            warnings.append(vanishing.message)
        if exploding is not None:
            severity = max(severity, exploding.severity)
            warnings.append(exploding.message)

        collapse = _detect_collapse(norms)
        if collapse is not None:
            severity = max(severity, collapse.severity)
            warnings.append(collapse.message)

        if not warnings:
            info.append(f"Gradient norm {mean_norm:.4f} -- within healthy range.")

        return severity, warnings, info


class _Flag:
    __slots__ = ("severity", "message")

    def __init__(self, severity: float, message: str) -> None:
        self.severity = severity
        self.message = message


def _compute_reference_norm(gradients: Dict[str, List[Observation]]) -> float:
    best = 0.0
    for obs_list in gradients.values():
        if not obs_list:
            continue
        norms = [o.stats.get("l2_norm", 0.0) for o in obs_list]
        mean_norm = float(np.mean(norms)) if norms else 0.0
        if mean_norm > best:
            best = mean_norm
    return best


def _compare_to_reference(
    mean_norm: float,
    reference_norm: float,
    cfg: ObservatoryConfig,
) -> Tuple:
    vanishing = None
    exploding = None

    if mean_norm < cfg.gradient_vanishing_threshold:
        ratio = cfg.gradient_vanishing_threshold / (mean_norm + 1e-15)
        sev = float(min(ratio / 100.0, 1.0))
        vanishing = _Flag(
            sev,
            f"Vanishing gradients -- mean norm {mean_norm:.2e} "
            f"(absolute threshold {cfg.gradient_vanishing_threshold:.2e}).",
        )
    elif mean_norm > cfg.gradient_exploding_threshold:
        log_ratio = math.log10(mean_norm / cfg.gradient_exploding_threshold + 1e-10)
        sev = float(min(log_ratio / 3.0, 1.0))
        exploding = _Flag(
            sev,
            f"Exploding gradients -- mean norm {mean_norm:.2e} "
            f"(absolute threshold {cfg.gradient_exploding_threshold:.2e}).",
        )

    if reference_norm > 0 and mean_norm > 0:
        log_ratio = math.log10(mean_norm / reference_norm)

        if log_ratio < -_VANISH_ORDERS_OF_MAGNITUDE:
            depth = (-log_ratio) / _VANISH_ORDERS_OF_MAGNITUDE
            sev = float(min(0.6 * depth, 1.0))
            if vanishing is None or sev > vanishing.severity:
                vanishing = _Flag(
                    sev,
                    f"Vanishing gradients -- mean norm {mean_norm:.2e} is "
                    f"{-log_ratio:.1f} orders of magnitude below the busiest "
                    f"layer ({reference_norm:.2e}).",
                )
        elif log_ratio > _EXPLODE_ORDERS_OF_MAGNITUDE:
            height = log_ratio / _EXPLODE_ORDERS_OF_MAGNITUDE
            sev = float(min(0.6 * height, 1.0))
            if exploding is None or sev > exploding.severity:
                exploding = _Flag(
                    sev,
                    f"Exploding gradients -- mean norm {mean_norm:.2e} is "
                    f"{log_ratio:.1f} orders of magnitude above the busiest "
                    f"layer ({reference_norm:.2e}).",
                )

    return vanishing, exploding


def _detect_collapse(norms: np.ndarray):
    if len(norms) < _TREND_MIN_POINTS:
        return None

    split = max(1, len(norms) // 4)
    recent = float(np.mean(norms[-split:]))
    earlier = float(np.mean(norms[:-split]))

    if earlier < 1e-12:
        return None

    ratio = recent / earlier
    if ratio < 0.1:
        sev = float(min(1.0, 0.6 + (0.1 - ratio) * 4.0))
        return _Flag(
            sev,
            f"Gradient norm collapsed recently -- last {split} steps average "
            f"{recent:.2e}, down from {earlier:.2e} earlier "
            f"({ratio:.1%} of baseline).",
        )
    return None
