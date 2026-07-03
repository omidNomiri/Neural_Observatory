"""
Neural Observatory — Activation Statistics Analyzer
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAnalyzer, AnalysisResult
from ..core.configuration import ObservatoryConfig
from ..collectors.base import Observation

logger = logging.getLogger(__name__)


class ActivationStatisticsAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "activation_statistics"

    def analyze(self, observations: Dict[str, Any]) -> List[AnalysisResult]:
        activations: Dict[str, List[Observation]] = observations.get("activations", {})
        results: List[AnalysisResult] = []

        for layer_name, obs_list in activations.items():
            if not obs_list:
                continue
            results.append(self._analyze_layer(layer_name, obs_list))

        return results

    def _analyze_layer(
        self, layer_name: str, obs_list: List[Observation]
    ) -> AnalysisResult:
        cfg = self._config

        all_means = [o.stats.get("mean", 0.0) for o in obs_list]
        all_stds  = [o.stats.get("std", 0.0)  for o in obs_list]
        all_abs   = [o.stats.get("abs_mean", 0.0) for o in obs_list]
        all_sp    = [o.stats.get("sparsity", 0.0) for o in obs_list]
        all_norms = [o.stats.get("norm", 0.0) for o in obs_list]

        mean_mean   = float(np.mean(all_means))
        mean_std    = float(np.mean(all_stds))
        mean_abs    = float(np.mean(all_abs))
        mean_sparse = float(np.mean(all_sp))
        mean_norm   = float(np.mean(all_norms))

        skewness = kurtosis = saturation_ratio = None
        arrays = [o.values for o in obs_list if o.values is not None]
        if arrays:
            try:
                flat = np.concatenate([a.ravel() for a in arrays])
                skewness         = float(self._skewness(flat))
                kurtosis         = float(self._kurtosis(flat))
                saturation_ratio = float(np.mean(np.abs(flat) > 0.99))
            except Exception:
                pass

        metrics: Dict[str, float] = {
            "mean": mean_mean,
            "std": mean_std,
            "abs_mean": mean_abs,
            "sparsity": mean_sparse,
            "norm": mean_norm,
            "num_observations": len(obs_list),
        }
        if skewness is not None:
            metrics["skewness"] = skewness
        if kurtosis is not None:
            metrics["kurtosis"] = kurtosis
        if saturation_ratio is not None:
            metrics["saturation_ratio"] = saturation_ratio

        severity = 0.0
        warnings: List[str] = []
        info: List[str] = []

        if mean_sparse >= cfg.high_sparsity_warning:
            severity = max(severity, 0.6)
            warnings.append(
                f"High sparsity {mean_sparse:.0%} — "
                "most activations are near zero."
            )
        elif mean_sparse >= 0.5:
            severity = max(severity, 0.25)
            info.append(f"Moderate sparsity {mean_sparse:.0%}.")

        if mean_abs < 1e-5:
            severity = max(severity, 0.3)
            info.append(
                f"Very low mean absolute activation {mean_abs:.2e} — "
                "layer may be inactive."
            )

        if saturation_ratio is not None and saturation_ratio > 0.5:
            severity = max(severity, 0.4)
            warnings.append(
                f"High saturation {saturation_ratio:.0%} — "
                "many activations near ±1 (may signal vanishing gradients)."
            )

        if not warnings and not info:
            info.append(
                f"Activations healthy — mean={mean_mean:.4f}, "
                f"std={mean_std:.4f}, sparsity={mean_sparse:.0%}."
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
    def _skewness(x: np.ndarray) -> float:
        mu = np.mean(x)
        sigma = np.std(x) + 1e-10
        return float(np.mean(((x - mu) / sigma) ** 3))

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        mu = np.mean(x)
        sigma = np.std(x) + 1e-10
        return float(np.mean(((x - mu) / sigma) ** 4) - 3.0)
