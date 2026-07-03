"""
Neural Observatory -- Dead Neuron Analyzer
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseAnalyzer, AnalysisResult
from ..core.configuration import ObservatoryConfig
from ..collectors.base import Observation

logger = logging.getLogger(__name__)


class DeadNeuronAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "dead_neurons"

    def analyze(self, observations: Dict[str, Any]) -> List[AnalysisResult]:
        activations: Dict[str, List[Observation]] = observations.get("activations", {})
        results: List[AnalysisResult] = []

        for layer_name, obs_list in activations.items():
            if not obs_list:
                continue
            result = self._analyze_layer(layer_name, obs_list)
            if result is not None:
                results.append(result)

        return results

    def _analyze_layer(
        self,
        layer_name: str,
        obs_list: List[Observation],
    ) -> Optional[AnalysisResult]:
        cfg = self._config
        arrays = [o.values for o in obs_list if o.values is not None]

        if arrays:
            dead_ratio, scale_used = self._dead_ratio_from_arrays(
                arrays, cfg.dead_neuron_threshold, cfg.dead_neuron_persistence,
            )
        else:
            abs_means = [o.stats.get("abs_mean") for o in obs_list if "abs_mean" in o.stats]
            if not abs_means:
                return None
            mean_abs = float(np.mean(abs_means))
            dead_ratio = 1.0 if mean_abs < cfg.dead_neuron_threshold else 0.0
            scale_used = "stats_only"

        metrics: Dict[str, float] = {
            "dead_ratio": round(dead_ratio, 4),
            "alive_ratio": round(1.0 - dead_ratio, 4),
            "num_observations": len(obs_list),
            # Always include scale so users in stats_only_mode know if this was per-neuron or coarse.
            "scale": scale_used,
        }

        if arrays:
            total_neurons = _count_neurons(arrays[0])
            metrics["total_neurons"] = int(total_neurons)
            metrics["dead_neurons"] = int(round(total_neurons * dead_ratio))

        warnings: List[str] = []
        info: List[str] = []

        if dead_ratio >= cfg.dead_neuron_critical_ratio:
            warnings.append(
                f"{dead_ratio:.0%} of neurons are dead -- "
                "this layer contributes almost nothing to learning."
            )
        elif dead_ratio >= cfg.dead_neuron_warning_ratio:
            warnings.append(
                f"{dead_ratio:.0%} of neurons appear dead -- "
                "consider reducing learning rate or checking weight init."
            )
        else:
            info.append(f"Dead neuron ratio {dead_ratio:.0%} -- within acceptable range.")

        return self._make_result(
            analyzer_name=self.name,
            layer_name=layer_name,
            severity=dead_ratio,
            metrics=metrics,
            warnings=warnings,
            info=info,
        )

    @staticmethod
    def _dead_ratio_from_arrays(
        arrays: List[np.ndarray],
        threshold: float,
        persistence: float,
    ) -> Tuple[float, str]:
        try:
            flattened = [
                a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1)
                for a in arrays
            ]
            stacked = np.concatenate(flattened, axis=0)
        except Exception:
            abs_mean = np.mean([np.mean(np.abs(a)) for a in arrays])
            return (1.0 if abs_mean < threshold else 0.0, "absolute")

        peak = float(np.max(np.abs(stacked)))
        if peak < 1e-12:
            return (1.0, "absolute")

        relative_floor = 1e-3 * peak
        effective_threshold = max(threshold, relative_floor)
        scale_used = "relative" if effective_threshold > threshold else "absolute"

        dead_mask = np.abs(stacked) < effective_threshold
        dead_fraction_per_neuron = np.mean(dead_mask, axis=0)
        dead_neurons = int(np.sum(dead_fraction_per_neuron >= persistence))

        dead_ratio = float(dead_neurons) / float(dead_fraction_per_neuron.size + 1e-10)
        return (dead_ratio, scale_used)


def _count_neurons(arr: np.ndarray) -> int:
    if arr.ndim > 1:
        return int(arr.reshape(arr.shape[0], -1).shape[-1])
    return int(arr.shape[-1])