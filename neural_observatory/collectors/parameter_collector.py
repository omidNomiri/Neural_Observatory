"""
Neural Observatory -- Parameter Collector
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BaseCollector, Observation
from ..core.configuration import ObservatoryConfig

logger = logging.getLogger(__name__)


class ParameterCollector(BaseCollector):
    def __init__(
        self,
        config: Optional[ObservatoryConfig] = None,
        storage: Optional[Any] = None,
    ) -> None:
        cfg = config or ObservatoryConfig()
        super().__init__(
            max_observations=cfg.max_observations,
            store_on_cpu=cfg.store_on_cpu,
        )
        self._config = cfg
        self._storage = storage
        self._baseline_values: Dict[str, np.ndarray] = {}
        self._baseline_norms: Dict[str, float] = {}

    def collect(
        self,
        layer_name: str,
        data: Any,
        step: int,
        epoch: int = 0,
    ) -> None:
        if not self._config.should_collect_parameters(step):
            return

        arr = self._safe_numpy(data, move_to_cpu=self._store_on_cpu)
        if arr is None:
            return

        stats = self._compute_basic_stats(arr)
        norm = stats["norm"]

        baseline_vals = self._baseline_values.get(layer_name)
        baseline_norm = self._baseline_norms.get(layer_name)
        if baseline_vals is None:
            self._baseline_values[layer_name] = arr.astype(np.float32, copy=True)
            self._baseline_norms[layer_name] = norm
            drift = 0.0
            norm_drift = 0.0
        else:
            drift = _true_drift(arr, baseline_vals)
            norm_drift = _norm_drift(norm, baseline_norm)

        stats["drift_from_init"] = drift
        stats["drift_from_init_norm"] = norm_drift
        stats["has_nan"] = int(np.any(np.isnan(arr)))
        stats["has_inf"] = int(np.any(np.isinf(arr)))

        obs = Observation(
            layer_name=layer_name,
            step=step,
            epoch=epoch,
            timestamp=time.time(),
            values=None if self._config.stats_only_mode else arr,
            stats=stats,
            metadata={"shape": list(arr.shape), "dtype": str(arr.dtype)},
        )

        buf = self._get_or_create_buffer(layer_name)
        buf.add(obs)
        self._total += 1

    def collect_model_parameters(
        self,
        model: nn.Module,
        step: int,
        epoch: int = 0,
    ) -> None:
        if not self._config.should_collect_parameters(step):
            return

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.collect(name, param, step, epoch)

    def get_drift(self, layer_name: str) -> Optional[float]:
        obs = self.latest(layer_name, n=1)
        if not obs:
            return None
        return obs[0].stats.get("drift_from_init")

    def get_norm_history(self, layer_name: str) -> List[float]:
        return [o.stats.get("norm", 0.0) for o in self.get(layer_name)]

    def clear(self) -> None:
        super().clear()
        self._baseline_values.clear()
        self._baseline_norms.clear()


def _true_drift(current: np.ndarray, baseline: np.ndarray) -> float:
    base_norm = float(np.linalg.norm(baseline))
    if base_norm < 1e-12:
        return 0.0
    diff_norm = float(np.linalg.norm(current.astype(baseline.dtype) - baseline))
    return diff_norm / base_norm


def _norm_drift(current_norm: float, baseline_norm: float) -> float:
    if baseline_norm < 1e-12:
        return 0.0
    return abs(current_norm - baseline_norm) / (baseline_norm + 1e-10)
