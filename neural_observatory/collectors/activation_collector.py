"""
Neural Observatory — Activation Collector
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from .base import BaseCollector, Observation
from ..core.configuration import ObservatoryConfig

logger = logging.getLogger(__name__)


class ActivationCollector(BaseCollector):
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

    def collect(self, layer_name: str, data: Any, step: int, epoch: int = 0) -> None:
        if not self._config.should_collect_activation(step):
            return

        arr = self._safe_numpy(data, move_to_cpu=self._store_on_cpu)
        if arr is None:
            logger.debug("Couldn't convert activation %s to numpy", type(data))
            return

        stats = self._compute_basic_stats(arr)
        stats["sparsity"] = self._compute_sparsity(arr)
        
        # Needed by AnomalyDetectionAnalyzer
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

    def get_sparsity(self, layer_name: str) -> Optional[float]:
        obs = self.get(layer_name)
        if not obs:
            return None
        sparsities = [o.stats.get("sparsity", 0.0) for o in obs]
        return float(np.mean(sparsities))

    def get_mean_activation(self, layer_name: str) -> Optional[float]:
        obs = self.get(layer_name)
        if not obs:
            return None
        return float(np.mean([o.stats.get("abs_mean", 0.0) for o in obs]))

    def _compute_sparsity(self, arr: np.ndarray) -> float:
        thr = self._config.activation_sparsity_threshold
        return float(np.mean(np.abs(arr) < thr))