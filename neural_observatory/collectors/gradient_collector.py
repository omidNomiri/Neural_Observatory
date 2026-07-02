"""
Neural Observatory — Gradient Collector
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseCollector, Observation
from ..core.configuration import ObservatoryConfig

logger = logging.getLogger(__name__)


class GradientCollector(BaseCollector):
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

    def collect(
        self,
        layer_name: str,
        data: Any,
        step: int,
        epoch: int = 0,
    ) -> None:
        if not self._config.should_collect_gradient(step):
            return

        arr = self._safe_numpy(data, move_to_cpu=self._store_on_cpu)
        if arr is None:
            logger.debug("GradientCollector: could not convert %s to numpy", type(data))
            return

        stats = self._compute_basic_stats(arr)
        stats["l2_norm"] = float(np.linalg.norm(arr.ravel()))
        stats["l1_norm"] = float(np.sum(np.abs(arr.ravel())))
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

    def get_norms(self, layer_name: str) -> List[float]:
        return [o.stats.get("l2_norm", 0.0) for o in self.get(layer_name)]

    def get_mean_norm(self, layer_name: str) -> Optional[float]:
        norms = self.get_norms(layer_name)
        return float(np.mean(norms)) if norms else None

    def has_anomalies(self, layer_name: str) -> bool:
        return any(
            o.stats.get("has_nan", 0) or o.stats.get("has_inf", 0)
            for o in self.get(layer_name)
        )
