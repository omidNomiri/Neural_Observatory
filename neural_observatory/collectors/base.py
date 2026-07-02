"""
Neural Observatory — Base Collector
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np


@dataclass
class Observation:
    """An atomic, memory-safe snapshot captured by a hook."""
    layer_name: str
    step: int
    epoch: int
    timestamp: float = field(default_factory=time.time)
    values: Optional[np.ndarray] = None
    stats: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerBuffer:
    """Fixed-size circular buffer for a single layer."""
    layer_name: str
    capacity: int
    _buffer: Deque[Observation] = field(init=False)

    def __post_init__(self) -> None:
        self._buffer = deque(maxlen=self.capacity)

    def add(self, obs: Observation) -> None:
        self._buffer.append(obs)

    def all(self) -> List[Observation]:
        return list(self._buffer)

    def latest(self, n: int = 1) -> List[Observation]:
        items = list(self._buffer)
        return items[-n:] if n < len(items) else items

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


class BaseCollector(ABC):
    """Base class for grabbing data off hooks and making it memory-safe."""

    def __init__(self, max_observations: int = 1000, store_on_cpu: bool = True) -> None:
        self._max_observations = max_observations
        self._store_on_cpu = store_on_cpu
        self._buffers: Dict[str, LayerBuffer] = {}
        self._total = 0

    @abstractmethod
    def collect(self, layer_name: str, data: Any, step: int, epoch: int = 0) -> None:
        pass

    def get(self, layer_name: str) -> List[Observation]:
        buf = self._buffers.get(layer_name)
        return buf.all() if buf else []

    def get_all(self) -> Dict[str, List[Observation]]:
        return {name: buf.all() for name, buf in self._buffers.items()}

    def latest(self, layer_name: str, n: int = 1) -> List[Observation]:
        buf = self._buffers.get(layer_name)
        return buf.latest(n) if buf else []

    @property
    def layer_names(self) -> List[str]:
        return list(self._buffers.keys())

    @property
    def total_observations(self) -> int:
        return self._total

    def clear(self) -> None:
        for buf in self._buffers.values():
            buf.clear()
        self._total = 0

    def _get_or_create_buffer(self, layer_name: str) -> LayerBuffer:
        if layer_name not in self._buffers:
            self._buffers[layer_name] = LayerBuffer(
                layer_name=layer_name,
                capacity=self._max_observations,
            )
        return self._buffers[layer_name]

    @staticmethod
    def _safe_numpy(tensor: Any, *, move_to_cpu: bool = True) -> Optional[np.ndarray]:
        """
        Convert tensor to numpy safely.
        
        detach() and cpu() can be no-ops. We MUST clone() because inplace ops 
        (like ReLU(inplace=True) or zero_grad()) will silently corrupt our 
        stored snapshots otherwise.
        """
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                detached = tensor.detach()
                if move_to_cpu and detached.device.type != "cpu":
                    detached = detached.cpu()
                # Force a copy to break aliasing with the model's live storage
                arr = detached.float().clone().numpy()
                return np.array(arr, copy=True)
        except Exception:
            pass
        
        if isinstance(tensor, np.ndarray):
            return tensor.copy()
        return None

    @staticmethod
    def _compute_basic_stats(arr: np.ndarray) -> Dict[str, float]:
        flat = arr.ravel().astype(np.float32)
        if flat.size == 0:
            return {}
        return {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "abs_mean": float(np.mean(np.abs(flat))),
            "norm": float(np.linalg.norm(flat)),
            "numel": int(flat.size),
        }