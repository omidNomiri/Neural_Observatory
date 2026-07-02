"""
Neural Observatory — In-Memory Storage Backend
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional

from ..collectors.base import Observation

logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, max_observations: int = 1000) -> None:
        self._max = max_observations
        self._data: Dict[str, Dict[str, deque]] = {}

    def put(
        self,
        collection: str,
        layer_name: str,
        observation: Observation,
    ) -> None:
        col = self._data.setdefault(collection, {})
        if layer_name not in col:
            col[layer_name] = deque(maxlen=self._max)
        col[layer_name].append(observation)

    def get(
        self,
        collection: str,
        layer_name: str,
    ) -> List[Observation]:
        try:
            return list(self._data[collection][layer_name])
        except KeyError:
            return []

    def get_collection(self, collection: str) -> Dict[str, List[Observation]]:
        if collection not in self._data:
            return {}
        return {
            layer: list(buf)
            for layer, buf in self._data[collection].items()
        }

    def layer_names(self, collection: str) -> List[str]:
        return list(self._data.get(collection, {}).keys())

    def collections(self) -> List[str]:
        return list(self._data.keys())

    def clear(self, collection: Optional[str] = None) -> None:
        if collection:
            self._data.pop(collection, None)
        else:
            self._data.clear()

    def stats(self) -> Dict[str, Any]:
        summary = {}
        for col, layers in self._data.items():
            summary[col] = {
                layer: len(buf)
                for layer, buf in layers.items()
            }
        return summary

    def total_observations(self) -> int:
        return sum(
            len(buf)
            for layers in self._data.values()
            for buf in layers.values()
        )