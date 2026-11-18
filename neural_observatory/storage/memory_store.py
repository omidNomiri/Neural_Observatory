"""
Neural Observatory — In-Memory Storage Backend
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional

from .base import BaseStore
from ..collectors.base import Observation

logger = logging.getLogger(__name__)


class MemoryStore(BaseStore):
    def __init__(self, max_observations: int = 1000) -> None:
        self._max = max_observations
        self._data: Dict[str, Dict[str, deque]] = {}
        self._keys: Dict[str, Dict[str, set]] = {}

    def put(
        self,
        collection: str,
        layer_name: str,
        observation: Observation,
    ) -> None:
        col = self._data.setdefault(collection, {})
        keys_col = self._keys.setdefault(collection, {})
        if layer_name not in col:
            col[layer_name] = deque(maxlen=self._max)
            keys_col[layer_name] = set()
        buf = col[layer_name]
        keys = keys_col[layer_name]

        key = (observation.step, observation.epoch, observation.seq)
        if key in keys:
            return
        # The deque silently evicts the oldest item once full; drop its key
        # too so the key set stays bounded and evicted data can return.
        if buf.maxlen and len(buf) == buf.maxlen:
            oldest = buf[0]
            keys.discard((oldest.step, oldest.epoch, oldest.seq))
        buf.append(observation)
        keys.add(key)

    def commit(self) -> None:
        """Memory store doesn't need to commit to disk, but provided for interface consistency."""
        pass

    def get(
        self,
        collection: str,
        layer_name: str,
        limit: Optional[int] = None,
    ) -> List[Observation]:
        try:
            items = list(self._data[collection][layer_name])
        except KeyError:
            return []
        return items[:limit] if limit is not None else items

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
            self._keys.pop(collection, None)
        else:
            self._data.clear()
            self._keys.clear()

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