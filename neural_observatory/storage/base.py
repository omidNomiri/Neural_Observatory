"""
Neural Observatory — Base Storage Backend
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..collectors.base import Observation


class BaseStore(ABC):
    """Abstract base class for all storage backends."""

    @abstractmethod
    def put(self, collection: str, layer_name: str, observation: Observation) -> None:
        """Store an observation."""
        pass

    @abstractmethod
    def get(self, collection: str, layer_name: str) -> List[Observation]:
        """Retrieve observations for a specific layer."""
        pass

    @abstractmethod
    def get_collection(self, collection: str) -> Dict[str, List[Observation]]:
        """Retrieve all observations for a specific collection type."""
        pass

    @abstractmethod
    def layer_names(self, collection: str) -> List[str]:
        """Get list of layer names in a collection."""
        pass

    @abstractmethod
    def clear(self, collection: Optional[str] = None) -> None:
        """Clear observations."""
        pass

    @abstractmethod
    def commit(self) -> None:
        """Commit pending transactions to disk."""
        pass

    def close(self) -> None:
        """Close the store connection. Default implementation does nothing."""
        pass