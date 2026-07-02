"""
Neural Observatory — Plugin Registry
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..analysis.base import BaseAnalyzer
    from ..collectors.base import BaseCollector
    from ..reporting.base import BaseReporter

logger = logging.getLogger(__name__)


class RegistrationError(Exception):
    """Raised when a plugin registration fails."""


class Registry:
    def __init__(self) -> None:
        self._analyzers: Dict[str, "BaseAnalyzer"] = {}
        self._collectors: Dict[str, "BaseCollector"] = {}
        self._reporters: Dict[str, "BaseReporter"] = {}

    def register_analyzer(self, analyzer: "BaseAnalyzer", name: Optional[str] = None) -> None:
        # Importing here to avoid circular dependency hell
        from ..analysis.base import BaseAnalyzer
        
        if not isinstance(analyzer, BaseAnalyzer):
            raise RegistrationError(f"{analyzer!r} does not inherit from BaseAnalyzer.")
        
        key = name or analyzer.name
        if key in self._analyzers:
            logger.warning("Overwriting existing analyzer '%s'.", key)
            
        self._analyzers[key] = analyzer
        logger.debug("Registered analyzer: %s", key)

    def get_analyzer(self, name: str) -> Optional["BaseAnalyzer"]:
        return self._analyzers.get(name)

    @property
    def analyzers(self) -> List["BaseAnalyzer"]:
        return list(self._analyzers.values())

    def register_collector(self, collector: "BaseCollector", name: Optional[str] = None) -> None:
        from ..collectors.base import BaseCollector
        
        if not isinstance(collector, BaseCollector):
            raise RegistrationError(f"{collector!r} does not inherit from BaseCollector.")
        
        key = name or collector.__class__.__name__
        if key in self._collectors:
            logger.warning("Overwriting existing collector '%s'.", key)
            
        self._collectors[key] = collector
        logger.debug("Registered collector: %s", key)

    def get_collector(self, name: str) -> Optional["BaseCollector"]:
        return self._collectors.get(name)

    @property
    def collectors(self) -> List["BaseCollector"]:
        return list(self._collectors.values())

    def register_reporter(self, reporter: "BaseReporter", name: Optional[str] = None) -> None:
        from ..reporting.base import BaseReporter
        
        if not isinstance(reporter, BaseReporter):
            raise RegistrationError(f"{reporter!r} does not inherit from BaseReporter.")
        
        key = name or reporter.__class__.__name__
        if key in self._reporters:
            logger.warning("Overwriting existing reporter '%s'.", key)
            
        self._reporters[key] = reporter
        logger.debug("Registered reporter: %s", key)

    def get_reporter(self, name: str) -> Optional["BaseReporter"]:
        return self._reporters.get(name)

    @property
    def reporters(self) -> List["BaseReporter"]:
        return list(self._reporters.values())

    def summary(self) -> Dict[str, List[str]]:
        return {
            "analyzers": list(self._analyzers.keys()),
            "collectors": list(self._collectors.keys()),
            "reporters": list(self._reporters.keys()),
        }

    def clear(self) -> None:
        self._analyzers.clear()
        self._collectors.clear()
        self._reporters.clear()