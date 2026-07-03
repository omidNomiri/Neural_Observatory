"""
Neural Observatory
==================

Research-grade neural network observability, diagnostics, and analysis framework for PyTorch.
"""

from .observatory import Observatory
from .core import (
    ObservatoryConfig,
    Event,
    EventBus,
    EventType,
    LifecycleManager,
    MonitoringState,
    Registry,
)
from .collectors import (
    BaseCollector,
    Observation,
    ActivationCollector,
    GradientCollector,
    ParameterCollector,
    MetadataCollector,
)
from .analysis import (
    BaseAnalyzer,
    AnalysisStatus,
    AnalysisResult,
    DeadNeuronAnalyzer,
    GradientHealthAnalyzer,
    ActivationStatisticsAnalyzer,
    AnomalyDetectionAnalyzer,
    ParameterHealthAnalyzer,
)
from .reporting import (
    BaseReporter,
    Report,
    ConsoleReporter,
    JSONReporter,
    HTMLReporter,
)

__version__ = "0.1.0"
__author__ = "Neural Observatory Contributors"

__all__ = [
    "Observatory",
    "ObservatoryConfig",
    "Event",
    "EventBus",
    "EventType",
    "LifecycleManager",
    "MonitoringState",
    "Registry",
    "BaseCollector",
    "Observation",
    "ActivationCollector",
    "GradientCollector",
    "ParameterCollector",
    "MetadataCollector",
    "BaseAnalyzer",
    "AnalysisStatus",
    "AnalysisResult",
    "DeadNeuronAnalyzer",
    "GradientHealthAnalyzer",
    "ActivationStatisticsAnalyzer",
    "AnomalyDetectionAnalyzer",
    "ParameterHealthAnalyzer",
    "BaseReporter",
    "Report",
    "ConsoleReporter",
    "JSONReporter",
    "HTMLReporter",
]
