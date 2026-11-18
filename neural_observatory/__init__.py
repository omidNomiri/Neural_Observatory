"""
Neural Observatory
==================

neural network observability, diagnostics, and analysis framework for PyTorch.
"""

from .analysis import (
    ActivationStatisticsAnalyzer,
    AnalysisResult,
    AnalysisStatus,
    AnomalyDetectionAnalyzer,
    BaseAnalyzer,
    DeadNeuronAnalyzer,
    EmbeddingDriftAnalyzer,
    GradientHealthAnalyzer,
    ParameterHealthAnalyzer,
)
from .collectors import (
    ActivationCollector,
    BaseCollector,
    GradientCollector,
    MetadataCollector,
    Observation,
    ParameterCollector,
)
from .core import (
    Event,
    EventBus,
    EventType,
    LifecycleManager,
    MonitoringState,
    ObservatoryConfig,
    Registry,
)
from .observatory import Observatory
from .reporting import (
    BaseReporter,
    ConsoleReporter,
    HTMLReporter,
    JSONReporter,
    Report,
)

__version__ = "0.5.2"
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
    "EmbeddingDriftAnalyzer",
    "ActivationStatisticsAnalyzer",
    "AnomalyDetectionAnalyzer",
    "ParameterHealthAnalyzer",
    "BaseReporter",
    "Report",
    "ConsoleReporter",
    "JSONReporter",
    "HTMLReporter",
]
