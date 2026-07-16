"""
Neural Observatory — Main public API
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch.nn as nn

from .core import (
    EventBus,
    EventType,
    LifecycleError,
    LifecycleManager,
    MonitoringState,
    ObservatoryConfig,
    Registry,
)
from .collectors import (
    ActivationCollector,
    GradientCollector,
    MetadataCollector,
    ParameterCollector,
)
from .analysis import (
    ActivationStatisticsAnalyzer,
    AnomalyDetectionAnalyzer,
    AttentionHealthAnalyzer,
    BaseAnalyzer,
    DeadNeuronAnalyzer,
    EmbeddingDriftAnalyzer,
    GradientHealthAnalyzer,
    ParameterHealthAnalyzer,
)
from .hooks import HookManager
from .reporting import (
    ConsoleReporter,
    HTMLReporter,
    JSONReporter,
    Report,
    BaseReporter,
)
from .storage import MemoryStore
from .storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class Observatory:
    """
    The main orchestrator for neural network monitoring and diagnostics.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ObservatoryConfig] = None,
    ) -> None:
        # Handle torch.compile() models.
        # Compiled models wrap the original nn.Module in an OptimizedModule.
        # We must hook the original module to avoid graph breaks and missed data.
        if hasattr(model, "_orig_mod"):
            logger.info(
                "torch.compile detected: Observatory will monitor the underlying original module. "
                "Note: Forward hooks may cause graph breaks in compiled models."
            )
            self._model = model._orig_mod
        else:
            self._model = model

        self._config = config or ObservatoryConfig()
        self._lifecycle = LifecycleManager()
        self._registry = Registry()
        self._event_bus = EventBus()

        # Collectors
        self._activation_collector = ActivationCollector(config=self._config)
        self._gradient_collector = GradientCollector(config=self._config)
        self._parameter_collector = ParameterCollector(config=self._config)
        self._metadata_collector = MetadataCollector()

        # Hooks
        self._hook_manager = HookManager(
            model=model,
            activation_collector=self._activation_collector,
            gradient_collector=self._gradient_collector,
            config=self._config,
        )

        # Storage — actually respect `config.storage_backend` instead of
        # unconditionally creating (and never using) a MemoryStore. Collectors
        # remain the source of truth for in-flight analysis (they keep their
        # own bounded per-layer buffers), but the configured backend is now a
        # real, functioning persistence layer that report-time snapshots are
        # mirrored into, so `storage_backend="sqlite"` genuinely persists data
        # instead of silently behaving like "memory" regardless of config.
        self._store = self._create_store(self._config)

        # State
        self._step = 0
        self._epoch = 0
        self._last_report: Optional[Report] = None

        # Register default analyzers
        self._register_default_analyzers()

        # Register default reporters
        self._register_default_reporters()

        logger.info(
            "Observatory initialized for model with %d parameters", self._count_params())

    # ==================================================================
    # Public API: Lifecycle
    # ==================================================================

    def watch(self) -> None:
        """
        Attach hooks to the model and begin monitoring.

        Raises:
            LifecycleError: if watch() has already been called.
        """
        try:
            self._lifecycle.start()
        except LifecycleError as e:
            logger.error("Lifecycle error during watch(): %s", e)
            raise

        try:
            self._metadata_collector.collect_model_metadata(self._model)
            self._hook_manager.attach()
            
            # IMPORTANT: Capture initial parameter baseline right after watch()
            # so drift_from_init compares against step 0, not the final step.
            self._parameter_collector.collect_model_parameters(
                self._model, step=0, epoch=0
            )
        except Exception as e:
            logger.error("Initialization failed during watch(): %s", e, exc_info=True)
            self._lifecycle.stop()
            raise RuntimeError(f"Failed to start monitoring: {e}") from e

        logger.info("Observatory watching model — %d layers instrumented", len(
            self._hook_manager.hook_handles))

    def stop(self) -> None:
        """
        Remove all hooks and stop monitoring.

        Safe to call multiple times; idempotent.
        """
        self._hook_manager.detach()
        self._lifecycle.stop()
        logger.info("Observatory stopped")

    def clear(self) -> None:
        """Clear all collected observations while keeping hooks active."""
        self._activation_collector.clear()
        self._gradient_collector.clear()
        self._parameter_collector.clear()
        logger.debug("Observatory cleared all observations")

    # ==================================================================
    # Public API: Training loop integration
    # ==================================================================

    def step(self, step: int = 1, epoch: int = 0) -> None:
        """
        Update the internal step and epoch counters.

        Call this at the end of each training step or batch.

        Args:
            step: Global training step number.
            epoch: Current epoch number.

        Note:
            This also forwards the step/epoch to the HookManager, whose
            forward/backward hook closures read these values to stamp
            each Observation and to evaluate the configured sample rate
            (``should_collect_activation``/``should_collect_gradient``).
            Without this forwarding, hooks would always see step=0, which
            silently disables sample-rate-based skipping (0 % N == 0 is
            always true) and mislabels every stored observation's step.
        """
        self._step = step
        self._epoch = epoch
        self._hook_manager.set_step(step)
        self._hook_manager.set_epoch(epoch)

    # ==================================================================
    # Public API: Analysis and Reporting
    # ==================================================================

    def report(self) -> Report:
        """
        Execute all registered analyzers and return a Report.

        Returns:
            A Report object containing all analysis results.

        Raises:
            LifecycleError: if called before watch().
        """
        self._lifecycle.require_reportable("report")

        if not self._has_data():
            raise LifecycleError("No data to report")

        self._lifecycle.begin_reporting()

        try:
            observations = self._collect_observations()
            self._persist_observations(observations)
            results = []

            for analyzer in self._registry.analyzers:
                try:
                    analyzer_results = analyzer.analyze(observations)
                    results.extend(analyzer_results)
                except Exception as e:
                    logger.error("Analyzer %s raised exception: %s",
                                 analyzer.name, e, exc_info=True)

            report = Report(
                timestamp=time.time(),
                step=self._step,
                epoch=self._epoch,
                results=results,
            )

            self._last_report = report
            logger.info("Report generated: %d results, %s status",
                        len(results), report.overall_status().value)
            return report

        finally:
            self._lifecycle.end_reporting()

    def console_report(self) -> None:
        """Render the latest report to stdout using ANSI formatting."""
        if self._last_report is None:
            report = self.report()
        else:
            report = self._last_report

        reporter = ConsoleReporter()
        reporter.report(report, self._config)

    def json_report(self, output: Optional[str] = None) -> None:
        """
        Render the latest report as JSON.

        Args:
            output: File path to write JSON to, or None for stdout.
        """
        if self._last_report is None:
            report = self.report()
        else:
            report = self._last_report

        reporter = JSONReporter(output=output)
        reporter.report(report, self._config)

    def html_report(self, output: Optional[str] = None) -> None:
        """
        Render the latest report as HTML.

        Args:
            output: File path to write HTML to (defaults to observatory_report.html).
        """
        if self._last_report is None:
            report = self.report()
        else:
            report = self._last_report

        reporter = HTMLReporter(output_path=output)
        reporter.report(report, self._config)

    # ==================================================================
    # Public API: Registry access
    # ==================================================================

    def register_analyzer(self, analyzer: BaseAnalyzer, name: Optional[str] = None) -> None:
        """Register a custom analyzer."""
        self._registry.register_analyzer(analyzer, name)

    def register_reporter(self, reporter: BaseReporter, name: Optional[str] = None) -> None:
        """Register a custom reporter."""
        self._registry.register_reporter(reporter, name)

    def registry_summary(self) -> Dict[str, List[str]]:
        """Get summary of all registered plugins."""
        return self._registry.summary()

    # ==================================================================
    # Public API: Accessors
    # ==================================================================

    @property
    def config(self) -> ObservatoryConfig:
        """Get the configuration object."""
        return self._config

    @property
    def state(self) -> MonitoringState:
        """Get the current monitoring state."""
        return self._lifecycle.state

    @property
    def is_active(self) -> bool:
        """Return True if currently monitoring."""
        return self._lifecycle.is_active

    @property
    def activation_collector(self) -> ActivationCollector:
        """Access the activation collector."""
        return self._activation_collector

    @property
    def gradient_collector(self) -> GradientCollector:
        """Access the gradient collector."""
        return self._gradient_collector

    @property
    def parameter_collector(self) -> ParameterCollector:
        """Access the parameter collector."""
        return self._parameter_collector

    @property
    def metadata(self) -> MetadataCollector:
        """Access model metadata."""
        return self._metadata_collector

    @property
    def store(self):
        """Access the configured storage backend (MemoryStore or SQLiteStore)."""
        return self._store

    @property
    def last_report(self) -> Optional[Report]:
        """Get the last generated report, or None."""
        return self._last_report

    # ==================================================================
    # Internal: Initialization
    # ==================================================================

    def _has_data(self) -> bool:
        return (
            len(self._activation_collector.get_all()) > 0 or
            len(self._gradient_collector.get_all()) > 0 or
            len(self._parameter_collector.get_all()) > 0
        )

    @staticmethod
    def _create_store(config: ObservatoryConfig):
        """Instantiate the storage backend the config actually asks for."""
        if config.storage_backend == "sqlite":
            return SQLiteStore(db_path=config.storage_path)
        return MemoryStore(max_observations=config.max_observations)

    def _register_default_analyzers(self) -> None:
        """Register the built-in analyzers."""
        analyzers = [
            DeadNeuronAnalyzer(config=self._config),
            GradientHealthAnalyzer(config=self._config),
            ActivationStatisticsAnalyzer(config=self._config),
            AnomalyDetectionAnalyzer(config=self._config),
            ParameterHealthAnalyzer(config=self._config),
            EmbeddingDriftAnalyzer(config=self._config),
            AttentionHealthAnalyzer(config=self._config),
        ]
        for analyzer in analyzers:
            self._registry.register_analyzer(analyzer)
        logger.debug("Registered %d default analyzers", len(analyzers))

    def _register_default_reporters(self) -> None:
        """Register the built-in reporters."""
        reporters = [
            ConsoleReporter(),
            JSONReporter(),
            HTMLReporter(),
        ]
        for reporter in reporters:
            self._registry.register_reporter(reporter)
        logger.debug("Registered %d default reporters", len(reporters))

    # ==================================================================
    # Internal: Data collection
    # ==================================================================

    def _collect_observations(self) -> Dict[str, Any]:
        """
        Gather all observations from collectors into a single dict.

        Returns a structure like:
        {
            "activations": {layer_name: [Observation, ...]},
            "gradients": {layer_name: [Observation, ...]},
            "parameters": {param_name: [Observation, ...]},
        }
        """
        self._parameter_collector.collect_model_parameters(
            self._model,
            step=self._step,
            epoch=self._epoch,
        )

        return {
            "activations": self._activation_collector.get_all(),
            "gradients": self._gradient_collector.get_all(),
            "parameters": self._parameter_collector.get_all(),
        }

    def _persist_observations(self, observations: Dict[str, Any]) -> None:
        """Mirror the current observation snapshot into the configured store."""
        for collection, by_layer in observations.items():
            for layer_name, obs_list in by_layer.items():
                for obs in obs_list:
                    self._store.put(collection, layer_name, obs)

    # ==================================================================
    # Internal: Utilities
    # ==================================================================

    def _count_params(self) -> int:
        """Count total parameters in the monitored model."""
        return sum(p.numel() for p in self._model.parameters())
