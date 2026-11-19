"""
Neural Observatory — Main public API
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .analysis import (
    ActivationStatisticsAnalyzer,
    AnomalyDetectionAnalyzer,
    AttentionHealthAnalyzer,
    BaseAnalyzer,
    DeadNeuronAnalyzer,
    EmbeddingDriftAnalyzer,
    GradientHealthAnalyzer,
    NeuralCollapseAnalyzer,
    ParameterHealthAnalyzer,
)
from .collectors import (
    ActivationCollector,
    GradientCollector,
    MetadataCollector,
    ParameterCollector,
)
from .core import (
    EventBus,
    LifecycleError,
    LifecycleManager,
    MonitoringState,
    ObservatoryConfig,
    Registry,
)
from .hooks import HookManager
from .hooks._compiler import unwrap_model
from .reporting import (
    BaseReporter,
    ConsoleReporter,
    HTMLReporter,
    JSONReporter,
    Report,
)
from .storage import MemoryStore
from .storage.base import BaseStore
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
        self._model = unwrap_model(model)

        self._config = config or ObservatoryConfig()
        self._lifecycle = LifecycleManager()
        self._registry = Registry()
        self._event_bus = EventBus()

        # One id per monitoring session, shared by all collectors so their
        # observations stay distinguishable when runs share a storage backend.
        self._run_id = uuid.uuid4().hex

        # Collectors
        self._activation_collector = ActivationCollector(
            config=self._config, run_id=self._run_id
        )
        self._gradient_collector = GradientCollector(
            config=self._config, run_id=self._run_id
        )
        self._parameter_collector = ParameterCollector(
            config=self._config, run_id=self._run_id
        )
        self._metadata_collector = MetadataCollector()

        # Hooks
        self._hook_manager = HookManager(
            model=self._model,
            activation_collector=self._activation_collector,
            gradient_collector=self._gradient_collector,
            config=self._config,
        )

        # Collectors own the in-flight buffers; the store only receives
        # report-time snapshots.
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
            "Observatory initialized for model with %d parameters",
            self._count_params(),
        )

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
            logger.error(
                "Initialization failed during watch(): %s", e, exc_info=True
            )
            self.stop()
            raise RuntimeError(f"Failed to start monitoring: {e}") from e

        logger.info(
            "Observatory watching model — %d layers instrumented",
            len(self._hook_manager.hook_handles),
        )

    def stop(self) -> None:
        """
        Remove all hooks and stop monitoring.

        Safe to call multiple times; idempotent. The store stays open so
        report() can still run after stop(); use close() to release it.
        """
        try:
            self._hook_manager.detach()
        finally:
            self._lifecycle.stop()
        logger.info("Observatory stopped")

    def close(self) -> None:
        """Stop monitoring and release the storage backend."""
        try:
            self.stop()
        finally:
            self._store.close()

    def clear(self) -> None:
        """Empty the in-memory collector buffers; hooks stay attached.

        Persisted history in the storage backend is intentionally kept.
        Sequence numbers keep increasing, so data collected after a clear()
        never collides with what was already stored.
        """
        self._activation_collector.clear()
        self._gradient_collector.clear()
        self._parameter_collector.clear()
        logger.debug("Observatory cleared all observations")

    # ==================================================================
    # Public API: Training loop integration
    # ==================================================================

    def step(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update the internal step and epoch counters.

        Call this *before* the forward pass of each training step, so the
        hooks stamp observations with the step that produced them.

        Args:
            step: Global training step number.
            epoch: Current epoch number.
            targets: Optional label tensor, forwarded to the HookManager for
                the configured Neural Collapse layer.

        Note:
            This also forwards the step/epoch to the HookManager, whose
            forward/backward hook closures read these values to stamp
            each Observation and to evaluate the configured sample rate
            (``should_collect_activation``/``should_collect_gradient``).
            If `step` is not provided, it auto-increments the internal counter.

        Gradient timing contract:
            Activations are stamped at forward time; gradients are stamped
            with the step active at backward time. Finish each step's
            backward pass before calling ``step()`` again. With gradient
            accumulation, call ``step()`` before every micro-batch forward
            and run that micro-batch's backward before moving on. Delaying
            backward across ``step()`` calls attributes the gradient to the
            later step; that pattern is not supported.
        """

        if not self._lifecycle.is_active:
            logger.warning(
                "Observatory.step() called while not active. "
                "Call obs.watch() first."
            )
            return

        if step is not None:
            self._step = step
        else:
            self._step += 1

        if epoch is not None:
            self._epoch = epoch

        self._hook_manager.set_step(self._step)
        self._hook_manager.set_epoch(self._epoch)
        # Always forward targets, including None: labels belong to one step
        # only and must not leak into the next forward.
        self._hook_manager.set_targets(targets)

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

        # UX Guard: Warn if step was never updated
        if self._step == 0 and (
            self._config.activation_sample_rate > 1
            or self._config.gradient_sample_rate > 1
        ):
            logger.warning(
                "Observatory.step() was never called (step=0). "
                "Sampling rates will not work properly. "
                "Please call obs.step() in your training loop."
            )

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
                    logger.error(
                        "Analyzer %s raised exception: %s",
                        analyzer.name,
                        e,
                        exc_info=True,
                    )

            report = Report(
                timestamp=time.time(),
                step=self._step,
                epoch=self._epoch,
                results=results,
            )

            self._last_report = report
            logger.info(
                "Report generated: %d results, %s status",
                len(results),
                report.overall_status().value,
            )
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
            output: HTML file path (defaults to observatory_report.html).
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

    def register_analyzer(
        self, analyzer: BaseAnalyzer, name: Optional[str] = None
    ) -> None:
        """Register a custom analyzer."""
        self._registry.register_analyzer(analyzer, name)

    def register_reporter(
        self, reporter: BaseReporter, name: Optional[str] = None
    ) -> None:
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
    def run_id(self) -> str:
        """Unique identifier of this monitoring session."""
        return self._run_id

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
        """Access the configured MemoryStore or SQLiteStore backend."""
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
            len(self._activation_collector.get_all()) > 0
            or len(self._gradient_collector.get_all()) > 0
            or len(self._parameter_collector.get_all()) > 0
        )

    @staticmethod
    def _create_store(config: ObservatoryConfig) -> BaseStore:
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
            NeuralCollapseAnalyzer(config=self._config),
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

    def __enter__(self) -> "Observatory":
        try:
            self.watch()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

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
        """Mirror the current snapshot into the configured store."""
        for collection, by_layer in observations.items():
            for layer_name, obs_list in by_layer.items():
                for obs in obs_list:
                    self._store.put(collection, layer_name, obs)

        # One batch commit per report keeps disk I/O bounded.
        self._store.commit()

    # ==================================================================
    # Internal: Utilities
    # ==================================================================

    def _count_params(self) -> int:
        """Count total parameters in the monitored model."""
        return sum(p.numel() for p in self._model.parameters())
