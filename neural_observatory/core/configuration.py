"""
Neural Observatory — Configuration
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ObservatoryConfig:
    """
    Central config for the framework. 
    """

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    # Sample activations every N forward passes.
    # 1 means collect on every forward pass.
    activation_sample_rate: int = 1
    
    # Backward-pass sampling interval.
    gradient_sample_rate: int = 1
    
    # How often to snapshot model weights.
    parameter_sample_rate: int = 10

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    # Limit memory usage per layer.
    max_observations: int = 1000
    
    # Store observations on CPU instead of keeping GPU memory occupied.
    store_on_cpu: bool = True
    
    # Skip storing raw tensors and keep only summary statistics.
    stats_only_mode: bool = False

    # ------------------------------------------------------------------
    # Layer filtering
    # ------------------------------------------------------------------
    # Layers to include. Empty means "all".
    include_layer_types: List[str] = field(default_factory=list)
    
    # Ignore these layer types by default to save space.
    exclude_layer_types: List[str] = field(default_factory=lambda: [
        "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
        "LayerNorm", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d",
        "Dropout", "Dropout2d", "Dropout3d", "AlphaDropout",
        "Identity", "Sequential", "ModuleList", "ModuleDict",
    ])
    
    # Only monitor layers with these names.
    include_layers: List[str] = field(default_factory=list)
    
    # Skip specific layers by name.
    exclude_layers: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Dead-neuron detection
    # ------------------------------------------------------------------
    # Values smaller than this are considered inactive.
    dead_neuron_threshold: float = 1e-6
    
    # How often a neuron must appear dead before it's reported.
    dead_neuron_persistence: float = 0.95
    
    # Warn if this fraction of a layer is dead.
    dead_neuron_warning_ratio: float = 0.20
    
    # Critical alert threshold for dead neurons.
    dead_neuron_critical_ratio: float = 0.50

    # ------------------------------------------------------------------
    # Gradient health
    # ------------------------------------------------------------------
    # Gradients smaller than this mean learning has probably stalled.
    gradient_vanishing_threshold: float = 1e-7
    
    # Gradients larger than this will likely destabilize training.
    gradient_exploding_threshold: float = 1e3
    
    # Fraction of layers with issues before a global warning.
    gradient_warning_ratio: float = 0.3

    # ------------------------------------------------------------------
    # Activation statistics
    # ------------------------------------------------------------------
    # Cutoff for considering an activation as zero.
    activation_sparsity_threshold: float = 1e-3
    
    # Warn if a layer is mostly inactive.
    high_sparsity_warning: float = 0.8

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------
    # Detect unusually large activation spikes.
    spike_std_multiplier: float = 5.0

    # ------------------------------------------------------------------
    # Parameter health
    # ------------------------------------------------------------------
    # Large changes usually indicate unstable optimization.
    parameter_drift_threshold: float = 0.5
    
    # High ratios mean the optimizer is taking steps that are too large.
    update_to_param_ratio_warning: float = 0.1

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    # Where to keep data: 'memory' or 'sqlite'.
    storage_backend: str = "memory"
    
    # Path for the sqlite db. If None, uses a temp file.
    storage_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    # Filter out low-severity results. 0.0 shows everything.
    report_min_severity: float = 0.0
    
    # Truncate long reports to keep them readable.
    report_max_layers: int = 100
    
    # Include healthy layers in the final report.
    report_include_healthy: bool = True
    
    # Add extra details like histograms to reports.
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate config right after initialization."""
        _validate_positive("activation_sample_rate", self.activation_sample_rate)
        _validate_positive("gradient_sample_rate", self.gradient_sample_rate)
        _validate_positive("parameter_sample_rate", self.parameter_sample_rate)
        _validate_positive("max_observations", self.max_observations)
        _validate_non_negative("dead_neuron_threshold", self.dead_neuron_threshold)
        _validate_range("dead_neuron_persistence", self.dead_neuron_persistence, 0.0, 1.0)
        _validate_range("dead_neuron_warning_ratio", self.dead_neuron_warning_ratio, 0.0, 1.0)
        _validate_range("dead_neuron_critical_ratio", self.dead_neuron_critical_ratio, 0.0, 1.0)
        
        if self.dead_neuron_critical_ratio <= self.dead_neuron_warning_ratio:
            raise ValueError(
                f"dead_neuron_critical_ratio ({self.dead_neuron_critical_ratio}) "
                f"must be greater than dead_neuron_warning_ratio ({self.dead_neuron_warning_ratio})"
            )
            
        _validate_non_negative("gradient_vanishing_threshold", self.gradient_vanishing_threshold)
        _validate_non_negative("gradient_exploding_threshold", self.gradient_exploding_threshold)
        
        if self.gradient_exploding_threshold <= self.gradient_vanishing_threshold:
            raise ValueError(
                f"gradient_exploding_threshold ({self.gradient_exploding_threshold}) "
                f"must be greater than gradient_vanishing_threshold ({self.gradient_vanishing_threshold})"
            )
            
        _validate_range("gradient_warning_ratio", self.gradient_warning_ratio, 0.0, 1.0)
        _validate_non_negative("activation_sparsity_threshold", self.activation_sparsity_threshold)
        _validate_range("high_sparsity_warning", self.high_sparsity_warning, 0.0, 1.0)
        _validate_positive("spike_std_multiplier", self.spike_std_multiplier)
        _validate_non_negative("parameter_drift_threshold", self.parameter_drift_threshold)
        _validate_non_negative("update_to_param_ratio_warning", self.update_to_param_ratio_warning)
        _validate_range("report_min_severity", self.report_min_severity, 0.0, 1.0)
        _validate_positive("report_max_layers", self.report_max_layers)
        
        if self.storage_backend not in ("memory", "sqlite"):
            raise ValueError(
                f"storage_backend must be 'memory' or 'sqlite', got '{self.storage_backend}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObservatoryConfig":
        return cls(**data)

    def save_json(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str) -> "ObservatoryConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def should_monitor_layer(self, layer_name: str, layer_type: str) -> bool:
        """Check if we should hook into this layer."""
        if layer_name in self.exclude_layers:
            return False
        if layer_type in self.exclude_layer_types:
            return False
        if self.include_layers and layer_name not in self.include_layers:
            return False
        if self.include_layer_types and layer_type not in self.include_layer_types:
            return False
        return True

    def should_collect_activation(self, step: int) -> bool:
        return self.activation_sample_rate <= 1 or step % self.activation_sample_rate == 0

    def should_collect_gradient(self, step: int) -> bool:
        return self.gradient_sample_rate <= 1 or step % self.gradient_sample_rate == 0

    def should_collect_parameters(self, step: int) -> bool:
        return self.parameter_sample_rate <= 1 or step % self.parameter_sample_rate == 0


# Simple validation helpers
def _validate_positive(name: str, value: float) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")

def _validate_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")

def _validate_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")