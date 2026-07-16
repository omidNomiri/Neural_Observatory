"""
Neural Observatory -- Neural Collapse Analyzer
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAnalyzer, AnalysisResult
from ..core.configuration import ObservatoryConfig
from ..collectors.base import Observation

logger = logging.getLogger(__name__)


class NeuralCollapseAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "neural_collapse"

    def analyze(self, observations: Dict[str, Any]) -> List[AnalysisResult]:
        cfg = self._config
        if not cfg.neural_collapse_layer:
            return []

        activations: Dict[str, List[Observation]] = observations.get("activations", {})
        layer_name = cfg.neural_collapse_layer
        obs_list = activations.get(layer_name, [])

        if not obs_list:
            return []

        results = []
        for obs in obs_list:
            if "targets" not in obs.metadata:
                continue
            result = self._analyze_observation(obs)
            if result:
                results.append(result)

        return results

    def _analyze_observation(self, obs: Observation) -> Optional[AnalysisResult]:
        cfg = self._config
        if obs.values is None:
            return None

        features = obs.values
        targets = obs.metadata["targets"]

        # Handle batching: features might be (batch, feature_dim) or (batch, seq, dim)
        # We flatten if it's 3D for simplicity in this analysis
        if features.ndim == 3:
            features = features.mean(axis=1)  # Global average pooling over sequence
        
        if features.ndim != 2 or targets.ndim != 1:
            return None

        unique_classes = np.unique(targets)
        if len(unique_classes) < 2:
            return None

        # Calculate Within-Class Variance (WC)
        class_means = []
        within_class_var = 0.0
        total_samples = 0

        for c in unique_classes:
            mask = targets == c
            class_features = features[mask]
            if len(class_features) == 0:
                continue
            
            mean = np.mean(class_features, axis=0)
            class_means.append(mean)
            
            # Variance of features within this class
            var = np.var(class_features, axis=0)
            within_class_var += np.sum(var) * len(class_features)
            total_samples += len(class_features)

        if total_samples == 0:
            return None

        within_class_var /= total_samples

        # Calculate Between-Class Variance (BC)
        global_mean = np.mean(features, axis=0)
        between_class_var = 0.0
        for mean in class_means:
            diff = mean - global_mean
            between_class_var += np.sum(diff ** 2)
        between_class_var /= len(class_means)

        # NC Metric: Lower ratio means higher collapse
        if between_class_var < 1e-10:
            nc_ratio = 1.0
        else:
            nc_ratio = float(within_class_var / (between_class_var + 1e-10))

        metrics = {
            "within_class_variance": float(within_class_var),
            "between_class_variance": float(between_class_var),
            "nc_variance_ratio": round(nc_ratio, 4),
        }

        severity = 0.0
        warnings = []
        info = []

        if nc_ratio < cfg.neural_collapse_variance_threshold:
            severity = 0.3
            info.append(
                f"Neural Collapse detected (Variance Ratio: {nc_ratio:.3f}). "
                "Features are aligning to class means."
            )
        else:
            info.append(f"No Neural Collapse yet (Variance Ratio: {nc_ratio:.3f}).")

        return self._make_result(
            analyzer_name=self.name,
            layer_name=obs.layer_name,
            severity=severity,
            metrics=metrics,
            warnings=warnings,
            info=info,
        )