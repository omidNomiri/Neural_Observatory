"""
Neural Observatory -- Attention Health Analyzer
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .base import BaseAnalyzer, AnalysisResult
from ..core.configuration import ObservatoryConfig
from ..collectors.base import Observation

logger = logging.getLogger(__name__)


class AttentionHealthAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "attention_health"

    def analyze(self, observations: Dict[str, Any]) -> List[AnalysisResult]:
        activations: Dict[str, List[Observation]] = observations.get("activations", {})
        results: List[AnalysisResult] = []

        # We only care about layers ending with our special suffix
        for layer_name, obs_list in activations.items():
            if not layer_name.endswith("_attn_weights") or not obs_list:
                continue

            # Remove suffix for the final report so it looks clean
            clean_name = layer_name.replace("_attn_weights", "")
            result = self._analyze_layer(clean_name, obs_list)
            if result:
                results.append(result)

        return results

    def _analyze_layer(self, layer_name: str, obs_list: List[Observation]) -> AnalysisResult:
        cfg = self._config
        entropies = []

        for obs in obs_list:
            if obs.values is None:
                continue
            
            # Attention weights shape is typically (batch, num_heads, seq_len, seq_len)
            # or (batch, seq_len, seq_len). We add a dummy axis if 3D for consistency.
            arr = obs.values
            if arr.ndim == 3:
                arr = arr[:, np.newaxis, :, :]
            
            # To compute entropy, we need probability distributions along the last axis.
            # We clip to avoid log(0).
            eps = 1e-9
            probs = np.clip(arr, eps, 1.0)
            
            # Vectorized Entropy calculation: H = -sum(p * log(p))
            entropy = -np.sum(probs * np.log(probs), axis=-1)
            mean_entropy = float(np.mean(entropy))
            entropies.append(mean_entropy)

        if not entropies:
            return self._make_result(
                analyzer_name=self.name,
                layer_name=layer_name,
                severity=0.0,
                metrics={},
                info=["No raw attention weights available (stats_only_mode might be on)."]
            )

        avg_entropy = float(np.mean(entropies))
        
        metrics: Dict[str, float] = {
            "mean_entropy": round(avg_entropy, 4),
            "min_entropy": round(float(np.min(entropies)), 4),
            "max_entropy": round(float(np.max(entropies)), 4),
        }

        severity = 0.0
        warnings: List[str] = []
        info: List[str] = []

        if avg_entropy < cfg.attention_low_entropy_warning:
            severity = 0.7
            warnings.append(
                f"Low attention entropy ({avg_entropy:.2f}). The model is focusing too "
                "hard on a single token (attention collapse)."
            )
        elif avg_entropy > cfg.attention_high_entropy_warning:
            severity = 0.5
            warnings.append(
                f"High attention entropy ({avg_entropy:.2f}). Attention is too uniform, "
                "the model might not be learning meaningful patterns."
            )
        else:
            info.append(f"Attention is healthy (entropy={avg_entropy:.2f}).")

        return self._make_result(
            analyzer_name=self.name,
            layer_name=layer_name,
            severity=severity,
            metrics=metrics,
            warnings=warnings,
            info=info,
        )