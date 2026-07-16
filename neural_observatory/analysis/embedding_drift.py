"""
Neural Observatory -- Embedding Drift Analyzer
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from .base import BaseAnalyzer, AnalysisResult
from ..collectors.base import Observation

logger = logging.getLogger(__name__)


class EmbeddingDriftAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "embedding_drift"

    def analyze(self, observations: Dict[str, Any]) -> List[AnalysisResult]:
        parameters: Dict[str, List[Observation]] = observations.get("parameters", {})
        results: List[AnalysisResult] = []

        for param_name, obs_list in parameters.items():
            if not obs_list:
                continue
            
            # Heuristic: Embedding weights are 2D matrices where dim 0 (vocab) > dim 1 (embed_dim)
            # Fallback to values.shape if metadata is missing
            shape = obs_list[0].metadata.get("shape", [])
            if not shape and obs_list[0].values is not None:
                shape = list(obs_list[0].values.shape)
                
            if len(shape) != 2 or shape[0] <= shape[1]:
                continue
                
            result = self._analyze_param(param_name, obs_list)
            if result:
                results.append(result)

        return results

    def _analyze_param(
        self, 
        param_name: str, 
        obs_list: List[Observation]
    ) -> Optional[AnalysisResult]:
        cfg = self._config
        latest = obs_list[-1]
        first = obs_list[0]

        # We need raw values to compute cosine similarity.
        # If stats_only_mode is True, values will be None.
        if latest.values is None or first.values is None:
            # Fallback to L2 drift if raw tensors are not available
            drift = latest.stats.get("drift_from_init", 0.0)
            return self._make_result(
                analyzer_name=self.name,
                layer_name=param_name,
                severity=0.0,
                metrics={"mean_cosine_similarity": -1.0, "l2_drift_fallback": drift},
                info=["Cosine similarity unavailable (stats_only_mode). Showing L2 drift instead."]
            )

        try:
            # Convert back to PyTorch tensors for fast vectorized math
            curr_tensor = torch.from_numpy(latest.values)
            base_tensor = torch.from_numpy(first.values)

            # Calculate cosine similarity for each row (token) 
            # Shape returned will be (vocab_size,)
            cos_sim = torch.nn.functional.cosine_similarity(
                curr_tensor, base_tensor, dim=1
            )
            
            # Average similarity across all tokens
            mean_sim = float(torch.mean(cos_sim))
            min_sim = float(torch.min(cos_sim))
            
        except Exception as e:
            logger.debug("Failed to calculate embedding drift for %s: %s", param_name, e)
            return None

        metrics: Dict[str, float] = {
            "mean_cosine_similarity": round(mean_sim, 4),
            "min_cosine_similarity": round(min_sim, 4),
            "num_observations": len(obs_list),
        }

        severity = 0.0
        warnings: List[str] = []
        info: List[str] = []

        # Lower cosine similarity means higher drift/severity
        if mean_sim < cfg.embedding_drift_critical:
            severity = 0.9
            warnings.append(
                f"Severe embedding drift -- mean cosine similarity dropped to {mean_sim:.2f}. "
                "Token representations have changed drastically from initialization."
            )
        elif mean_sim < cfg.embedding_drift_warning:
            severity = 0.5
            warnings.append(
                f"Moderate embedding drift -- mean cosine similarity is {mean_sim:.2f}."
            )
        else:
            info.append(f"Embeddings stable -- mean cosine similarity {mean_sim:.2f}.")

        return self._make_result(
            analyzer_name=self.name,
            layer_name=param_name,
            severity=severity,
            metrics=metrics,
            warnings=warnings,
            info=info,
        )