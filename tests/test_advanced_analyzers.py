"""Deep dive tests for advanced analyzers: Embedding Drift, Attention, Neural Collapse."""
import numpy as np
import torch

from neural_observatory import ObservatoryConfig
from neural_observatory.analysis import (
    EmbeddingDriftAnalyzer,
    AttentionHealthAnalyzer,
    NeuralCollapseAnalyzer,
)
from neural_observatory.collectors.base import Observation


def _obs(step, values=None, metadata=None, **stats):
    return Observation(layer_name="L", step=step, epoch=0, values=values, stats=stats, metadata=metadata or {})


# ======================================================================
# 1. Embedding Drift Analyzer
# ======================================================================
def test_embedding_drift_analyzer_detects_flip():
    """Cosine similarity should catch a 180-degree flip (sign reversal)."""
    cfg = ObservatoryConfig(embedding_drift_warning=0.7, embedding_drift_critical=0.4)
    analyzer = EmbeddingDriftAnalyzer(config=cfg)

    # Simulate an embedding matrix (vocab_size=10, embed_dim=5)
    baseline = np.random.randn(10, 5).astype(np.float32)
    flipped = -baseline  # Cosine similarity will be -1.0

    observations = {
        "parameters": {
            "embedding.weight": [
                _obs(0, values=baseline),
                _obs(1, values=flipped)
            ]
        }
    }
    results = analyzer.analyze(observations)
    assert len(results) == 1
    assert results[0].metrics["mean_cosine_similarity"] == -1.0
    assert results[0].status.value == "critical"


# ======================================================================
# 2. Attention Health Analyzer
# ======================================================================
def test_attention_health_analyzer_detects_collapse():
    """Low entropy (all attention on one token) should trigger a warning."""
    cfg = ObservatoryConfig(attention_low_entropy_warning=0.5, attention_high_entropy_warning=3.0)
    analyzer = AttentionHealthAnalyzer(config=cfg)

    # Shape: (batch=2, heads=1, seq=4, seq=4)
    # Collapsed: every token attends ONLY to token 0
    collapsed_weights = np.zeros((2, 1, 4, 4), dtype=np.float32)
    collapsed_weights[:, :, :, 0] = 1.0  # One-hot distribution

    observations = {
        "activations": {
            "attn_attn_weights": [_obs(0, values=collapsed_weights)]
        }
    }
    results = analyzer.analyze(observations)
    assert len(results) == 1
    assert results[0].metrics["mean_entropy"] == 0.0  # Perfectly predictable = 0 entropy
    assert results[0].status.value == "warning"


def test_attention_health_analyzer_detects_uniform():
    """High entropy (attention equally spread everywhere) should trigger a warning."""
    cfg = ObservatoryConfig(attention_low_entropy_warning=0.5, attention_high_entropy_warning=1.0)
    analyzer = AttentionHealthAnalyzer(config=cfg)

    # Shape: (batch=2, heads=1, seq=4, seq=4)
    # Uniform: every token attends equally to all tokens (prob = 0.25)
    uniform_weights = np.ones((2, 1, 4, 4), dtype=np.float32) * 0.25

    observations = {
        "activations": {
            "attn_attn_weights": [_obs(0, values=uniform_weights)]
        }
    }
    results = analyzer.analyze(observations)
    assert len(results) == 1
    # Entropy of uniform distribution over 4 items is ln(4) ~ 1.386
    assert results[0].metrics["mean_entropy"] > 1.0
    assert results[0].status.value == "warning"


# ======================================================================
# 3. Neural Collapse Analyzer
# ======================================================================
def test_neural_collapse_analyzer_detects_collapse():
    """Should flag when within-class variance drops to near zero."""
    cfg = ObservatoryConfig(neural_collapse_layer="penultimate", neural_collapse_variance_threshold=0.1)
    analyzer = NeuralCollapseAnalyzer(config=cfg)

    # 10 samples, 4 features, 2 classes
    targets = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Case 1: Collapsed. All class 0 features are exactly [1,0,0,0], class 1 are [0,1,0,0]
    collapsed_features = np.zeros((10, 4), dtype=np.float32)
    collapsed_features[:5] = np.array([1.0, 0.0, 0.0, 0.0])
    collapsed_features[5:] = np.array([0.0, 1.0, 0.0, 0.0])

    obs_collapsed = _obs(0, values=collapsed_features, metadata={"targets": targets})
    result = analyzer._analyze_observation(obs_collapsed)
    
    assert result.metrics["within_class_variance"] == 0.0
    assert result.metrics["nc_variance_ratio"] == 0.0
    # Status severity 0.4 maps to INFO
    assert result.status.value == "info"


def test_neural_collapse_analyzer_no_collapse():
    """Should NOT flag when features are randomly spread out."""
    cfg = ObservatoryConfig(neural_collapse_layer="penultimate", neural_collapse_variance_threshold=0.1)
    analyzer = NeuralCollapseAnalyzer(config=cfg)

    targets = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Case 2: Random noise. High within-class variance.
    np.random.seed(42)
    random_features = np.random.randn(10, 4).astype(np.float32)

    obs_random = _obs(0, values=random_features, metadata={"targets": targets})
    result = analyzer._analyze_observation(obs_random)
    
    assert result.metrics["nc_variance_ratio"] > 0.1
    assert result.status.value == "healthy"