"""Correctness tests for the analysis engine."""
import numpy as np
import torch
import torch.nn as nn

from neural_observatory import Observatory, ObservatoryConfig
from neural_observatory.analysis import (
    DeadNeuronAnalyzer,
    GradientHealthAnalyzer,
    ParameterHealthAnalyzer,
)
from neural_observatory.collectors.base import Observation
from conftest import run_training_steps


def _obs(step, values=None, **stats):
    return Observation(layer_name="L", step=step, epoch=0, values=values, stats=stats)


def test_dead_neuron_analyzer_flags_all_zero_layer():
    cfg = ObservatoryConfig()
    analyzer = DeadNeuronAnalyzer(config=cfg)

    zeros = np.zeros((4, 10), dtype=np.float32)
    observations = {
        "activations": {
            "L": [_obs(i, values=zeros, mean=0.0, abs_mean=0.0) for i in range(5)]
        }
    }
    results = analyzer.analyze(observations)
    assert len(results) == 1
    assert results[0].metrics["dead_ratio"] == 1.0
    assert results[0].status.value in ("critical",)


def test_dead_neuron_analyzer_healthy_layer():
    cfg = ObservatoryConfig()
    analyzer = DeadNeuronAnalyzer(config=cfg)

    rng = np.random.default_rng(0)
    active = rng.normal(size=(4, 10)).astype(np.float32)
    observations = {
        "activations": {
            "L": [_obs(i, values=active) for i in range(5)]
        }
    }
    results = analyzer.analyze(observations)
    assert results[0].metrics["dead_ratio"] < 0.2
    assert results[0].status.value == "healthy"


def test_dead_neuron_analyzer_is_scale_invariant():
    """Scale-invariance: a 1000x rescaled layer shouldn't be flagged dead."""
    cfg = ObservatoryConfig(dead_neuron_threshold=1e-6)
    analyzer = DeadNeuronAnalyzer(config=cfg)

    rng = np.random.default_rng(1)
    small_scale = rng.normal(size=(8, 10)).astype(np.float32) * 1e-4
    large_scale = small_scale * 1e7

    obs_small = {"activations": {"L": [_obs(i, values=small_scale) for i in range(5)]}}
    obs_large = {"activations": {"L": [_obs(i, values=large_scale) for i in range(5)]}}

    r_small = analyzer.analyze(obs_small)[0]
    r_large = analyzer.analyze(obs_large)[0]

    assert abs(r_small.metrics["dead_ratio"] - r_large.metrics["dead_ratio"]) < 0.05


def test_gradient_health_flags_nan():
    cfg = ObservatoryConfig()
    analyzer = GradientHealthAnalyzer(config=cfg)
    observations = {
        "gradients": {
            "L": [_obs(i, l2_norm=1.0, has_nan=(i == 2)) for i in range(5)]
        },
        "activations": {},
    }
    results = analyzer.analyze(observations)
    assert results[0].status.value == "critical"
    assert any("NaN" in w for w in results[0].warnings)


def test_gradient_health_flags_collapse():
    cfg = ObservatoryConfig()
    analyzer = GradientHealthAnalyzer(config=cfg)
    norms = [1.0, 1.1, 0.9, 1.0, 0.001, 0.001]
    observations = {
        "gradients": {"L": [_obs(i, l2_norm=n) for i, n in enumerate(norms)]},
        "activations": {},
    }
    results = analyzer.analyze(observations)
    assert any("collapsed" in w for w in results[0].warnings)


def test_parameter_health_true_drift_catches_sign_flip():
    """True drift should catch sign flips even if norm stays the same."""
    cfg = ObservatoryConfig(parameter_drift_threshold=0.1)
    analyzer = ParameterHealthAnalyzer(config=cfg)

    baseline = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    flipped = -baseline

    same_norm = float(np.linalg.norm(baseline))
    true_drift = float(np.linalg.norm(flipped - baseline) / same_norm)

    observations = {
        "parameters": {
            "w": [
                _obs(0, norm=same_norm, drift_from_init=0.0),
                _obs(1, norm=same_norm, drift_from_init=true_drift),
            ]
        }
    }
    results = analyzer.analyze(observations)
    assert results[0].metrics["drift_from_init"] == true_drift
    assert any("drift" in w.lower() for w in results[0].warnings)


def test_end_to_end_report_is_internally_consistent(tiny_model):
    """Smoke test: ensure report summary counts match actual results."""
    obs = Observatory(tiny_model, config=ObservatoryConfig(max_observations=50))
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=5)
    obs.stop()
    report = obs.report()

    summary = report.summary_dict()
    total_by_status = sum(summary["status_counts"].values())
    assert total_by_status == summary["total_results"] == len(report.results)
    assert summary["max_severity"] == (
        round(max((r.severity for r in report.results), default=0.0), 4)
    )