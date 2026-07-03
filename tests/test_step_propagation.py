"""Regression tests for Observatory.step() propagation to HookManager."""
import torch

from neural_observatory import Observatory, ObservatoryConfig
from conftest import run_training_steps


def test_activation_observation_step_tracks_training_step(tiny_model):
    obs = Observatory(tiny_model, config=ObservatoryConfig(max_observations=100))
    obs.watch()

    run_training_steps(obs, tiny_model, n_steps=5)

    layer = obs.activation_collector.layer_names[0]
    observed_steps = sorted({o.step for o in obs.activation_collector.get(layer)})

    assert observed_steps == [0, 1, 2, 3, 4], (
        "Activation observations are not tracking the actual training step."
    )


def test_gradient_observation_step_tracks_training_step(tiny_model):
    obs = Observatory(tiny_model, config=ObservatoryConfig(max_observations=100))
    obs.watch()

    run_training_steps(obs, tiny_model, n_steps=5)

    layer = obs.gradient_collector.layer_names[0]
    observed_steps = sorted({o.step for o in obs.gradient_collector.get(layer)})

    assert observed_steps == [0, 1, 2, 3, 4]


def test_activation_sample_rate_actually_skips_collection(tiny_model):
    """With sample_rate=2, only every-other step should be collected."""
    cfg = ObservatoryConfig(activation_sample_rate=2, max_observations=100)
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()

    run_training_steps(obs, tiny_model, n_steps=6)  # steps 0..5

    layer = obs.activation_collector.layer_names[0]
    observed_steps = sorted({o.step for o in obs.activation_collector.get(layer)})

    assert observed_steps == [0, 2, 4], (
        f"Expected only even steps collected, got {observed_steps}."
    )


def test_anomaly_warning_reports_correct_step(tiny_model):
    """AnomalyDetectionAnalyzer must include the correct step number."""
    obs = Observatory(tiny_model, config=ObservatoryConfig(max_observations=100))
    obs.watch()

    obs.step(step=3, epoch=0)
    with torch.no_grad():
        tiny_model[0].weight[0, 0] = float("nan")
    x = torch.randn(4, 8)
    _ = tiny_model(x)

    report = obs.report()
    anomaly_results = [r for r in report.results if r.analyzer_name == "anomaly_detection"]
    nan_result = next((r for r in anomaly_results if r.metrics.get("act_nan_count", 0) > 0), None)

    assert nan_result is not None
    assert any("steps [3]" in w or "step 3" in w or "[3]" in w for w in nan_result.warnings), (
        f"Expected warning to reference step 3, got: {nan_result.warnings}"
    )