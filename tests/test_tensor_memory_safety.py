"""Regression tests for BaseCollector._safe_numpy copy behavior."""
import numpy as np
import torch
import torch.nn as nn

from neural_observatory import Observatory, ObservatoryConfig
from neural_observatory.collectors.base import BaseCollector
from conftest import run_training_steps


def test_safe_numpy_returns_independent_copy_for_cpu_float32_tensor():
    """Mutating source tensor in place must not change the returned array."""
    t = torch.ones(4, 4, dtype=torch.float32)
    arr = BaseCollector._safe_numpy(t, move_to_cpu=True)

    assert arr is not None
    snapshot = arr.copy()

    t.add_(100.0)

    np.testing.assert_array_equal(
        arr, snapshot,
        err_msg="_safe_numpy returned an array that aliased the source storage.",
    )


def test_stored_activation_history_survives_inplace_relu(inplace_model):
    """Early activation snapshots must not change after later inplace passes."""
    cfg = ObservatoryConfig(max_observations=100)
    obs = Observatory(inplace_model, config=cfg)
    obs.watch()

    x = torch.randn(4, 8)
    obs.step(step=0, epoch=0)
    _ = inplace_model(x)

    layer = obs.activation_collector.layer_names[0]
    early_obs = obs.activation_collector.get(layer)[0]
    assert early_obs.values is not None
    snapshot = early_obs.values.copy()

    for step in range(1, 6):
        obs.step(step=step, epoch=0)
        _ = inplace_model(torch.randn(4, 8))

    np.testing.assert_array_equal(
        early_obs.values, snapshot,
        err_msg="Historical activation snapshot was corrupted by in-place reuse.",
    )


def test_stored_parameter_history_survives_optimizer_step(tiny_model):
    """Parameter snapshots must not change when optimizer mutates param.data."""
    cfg = ObservatoryConfig(max_observations=100, parameter_sample_rate=1)
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()

    obs.parameter_collector.collect_model_parameters(tiny_model, step=0, epoch=0)
    param_name = obs.parameter_collector.layer_names[0]
    first_obs = obs.parameter_collector.get(param_name)[0]
    assert first_obs.values is not None
    snapshot = first_obs.values.copy()

    run_training_steps(obs, tiny_model, n_steps=5)
    obs.parameter_collector.collect_model_parameters(tiny_model, step=99, epoch=0)

    np.testing.assert_array_equal(
        first_obs.values, snapshot,
        err_msg="Parameter snapshot aliased live storage instead of copying.",
    )