"""Regression tests for parameter snapshot scheduling and drift precision."""

import numpy as np
import pytest
import torch
from conftest import run_training_steps

from neural_observatory import Observatory, ObservatoryConfig
from neural_observatory.collectors.parameter_collector import (
    ParameterCollector,
)


def test_repeated_report_does_not_resnapshot_parameters(tiny_model):
    """report() must not duplicate params at the same (step, epoch)."""
    cfg = ObservatoryConfig(parameter_sample_rate=1, max_observations=100)
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=2)

    obs.report()
    name = obs.parameter_collector.layer_names[0]
    first_count = len(obs.parameter_collector.get(name))
    assert (
        first_count == 2
    ), "Expected baseline (step 0) + one report snapshot (step 1)."

    obs.report()
    obs.report()
    assert (
        len(obs.parameter_collector.get(name)) == first_count
    ), "Repeated report() re-snapshotted unchanged parameters."


def test_new_step_after_report_is_snapshotted(tiny_model):
    cfg = ObservatoryConfig(parameter_sample_rate=1, max_observations=100)
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=2)
    obs.report()

    run_training_steps(obs, tiny_model, n_steps=1)
    obs.step(step=5, epoch=0)
    obs.report()

    name = obs.parameter_collector.layer_names[0]
    steps = [o.step for o in obs.parameter_collector.get(name)]
    assert steps == [
        0,
        1,
        5,
    ], f"Parameter snapshots lost or duplicated steps: {steps}"


def test_float64_baseline_keeps_tiny_updates():
    """A 1e-10 update on a float64 weight is invisible once cast to float32."""
    pc = ParameterCollector(config=ObservatoryConfig(parameter_sample_rate=1))
    weight = torch.ones(1, dtype=torch.float64)

    pc.collect("w", weight, step=0)
    weight += 1e-10
    pc.collect("w", weight, step=1)

    drift = pc.get_drift("w")
    assert (
        drift is not None and drift > 0
    ), "Baseline downcast erased a real float64 parameter update."


@pytest.mark.parametrize(
    "initial, current, expected",
    [(300.0, 330.0, 0.1), (300.0, -300.0, 2.0), (60000.0, -60000.0, 2.0)],
)
def test_float16_drift_uses_widened_arithmetic(initial, current, expected):
    pc = ParameterCollector(config=ObservatoryConfig(parameter_sample_rate=1))
    weight = torch.full((4,), initial, dtype=torch.float16)

    pc.collect("w", weight, step=0)
    weight.fill_(current)
    pc.collect("w", weight, step=1)

    snapshots = pc.get("w")
    assert all(snapshot.values.dtype == np.float16 for snapshot in snapshots)
    np.testing.assert_array_equal(snapshots[0].values, [initial] * 4)
    np.testing.assert_array_equal(snapshots[1].values, [current] * 4)
    assert pc._baseline_values["w"].dtype == np.float16
    np.testing.assert_array_equal(pc._baseline_values["w"], [initial] * 4)
    assert pc.get_drift("w") == pytest.approx(expected)


def test_clear_keeps_the_init_baseline():
    """drift_from_init stays measured against initialization after clear()."""
    pc = ParameterCollector(config=ObservatoryConfig(parameter_sample_rate=1))
    weight = torch.ones(4, dtype=torch.float32)

    pc.collect("w", weight, step=0)
    weight += 0.5
    pc.collect("w", weight, step=1)
    drift_before = pc.get_drift("w")

    pc.clear()
    pc.collect("w", weight, step=2)

    assert (
        len(pc.get("w")) == 1
    ), "clear() did not empty the observation buffer."
    assert pc.get_drift("w") == drift_before, (
        "clear() re-baselined; drift_from_init must stay relative "
        "to initialization."
    )
