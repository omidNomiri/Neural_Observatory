"""Regression tests for parameter snapshot scheduling (0.5.2)."""
from neural_observatory import Observatory, ObservatoryConfig
from conftest import run_training_steps


def test_repeated_report_does_not_resnapshot_parameters(tiny_model):
    """report() at an already-snapshotted (step, epoch) must not duplicate params."""
    cfg = ObservatoryConfig(parameter_sample_rate=1, max_observations=100)
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=2)

    obs.report()
    name = obs.parameter_collector.layer_names[0]
    first_count = len(obs.parameter_collector.get(name))
    assert first_count == 2, "Expected baseline (step 0) + one report snapshot (step 1)."

    obs.report()
    obs.report()
    assert len(obs.parameter_collector.get(name)) == first_count, (
        "Repeated report() re-snapshotted unchanged parameters."
    )


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
    assert steps == [0, 1, 5], f"Parameter snapshots lost or duplicated steps: {steps}"