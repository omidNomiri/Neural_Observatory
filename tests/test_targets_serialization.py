"""Captured targets must be JSON-safe and still feed the Neural Collapse analyzer."""
import json

import numpy as np
import torch

from neural_observatory import Observatory, ObservatoryConfig
from neural_observatory.analysis.neural_collapse import NeuralCollapseAnalyzer
from neural_observatory.collectors.base import Observation


def test_report_with_targets_and_sqlite_does_not_crash(tiny_model, tmp_path):
    """ndarray targets in metadata used to crash json.dumps inside store.put()."""
    cfg = ObservatoryConfig(
        storage_backend="sqlite",
        storage_path=str(tmp_path / "obs.db"),
        neural_collapse_layer="0",
        max_observations=50,
    )
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()

    y = torch.tensor([0, 1, 0, 1])
    obs.step(step=0, epoch=0, targets=y)
    _ = tiny_model(torch.randn(4, 8))

    report = obs.report()

    layer = obs.activation_collector.layer_names[0]
    stored = obs.store.get("activations", layer)
    assert len(stored) > 0
    assert stored[0].metadata["targets"] == [0, 1, 0, 1]

    nc_results = [r for r in report.results if r.analyzer_name == "neural_collapse"]
    assert nc_results, "Neural Collapse analyzer produced nothing after targets round-trip."
    assert "nc_variance_ratio" in nc_results[0].metrics


def test_targets_metadata_is_json_serializable(tiny_model):
    cfg = ObservatoryConfig(neural_collapse_layer="0", max_observations=50)
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()

    y = torch.tensor([2, 3])
    obs.step(step=0, epoch=0, targets=y)
    _ = tiny_model(torch.randn(2, 8))

    metadata = obs.activation_collector.get("0")[0].metadata
    json.dumps(metadata)
    assert metadata["targets"] == [2, 3]


def test_targets_snapshot_is_independent_of_user_tensor(tiny_model):
    cfg = ObservatoryConfig(neural_collapse_layer="0", max_observations=50)
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()

    y = torch.tensor([0, 1])
    obs.step(step=0, epoch=0, targets=y)
    _ = tiny_model(torch.randn(2, 8))

    y.add_(100)

    metadata = obs.activation_collector.get("0")[0].metadata
    assert metadata["targets"] == [0, 1], (
        "Captured targets alias the user's live tensor storage."
    )


def test_targets_do_not_leak_into_later_steps(tiny_model):
    cfg = ObservatoryConfig(neural_collapse_layer="0", max_observations=50)
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()

    obs.step(step=0, epoch=0, targets=torch.tensor([0, 1]))
    _ = tiny_model(torch.randn(2, 8))
    obs.step(step=1, epoch=0)
    _ = tiny_model(torch.randn(2, 8))
    obs.stop()

    acts = obs.activation_collector.get("0")
    assert acts[0].metadata["targets"] == [0, 1]
    assert "targets" not in acts[1].metadata, (
        "Labels from step 0 were attached to a step that provided none."
    )


def test_neural_collapse_analyzer_accepts_list_targets():
    """After a SQLite round-trip targets are lists; the analyzer must cope."""
    cfg = ObservatoryConfig(neural_collapse_layer="l")
    analyzer = NeuralCollapseAnalyzer(config=cfg)

    features = np.array(
        [[1.0, 0.0], [1.1, 0.0], [0.0, 1.0], [0.0, 1.1]], dtype=np.float32
    )
    obs = Observation(
        layer_name="l",
        step=0,
        epoch=0,
        values=features,
        metadata={"targets": [0, 0, 1, 1]},
    )

    results = analyzer.analyze({"activations": {"l": [obs]}})
    assert len(results) == 1
    assert results[0].metrics["nc_variance_ratio"] < 1.0