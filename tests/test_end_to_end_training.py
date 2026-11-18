"""End-to-end training loops through the public API.

These drive real training on every supported execution mode: CPU eager,
inplace ops, inference (no grad), torch.compile, stats-only snapshots,
and CUDA when a device is present.
"""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from conftest import run_training_steps

from neural_observatory import Observatory, ObservatoryConfig
from neural_observatory.storage.sqlite_store import SQLiteStore


def test_full_training_loop_memory_backend(tiny_model, tmp_path):
    cfg = ObservatoryConfig(
        storage_backend="memory", parameter_sample_rate=1, max_observations=100
    )
    with Observatory(tiny_model, config=cfg) as obs:
        run_training_steps(obs, tiny_model, n_steps=5)
        report = obs.report()
        assert report.results, "Analyzers produced no results on a real training loop."

        total_before = obs.store.total_observations()
        obs.report()
        assert obs.store.total_observations() == total_before

        layer = obs.activation_collector.layer_names[0]
        assert sorted({o.step for o in obs.activation_collector.get(layer)}) == [
            0,
            1,
            2,
            3,
            4,
        ]
        assert obs.gradient_collector.total_observations > 0

        param = obs.parameter_collector.layer_names[0]
        assert (
            obs.parameter_collector.get_drift(param) > 0
        ), "SGD updated the weights but drift_from_init stayed zero."

        out = str(tmp_path / "report.json")
        obs.json_report(output=out)
        with open(out, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload, "JSON report is empty."


def test_full_training_loop_sqlite_round_trip(tiny_model, tmp_path):
    db_path = str(tmp_path / "obs.db")
    cfg = ObservatoryConfig(
        storage_backend="sqlite", storage_path=db_path, max_observations=100
    )
    with Observatory(tiny_model, config=cfg) as obs:
        run_training_steps(obs, tiny_model, n_steps=4)
        obs.report()
        layer = obs.activation_collector.layer_names[0]
        live = obs.activation_collector.get(layer)

    reopened = SQLiteStore(db_path=db_path)
    stored = reopened.get("activations", layer)
    assert len(stored) == len(live)
    for original, restored in zip(live, stored):
        assert restored.step == original.step
        assert restored.seq == original.seq
        assert restored.stats == original.stats
        np.testing.assert_array_equal(restored.values, original.values)
    reopened.close()


def test_inplace_model_training_loop(inplace_model):
    cfg = ObservatoryConfig(max_observations=100)
    with Observatory(inplace_model, config=cfg) as obs:
        run_training_steps(obs, inplace_model, n_steps=5)
        report = obs.report()
        assert report.results
        assert obs.activation_collector.total_observations > 0
        assert obs.gradient_collector.total_observations > 0


def test_inference_without_gradients_collects_activations_only(tiny_model):
    cfg = ObservatoryConfig(max_observations=100)
    with Observatory(tiny_model, config=cfg) as obs:
        tiny_model.eval()
        with torch.no_grad():
            for step in range(3):
                obs.step(step=step, epoch=0)
                _ = tiny_model(torch.randn(4, 8))
        report = obs.report()
        assert report.results
        assert obs.activation_collector.total_observations > 0
        assert obs.gradient_collector.total_observations == 0


@pytest.mark.filterwarnings("ignore:Dynamo does not know how to trace:UserWarning")
def test_torch_compile_model_is_monitored(tiny_model):
    compiled = torch.compile(tiny_model, backend="eager")
    cfg = ObservatoryConfig(max_observations=100)
    with Observatory(compiled, config=cfg) as obs:
        assert obs._model is tiny_model, "Compiled wrapper was not unwrapped."
        run_training_steps(obs, compiled, n_steps=3)
        report = obs.report()
        assert report.results
        layer = obs.activation_collector.layer_names[0]
        assert sorted({o.step for o in obs.activation_collector.get(layer)}) == [
            0,
            1,
            2,
        ]


def test_stats_only_mode_round_trip(tiny_model, tmp_path):
    db_path = str(tmp_path / "obs.db")
    cfg = ObservatoryConfig(
        storage_backend="sqlite",
        storage_path=db_path,
        stats_only_mode=True,
        max_observations=100,
    )
    with Observatory(tiny_model, config=cfg) as obs:
        run_training_steps(obs, tiny_model, n_steps=3)
        obs.report()
        layer = obs.activation_collector.layer_names[0]
        assert all(o.values is None for o in obs.activation_collector.get(layer))

    reopened = SQLiteStore(db_path=db_path)
    stored = reopened.get("activations", layer)
    assert len(stored) > 0
    assert all(o.values is None for o in stored)
    assert all("mean" in o.stats for o in stored)
    reopened.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_cuda_training_loop():
    device = torch.device("cuda")
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(device)
    cfg = ObservatoryConfig(max_observations=100)
    with Observatory(model, config=cfg) as obs:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        for step in range(3):
            obs.step(step=step, epoch=0)
            x = torch.randn(4, 8, device=device)
            y = torch.randint(0, 4, (4,), device=device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        report = obs.report()
        assert report.results
        layer = obs.activation_collector.layer_names[0]
        for o in obs.activation_collector.get(layer):
            assert o.values is not None
            assert isinstance(o.values, np.ndarray)
