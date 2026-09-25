"""Regression tests for storage backend wiring and misc bugfixes."""
import inspect

import numpy as np
import torch.nn as nn

from neural_observatory import Observatory, ObservatoryConfig
from neural_observatory.collectors.base import Observation
from neural_observatory.storage.memory_store import MemoryStore
from neural_observatory.storage.sqlite_store import SQLiteStore
from conftest import run_training_steps


def test_memory_backend_is_actually_populated(tiny_model):
    cfg = ObservatoryConfig(storage_backend="memory", max_observations=100)
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=3)
    obs.report()

    assert isinstance(obs.store, MemoryStore)
    assert obs.store.total_observations() > 0, "MemoryStore is empty."


def test_sqlite_backend_is_selected_and_populated(tiny_model, tmp_path):
    db_path = str(tmp_path / "obs.db")
    cfg = ObservatoryConfig(storage_backend="sqlite", storage_path=db_path, max_observations=100)
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=3)
    obs.report()

    assert isinstance(obs.store, SQLiteStore), "Config did not select SQLiteStore."
    layer = obs.activation_collector.layer_names[0]
    stored = obs.store.get("activations", layer)
    assert len(stored) > 0, "SQLiteStore contains no rows."


def test_count_params_has_single_correct_definition(tiny_model):
    """Ensure the dead duplicate staticmethod is gone."""
    obs = Observatory(tiny_model)
    expected = sum(p.numel() for p in tiny_model.parameters())
    assert obs._count_params() == expected

    sig = inspect.signature(Observatory._count_params)
    params = list(sig.parameters)
    assert params == ["self"], (
        f"Expected _count_params(self), got {params}."
    )


def test_hook_manager_does_not_double_attach_shared_module():
    """A module referenced under two names should only get one set of hooks."""
    shared_relu = nn.ReLU()

    class TiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.act_a = shared_relu
            self.act_b = shared_relu
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x):
            x = self.act_a(self.fc1(x))
            x = self.act_b(x)
            return self.fc2(x)

    model = TiedModel()
    obs = Observatory(model, config=ObservatoryConfig(max_observations=50))
    obs.watch()

    attached_names = set(obs._hook_manager.hook_handles.keys())
    assert not {"act_a", "act_b"} <= attached_names, (
        "Both names got hooks attached for a single shared module instance."
    )


def test_repeated_observation_persistence_does_not_duplicate(tmp_path):
    db_path = str(tmp_path / "observatory.db")
    store = SQLiteStore(db_path=db_path)

    observation = Observation(
        layer_name="layer_1",
        step=1,
        epoch=0,
        timestamp=1.0,
        stats={"mean": 1.0},
        metadata={},
        values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )

    store.put(
        "activations",
        "layer_1",
        observation,
    )
    store.commit()

    store.put(
        "activations",
        "layer_1",
        observation,
    )
    store.commit()

    stored = store.get(
        collection="activations",
        layer_name="layer_1",
    )

    assert len(stored) == 1


def test_different_steps_are_stored_separately(tmp_path):
    db_path = str(tmp_path / "observatory.db")
    store = SQLiteStore(db_path=db_path)

    first = Observation(
        layer_name="layer_1",
        step=1,
        epoch=0,
        timestamp=1.0,
        stats={"mean": 1.0},
        metadata={},
        values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )

    second = Observation(
        layer_name="layer_1",
        step=2,
        epoch=0,
        timestamp=2.0,
        stats={"mean": 2.0},
        metadata={},
        values=np.array([4.0, 5.0, 6.0], dtype=np.float32),
    )

    store.put(
        "activations",
        "layer_1",
        first,
    )
    store.put(
        "activations",
        "layer_1",
        second,
    )
    store.commit()

    stored = store.get(
        collection="activations",
        layer_name="layer_1",
    )

    assert len(stored) == 2
    assert stored[0].step == 1
    assert stored[1].step == 2