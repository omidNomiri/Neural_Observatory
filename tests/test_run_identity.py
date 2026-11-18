"""Regression tests for run identity across shared storage."""

import sqlite3

import numpy as np
import torch
import torch.nn as nn
from conftest import run_training_steps

from neural_observatory import Observatory, ObservatoryConfig
from neural_observatory.collectors.base import Observation
from neural_observatory.storage.memory_store import MemoryStore
from neural_observatory.storage.sqlite_store import _SCHEMA, SQLiteStore


def _make_obs(step=0, epoch=0, seq=0, run_id="", value=1.0, layer="layer_1"):
    return Observation(
        layer_name=layer,
        step=step,
        epoch=epoch,
        seq=seq,
        run_id=run_id,
        timestamp=float(step),
        stats={"mean": value},
        metadata={},
        values=np.array([value], dtype=np.float32),
    )


def _fresh_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_observations_carry_the_session_run_id():
    obs = Observatory(_fresh_model(), config=ObservatoryConfig(max_observations=100))
    assert obs.run_id, "Observatory must expose a non-empty run id."
    obs.watch()
    run_training_steps(obs, obs._model, n_steps=2)

    for layer in obs.activation_collector.layer_names:
        for o in obs.activation_collector.get(layer):
            assert o.run_id == obs.run_id
    obs.close()


def test_two_runs_sharing_one_sqlite_database_are_both_kept(tmp_path):
    db_path = str(tmp_path / "shared.db")
    cfg = ObservatoryConfig(
        storage_backend="sqlite", storage_path=db_path, max_observations=100
    )

    model_a = _fresh_model()
    obs_a = Observatory(model_a, config=cfg)
    obs_a.watch()
    run_training_steps(obs_a, model_a, n_steps=2)
    obs_a.report()
    obs_a.report()
    obs_a.close()

    model_b = _fresh_model()
    obs_b = Observatory(model_b, config=cfg)
    obs_b.watch()
    run_training_steps(obs_b, model_b, n_steps=2)
    obs_b.report()
    obs_b.report()

    assert obs_a.run_id != obs_b.run_id

    stored = obs_b.store.get("activations", "0")
    assert (
        len(stored) == 4
    ), f"Second run was dropped as a duplicate of the first: {len(stored)} rows."
    by_run = {}
    for o in stored:
        by_run.setdefault(o.run_id, []).append(o.step)
    assert set(by_run) == {obs_a.run_id, obs_b.run_id}
    assert sorted(by_run[obs_a.run_id]) == [0, 1]
    assert sorted(by_run[obs_b.run_id]) == [0, 1]
    obs_b.close()


def test_memory_store_dedup_is_per_run():
    store = MemoryStore(max_observations=10)
    store.put("activations", "layer_1", _make_obs(run_id="run-a"))
    store.put("activations", "layer_1", _make_obs(run_id="run-b"))
    assert (
        len(store.get("activations", "layer_1")) == 2
    ), "Same step/seq from a different run must not be treated as a duplicate."

    store.put("activations", "layer_1", _make_obs(run_id="run-a"))
    assert len(store.get("activations", "layer_1")) == 2


_V1_SCHEMA = """
CREATE TABLE observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    collection  TEXT    NOT NULL,
    layer_name  TEXT    NOT NULL,
    step        INTEGER NOT NULL,
    epoch       INTEGER NOT NULL,
    seq         INTEGER NOT NULL DEFAULT 0,
    timestamp   REAL    NOT NULL,
    stats       TEXT    NOT NULL,
    metadata    TEXT    NOT NULL,
    values_blob BLOB,
    UNIQUE (collection, layer_name, step, epoch, seq)
);
PRAGMA user_version = 1;
"""


def test_v1_schema_gains_run_identity(tmp_path):
    db_path = str(tmp_path / "v1.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(_V1_SCHEMA)
    conn.execute(
        "INSERT INTO observations (collection, layer_name, step, epoch, seq, timestamp, stats, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("activations", "layer_1", 0, 0, 0, 0.0, '{"mean": 1.0}', "{}"),
    )
    conn.commit()
    conn.close()

    store = SQLiteStore(db_path=db_path)
    version = sqlite3.connect(db_path).execute("PRAGMA user_version").fetchone()[0]
    assert version >= 2, "v1 database was not migrated to the run-identity schema."

    legacy = store.get("activations", "layer_1")
    assert len(legacy) == 1
    assert legacy[0].run_id == "", "Legacy rows should keep an empty run marker."

    store.put(
        "activations", "layer_1", _make_obs(step=0, seq=0, run_id="new-run", value=9.0)
    )
    store.commit()
    assert (
        len(store.get("activations", "layer_1")) == 2
    ), "A new run must coexist with legacy rows at the same step/seq."
    store.close()


def test_remigration_keeps_existing_run_ids(tmp_path):
    """A run-identity database stamped with an old version must survive re-migration."""
    db_path = str(tmp_path / "restamped.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    for run in ("run-a", "run-b"):
        conn.execute(
            "INSERT INTO observations (collection, layer_name, step, epoch, seq, run_id, timestamp, stats, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("activations", "layer_1", 0, 0, 0, run, 0.0, '{"mean": 1.0}', "{}"),
        )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    store = SQLiteStore(db_path=db_path)
    stored = store.get("activations", "layer_1")
    assert len(stored) == 2, "Re-migration collapsed rows that differ only by run_id."
    assert {o.run_id for o in stored} == {"run-a", "run-b"}
    store.close()
