"""Observation identity, dedup, buffer limits, and schema migration."""
import sqlite3

import numpy as np
import pytest

from neural_observatory import Observatory, ObservatoryConfig
from neural_observatory.collectors.base import LayerBuffer, Observation
from neural_observatory.storage.memory_store import MemoryStore
from neural_observatory.storage.sqlite_store import SQLiteStore
from conftest import run_training_steps


def _make_obs(step=1, epoch=0, seq=0, value=1.0, layer="layer_1"):
    return Observation(
        layer_name=layer,
        step=step,
        epoch=epoch,
        seq=seq,
        timestamp=float(step),
        stats={"mean": value},
        metadata={},
        values=np.array([value], dtype=np.float32),
    )


def test_collector_assigns_monotonic_sequence(tiny_model):
    """Every observation gets a unique, strictly increasing seq per collector."""
    obs = Observatory(tiny_model, config=ObservatoryConfig(max_observations=100))
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=3)

    all_seqs = []
    for layer in obs.activation_collector.layer_names:
        seqs = [o.seq for o in obs.activation_collector.get(layer)]
        assert seqs == sorted(set(seqs)), f"seq not strictly increasing for {layer}: {seqs}"
        all_seqs.extend(seqs)

    assert sorted(all_seqs) == list(range(len(all_seqs))), (
        "Collector-wide sequence numbers must be unique and contiguous from 0."
    )


def test_sequence_does_not_restart_after_clear(tiny_model):
    """clear() empties buffers but must not recycle identity numbers."""
    obs = Observatory(tiny_model, config=ObservatoryConfig(max_observations=100))
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=2)

    old_seqs = [
        o.seq
        for layer in obs.activation_collector.layer_names
        for o in obs.activation_collector.get(layer)
    ]
    obs.clear()
    run_training_steps(obs, tiny_model, n_steps=2)

    new_seqs = [
        o.seq
        for layer in obs.activation_collector.layer_names
        for o in obs.activation_collector.get(layer)
    ]
    assert new_seqs, "No observations collected after clear()."
    assert min(new_seqs) > max(old_seqs), (
        "seq restarted after clear(); persisted history would collide with fresh data."
    )


def test_sqlite_keeps_distinct_observations_with_same_step(tmp_path):
    """Two different observations at the same (layer, step, epoch) must both survive."""
    store = SQLiteStore(db_path=str(tmp_path / "obs.db"))
    store.put("activations", "layer_1", _make_obs(seq=0, value=1.0))
    store.put("activations", "layer_1", _make_obs(seq=1, value=2.0))
    store.commit()

    stored = store.get("activations", "layer_1")
    assert len(stored) == 2, "Distinct same-step observation was silently dropped."
    assert [o.seq for o in stored] == [0, 1]
    assert [float(o.values[0]) for o in stored] == [1.0, 2.0]


def test_memory_store_repeated_report_does_not_duplicate(tiny_model):
    """report() mirrors the same buffers again; totals must stay stable."""
    cfg = ObservatoryConfig(storage_backend="memory", max_observations=100)
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=3)

    obs.report()
    first_total = obs.store.total_observations()
    assert first_total > 0

    obs.report()
    obs.report()
    assert obs.store.total_observations() == first_total, (
        "Repeated report() duplicated observations in MemoryStore."
    )

    expected = sum(
        len(obs.activation_collector.get(l)) for l in obs.activation_collector.layer_names
    ) + sum(
        len(obs.gradient_collector.get(l)) for l in obs.gradient_collector.layer_names
    ) + sum(
        len(obs.parameter_collector.get(l)) for l in obs.parameter_collector.layer_names
    )
    assert first_total == expected


def test_sqlite_repeated_report_does_not_duplicate(tiny_model, tmp_path):
    cfg = ObservatoryConfig(
        storage_backend="sqlite",
        storage_path=str(tmp_path / "obs.db"),
        max_observations=100,
    )
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=3)

    obs.report()
    layer = obs.activation_collector.layer_names[0]
    first_count = len(obs.store.get("activations", layer))
    assert first_count > 0

    obs.report()
    obs.report()
    assert len(obs.store.get("activations", layer)) == first_count
    assert first_count == len(obs.activation_collector.get(layer))


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_clear_then_report_persists_fresh_observations(tiny_model, tmp_path, backend):
    """After clear(), a new run at the same step numbers must still persist."""
    cfg = ObservatoryConfig(
        storage_backend=backend,
        storage_path=str(tmp_path / "obs.db"),
        max_observations=100,
    )
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=2)
    obs.report()

    layer = obs.activation_collector.layer_names[0]
    first_run = obs.store.get("activations", layer)
    assert len(first_run) == 2
    first_max_seq = max(o.seq for o in first_run)

    obs.clear()
    run_training_steps(obs, tiny_model, n_steps=2)
    obs.report()

    stored = obs.store.get("activations", layer)
    assert len(stored) == 4, "Fresh post-clear observations were dropped as duplicates."
    assert max(o.seq for o in stored) > first_max_seq


_OLD_SCHEMA = """
CREATE TABLE observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    collection  TEXT    NOT NULL,
    layer_name  TEXT    NOT NULL,
    step        INTEGER NOT NULL,
    epoch       INTEGER NOT NULL,
    timestamp   REAL    NOT NULL,
    stats       TEXT    NOT NULL,
    metadata    TEXT    NOT NULL,
    values_blob BLOB,
    UNIQUE (collection, layer_name, step, epoch)
);
"""


def test_old_schema_database_is_migrated(tmp_path):
    """A legacy database must keep its rows and gain the seq identity column."""
    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(_OLD_SCHEMA)
    for step in (0, 1):
        conn.execute(
            "INSERT INTO observations (collection, layer_name, step, epoch, timestamp, stats, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("activations", "layer_1", step, 0, float(step), '{"mean": 1.0}', "{}"),
        )
    conn.commit()
    conn.close()

    store = SQLiteStore(db_path=db_path)
    version = sqlite3.connect(db_path).execute("PRAGMA user_version").fetchone()[0]
    assert version >= 1, "Schema migration did not run on the legacy database."

    stored = store.get("activations", "layer_1")
    assert len(stored) == 2, "Migration lost legacy rows."
    assert [o.step for o in stored] == [0, 1]

    store.put("activations", "layer_1", _make_obs(step=0, seq=1, value=9.0))
    store.commit()
    assert len(store.get("activations", "layer_1")) == 3, (
        "Migrated table does not accept distinct same-step observations."
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_get_limit_zero_returns_empty(tmp_path, backend):
    if backend == "memory":
        store = MemoryStore(max_observations=10)
    else:
        store = SQLiteStore(db_path=str(tmp_path / "obs.db"))
    store.put("activations", "layer_1", _make_obs(step=0, seq=0))
    store.put("activations", "layer_1", _make_obs(step=1, seq=1))
    store.commit()

    assert store.get("activations", "layer_1", limit=0) == []


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_get_limit_returns_first_n(tmp_path, backend):
    if backend == "memory":
        store = MemoryStore(max_observations=10)
    else:
        store = SQLiteStore(db_path=str(tmp_path / "obs.db"))
    for step in range(3):
        store.put("activations", "layer_1", _make_obs(step=step, seq=step))
    store.commit()

    limited = store.get("activations", "layer_1", limit=2)
    assert [o.step for o in limited] == [0, 1]


def test_layer_buffer_latest_zero_returns_empty():
    buf = LayerBuffer(layer_name="l", capacity=5)
    for step in range(3):
        buf.add(_make_obs(step=step, seq=step))
    assert buf.latest(0) == []
    assert len(buf.latest(2)) == 2


def test_memory_store_dedup_survives_eviction():
    """Dedup keys must be bounded: an evicted observation can be ingested again."""
    store = MemoryStore(max_observations=3)
    for seq in range(5):
        store.put("activations", "layer_1", _make_obs(step=seq, seq=seq))

    assert [o.seq for o in store.get("activations", "layer_1")] == [2, 3, 4]

    store.put("activations", "layer_1", _make_obs(step=4, seq=4))
    assert [o.seq for o in store.get("activations", "layer_1")] == [2, 3, 4], (
        "Re-put of a still-buffered observation must be ignored."
    )

    store.put("activations", "layer_1", _make_obs(step=0, seq=0))
    assert [o.seq for o in store.get("activations", "layer_1")] == [3, 4, 0], (
        "An evicted observation must be re-ingestible (bounded key set)."
    )