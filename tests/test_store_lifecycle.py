"""Close and exit paths: pending writes are committed and nothing is orphaned."""

import sqlite3

import pytest
from conftest import run_training_steps

from neural_observatory import Observatory, ObservatoryConfig
from neural_observatory.storage.sqlite_store import SQLiteStore


def test_sqlite_close_commits_pending_writes(tmp_path):
    """close() must flush uncommitted rows; a shadowed close() lost them."""
    db_path = str(tmp_path / "obs.db")
    store = SQLiteStore(db_path=db_path)
    from neural_observatory.collectors.base import Observation

    store.put(
        "activations",
        "layer_1",
        Observation(layer_name="layer_1", step=0, epoch=0, stats={"mean": 1.0}),
    )
    store.close()

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    conn.close()
    assert count == 1, "close() discarded uncommitted observations."


def test_context_manager_closes_and_persists_sqlite(tiny_model, tmp_path):
    db_path = str(tmp_path / "obs.db")
    cfg = ObservatoryConfig(
        storage_backend="sqlite", storage_path=db_path, max_observations=100
    )

    with Observatory(tiny_model, config=cfg) as obs:
        run_training_steps(obs, tiny_model, n_steps=3)
        obs.report()
        layer = obs.activation_collector.layer_names[0]

    with pytest.raises(sqlite3.ProgrammingError):
        obs.store.get("activations", layer)

    reopened = SQLiteStore(db_path=db_path)
    assert len(reopened.get("activations", layer)) > 0, "Exit did not persist data."
    reopened.close()


def test_close_method_stops_and_closes(tiny_model, tmp_path):
    db_path = str(tmp_path / "obs.db")
    cfg = ObservatoryConfig(
        storage_backend="sqlite", storage_path=db_path, max_observations=100
    )
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=2)
    obs.report()

    obs.close()

    assert not obs.is_active
    with pytest.raises(sqlite3.ProgrammingError):
        obs.store.get("activations", obs.activation_collector.layer_names[0])


def test_stop_then_report_still_works(tiny_model, tmp_path):
    """README flow: stop() detaches hooks but report() must still read the store."""
    cfg = ObservatoryConfig(
        storage_backend="sqlite",
        storage_path=str(tmp_path / "obs.db"),
        max_observations=100,
    )
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=2)
    obs.stop()

    report = obs.report()
    assert report is not None
    layer = obs.activation_collector.layer_names[0]
    assert len(obs.store.get("activations", layer)) > 0


def test_stop_is_idempotent(tiny_model):
    obs = Observatory(tiny_model, config=ObservatoryConfig(max_observations=50))
    obs.watch()
    obs.stop()
    obs.stop()
    assert not obs.is_active


def test_close_after_stop_is_safe(tiny_model, tmp_path):
    cfg = ObservatoryConfig(
        storage_backend="sqlite",
        storage_path=str(tmp_path / "obs.db"),
        max_observations=50,
    )
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()
    obs.stop()
    obs.close()
    assert not obs.is_active


def test_report_commits_exactly_once(tiny_model, tmp_path):
    class CountingStore(SQLiteStore):
        def __init__(self, db_path):
            super().__init__(db_path=db_path)
            self.commit_calls = 0

        def commit(self):
            self.commit_calls += 1
            super().commit()

    db_path = str(tmp_path / "obs.db")
    cfg = ObservatoryConfig(
        storage_backend="sqlite", storage_path=db_path, max_observations=100
    )
    obs = Observatory(tiny_model, config=cfg)
    obs._store.close()
    obs._store = CountingStore(db_path)

    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=2)
    obs.report()

    assert (
        obs._store.commit_calls == 1
    ), f"report() committed {obs._store.commit_calls} times; expected exactly one batch commit."


def test_memory_backend_context_exit_keeps_store_readable(tiny_model):
    cfg = ObservatoryConfig(storage_backend="memory", max_observations=100)
    with Observatory(tiny_model, config=cfg) as obs:
        run_training_steps(obs, tiny_model, n_steps=2)
        obs.report()
        layer = obs.activation_collector.layer_names[0]

    assert len(obs.store.get("activations", layer)) > 0


def test_temporary_database_is_removed_on_close():
    store = SQLiteStore()
    path = store._path
    assert path.exists()
    store.close()
    assert not path.exists(), "close() left an orphaned temporary database behind."


def test_explicit_database_survives_close(tmp_path):
    db_path = tmp_path / "obs.db"
    store = SQLiteStore(db_path=str(db_path))
    store.close()
    assert db_path.exists(), "close() deleted a user-owned database file."


def test_store_close_is_idempotent(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "obs.db"))
    store.close()
    store.close()


def test_observatory_close_is_idempotent_and_removes_temp_database(tiny_model):
    cfg = ObservatoryConfig(storage_backend="sqlite", max_observations=50)
    obs = Observatory(tiny_model, config=cfg)
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=1)
    obs.report()
    path = obs.store._path

    obs.close()
    obs.close()

    assert not obs.is_active
    assert not path.exists(), "Observatory.close() orphaned its temporary database."
