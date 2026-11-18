"""Collectors and lifecycle calls must stay correct under concurrent use."""

import sys
import threading
import time

import torch
from conftest import run_training_steps

from neural_observatory import Observatory, ObservatoryConfig
from neural_observatory.analysis.base import BaseAnalyzer
from neural_observatory.collectors.activation_collector import ActivationCollector
from neural_observatory.collectors.base import Observation
from neural_observatory.core.events import Event, EventBus, EventType
from neural_observatory.core.lifecycle import LifecycleManager
from neural_observatory.storage.sqlite_store import SQLiteStore


def test_collector_reads_are_safe_under_concurrent_writes():
    """get_all()/layer_names must not crash while new layers are being added."""
    col = ActivationCollector(config=ObservatoryConfig(max_observations=10))
    n_layers = 2000

    def writer():
        for i in range(n_layers):
            col.collect(f"layer_{i}", torch.randn(4), step=i)

    t = threading.Thread(target=writer)
    # Fine-grained GIL switching makes an unlocked dict iteration observe
    # concurrent inserts reliably instead of by luck.
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    t.start()
    errors = []
    try:
        while t.is_alive():
            col.get_all()
            col.layer_names
            col.latest("layer_0", n=2)
            # Yield so the reader doesn't starve the writer under the lock.
            time.sleep(0.001)
    except Exception as exc:
        errors.append(exc)
    finally:
        sys.setswitchinterval(old_interval)
        t.join(10)

    assert not errors, f"Unlocked collector reads raced with writes: {errors}"
    assert len(col.layer_names) == n_layers


def test_sqlite_puts_are_serialized(tmp_path):
    """A second put must wait while the first is inside the connection."""
    store = SQLiteStore(db_path=str(tmp_path / "obs.db"))
    real_conn = store._conn
    entered = threading.Event()
    gate = threading.Event()

    class GateConn:
        def execute(self, *args, **kwargs):
            if not entered.is_set():
                entered.set()
                gate.wait(5)
            return real_conn.execute(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_conn, name)

    store._conn = GateConn()
    order = []

    def puter(step):
        store.put(
            "activations",
            "l",
            Observation(layer_name="l", step=step, epoch=0, seq=step, stats={}),
        )
        order.append(step)

    t1 = threading.Thread(target=puter, args=(0,))
    t1.start()
    assert entered.wait(5), "First put never reached the connection."
    t2 = threading.Thread(target=puter, args=(1,))
    t2.start()
    t2.join(0.3)
    second_blocked = t2.is_alive()
    gate.set()
    t1.join(5)
    t2.join(5)

    assert second_blocked, "Second put ran while the first was still in the connection."
    assert order == [0, 1]
    assert len(store.get("activations", "l")) == 2


def test_event_bus_subscribe_is_locked():
    bus = EventBus()
    done = []

    def handler(event):
        done.append(event)

    with bus._lock:
        t = threading.Thread(
            target=bus.subscribe, args=(EventType.BATCH_FINISHED, handler)
        )
        t.start()
        t.join(0.3)
        blocked = t.is_alive()

    t.join(5)
    assert blocked, "subscribe() mutated subscriber state without holding the lock."
    bus.publish(Event(event_type=EventType.BATCH_FINISHED, source="test"))
    assert len(done) == 1


def test_end_reporting_tolerates_stop_during_report():
    lm = LifecycleManager()
    lm.start()
    lm.begin_reporting()
    lm.stop()
    lm.end_reporting()
    assert lm.is_stopped


def test_report_survives_analyzer_that_stops_monitoring(tiny_model):
    """stop() inside an analyzer must not make report()'s finally clause raise."""

    class StoppingAnalyzer(BaseAnalyzer):
        def __init__(self, obs, config):
            super().__init__(config=config)
            self._obs = obs

        @property
        def name(self):
            return "stopping"

        def analyze(self, observations):
            self._obs.stop()
            return []

    cfg = ObservatoryConfig(max_observations=50)
    obs = Observatory(tiny_model, config=cfg)
    obs.register_analyzer(StoppingAnalyzer(obs, cfg))
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=1)

    report = obs.report()
    assert report is not None
    assert obs.state.name == "STOPPED"
