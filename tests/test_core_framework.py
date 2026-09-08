"""Comprehensive tests for the core framework components."""
import os
import pytest
import logging

from neural_observatory.core.lifecycle import LifecycleManager, LifecycleError, MonitoringState
from neural_observatory.core.events import EventBus, Event, EventType
from neural_observatory.core.registry import Registry, RegistrationError
from neural_observatory.core.configuration import ObservatoryConfig
from neural_observatory.analysis.base import BaseAnalyzer, AnalysisResult, AnalysisStatus
from neural_observatory.storage.sqlite_store import SQLiteStore
from neural_observatory.collectors.base import Observation


# ======================================================================
# 1. LifecycleManager Tests
# ======================================================================
class TestLifecycleManager:
    def test_valid_transitions(self):
        lm = LifecycleManager()
        assert lm.is_idle
        lm.start()
        assert lm.is_active
        lm.stop()
        assert lm.is_stopped

    def test_invalid_transition_raises_error(self):
        lm = LifecycleManager()
        # IDLE -> REPORTING is an invalid transition
        with pytest.raises(LifecycleError):
            lm.begin_reporting()

    def test_reportable_states(self):
        lm = LifecycleManager()
        with pytest.raises(LifecycleError):
            lm.require_reportable("test_op") # Fails on IDLE
            
        lm.start()
        lm.require_reportable("test_op") # Passes on ACTIVE
        lm.stop()
        lm.require_reportable("test_op") # Passes on STOPPED


# ======================================================================
# 2. EventBus Tests
# ======================================================================
class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received_events = []
        bus.subscribe(EventType.ANALYSIS_COMPLETED, lambda e: received_events.append(e))
        
        event = Event(event_type=EventType.ANALYSIS_COMPLETED, source="test")
        bus.publish(event)
        
        assert len(received_events) == 1
        assert received_events[0].source == "test"

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        handler = lambda e: received.append(e)
        
        bus.subscribe(EventType.ANOMALY_DETECTED, handler)
        bus.unsubscribe(EventType.ANOMALY_DETECTED, handler)
        bus.publish(Event(event_type=EventType.ANOMALY_DETECTED, source="test"))
        
        assert len(received) == 0

    def test_handler_exception_does_not_crash_bus(self, caplog):
        bus = EventBus()
        def bad_handler(e): raise ValueError("Boom!")
        
        good_handler_called = False
        def good_handler(e): 
            nonlocal good_handler_called
            good_handler_called = True
            
        bus.subscribe(EventType.OBSERVATION_COLLECTED, bad_handler)
        bus.subscribe(EventType.OBSERVATION_COLLECTED, good_handler)
        
        # Should not raise
        bus.publish(Event(event_type=EventType.OBSERVATION_COLLECTED, source="test"))
        
        # Good handler should still be called
        assert good_handler_called


# ======================================================================
# 3. Registry Tests
# ======================================================================
class DummyAnalyzer(BaseAnalyzer):
    @property
    def name(self): return "dummy"
    def analyze(self, observations): return []

class TestRegistry:
    def test_register_and_get(self):
        reg = Registry()
        analyzer = DummyAnalyzer(config=ObservatoryConfig())
        reg.register_analyzer(analyzer)
        assert reg.get_analyzer("dummy") is analyzer

    def test_register_invalid_type_raises(self):
        reg = Registry()
        with pytest.raises(RegistrationError):
            reg.register_analyzer("not_an_analyzer")

    def test_overwrite_warning(self, caplog):
        reg = Registry()
        analyzer1 = DummyAnalyzer(config=ObservatoryConfig())
        analyzer2 = DummyAnalyzer(config=ObservatoryConfig())
        
        with caplog.at_level(logging.WARNING):
            reg.register_analyzer(analyzer1, name="dummy")
            reg.register_analyzer(analyzer2, name="dummy") # Overwrite
            
        assert "Overwriting existing analyzer" in caplog.text
        assert reg.get_analyzer("dummy") is analyzer2


# ======================================================================
# 4. SQLiteStore Advanced Tests
# ======================================================================
class TestSQLiteStore:
    def test_commit_and_close_persists_data(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = SQLiteStore(db_path=db_path)

        obs = Observation(layer_name="L", step=0, epoch=0)
        store.put("activations", "L", obs)

        # Commit explicitly
        store.commit()
        store.close()

        # Reopen and check
        store2 = SQLiteStore(db_path=db_path)
        results = store2.get("activations", "L")
        assert len(results) == 1
        store2.close()

    def test_clear_specific_collection(self, tmp_path):
        db_path = str(tmp_path / "test_clear.db")
        store = SQLiteStore(db_path=db_path)

        store.put("activations", "L", Observation(layer_name="L", step=0, epoch=0))
        store.put("gradients", "L", Observation(layer_name="L", step=0, epoch=0))
        store.commit()

        store.clear("activations")
        store.commit()

        assert len(store.get("activations", "L")) == 0
        assert len(store.get("gradients", "L")) == 1
        store.close()