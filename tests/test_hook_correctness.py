"""All monitored modules get hooks; gradient hooks follow the live step."""

import logging

import pytest
import torch
import torch.nn as nn

from neural_observatory import Observatory, ObservatoryConfig


class Passthrough(nn.Module):
    """Returns the very same tensor object on every forward call."""

    def forward(self, x):
        return x


@pytest.mark.parametrize("compiler_api", ["public", "private", "none"])
def test_root_leaf_module_is_monitored(compiler_api, monkeypatch):
    """A bare nn.Linear root must not be silently skipped."""
    model = nn.Linear(8, 4)
    obs = Observatory(model, config=ObservatoryConfig(max_observations=50))
    with monkeypatch.context() as patch:
        if compiler_api != "public":
            patch.setattr(torch, "compiler", None, raising=False)
        if compiler_api == "none":
            patch.setattr(torch, "_dynamo", None, raising=False)
        obs.watch()

    obs.step(step=0, epoch=0)
    out = model(torch.randn(2, 8))
    out.sum().backward()

    assert obs.activation_collector.layer_names == ["root"]
    assert len(obs.activation_collector.get("root")) == 1
    assert len(obs.gradient_collector.get("root")) == 1


def test_root_container_is_not_hooked(tiny_model):
    """A root with children must not get a duplicate 'root' hook."""
    obs = Observatory(
        tiny_model, config=ObservatoryConfig(max_observations=50)
    )
    obs.watch()
    assert "root" not in obs._hook_manager.hook_handles


def test_gradient_hooks_do_not_accumulate_on_reused_tensor():
    """Three forwards returning one tensor must yield one gradient hook."""
    model = Passthrough()
    obs = Observatory(model, config=ObservatoryConfig(max_observations=50))
    obs.watch()

    x = torch.randn(4, requires_grad=True)
    obs.step(step=0, epoch=0)
    for _ in range(3):
        out = model(x)
    out.sum().backward()

    grads = obs.gradient_collector.get("root")
    assert len(grads) == 1, (
        "Gradient hook accumulated on a reused tensor: "
        f"{len(grads)} observations."
    )


def test_forward_hook_does_not_collect_after_detach(tiny_model):
    obs = Observatory(
        tiny_model, config=ObservatoryConfig(max_observations=50)
    )
    obs.watch()
    hook = obs._hook_manager._make_forward_hook("0")
    obs.stop()

    hook(tiny_model[0], None, torch.randn(2, 16))
    assert obs.activation_collector.get("0") == []


def test_attach_failure_leaves_no_partial_hooks(tiny_model, monkeypatch):
    """If attach() dies mid-way, already-attached hooks must be rolled back."""
    obs = Observatory(
        tiny_model, config=ObservatoryConfig(max_observations=50)
    )

    original = nn.Linear.register_forward_hook
    calls = {"n": 0}

    def flaky(module_self, hook):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated attach failure")
        return original(module_self, hook)

    monkeypatch.setattr(nn.Linear, "register_forward_hook", flaky)
    with pytest.raises(RuntimeError):
        obs.watch()
    monkeypatch.undo()

    assert obs._hook_manager.hook_handles == {}
    _ = tiny_model(torch.randn(2, 8))
    assert (
        obs.activation_collector.total_observations == 0
    ), "A failed watch() left live hooks on the model."


def test_watch_failure_after_attach_detaches_hooks(tiny_model, monkeypatch):
    """watch() must roll back if baseline capture fails after attach()."""
    obs = Observatory(
        tiny_model, config=ObservatoryConfig(max_observations=50)
    )

    def boom(*args, **kwargs):
        raise RuntimeError("baseline capture failed")

    monkeypatch.setattr(
        obs._parameter_collector, "collect_model_parameters", boom
    )
    with pytest.raises(RuntimeError):
        obs.watch()
    monkeypatch.undo()

    assert obs._hook_manager.hook_handles == {}
    _ = tiny_model(torch.randn(2, 8))
    assert obs.activation_collector.total_observations == 0


def test_reused_tensor_gradients_follow_the_current_step():
    """A reused tensor must stamp each backward with its current step."""
    model = Passthrough()
    obs = Observatory(model, config=ObservatoryConfig(max_observations=50))
    obs.watch()

    x = torch.randn(4, requires_grad=True)
    obs.step(step=0, epoch=0)
    model(x).sum().backward()
    x.grad = None

    obs.step(step=1, epoch=0)
    model(x).sum().backward()

    grads = obs.gradient_collector.get("root")
    assert (
        len(grads) == 2
    ), f"Expected one gradient per step, got {len(grads)}."
    assert [g.step for g in grads] == [
        0,
        1,
    ], "Gradient hook stayed bound to the step of the first forward."
    obs.stop()


def test_stop_then_watch_recollects_gradients_cleanly():
    model = Passthrough()
    obs = Observatory(model, config=ObservatoryConfig(max_observations=50))
    x = torch.randn(4, requires_grad=True)

    obs.watch()
    obs.step(step=0, epoch=0)
    model(x).sum().backward()
    obs.stop()

    obs.watch()
    obs.step(step=1, epoch=0)
    model(x).sum().backward()

    grads = obs.gradient_collector.get("root")
    assert len(grads) == 2, (
        "Re-attach left stale gradient hooks behind: "
        f"{len(grads)} observations."
    )
    assert [g.step for g in grads] == [0, 1]


def test_shared_tensor_reaches_every_monitored_layer():
    """Both Identity layers sharing a tensor must see its gradient."""
    model = nn.Sequential(nn.Identity(), nn.Identity())
    cfg = ObservatoryConfig(max_observations=50, exclude_layer_types=[])
    obs = Observatory(model, config=cfg)
    obs.watch()

    x = torch.randn(4, requires_grad=True)
    obs.step(step=0, epoch=0)
    for _ in range(3):
        out = model(x)
    out.sum().backward()

    assert len(obs.gradient_collector.get("0")) == 1
    assert len(obs.gradient_collector.get("1")) == 1, (
        "Only the first layer attached to the shared tensor "
        "received a gradient."
    )
    obs.stop()


def test_gradient_accumulation_follows_the_documented_contract(tiny_model):
    """step() before each micro-batch, backward before the next step()."""
    obs = Observatory(
        tiny_model, config=ObservatoryConfig(max_observations=100)
    )
    obs.watch()
    optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    for micro in range(2):
        obs.step(step=micro, epoch=0)
        out = tiny_model(torch.randn(4, 8))
        criterion(out, torch.randint(0, 4, (4,))).backward()
    optimizer.step()

    layer = obs.gradient_collector.layer_names[0]
    grads = obs.gradient_collector.get(layer)
    assert [g.step for g in grads] == [0, 1], (
        "Per-micro-batch step() + immediate backward "
        "must attribute each gradient."
    )
    obs.stop()


@pytest.mark.parametrize(
    "collector_name", ["activation_collector", "gradient_collector"]
)
def test_collector_failure_is_visible_without_breaking_training(
    collector_name, monkeypatch, caplog
):
    model = nn.Linear(8, 4)
    obs = Observatory(model)
    obs.watch()
    caplog.set_level(logging.WARNING)

    def fail(*args, **kwargs):
        raise RuntimeError("collector unavailable")

    monkeypatch.setattr(getattr(obs, collector_name), "collect", fail)
    try:
        model(torch.randn(2, 8)).sum().backward()
        assert any(
            "collector unavailable" in record.getMessage()
            and record.levelno >= logging.WARNING
            for record in caplog.records
        )
    finally:
        obs.close()


def test_close_attempts_all_handles_and_closes_store_on_remove_failure(
    monkeypatch,
):
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
    obs = Observatory(model, ObservatoryConfig(storage_backend="sqlite"))
    obs.watch()
    output = model(torch.randn(2, 8))
    path = obs.store._path
    handle = obs._hook_manager.hook_handles["0"][0]
    original_remove = handle.remove

    def fail():
        raise RuntimeError("remove failed")

    monkeypatch.setattr(handle, "remove", fail)
    with pytest.raises(RuntimeError, match="remove failed"):
        obs.close()
    assert not obs.is_active
    assert not model[1]._forward_hooks
    assert not obs._hook_manager._grad_hook_handles
    assert not path.exists()
    totals = obs.gradient_collector.total_observations
    output.sum().backward()
    assert obs.gradient_collector.total_observations == totals
    monkeypatch.setattr(handle, "remove", original_remove)
    obs.close()
    assert not model[0]._forward_hooks


def test_failed_watch_stops_lifecycle_even_if_rollback_fails(monkeypatch):
    model = nn.Linear(8, 4)
    obs = Observatory(model)
    handle = None
    original_remove = None

    def fail_remove():
        raise RuntimeError("remove failed")

    def fail_baseline(*args, **kwargs):
        nonlocal handle, original_remove
        handle = next(iter(obs._hook_manager.hook_handles.values()))[0]
        original_remove = handle.remove
        monkeypatch.setattr(handle, "remove", fail_remove)
        raise RuntimeError("baseline failed")

    monkeypatch.setattr(
        obs.parameter_collector, "collect_model_parameters", fail_baseline
    )
    try:
        with pytest.raises(RuntimeError, match="remove failed"):
            obs.watch()
        assert not obs.is_active
        assert not obs._hook_manager._active
    finally:
        if handle is not None:
            monkeypatch.setattr(handle, "remove", original_remove)
        obs.close()
    assert not model._forward_hooks


def test_gradient_handle_removal_failure_can_be_retried(monkeypatch):
    model = nn.Linear(8, 4)
    obs = Observatory(model)
    obs.watch()
    output = model(torch.randn(2, 8))
    handle = next(iter(obs._hook_manager._grad_hook_handles.values()))[1]
    original_remove = handle.remove

    def fail():
        raise RuntimeError("gradient remove failed")

    try:
        monkeypatch.setattr(handle, "remove", fail)
        with pytest.raises(RuntimeError, match="gradient remove failed"):
            obs.close()
        assert not obs.is_active
        assert not model._forward_hooks
        assert obs._hook_manager._grad_hook_handles
        output.sum().backward()
        assert obs.gradient_collector.total_observations == 0
    finally:
        monkeypatch.setattr(handle, "remove", original_remove)
        obs.close()
    assert not obs._hook_manager._grad_hook_handles


def test_failed_context_entry_closes_temporary_store(monkeypatch):
    model = nn.Linear(8, 4)
    obs = Observatory(model, ObservatoryConfig(storage_backend="sqlite"))
    path = obs.store._path

    def fail(*args, **kwargs):
        raise RuntimeError("baseline failed")

    monkeypatch.setattr(
        obs.parameter_collector, "collect_model_parameters", fail
    )
    with pytest.raises(RuntimeError, match="baseline failed"):
        with obs:
            pytest.fail("context must not enter")
    assert not path.exists()
    assert not model._forward_hooks
