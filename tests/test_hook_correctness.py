"""Regression tests for hook attachment coverage and gradient-hook lifecycle (0.5.2)."""
import pytest
import torch
import torch.nn as nn

from neural_observatory import Observatory, ObservatoryConfig


class Passthrough(nn.Module):
    """Returns the very same tensor object on every forward call."""

    def forward(self, x):
        return x


def test_root_leaf_module_is_monitored():
    """A bare nn.Linear (the root module itself) must not be silently skipped."""
    model = nn.Linear(8, 4)
    obs = Observatory(model, config=ObservatoryConfig(max_observations=50))
    obs.watch()

    obs.step(step=0, epoch=0)
    out = model(torch.randn(2, 8))
    out.sum().backward()

    assert obs.activation_collector.layer_names == ["root"]
    assert len(obs.activation_collector.get("root")) == 1
    assert len(obs.gradient_collector.get("root")) == 1


def test_root_container_is_not_hooked(tiny_model):
    """A root with children must not get a duplicate 'root' hook."""
    obs = Observatory(tiny_model, config=ObservatoryConfig(max_observations=50))
    obs.watch()
    assert "root" not in obs._hook_manager.hook_handles


def test_gradient_hooks_do_not_accumulate_on_reused_tensor():
    """Three forwards returning one tensor must yield one gradient hook, not three."""
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
        f"Gradient hook accumulated on a reused tensor: {len(grads)} observations."
    )


def test_forward_hook_does_not_collect_after_detach(tiny_model):
    obs = Observatory(tiny_model, config=ObservatoryConfig(max_observations=50))
    obs.watch()
    hook = obs._hook_manager._make_forward_hook("0")
    obs.stop()

    hook(tiny_model[0], None, torch.randn(2, 16))
    assert obs.activation_collector.get("0") == []


def test_attach_failure_leaves_no_partial_hooks(tiny_model, monkeypatch):
    """If attach() dies mid-way, already-attached hooks must be rolled back."""
    obs = Observatory(tiny_model, config=ObservatoryConfig(max_observations=50))

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
    assert obs.activation_collector.total_observations == 0, (
        "A failed watch() left live hooks on the model."
    )


def test_watch_failure_after_attach_detaches_hooks(tiny_model, monkeypatch):
    """If baseline capture fails after attach(), watch() must still roll back."""
    obs = Observatory(tiny_model, config=ObservatoryConfig(max_observations=50))

    def boom(*args, **kwargs):
        raise RuntimeError("baseline capture failed")

    monkeypatch.setattr(obs._parameter_collector, "collect_model_parameters", boom)
    with pytest.raises(RuntimeError):
        obs.watch()
    monkeypatch.undo()

    assert obs._hook_manager.hook_handles == {}
    _ = tiny_model(torch.randn(2, 8))
    assert obs.activation_collector.total_observations == 0


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
        f"Re-attach left stale gradient hooks behind: {len(grads)} observations."
    )
    assert [g.step for g in grads] == [0, 1]