import copy
from collections import Counter

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import neural_observatory.collectors as collectors
from neural_observatory import Observatory, ObservatoryConfig
from neural_observatory.hooks.hook_manager import HookManager


@pytest.mark.parametrize("entry", ["observatory", "hooks"])
def test_unsupported_compiled_wrapper_is_rejected_without_mutation(
    entry, monkeypatch
):
    model = nn.Linear(8, 4)
    compiled = torch.compile(model, backend="eager")
    parameter_ids = [id(p) for p in model.parameters()]
    monkeypatch.setattr(torch, "__version__", "2.8.0+cpu")
    with pytest.raises(RuntimeError, match=r"PyTorch.*2\.12.*eager"):
        if entry == "observatory":
            Observatory(compiled)
        else:
            manager = HookManager(
                compiled,
                collectors.ActivationCollector(),
                collectors.GradientCollector(),
            )
            manager.attach()
    assert not model._forward_hooks
    assert [id(p) for p in model.parameters()] == parameter_ids
    assert "forward" not in model.__dict__


def test_old_torch_eager_monitoring_warns_about_later_compilation(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.8.0+cpu")
    model = nn.Linear(8, 4)
    obs = Observatory(model)
    try:
        with pytest.warns(UserWarning, match=r"torch.compile.*unsupported"):
            obs.watch()
        obs.step(0)
        model(torch.randn(2, 8)).sum().backward()
        assert len(obs.activation_collector.get("root")) == 1
        assert len(obs.gradient_collector.get("root")) == 1
    finally:
        obs.close()


@pytest.mark.parametrize(
    "mode", ["bare", "observatory", "hooks", "compile_later"]
)
@pytest.mark.parametrize(
    "sample_rate, expected_steps", [(1, [0, 1, 2]), (2, [0, 2])]
)
@pytest.mark.parametrize("grad_enabled", [True, False])
@pytest.mark.parametrize("compiler_api", ["public", "private"])
def test_real_model_graphs_and_observation_fidelity(
    mode, sample_rate, expected_steps, grad_enabled, compiler_api, monkeypatch
):
    torch.compiler.reset()
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    counts = Counter()
    executed_operations = Counter()

    def backend(graph, example_inputs):
        counts["compiled"] += 1
        operations = [
            n.target for n in graph.graph.nodes if n.op == "call_function"
        ]

        def execute(*args):
            counts["executed"] += 1
            executed_operations.update(operations)
            return graph.forward(*args)

        return execute

    config = ObservatoryConfig(
        max_observations=100,
        activation_sample_rate=sample_rate,
        gradient_sample_rate=sample_rate,
    )
    obs = None
    manager = None
    activation = collectors.ActivationCollector(config=config)
    gradient = collectors.GradientCollector(config=config)
    compiled = torch.compile(model, backend=backend)
    version = tuple(int(v) for v in torch.__version__.split(".")[:2])
    try:
        if version < (2, 12) and mode != "bare":
            with pytest.raises(RuntimeError, match=r"PyTorch.*2\.12.*eager"):
                if mode in ("observatory", "compile_later"):
                    Observatory(compiled, config=config)
                else:
                    HookManager(
                        compiled, activation, gradient, config
                    ).attach()
            assert not any(m._forward_hooks for m in model.modules())
            return
        with monkeypatch.context() as patch:
            if compiler_api == "private":
                patch.setattr(torch, "compiler", None, raising=False)
            if mode in ("observatory", "compile_later"):
                obs = Observatory(
                    model if mode == "compile_later" else compiled, config
                )
                obs.watch()
                manager = obs._hook_manager
                activation = obs.activation_collector
                gradient = obs.gradient_collector
            elif mode == "hooks":
                manager = HookManager(compiled, activation, gradient, config)
                manager.attach()
        if mode == "compile_later":
            compiled = torch.compile(model, backend=backend)

        with torch.set_grad_enabled(grad_enabled):
            for step in range(3):
                if obs is not None:
                    obs.step(step, epoch=0)
                elif manager is not None:
                    manager.set_step(step)
                optimizer.zero_grad()
                reference_optimizer.zero_grad()
                x = torch.randn(4, 8, requires_grad=grad_enabled)
                reference_x = x.detach().clone().requires_grad_(grad_enabled)
                actual = compiled(x)
                expected = reference_x
                reference_outputs = []
                for layer in reference:
                    expected = layer(expected)
                    if grad_enabled:
                        expected.retain_grad()
                    reference_outputs.append(expected)
                torch.testing.assert_close(actual, expected)
                if grad_enabled:
                    targets = torch.randint(0, 4, (4,))
                    F.cross_entropy(actual, targets).backward()
                    F.cross_entropy(expected, targets).backward()
                    torch.testing.assert_close(x.grad, reference_x.grad)
                    for p, reference_p in zip(
                        model.parameters(), reference.parameters()
                    ):
                        torch.testing.assert_close(p.grad, reference_p.grad)
                    optimizer.step()
                    reference_optimizer.step()
                    for p, reference_p in zip(
                        model.parameters(), reference.parameters()
                    ):
                        torch.testing.assert_close(p, reference_p)
                if manager is not None and step in expected_steps:
                    for name, output in zip(
                        ("0", "1", "2"), reference_outputs
                    ):
                        torch.testing.assert_close(
                            torch.from_numpy(
                                activation.latest(name)[0].values
                            ),
                            output.detach(),
                        )
                        if grad_enabled:
                            torch.testing.assert_close(
                                torch.from_numpy(
                                    gradient.latest(name)[0].values
                                ),
                                output.grad,
                            )
        assert counts["compiled"] > 0
        assert counts["executed"] > 0
        assert executed_operations[F.linear] == 6
        assert executed_operations[F.relu] == 3
        if manager is not None:
            for name in ("0", "1", "2"):
                assert [o.step for o in activation.get(name)] == expected_steps
                assert [o.step for o in gradient.get(name)] == (
                    expected_steps if grad_enabled else []
                )
    finally:
        try:
            if obs is not None:
                obs.close()
            elif manager is not None:
                manager.detach()
            assert not any(m._forward_hooks for m in model.modules())
            if manager is not None:
                assert not manager._grad_hook_handles
                totals = (
                    activation.total_observations,
                    gradient.total_observations,
                )
                compiled(
                    torch.randn(4, 8, requires_grad=True)
                ).sum().backward()
                assert (
                    activation.total_observations,
                    gradient.total_observations,
                ) == totals
        finally:
            torch.compiler.reset()
