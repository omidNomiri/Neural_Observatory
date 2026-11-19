from typing import Callable

import torch
from torch import nn


def supports_monitored_compile() -> bool:
    return tuple(int(part) for part in torch.__version__.split(".")[:2]) >= (
        2,
        12,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    # PyTorch has no public accessor for an OptimizedModule's original module.
    original = getattr(model, "_orig_mod", None)
    if isinstance(original, nn.Module):
        if not supports_monitored_compile():
            raise RuntimeError(
                "Monitoring torch.compile models requires PyTorch >= 2.12; "
                "older Dynamo versions can silently run monitored models in "
                "eager mode. Pass the original eager model instead."
            )
        return original
    return model


def eager_hook(hook: Callable) -> Callable:
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if disable is None:
        # Older PyTorch exposes the callback boundary only through Dynamo.
        disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    return disable(hook) if disable is not None else hook
