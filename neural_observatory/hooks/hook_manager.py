"""
Neural Observatory — Hook Manager
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ..collectors.activation_collector import ActivationCollector
from ..collectors.gradient_collector import GradientCollector
from ..core.configuration import ObservatoryConfig

logger = logging.getLogger(__name__)


class HookManager:
    def __init__(
        self,
        model: nn.Module,
        activation_collector: ActivationCollector,
        gradient_collector: GradientCollector,
        config: Optional[ObservatoryConfig] = None,
    ) -> None:
        self._model = model
        self._act_col = activation_collector
        self._grad_col = gradient_collector
        self._config = config or ObservatoryConfig()
        self._step: int = 0
        self._epoch: int = 0
        self._active: bool = False
        self.hook_handles: Dict[str, List[Any]] = {}

    def attach(self, layer_filter: Optional[List[str]] = None) -> None:
        self.detach()

        cfg = self._config
        attached = 0
        inplace_layers: List[str] = []
        seen_module_ids: set = set()

        for name, module in self._model.named_modules():
            if not name:
                continue

            layer_type = module.__class__.__name__

            if layer_filter and name not in layer_filter:
                continue

            if not cfg.should_monitor_layer(name, layer_type):
                continue

            # Skip tied/shared weights to avoid duplicate hooks
            module_id = id(module)
            if module_id in seen_module_ids:
                continue
            seen_module_ids.add(module_id)

            if getattr(module, "inplace", False):
                inplace_layers.append(name)

            handles: List[Any] = []

            # We use tensor-level hooks instead of module.register_full_backward_hook.
            # Module backward hooks wrap the output in an autograd Function, which
            # crashes hard if the user has inplace=True ops downstream (e.g. ResNet).
            # Tensor hooks just bind to the specific output tensor and are GC'd normally.
            fwd_handle = module.register_forward_hook(
                self._make_forward_hook(name)
            )
            handles.append(fwd_handle)

            self.hook_handles[name] = handles
            attached += 1

        if inplace_layers:
            logger.warning(
                "Monitoring %d inplace layers (%s...). Tensor hooks handle this, "
                "but custom inplace ops in forward() might still act weird.",
                len(inplace_layers), ", ".join(inplace_layers[:3]),
            )

        self._active = True
        logger.info("Attached hooks to %d layers.", attached)

    def detach(self) -> None:
        self._active = False
        count = 0
        for _, handles in self.hook_handles.items():
            for h in handles:
                h.remove()
                count += 1
        self.hook_handles.clear()
        if count:
            logger.info("Removed %d hooks.", count)

    def set_step(self, step: int) -> None:
        self._step = step

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _make_forward_hook(self, layer_name: str) -> Callable:
        act_col = self._act_col
        grad_col = self._grad_col

        def _forward_hook(module: nn.Module, inp: Any, output: Any) -> None:
            # Capture step/epoch right now so backward() ties to this exact iteration
            step = self._step
            epoch = self._epoch

            tensor = output
            attn_weights = None
            
            if isinstance(output, (tuple, list)):
                # Special handling for nn.MultiheadAttention which returns (output, weights)
                if len(output) == 2 and isinstance(output[1], torch.Tensor):
                    tensor = output[0]
                    attn_weights = output[1]
                else:
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            tensor = item
                            break
                    else:
                        return

            if not isinstance(tensor, torch.Tensor):
                return

            try:
                act_col.collect(
                    layer_name=layer_name,
                    data=tensor,
                    step=step,
                    epoch=epoch,
                )
                # If we captured attention weights, store them under a special collection name
                if attn_weights is not None:
                    act_col.collect(
                        layer_name=f"{layer_name}_attn_weights",
                        data=attn_weights,
                        step=step,
                        epoch=epoch,
                    )
            except Exception as exc:
                logger.debug("Forward hook error on %s: %s", layer_name, exc)

            # THIS WAS MISSING! We need to re-attach the gradient hook.
            if tensor.requires_grad:
                tensor.register_hook(
                    self._make_grad_hook(layer_name, step, epoch)
                )

        return _forward_hook

    def _make_grad_hook(self, layer_name: str, step: int, epoch: int) -> Callable:
        grad_col = self._grad_col

        def _grad_hook(grad: torch.Tensor) -> None:
            if not self._active:
                return
            try:
                grad_col.collect(
                    layer_name=layer_name,
                    data=grad,
                    step=step,
                    epoch=epoch,
                )
            except Exception as exc:
                logger.debug("Gradient hook error on %s: %s", layer_name, exc)

        return _grad_hook

    @property
    def monitored_layers(self) -> List[str]:
        return sorted(self.hook_handles.keys())

    def __repr__(self) -> str:
        return f"HookManager(layers={len(self.hook_handles)}, step={self._step})"