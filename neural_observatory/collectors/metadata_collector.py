"""
Neural Observatory — Metadata Collector
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerMeta:
    name: str
    layer_type: str
    num_parameters: int
    parameter_shapes: Dict[str, List[int]]
    trainable: bool
    depth: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class MetadataCollector:
    def __init__(self) -> None:
        self._layers: Dict[str, LayerMeta] = {}

    def collect_model_metadata(self, model: nn.Module) -> None:
        self._layers.clear()

        for name, module in model.named_modules():
            if name == "":
                continue

            depth = name.count(".")
            param_shapes = {
                pname: list(p.shape)
                for pname, p in module.named_parameters(recurse=False)
            }
            num_params = sum(
                p.numel()
                for p in module.parameters(recurse=False)
            )
            trainable = any(
                p.requires_grad
                for p in module.parameters(recurse=False)
            )

            extra: Dict[str, Any] = {}
            if hasattr(module, "in_features"):
                extra["in_features"] = module.in_features
            if hasattr(module, "out_features"):
                extra["out_features"] = module.out_features
            if hasattr(module, "in_channels"):
                extra["in_channels"] = module.in_channels
            if hasattr(module, "out_channels"):
                extra["out_channels"] = module.out_channels
            if hasattr(module, "kernel_size"):
                extra["kernel_size"] = module.kernel_size
            if hasattr(module, "num_heads"):
                extra["num_heads"] = module.num_heads
            if hasattr(module, "embed_dim"):
                extra["embed_dim"] = module.embed_dim

            meta = LayerMeta(
                name=name,
                layer_type=module.__class__.__name__,
                num_parameters=num_params,
                parameter_shapes=param_shapes,
                trainable=trainable,
                depth=depth,
                extra=extra,
            )
            self._layers[name] = meta

        logger.debug(
            "MetadataCollector: catalogued %d layers.", len(self._layers)
        )

    def get(self, layer_name: str) -> Optional[LayerMeta]:
        return self._layers.get(layer_name)

    def get_all(self) -> Dict[str, LayerMeta]:
        return dict(self._layers)

    @property
    def layer_names(self) -> List[str]:
        return list(self._layers.keys())

    def total_parameters(self) -> int:
        return sum(m.num_parameters for m in self._layers.values())

    def model_summary(self) -> Dict[str, Any]:
        total = self.total_parameters()
        trainable = sum(
            m.num_parameters for m in self._layers.values() if m.trainable
        )
        return {
            "total_layers": len(self._layers),
            "total_parameters": total,
            "trainable_parameters": trainable,
            "non_trainable_parameters": total - trainable,
            "layer_types": list({m.layer_type for m in self._layers.values()}),
        }
