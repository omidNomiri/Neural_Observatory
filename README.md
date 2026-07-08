# 🔭 Neural Observatory

**A tool for neural network observability, diagnostics, and analysis framework for PyTorch.**

A scientific instrumentation framework that allows researchers and engineers to inspect, analyze, monitor, debug, and understand the internal behavior of neural networks during training and inference.

## Features

### Core Diagnostics

- **Dead Neuron Detection** — Identify inactive neurons that contribute nothing to learning
- **Gradient Health Monitoring** — Detect vanishing and exploding gradients
- **Activation Analysis** — Statistical summaries and anomaly detection
- **Parameter Health Tracking** — Monitor weight drift and update magnitudes
- **Numerical Anomaly Detection** — Flag NaN/Inf and activation spikes

### Design Philosophy

- **Non-invasive** — Uses PyTorch's official hooks, leaves no trace
- **Extensible** — Plugin-based analyzer and reporter system
- **Memory-efficient** — Configurable sampling, circular buffers, stats-only mode
- **Production-capable** — Multiple storage backends and output formats

---

## Compatibility Matrix

| Feature | CPU | CUDA | MPS | DDP |
| ----------- | --- | ---- | --- | --- |
| Activations | ✅ | ✅ | ✅ | ❌ |
| Gradients | ✅ | ✅ | ✅ | ❌ |
| SQLite | ✅ | ✅ | ✅ | ✅ |
| HTML Report | ✅ | ✅ | ✅ | ✅ |

## Version Compatibility

| PyTorch | Status |
| ------- | ------------ |
| 2.1 | ✅ |
| 2.2 | ✅ |
| 2.3 | ✅ |
| Nightly | Experimental |

## Supported Layers

Currently monitors the following standard PyTorch layer types (custom layers are also supported if they output tensors):

- `Linear`
- `Conv` (Conv1d, Conv2d, Conv3d, ConvTranspose)
- `Embedding`
- `LayerNorm` & `BatchNorm`
- `LSTM` / `GRU`
- `Transformer` & `MultiheadAttention`

## Limitations

To ensure stability, please be aware of the following current limitations:

- No DDP (Distributed Data Parallel) support yet
- No FSDP support
- TorchScript not tested
- `torch.compile` compatibility is experimental
- Quantized models are only partially supported

---

## Quick Start

```python
from neural_observatory import Observatory
import torch.nn as nn

# Create model
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)

# Initialize and start monitoring
obs = Observatory(model)
obs.watch()

# Training loop
for epoch in range(num_epochs):
    for X, y in train_loader:
        # IMPORTANT: Call step() *before* the forward pass
        obs.step(step=global_step, epoch=epoch)
        
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

# Generate diagnostics
obs.stop()
report = obs.report()
obs.console_report()
```

## Installation

```bash
pip install neural-observatory
```

## Documentation

### Architecture

Neural Observatory is built in modular layers:

```text
Model Execution
     ↓
Hooks (observation points)
     ↓
Collectors (raw data capture)
     ↓
Storage (in-memory/disk)
     ↓
Analysis Engine (compute diagnostics)
     ↓
Reporting (human/machine output)
```

### Configuration

```python
from neural_observatory import Observatory, ObservatoryConfig

config = ObservatoryConfig(
    activation_sample_rate=1,  # Collect every forward pass
    gradient_sample_rate=1,    # Collect every backward pass
    parameter_sample_rate=10,  # Snapshot parameters every N steps
    max_observations=1000,     # Circular buffer size per layer
    stats_only_mode=False,     # Keep raw tensors
    store_on_cpu=True,         # Move tensors off GPU
)

obs = Observatory(model, config=config)
```

### Custom Analyzers

Create custom analyzers by subclassing `BaseAnalyzer`:

```python
from neural_observatory import BaseAnalyzer

class CustomAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "my_custom_analyzer"

    def analyze(self, observations):
        results = []
        activations = observations.get("activations", {})
        
        for layer_name, obs_list in activations.items():
            result = self._make_result(
                analyzer_name=self.name,
                layer_name=layer_name,
                severity=0.5,
                metrics={"custom_metric": 42.0},
                warnings=["Custom warning"],
            )
            results.append(result)
        
        return results

obs.register_analyzer(CustomAnalyzer(config=obs.config))
```

## Memory Safety

Neural Observatory implements strict memory safety:

1. **No Graph Retention** — All tensors are detached immediately after observation
2. **CPU Movement** — Tensors are moved to CPU before storage (configurable)
3. **Circular Buffers** — Fixed-size buffers prevent unbounded memory growth
4. **Deep Copying** — Tensor data is safely copied to prevent inplace-op corruption
5. **Stats-Only Mode** — Discard raw tensors, keep only statistics

## Testing

```bash
pytest tests/
pytest --cov=neural_observatory tests/
```

## Roadmap

- [x] Core activation, gradient, and parameter monitoring
- [x] Dead neuron and vanishing/exploding gradient detection
- [x] Inplace operation compatibility (`inplace=True`)
- [x] Memory-safe tensor copying (prevent silent data corruption)
- [x] SQLite and In-Memory storage backends
- [x] Add embedding drift detection over training epochs
- [ ] Add support for Distributed Data Parallel (DDP) and FSDP
- [ ] Deep compatibility testing with `torch.compile`
- [ ] Implement attention weight visualization tools
- [ ] Add Neural Collapse detection metrics
- [ ] Write integration tests for Vision Transformers (ViT) and LLMs
- [ ] Set up automated Sphinx documentation hosting
