# 🔭 Neural Observatory

**A tool for neural network observability, diagnostics, and analysis framework for PyTorch.**

A scientific instrumentation framework that allows researchers and engineers to inspect, analyze, monitor, debug, and understand the internal behavior of neural networks during training and inference.

## Features

### Core Diagnostics

- **Dead Neuron Detection** — Identify inactive neurons that contribute nothing to learning
- **Gradient Health Monitoring** — Detect vanishing and exploding gradients
- **Activation Analysis** — Statistical summaries and anomaly detection
- **Parameter Health Tracking** — Monitor weight drift and update magnitudes
- **Embedding Drift Detection** — Track semantic shifts in NLP models using cosine similarity
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
| SQLite | ✅ | ✅ | ✅ | ❌ |
| HTML Report | ✅ | ✅ | ✅ | ❌ |

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
num_epochs = 5
global_step = 0

for epoch in range(num_epochs):
    for X, y in train_loader:
        # IMPORTANT: Call step() *before* the forward pass
        obs.step(step=global_step, epoch=epoch)
        
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        global_step += 1

# Generate diagnostics
obs.stop()
report = obs.report()
obs.console_report()

# Release storage resources (also done automatically by `with Observatory(...)`)
obs.close()
```

## Installation

```bash
pip install neural-observatory
```

## Documentation

### Configuration

```python
from neural_observatory import Observatory, ObservatoryConfig

config = ObservatoryConfig(
    activation_sample_rate=1,  # Collect every forward pass
    gradient_sample_rate=1,    # Collect every backward pass
    parameter_sample_rate=10,  # Snapshot parameters every N steps
    max_observations=1000,     # Circular buffer size per layer
    stats_only_mode=False,     # Keep raw tensors (needed for Embedding Drift)
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
2. **CPU Movement** — Snapshots are always stored in CPU memory (`store_on_cpu` is a deprecated no-op)
3. **Circular Buffers** — Fixed-size buffers prevent unbounded memory growth
4. **Deep Copying** — Tensor data is safely copied to prevent inplace-op corruption
5. **Stats-Only Mode** — Discard raw tensors, keep only statistics

## Testing

```bash
pytest tests/
pytest --cov=neural_observatory tests/
```

## Roadmap

The project evolves version by version along three phases:

### 0.5.x — Foundation Stabilization

- [x] **0.5.2** — Core correctness hotfix: observation identity, idempotent persistence, memory-safe snapshots, hook lifecycle
- [ ] **0.5.3** — Metric correctness: true parameter update ratio, gradient RMS, Conv neuron axis, normalized attention entropy
- [ ] **0.5.4** — Layer semantics: `LayerDescriptor`, semantic axes, stable layer/parameter IDs, module graph
- [ ] **0.5.5** — Memory & performance: memory budgets, capture modes, overhead benchmarks
- [ ] **0.5.6** — API/plugin hardening: collector/reporter registries, EventBus integration, plugin lifecycle

### Phase 1 — Model & Training Intelligence

- [ ] **0.6.0** — Model intelligence: model/layer graph, layer role classification
- [ ] **0.7.0** — Training timeline: `Run → Step → Layer → Observation`, training metadata (loss, LR, throughput)
- [ ] **0.8.0** — Diagnostic engine: evidence graph, temporal correlation, root-cause candidates
- [ ] **0.9.0** — Next-run recommendation engine
- [ ] **1.0.0** — Phase 1 complete: Training Observatory
- [ ] **1.1–1.4** — Maturity: multi-run comparison, dashboard, recommendation validation, DDP foundation

### Phase 2 — Neuron Observatory

- [ ] **2.0.0** — Neuron identity & ledger
- [ ] **2.1.0** — Exact weight history (chunked tensor storage beside SQLite metadata)
- [ ] **2.2.0** — Neuron dynamics
- [ ] **2.3.0** — Replay & forensics
- [ ] **2.4.0** — Data-aware neuron analysis
- [ ] **2.5.0** — Massive storage (`SUMMARY` / `TRACE` / `EXACT` modes)
- [ ] **2.6.0** — Causal interventions (ablation, suppression, effect measurement)
- [ ] **2.9.0** — Phase 2 release candidate

### Phase 3 — Neuron Intelligence

- [ ] **3.0.0** — Neuron profiles, spatial localization, LLM explanation of structured evidence, intervention planning

From 0.6.0 onward, all data is organized around one chain:

```text
Run → Step → Layer → Parameter/Neuron → Observation → Evidence → Diagnosis → Intervention
```
