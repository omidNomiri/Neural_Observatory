# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0b1] - 2026-07-03

### Added

- Initial beta release of Neural Observatory.
- Core monitoring for activations, gradients, and parameters.
- Analyzers: Dead Neuron, Gradient Health, Activation Statistics, Anomaly Detection, Parameter Health.
- Reporters: Console (ANSI), JSON, and HTML.
- Storage backends: In-Memory and SQLite.
- Compatibility with `inplace=True` operations via tensor-level hooks.

## [0.1.0] - 2026-07-10

### Added

- Initial stable release.
- Core monitoring for activations, gradients, and parameters.
- Analyzers: Dead Neuron, Gradient Health, Activation Statistics, Anomaly Detection, Parameter Health.
- New EmbeddingDriftAnalyzer using cosine similarity.
- Full torch.compile compatibility.

## [0.2.0] - 2026-07-16

### Added

- New EmbeddingDriftAnalyzer using cosine similarity to detect semantic shifts.
- New AttentionHealthAnalyzer to detect attention collapse via entropy.
- New NeuralCollapseAnalyzer to monitor within-class variance in penultimate layers.
- Deep compatibility testing with torch.compile.
- Ability to pass targets to Observatory.step() for advanced analysis.

## [0.3.0] - 2026-07-31

### Added

- Automated Sphinx documentation hosted on GitHub Pages.
- Add Vit test.

## [0.4.0] - 2026-08-7

## Fixed

- AttentionHealthAnalyzer: Warns users if nn.MultiheadAttention is called with need_weights=False.
- NeuralCollapseAnalyzer: Correctly extracts the [CLS] token (index 0) for 3D NLP features instead of averaging.

## Added

- Observatory.step(): Auto-increments the step counter if no argument is provided. Warns if sampling rates are bypassed.

## Performance

- SQLiteStore: Implemented batch commits during report generation to drastically reduce disk I/O bottleneck.
