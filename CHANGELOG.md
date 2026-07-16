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
