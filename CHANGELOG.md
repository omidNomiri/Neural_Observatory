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

## [0.4.1] - 2026-08-15

## Fixed

- Critical: HookManager now correctly attaches hooks to the underlying original module (_orig_mod) when using torch.compile, preventing silent monitoring failures.
- Critical: Fixed duplicate close() method in SQLiteStore that could cause uncommitted data loss.
Observatory.step() now safely guards against being called before watch() or after stop().

## Changed

- Introduced BaseStore ABC to standardize the storage backend interface. MemoryStore and SQLiteStore now inherit from it, replacing fragile hasattr checks.

## [0.5.0] -  2026-08-16

## Added

- Thread-Safety: Added threading.RLock to Registry, EventBus, and BaseCollector to prevent crashes in multi-threaded environments (e.g., Pipeline Parallelism).
- Context Manager: You can now use with Observatory(model) as obs: for cleaner lifecycle management.

## Changed

- Attention Analysis: Replaced the fragile _attn_weights name suffix with a robust is_attention_weights metadata flag.
- Embedding Detection: Improved heuristic in EmbeddingDriftAnalyzer to check parameter names alongside shapes.

## Fixed

- Console Reporter: Long layer names are now safely truncated with ... to maintain table alignment.
- Attention Buffer: Fixed a bug where AttentionHealthAnalyzer would silently skip analysis if the layer buffer contained mixed tensors (output + weights).

## [0.5.1] - 2026-09-8

## Added

- Comprehensive test suite for core framework components (LifecycleManager, EventBus, Registry, SQLiteStore).
- Verified thread-safety and exception handling in core systems.

## [0.5.2] - 2026-11-18

### Fixed

- Persistence: every observation carries a session `run_id` plus a monotonic per-collector sequence number; SQLite and In-Memory stores deduplicate on `(collection, layer, step, epoch, seq, run_id)`, so repeated `report()` calls are idempotent, distinct observations at the same step are kept, and two runs sharing one database never overwrite each other.
- SQLite: legacy databases are migrated in place to schema version 2 (`PRAGMA user_version`), adding the `seq`/`run_id` identity columns and collapsing coarse-key duplicates during the rebuild.
- SQLite: re-running the migration preserves existing `run_id` values, and the version stamp commits in the same transaction as the table rebuild, so an interrupted migration is retried without losing run identities.
- SQLite: removed the duplicate `close()` definition that shadowed the committing implementation and silently lost uncommitted rows.
- Storage lifecycle: `Observatory.close()` commits and closes the store; context-manager exit and explicit `close()` no longer leak connections, and `stop()` leaves the store readable for `report()`.
- SQLite: `close()` is idempotent and removes temporary databases created with `db_path=None`; user-provided database files are never deleted.
- Hooks: bare leaf models (e.g. a single `nn.Linear`) are now monitored as `root`; container roots are still skipped to avoid double collection.
- Hooks: gradient hooks are registered once per output tensor and removed on `detach()`, preventing duplicate observations and handle accumulation across repeated forwards.
- Hooks: gradient hooks read the step counters at backward time, so a tensor reused across steps attributes each gradient to the step that produced it.
- Hooks: when several monitored layers return the same tensor object, every associated layer receives the gradient observation (still one hook per tensor).
- Hooks: a failure during `attach()`/`watch()` rolls back all partially attached hooks instead of leaving the model half-instrumented.
- Snapshots: `_safe_numpy()` preserves float64/float16 dtypes, casts bfloat16 to float32, and rejects complex tensors instead of silently dropping the imaginary part.
- Collectors: zero-element tensors/parameters no longer raise during statistics collection.
- Neural Collapse: targets are serialized via `.tolist()` so SQLite metadata stays JSON-safe and later mutations of the user's tensor cannot leak into stored observations.
- Neural Collapse: targets are reset on every `step()`; a step that provides no labels no longer inherits the previous step's targets.
- Concurrency: collector reads, SQLite access, and EventBus subscription changes are now lock-protected.
- Lifecycle: `stop()` is idempotent and `end_reporting()` tolerates a `stop()` that happened mid-report instead of masking the report outcome.
- Parameters: repeated `report()` calls no longer re-snapshot the same `(step, epoch)`; `clear()` resets the snapshot guard.
- Parameters: baselines keep their original dtype, so tiny float64 updates are no longer erased by a float32 downcast when computing drift.
- Parameters: `clear()` keeps the initialization baselines, so `drift_from_init` stays measured against initialization after clearing observation buffers.
- In-Memory store: eviction keeps the dedup key set bounded; `get(limit=0)` and `latest(0)` return empty results consistently across backends.

### Changed

- `store_on_cpu` is now a documented deprecated no-op: snapshots always live in CPU memory. Setting it to `False` emits a `UserWarning`.
- `Observatory.step()` docstring now matches the documented convention: call it *before* the forward pass.
- `Observatory.clear()` docstring states the contract: in-memory buffers are emptied, persisted history is kept, and sequence numbers never restart.
- Gradient timing contract documented: gradients are stamped with the step active at backward time; finish each step's backward before calling `step()` again, and call `step()` per micro-batch when accumulating gradients. Delayed backward across steps is explicitly unsupported.
- SQLite `get()`/`get_collection()` order by `(step, seq, run_id)` for deterministic multi-run retrieval and accept an optional `run_id` filter.
- README compatibility matrix no longer claims DDP support for SQLite/HTML Report; DDP is unsupported across the board.
- README roadmap replaced with the phased version plan (0.5.x stabilization → Phase 1 model intelligence → Phase 2 neuron observatory → Phase 3 neuron intelligence).
- All modules touched since 0.5.1 formatted with Black and isort (profile: black).

### Added

- End-to-end training regression tests: full training loops on both storage backends, inplace ops, `no_grad` inference, `torch.compile`, stats-only mode, and round-trip persistence equality.
- Regression suites for run identity, persistence identity/migration, store lifecycle, targets serialization, hook correctness, snapshot fidelity, concurrency safety, and parameter snapshot guards (123 tests passing, 1 CUDA-only skip).
