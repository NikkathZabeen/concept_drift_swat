# CHANGELOG — Continual Learning for Concept-Drifting SWaT Streams

All notable changes, bug fixes, and methodological improvements to this repository are documented in this file.

---

## [1.2.0] — 2026-10-18

### Pipeline Discrepancy Resolution & Audit Completion
- **Strictly Causal Isolated Online Preprocessing (Priority 1):**
  - Eliminated shared preprocessor state passed by `run_experiment`; each system (`BaselineADWINSystem` and `EnhancedAdaptiveSystem`) now owns its own independent `OnlineStandardScaler`.
  - Enforced strict causal ordering: (1) `transform_one`, (2) `predict_proba`, (3) detector update, (4) drift adaptation, (5) `learn_one`, (6) `update_one`.
  - Added regression test `test_online_scaling_strictly_causal_no_retrospective_fit` verifying that future observations cannot alter past predictions and that models maintain isolated scaler states.
- **Index Semantics and Coordinate Systems (Priority 2):**
  - Mapped global reference indices ($G_{\text{ref}}$) into run-local steps ($S_{\text{ref}} = G_{\text{ref}} - G_{\text{start}}$) for recovery and delay metrics.
  - Flagged out-of-window references explicitly as `outside_evaluation_window` without miscalculating delay or crashing with index errors.
  - Added unit test `test_smoke_window_index_semantics_and_coordinate_mapping` verifying a run starting at global index 1,385,000 and containing reference 1,387,098.
- **Meaningful Attack Detection & Window Balancing (Priority 3):**
  - Re-anchored the synthetic verification fixture to place the concept drift transition at row 2,000 (2,000 Normal, 4,000 Attack).
  - Evaluated 4,000 scored events (1,000 Normal, 3,000 Attack) across the reference boundary, replacing the degenerate 3,500-all-Normal window with genuine, non-zero Attack Recall (99.47% vs 98.77%), Precision (100%), and F1 (0.9973 vs 0.9938).
  - Added pre-run validation (`validate_scored_window_labels`) inspecting and logging upcoming class distribution before model execution.
- **Validated Continuous Metrics & Memory Profiling (Priority 4):**
  - Confirmed $P(\text{Attack})$ is recorded before model learning.
  - Recomputed true continuous ROC-AUC (0.9998) and PR-AUC (0.9999).
  - Profiled memory using `psutil` measuring RSS and peak working set (`peak_wset` on Windows) with explicit documentation of measurement methodology.
- **Drift Metrics & Operational False Alarm Accounting (Priority 5):**
  - Added signed delay metrics (negative = early alarm, positive = late alarm).
  - Defined operational false alarms as alarms occurring outside $\pm \text{tolerance}$ of in-window references.
  - Separated raw alarms (14), filtered triggers (7), and model resets (7).
- **Academic Paper & Artifact Reconciled (Priority 6):**
  - Updated `reports/report.tex` Table I with exact newly measured numbers from the completed run.
  - Replaced unsupported statements with measured findings; disclosed synthetic provenance and Spark placeholder status.
  - Regenerated comparison table CSV, summaries, metadata, and publication plots.
- **Test Suite Expanded (Priority 7):**
  - Expanded test suite to 23 / 23 passing tests.

## [1.1.0] — 2026-10-18


### Critical Correctness Fixes
- **Eliminated Preprocessing Data Leakage (Phase 2):**
  - Replaced retrospective slice-fitting with `OnlineStandardScaler` implementing Welford's algorithm for causal running mean and variance computation.
  - Implemented explicit separation between warm-up prefix events and scored evaluation events.
  - Ensured evaluation predictions are normalized strictly using statistics accumulated prior to the observation being scored.
  - Added unit test `test_online_scaler_no_future_leakage` verifying that future observations cannot alter past normalized representations.
- **Fixed Smoke-Run Warm-Up Bypass Bug (Phase 2):**
  - Replaced faulty global row index check (`event_index >= warmup`) with local stream progress counter (`stream_step < warmup_steps`).
  - Guaranteed that drift detectors remain inactive during the intended calibration window regardless of the global starting row index.
  - Added regression test `test_warmup_semantics_smoke_mode_bug_fix`.
- **Corrected Probability-Based Ranking Metrics (Phase 3):**
  - Updated `predict_proba` in both `BaselineADWINSystem` and `EnhancedAdaptiveSystem` to extract continuous class posterior probabilities $P(\text{Attack}=1 \mid x)$ via River's `predict_proba_one`.
  - Replaced degenerate binary step-function ROC-AUC and PR-AUC calculations with true continuous probability evaluations.
  - Implemented safe handling of single-class windows, returning `None` rather than raising `ValueError` or fabricating approximations.
  - Added unit test comparing output against scikit-learn's `roc_auc_score` and `average_precision_score`.
- **Standardized Prequential Evaluation Sequence (Phase 4):**
  - Enforced strictly causal evaluation order: (1) causal scaling, (2) predict class and probability, (3) record metrics, (4) update detector, (5) handle adaptation/reset, (6) learn from true label.
  - Verified that the unsupervised ADWIN-U-inspired detector has zero access to ground-truth labels during monitoring.
- **Harmonized Latency & Memory Measurements (Phase 5):**
  - Synchronized measurement boundaries across both baseline and enhanced models: isolating prediction latency, detector latency, adaptation reset latency, and learning latency.
  - Added p50, p95, and p99 percentile latency reporting.
  - Replaced Unix-only `resource.getrusage` with cross-platform `psutil.Process().memory_info().rss`.
- **Enhanced Drift Detection Accounting (Phase 6):**
  - Added tracking of raw alarm events vs cooldown-filtered model resets.
  - Added metrics for matched reference events, unmatched alarms, and false alarm rate per 10,000 processed events.
  - Explicitly documented the detector as **ADWIN-U-inspired / ADWIN-U-style** monitoring pooled feature descriptive statistics.
- **Stream Ordering and Timestamp Validation (Phase 7):**
  - Added timestamp parsing, monotonicity validation, and gap detection (>2s).
  - Preserved canonical CSV source row order with documentation.
- **Windows Platform Portability (Phase 8):**
  - Fixed Windows privilege failure on symlink creation in `run_project.py` by implementing Directory Junction (`_winapi.CreateJunction`) and pointer fallback.
  - Added standalone `compare` subcommand to `run_project.py` to compare saved artifacts without retraining.
- **Synthetic Test Fixture Fallback (Phase 8):**
  - Added `generate_synthetic_swat_csv` conforming to the exact 53-column schema, enabling self-contained verification when the 407 MB external dataset is absent.
- **Academic Paper Reconciled (Phase 9):**
  - Updated `reports/report.tex` to accurately reflect leak-free methodology, measured performance comparison, continuous ranking metrics, and honest infrastructure disclosures.
- **Expanded Test Suite (Phase 11):**
  - Expanded test suite from 8 to 20 passing unit tests.
