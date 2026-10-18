# PROJECT_AUDIT — Concept Drift SWaT Stream Continual Learning

**Date:** 2026-10-18  
**Project:** Continual Learning for Concept-Drifting SWaT Streams Using Supervised and Unsupervised ADWIN Detectors  
**Course:** IOE413 — Streaming Data Analytics (Topic 10, Track B)  
**Auditor:** Senior ML Research Engineer + Streaming Systems Engineer + Academic Paper Reviewer  

---

## 1. Repository Architecture & File Inventory

The repository implements a prequential continual learning pipeline comparing two stream learning methodologies on the Secure Water Treatment (SWaT) industrial sensor dataset:

- **Baseline:** River `HoeffdingTreeClassifier` monitored by a supervised `ADWIN` drift detector tracking 0/1 classification loss, triggering a full classifier reset and replay from a bounded labelled buffer upon drift.
- **Enhanced:** River `ARFClassifier` (Adaptive Random Forest) monitored by an **ADWIN-U-inspired** unsupervised detector tracking streaming descriptive statistics (mean, standard deviation, skewness, kurtosis) across a sliding buffer of pooled standardized features, triggering an adaptation reset and replay upon drift without ground-truth label access during monitoring.

```
concept_drift_swat/
├── archive/
│   └── README.md                         # Dataset download and placement instructions
├── configs/
│   ├── project.yaml                      # Global dataset, stream windows, seeds, infra configs
│   ├── baseline.yaml                     # Hoeffding tree + supervised ADWIN parameters
│   └── enhanced.yaml                     # ARF + ADWIN-U-inspired unsupervised parameters
├── dashboard/
│   ├── dashboards/swat-dashboard.json    # Grafana dashboard definition
│   └── provisioning/datasources/influx.yml# InfluxDB 2.x datasource config
├── reports/
│   └── report.tex                        # Academic research paper (IEEEtran format)
├── results/
│   ├── dataset_inspection.json           # Offline inspection metadata of SWaT CSVs
│   ├── dataset_schema.json               # Schema (51 features, timestamp, label)
│   ├── comparison_table.csv              # Baseline vs Enhanced metrics summary
│   └── plots/                            # Generated publication figures
├── sda_project/                          # Course proposal and base research papers
├── src/
│   ├── data/
│   │   ├── inspect_dataset.py            # Dataset scan, timestamp parsing, column classifier
│   │   ├── preprocess.py                 # Feature vectorization, scaling, stream iterator
│   │   └── schema.py                     # Schema dataclass definitions
│   ├── drift/
│   │   ├── adwin_baseline.py             # Supervised ADWIN on 0/1 prediction error
│   │   └── adwin_u.py                    # ADWIN-U-inspired unsupervised feature-statistics detector
│   ├── evaluation/
│   │   ├── compare.py                    # Comparison table builder
│   │   ├── experiment_runner.py          # Prequential experiment loop and artifact generation
│   │   └── metrics.py                    # Classification, drift detection, and latency metrics
│   ├── models/
│   │   ├── baseline.py                   # BaselineADWINSystem wrapper
│   │   ├── enhanced.py                   # EnhancedAdaptiveSystem wrapper
│   │   └── online_learning.py            # River classifier factory
│   ├── storage/
│   │   └── influx_writer.py              # InfluxDB 2.x post-experiment metric exporter
│   ├── streaming/
│   │   ├── kafka_consumer.py             # Kafka online stream consumer
│   │   ├── spark_processor.py            # Spark Structured Streaming placeholder
│   │   └── stream_producer.py            # Kafka event replay producer
│   └── visualization/
│       └── plot_results.py               # Seaborn/matplotlib publication figures
├── tests/
│   ├── test_drift.py                     # Detector unit tests
│   ├── test_metrics.py                   # Metric computation unit tests
│   └── test_preprocess.py                # Preprocessor and feature isolation tests
├── docker-compose.yml                    # Zookeeper, Kafka, InfluxDB 2.x, Grafana stack
├── run_project.py                        # CLI entry point (inspect, experiment, plot, infra, demo)
├── requirements.txt                      # Pinned Python package dependencies
├── DEMO_GUIDE.md                         # 10-minute demo workflow
├── PROJECT_STATUS.md                     # Component implementation status
└── README.md                             # Quickstart and project documentation
```

---

## 2. Environment Audit

| Component | Status | Details |
|-----------|--------|---------|
| **Operating System** | Verified Active | Windows 11 (PowerShell) |
| **Python** | Verified Active | Python 3.12.6, virtualenv `.venv` created and populated |
| **Dependencies** | Verified Active | `river==0.22.0`, `scikit-learn==1.7.2`, `pandas==2.3.3`, `numpy==2.3.4`, `scipy==1.16.2`, `psutil==7.2.2` |
| **Java** | Verified Available | JDK 21.0.1 LTS (`C:\Program Files\Java\jdk-21\bin\java.exe`) |
| **Docker Engine** | Verified Active | Docker Desktop 29.1.3 running; Linux containers active |
| **LaTeX** | Not Available Locally | No native `pdflatex` on Windows PATH |
| **SWaT Dataset** | External / Git-Ignored | `archive/merged.csv` (407 MB) not present in local workspace; `.gitignore` excludes `*.csv` |

---

## 3. Verified Working Components

1. **Core ML Library Integration:** `river` 0.22.0 Hoeffding Tree, Adaptive Random Forest, and ADWIN detectors successfully imported and instantiated.
2. **Supervised Drift Detector:** `src/drift/adwin_baseline.py` detects error-rate shifts correctly in unit tests.
3. **Unsupervised Detector (ADWIN-U-inspired):** `src/drift/adwin_u.py` operates strictly without labels on feature buffers.
4. **Metric Algorithms:** Basic classification metrics (accuracy, precision, recall, F1) and delay calculations run without crash.
5. **Docker Infrastructure Config:** `docker-compose.yml` configures valid Zookeeper, Kafka 7.6.1, InfluxDB 2.7, and Grafana 11.2.0 services.

---

## 4. Known Defects & Critical Bugs Identified

### Defect 1: Data Leakage in Preprocessing (High Priority)
- **Problem:** In `src/evaluation/experiment_runner.py`, `fit_preprocess_state` fits mean and standard deviation on an initial slice of 5,000 events (`fit_start` to `fit_start + 5000`). In full mode (`start_index=0`), evaluation predictions start at row 0. As a result, predictions for rows 0 through 4,999 use scaling statistics computed from future rows up to row 4,999.
- **Impact:** Violates the fundamental streaming principle that preprocessing must not use future information.

### Defect 2: Smoke-Run Warm-Up Bug (High Priority)
- **Problem:** In `src/models/baseline.py` and `enhanced.py`, detector activation is guarded by `if event_index >= self.warmup:` where `self.warmup = 5000`. In smoke mode, `smoke_start_index` is set to `1,385,000`. On the very first event processed in smoke mode, `event_index = 1,385,000 >= 5000`, causing the detector to activate immediately on stream step 0 with zero warm-up.
- **Impact:** Detector is uncalibrated at start of smoke experiment; warm-up guarantee is bypassed.

### Defect 3: Probability Metrics Computed on Hard Binary Decisions (High Priority)
- **Problem:** In `src/evaluation/metrics.py`, `roc_auc_score(y_true, y_pred)` and `average_precision_score(y_true, y_pred)` receive integer binary predictions `y_pred` (0 or 1) rather than continuous probability scores `y_score`.
- **Impact:** ROC-AUC and Average Precision (PR-AUC) metrics are degenerated to step-function approximations rather than true ranking curves, producing misleading numbers.

### Defect 4: Asymmetric Latency Boundaries (Medium Priority)
- **Problem:** In `src/models/baseline.py`, `latency_ms` measures only `vectorize_event` and `predict`. In `src/models/enhanced.py`, `latency_ms` includes `vectorize_event`, `predict`, `detector.update`, and `on_drift`. Neither includes `learn_one`.
- **Impact:** Latency and throughput comparisons between baseline and enhanced are scientifically invalid and unfairly favor the baseline.

### Defect 5: Windows Platform Incompatibilities (Medium Priority)
- **Problem A:** `src/evaluation/experiment_runner.py` executes `import resource` and `resource.getrusage(resource.RUSAGE_SELF)`, which fails immediately on Windows with `ModuleNotFoundError: No module named 'resource'`.
- **Problem B:** `run_project.py` calls `latest.symlink_to(...)`, which fails on Windows with privilege violation `WinError 1314` unless Developer Mode / admin is enabled.
- **Impact:** Experiment runner and CLI fail on Windows.

### Defect 6: Drift Alarm Accounting & Ambiguity (Medium Priority)
- **Problem:** Drift matching only tracked matched delays and a single reference boundary. Enhanced ADWIN-U alarms were not broken down into raw vs cooldown-filtered, nor normalized per 10k events.
- **Impact:** Paper narrative risked conflating raw statistical alarms with physical concept drifts.

### Defect 7: Spark Structured Streaming Placeholder (Low Priority)
- **Problem:** `src/streaming/spark_processor.py` contains only a stub with `logger.info("Spark job placeholder...")`.
- **Impact:** Cannot be claimed as an operational end-to-end Spark streaming pipeline.

### Defect 8: Unit Tests Hard-Depend on Missing 407 MB Dataset (Medium Priority)
- **Problem:** `tests/test_preprocess.py` attempts to open `archive/merged.csv` directly without checking if the file exists. When run in a freshly cloned repository, tests fail with `FileNotFoundError`.
- **Impact:** Automated CI/CD and developer testing fail out of the box.

---

## 5. Proposed Fixes & Engineering Roadmap

1. **Phase 2 (Preprocessing & Warm-Up):**
   - Implement explicit stream step vs global index tracking.
   - Enforce causal scaling: statistics computed strictly from pre-evaluation warm-up prefix or causal online updating.
   - Guard detector activation by `stream_step >= warmup_steps` so warm-up is guaranteed in both full and smoke modes.
   - Exclude warm-up prefix events from scored evaluation metrics.

2. **Phase 3 (Probability Metrics):**
   - Update model interfaces to extract continuous probability `y_score = P(Attack=1 | x)` via `predict_proba_one`.
   - Record `y_score` before model updating.
   - Pass continuous scores to `roc_auc_score` and `average_precision_score`.
   - Safely handle single-class evaluation windows by reporting `NaN` / `None` without exception.

3. **Phase 4 (Prequential Evaluation):**
   - Standardize causal loop: `read -> scale -> predict(score & decision) -> record -> detector update -> drift response -> learn -> buffer`.
   - Verify unsupervised detector never touches ground truth `y_true`.

4. **Phase 5 (Fair Benchmarking):**
   - Instrument granular timing: `t_pred`, `t_detector`, `t_drift_response`, `t_learn`, `t_total`.
   - Compute p50, p95, p99 across all stages.
   - Use cross-platform `psutil` for process memory tracking with clear reporting disclaimers.

5. **Phase 6 (Drift Evaluation):**
   - Compute total alarms, matched alarms, unmatched (false) alarms, alarm rate per 10k events, and resets.
   - Clarify SWaT Normal-to-Attack transition as an evaluation reference proxy.
   - Document pooled feature statistics nature of the ADWIN-U-inspired detector.

6. **Phase 7 (Stream Order & Timestamps):**
   - Validate timestamp parsing, order monotonicity, and gap detection.
   - Preserve original source row order with documentation.

7. **Phase 8-12 (Testing, Infrastructure & Reporting):**
   - Create synthetic stream generator fixture so test suite and smoke demonstrations run deterministically even when 407 MB `archive/merged.csv` is absent.
   - Fix Windows symlink and resource issues.
   - Accurately update `report.tex`, `README.md`, `PROJECT_STATUS.md`, and `DEMO_GUIDE.md`.
