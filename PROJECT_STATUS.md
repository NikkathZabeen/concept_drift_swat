# PROJECT_STATUS — Continual Learning for Concept-Drifting SWaT Streams

**Last updated:** 2026-10-18 (Complete end-to-end audit, correctness fixes, regression testing, and smoke run execution)

---

## 1. Environment & Dependencies

| Component | Status | Details |
|-----------|--------|---------|
| **Operating System** | Verified Active | Windows 11 (PowerShell) |
| **Python** | Verified Active | Python 3.12.6 (virtual environment `.venv` configured) |
| **Core Libraries** | Verified Active | `river==0.22.0`, `scikit-learn==1.7.2`, `pandas==2.3.3`, `numpy==2.3.4`, `scipy==1.16.2`, `psutil==7.2.2` |
| **Test Suite** | Verified Active | `pytest==8.4.2` — 23 / 23 tests passing in ~6s |
| **Java** | Verified Available | Oracle JDK 21.0.1 LTS (`C:\Program Files\Java\jdk-21\bin\java.exe`) |
| **Docker Engine** | Verified Active | Docker Desktop 29.1.3 active with Linux containers daemon |
| **LaTeX Tooling** | Not Available Locally | No local `pdflatex` on Windows PATH; `report.tex` inspected and validated for syntax |
| **PySpark** | Blocked / Optional | Preserved as documented future work (pure-Python River pipeline is primary) |

---

## 2. Component Implementation Status

| Component | Implementation Status | Verification Status |
|-----------|----------------------|---------------------|
| **Causal Preprocessing** | Implemented (Isolated `OnlineStandardScaler` per model instance) | **Verified** — Zero retrospective leakage; independent instances; 23 tests pass |
| **Smoke Warm-Up Semantics** | Implemented (`stream_step` guard, local coordinate mapping) | **Verified** — Inactive during warm-up; tested on high global index 1,385,000 |
| **Scored Window Distribution** | Implemented (Pre-run class balance check) | **Verified** — 4,000 scored events (1,000 Normal, 3,000 Attack); non-zero Attack metrics |
| **Baseline ADWIN System** | Implemented (HT + Supervised ADWIN + Reset) | **Verified** — Predict-before-learn prequential protocol verified |
| **Enhanced System** | Implemented (ARF + ADWIN-U-inspired + Replay) | **Verified** — Operates strictly without labels in drift monitor |
| **Probability Metrics** | Implemented (`predict_proba_one` -> ROC-AUC/PR-AUC) | **Verified** — Tested against scikit-learn; safe single-class handling |
| **Granular Latencies** | Implemented (Pred, Detector, Drift, Learn, Total) | **Verified** — Synchronized timing boundaries; p50, p95, p99 percentiles |
| **Portable Memory Profiler** | Implemented (`psutil` Process RSS + Peak Working Set) | **Verified** — Cross-platform on Windows/Linux; no Unix-only crashes |
| **Dataset Inspection** | Implemented (`inspect_dataset.py`) | **Verified** — Offline schema JSON generated |
| **Synthetic Fixture Fallback** | Implemented (`generate_synthetic_swat_csv`) | **Verified** — Self-contained testing without 407 MB external download |
| **CLI & Compare Tool** | Implemented (`run_project.py` with `compare`) | **Verified** — Standalone comparison without retraining |
| **Visualization Module** | Implemented (`plot_results.py`) | **Verified** — 10 figures (PNG and PDF) generated in `results/plots/` |
| **Docker Compose Services** | Configured (`docker-compose.yml`) | **Verified Syntax** — Validated via `docker compose config` |
| **InfluxDB & Grafana** | Implemented (`influx_writer.py`, dashboards) | **Verified** — Provisioned datasource and dashboard definitions |
| **Kafka Producer / Consumer** | Implemented (`stream_producer.py`, `kafka_consumer.py`)| **Verified** — Local serialization & stream step handling verified |
| **Spark Structured Streaming**| Placeholder (`spark_processor.py`) | **Documented Future Work** — Documented in report; not claimed as executed |

---

## 3. Dataset Specifications

### Schema (53 Columns)
- **`Timestamp`**: Format `DD/MM/YYYY HH:MM:SS AM/PM` (whitespace-stripped).
- **`Normal/Attack`**: Label column (`Normal -> 0`, `Attack -> 1`).
- **51 Physical Features**:
  - 25 continuous sensors: `FIT*` (flow), `LIT*` (level), `AIT*` (analyzer), `PIT*` (pressure), `DPIT*` (differential pressure).
  - 26 discrete actuators: `MV*` (motor valves), `P*` (pumps), `UV*` (de-chlorination).

### Reference Transition & Canonical Stream
- Primary dataset file: `archive/merged.csv` (1,441,719 rows; Normal: 1,387,098 rows; Attack: 54,621 rows).
- Monolithic transition index: **1,387,098** (Normal $\rightarrow$ Attack).
- Scientific caveat: The label boundary serves as an evaluation reference proxy, not empirical proof of the exact hydrodynamic drift onset.
- Verification fixture: 6,000 rows (2,000 Normal, 4,000 Attack) generated with transition at index 2,000 for reproducible CI testing and smoke evaluation.

---

## 4. Key Correctness Fixes Completed

1. **Preprocessing Leakage:** Eliminated retrospective fitting on test rows; implemented isolated `OnlineStandardScaler` with Welford accumulators per model instance.
2. **Warm-Up Bypass & Index Semantics:** Replaced global index comparison with local `stream_step` comparison, preventing immediate detector activation in smoke runs; properly translated global references to local run steps.
3. **Attack Detection Window Balance:** Repositioned synthetic smoke evaluation window across the reference label transition, ensuring scored evaluations contain both Normal (1,000) and Attack (3,000) observations and producing genuine precision, recall, and F1 scores.
4. **Continuous Score Ranking:** Extracted posterior probabilities $P(\text{Attack}=1 \mid x)$ via `predict_proba_one`; eliminated degenerate binary ROC-AUC/PR-AUC calculations.
5. **Platform Robustness:** Replaced Unix `resource` with `psutil` (monitoring RSS and peak working set); replaced brittle symlinks with Windows directory junction fallback.
6. **Drift Accounting:** Separated raw statistical alarms from cooldown-filtered model resets; added false-alarm rates per 10k events with documented operational definitions.

