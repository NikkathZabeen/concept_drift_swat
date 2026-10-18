# Continual Learning for Concept-Drifting SWaT Streams

[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)]()
[![Framework](https://img.shields.io/badge/stream%20ML-River%200.22.0-orange.svg)]()

Academic Project for **IOE413 — Streaming Data Analytics (Topic 10, Track B)**.  
End-to-end streaming ML pipeline for the [SWaT dataset](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system) comparing a **supervised ADWIN baseline** with an **unsupervised ADWIN-U-inspired** continual learning system.

---

## 1. Research Question

> *Can an unsupervised feature-statistics drift detector adapt continuous stream classifiers as effectively as supervised error monitors while maintaining predictive quality, detection reliability, and acceptable resource use?*

---

## 2. Architecture & Pipeline

```
archive/merged.csv ──► Causal Online Scaler (Welford) ──► Stream Iterator
                             │
                             ▼
                 Prequential Evaluation Loop:
       1. Predict continuous Attack probability & class (predict_proba)
       2. Update drift detector:
          - Baseline: Supervised ADWIN on 0/1 prediction loss
          - Enhanced: ADWIN-U-inspired on pooled feature descriptive stats (mean, std)
       3. Trigger adaptation reset & replay from bounded memory buffer on alarm
       4. Update classifier (learn_one) strictly after predictions & decisions
                             │
                             ▼
          Metrics, Confusion Matrix, Latency Percentiles (p50, p95, p99)
                             │
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
results/runs/latest/   results/plots/      [Docker: InfluxDB / Grafana]
```

---

## 3. Methodological Integrity & Fixes

This codebase enforces strict streaming machine learning standards:
1. **Zero Future Data Leakage:** Features are normalized causally using `OnlineStandardScaler` (Welford's algorithm) or an isolated pre-evaluation warm-up prefix. Test events are never used to fit past normalization statistics.
2. **Warm-Up Isolation:** Warm-up prefix events are explicitly excluded from scored classification metrics. In smoke mode, `stream_step` tracking guarantees the detector remains inactive during the intended calibration window regardless of global starting row index.
3. **Probability-Based Metrics:** Evaluates continuous positive-class probabilities ($P(\text{Attack} \mid x)$) for true continuous ROC-AUC and Average Precision (PR-AUC), eliminating degenerate binary step-function approximations.
4. **Fair Timing Boundaries:** Synchronized, microsecond-accurate timers capture prediction, detector, drift response, and learning latency independently across both methods.
5. **Cross-Platform Compatibility:** Memory profiling via `psutil` Process RSS (no Unix-only `resource` crashes on Windows); directory junction fallback for Windows environments without developer mode.

---

## 4. Setup & Quickstart

### Prerequisites
- Python 3.11 or 3.12
- Optional: Docker Desktop (for Kafka, InfluxDB, Grafana)

### Installation

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m uv pip install -r requirements.txt --python .\.venv\Scripts\python.exe
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Dataset Placement

Place the SWaT CSV files in `archive/` (excluded from git via `.gitignore`):
- `archive/merged.csv` (407 MB, canonical stream: 1,441,719 rows)
- `archive/normal.csv` (384 MB)
- `archive/attack.csv` (14 MB)

*Note: If `archive/merged.csv` is absent, the experiment runner automatically generates a deterministic synthetic test fixture (53 columns, exact schema) for verification without crashing.*

---

## 6. Execution Commands

### Run Unit Test Suite (20 Tests)
```bash
pytest tests/ -v
```

### Run Smoke Experiment
```bash
python run_project.py experiment --mode smoke
```

### Generate Publication Plots
```bash
python run_project.py plot
```

### Standalone Model Comparison (Without Retraining)
```bash
python run_project.py compare --run-dir results/runs/latest
```

### 10-Minute Demonstration
```bash
python run_project.py demo --mode smoke
```

### Optional Streaming Infrastructure (Docker Compose)
```bash
python run_project.py infra up
# Open Grafana at http://localhost:3000 (admin / admin)
python run_project.py infra down
```

---

## 7. Outputs & Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| **Comparison Table** | `results/runs/latest/comparison_table.csv` | Side-by-side metric comparison |
| **Run Manifest** | `results/runs/latest/run_metadata.json` | Commit hash, versions, configs, status |
| **Event Logs** | `results/runs/latest/*_events.csv` | Per-event predictions, probabilities, latencies |
| **Summaries** | `results/runs/latest/*_summary.json` | Full metric distributions & drift indices |
| **Figures** | `results/plots/` | Accuracy curves, drift timelines, bar charts (PNG/PDF) |
| **Paper Report** | `reports/report.tex` | IEEEtran conference research paper |

---

## 8. Academic Attribution

- **ADWIN-U:** D. N. Assis and V. M. A. Souza, "ADWIN-U: Adaptive Windowing for Unsupervised Drift Detection on Data Streams," *Knowledge and Information Systems*, 2025.
- **SGBR:** N. Gunasekara et al., "Gradient boosted bagging for evolving data stream regression," *Data Mining and Knowledge Discovery*, 2025 (reviewed as regression ensemble context).
- **SWaT Testbed:** iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design.
