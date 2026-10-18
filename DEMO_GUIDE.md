# Demonstration Guide (< 10 Minutes)

This guide provides a rapid, step-by-step evaluation workflow for instructors, reviewers, and graders to test and verify the entire SWaT continual learning pipeline.

---

## Prerequisites

- Python 3.11 or 3.12 (virtual environment activated)
- Dataset: If `archive/merged.csv` is present, the pipeline runs on the real SWaT dataset; if absent, a self-contained synthetic fixture is automatically used.
- Optional: Docker Desktop running (for Path B streaming dashboard segment).

---

## Path A — Local Demonstration (Core ML Pipeline, ~3 Minutes)

### Step 1: Run Unit Test Suite
Execute the test suite to verify leak-free preprocessing, warm-up bug fixes, probability metrics, and drift detectors:

**Windows PowerShell:**
```powershell
.\.venv\Scripts\pytest -v
```

**Linux / macOS:**
```bash
pytest tests/ -v
```

*Expected output:* All 23 tests pass in ~5-6 seconds.


---

### Step 2: Run End-to-End Smoke Demonstration
Execute the smoke experiment comparing Baseline ADWIN vs Enhanced ADWIN-U-inspired, and automatically generate publication figures:

**Windows PowerShell:**
```powershell
.\.venv\Scripts\python run_project.py demo --mode smoke
```

**Linux / macOS:**
```bash
python run_project.py demo --mode smoke
```

*Expected output:*
- Causal preprocessing state is fitted on the warm-up prefix.
- Baseline (Hoeffding Tree + Supervised ADWIN) runs prequentially.
- Enhanced (Adaptive Random Forest + ADWIN-U-inspired) runs prequentially.
- Output artifacts saved to `results/runs/demo_smoke/` and mirrored to `results/runs/latest/`.
- 10 publication-quality plots (PNG + PDF) generated under `results/plots/`.

---

### Step 3: View Side-by-Side Comparison Table
Inspect the performance, latency, and drift detection metrics directly in your terminal:

**Windows PowerShell:**
```powershell
.\.venv\Scripts\python run_project.py compare --run-dir results/runs/latest
```

**Linux / macOS:**
```bash
python run_project.py compare --run-dir results/runs/latest
```

*Expected observations:*
1. **Prequential Integrity:** Accuracy, Recall, and F1 are evaluated exclusively on post-warmup scored observations.
2. **Unsupervised Sensitivity:** Enhanced ADWIN-U triggers drift adaptation based strictly on feature statistics without label feedback.
3. **Latency Breakdown:** Percentiles (p50, p95, p99) demonstrate that Hoeffding Trees achieve sub-millisecond predictions, whereas 10-tree ARFs incur higher latency in exchange for ensemble resilience.
4. **Memory Profiling:** Process-level peak RSS is reported using `psutil`.

---

### Step 4: Inspect Generated Figures
Open the figures generated in `results/plots/`:
- `04_prequential_accuracy.png` — Rolling accuracy curve over stream positions.
- `06_drift_timeline.png` — Timeline of baseline vs enhanced drift detection alarms.
- `12_baseline_vs_enhancement_summary.png` — Side-by-side metric difference bar chart.
- `01_label_distribution.png` — Class distribution in the evaluated stream.

---

## Path B — Streaming Infrastructure (Optional, Docker Desktop Required)

### Step 1: Launch Streaming Stack
Start Zookeeper, Kafka, InfluxDB 2.x, and Grafana:
```powershell
python run_project.py infra up
```

### Step 2: Export Run Metrics to InfluxDB
```powershell
python -c "from pathlib import Path; from src.storage.influx_writer import write_event_metrics; write_event_metrics(Path('results/runs/latest'), Path('configs/project.yaml'))"
```

### Step 3: View Grafana Dashboard
1. Open browser to: `http://localhost:3000`
2. Login credentials: `admin` / `admin` (or skip password change on first launch).
3. The provisioned datasource `InfluxDB-SWaT` connects to `http://influxdb:8086`.

### Step 4: Teardown Infrastructure
```powershell
python run_project.py infra down
```

---

## Troubleshooting

| Issue | Root Cause | Solution |
|-------|------------|----------|
| `ModuleNotFoundError: No module named 'river'` | Virtualenv not active | Run using `.\.venv\Scripts\python` or activate `.venv` |
| Symlink error on Windows | Unprivileged symlink | Handled automatically via Directory Junction / pointer fallback |
| Docker connection failure | Docker Desktop not running | Start Docker Desktop or use Path A (local mode) |
| Missing `archive/merged.csv` | Dataset git-ignored | Pipeline automatically falls back to synthetic verification fixture |
