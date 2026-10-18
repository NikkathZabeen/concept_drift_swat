# Continual Learning for Concept-Drifting SWaT Streams

IOE413 — Streaming Data Analytics (Topic 10, Track B).  
End-to-end streaming ML pipeline for the [SWaT dataset](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system) with **baseline supervised ADWIN** vs **enhanced ADWIN-U** continual learning.

## Research question

Can a continual learning system detect changes in industrial IoT sensor streams and adapt quickly while maintaining predictive quality, detection reliability, and acceptable resource use?

## Architecture

```
archive/merged.csv → preprocess (causal) → stream iterator
        → [Kafka producer] → [Spark*] → online learner + drift detector → metrics → [InfluxDB] → [Grafana]
        → local experiment_runner (default when infra unavailable)
```

\* Spark requires Java; not available in the primary dev environment (see `PROJECT_STATUS.md`).

## Dataset (required before running)

CSV files are **not** in Git (size limits). Download from Kaggle and put them in `archive/` — see [archive/README.md](archive/README.md).

## Quick start

```bash
cd concept_drift_swat
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Dataset inspection
python run_project.py inspect

# 2) Smoke experiment (drift-focused window around index 1,387,098)
python run_project.py experiment --mode smoke

# 3) Plots (set writable matplotlib cache if needed)
MPLCONFIGDIR=$PWD/.mplconfig python run_project.py plot

# 4) Full chronological stream (long-running)
python run_project.py experiment --mode full
```

## Tests

```bash
pytest tests/ -q
```

## Configuration

- `configs/project.yaml` — data paths, smoke/full stream windows, reference drift index
- `configs/baseline.yaml` — Hoeffding tree + supervised ADWIN
- `configs/enhanced.yaml` — ARF + ADWIN-U + buffer reset policy

## Outputs

| Artifact | Location |
|----------|----------|
| Inspection JSON | `results/dataset_inspection.json` |
| Event logs | `results/runs/latest/baseline_events.csv`, `enhanced_events.csv` |
| Summaries | `results/runs/latest/*_summary.json` |
| Comparison table | `results/runs/latest/comparison_table.csv` |
| Figures | `results/plots/*.png`, `*.pdf` |
| Report (LaTeX) | `reports/report.tex` |

## Infrastructure (optional)

```bash
cp .env.example .env
docker compose up -d          # Kafka, InfluxDB 2.x, Grafana
python -m src.streaming.stream_producer --max-events 5000 --kafka
python -m src.storage.influx_writer  # after enabling influx in project.yaml
```

Grafana: http://localhost:3000 (default admin/admin on first launch).  
Provisioned datasource: `dashboard/provisioning/datasources/influx.yml`.

## Attribution

SWaT dataset: iTrust, Singapore University of Technology and Design.  
Place CSV files under `archive/` as supplied by the course/Kaggle mirror.

## Reproducibility

- Fixed seed: `42` in `configs/project.yaml`
- Prequential protocol: predict → score → `learn_one`
- Preprocessing fit only on pre-window history (see `src/data/preprocess.py`)

## Known limitations

See `PROJECT_STATUS.md` and Section VII of `reports/report.tex`.
