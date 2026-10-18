# Demo Guide (< 10 minutes)

## Prerequisites

- Python 3.11+ (tested on 3.14)
- `archive/merged.csv` present
- Optional: Docker Desktop for Kafka/Grafana segment

## Path A — Local demo (no Docker)

```bash
source .venv/bin/activate
python run_project.py demo --mode smoke
open results/plots/04_prequential_accuracy.png
cat results/runs/latest/comparison_table.csv
```

**Expected observations**

1. Inspection already recorded in `results/dataset_inspection.json`.
2. Stream window includes Normal→Attack transition (index 1,387,098).
3. Baseline detects drift via **prediction error** (supervised ADWIN).
4. Enhanced detects drift via **feature statistics** (ADWIN-U) without using labels in the monitor.
5. Comparison table shows F1/recall and drift counts (actual numbers, not hardcoded).

## Path B — Streaming infra (requires Docker daemon)

```bash
docker compose up -d
python run_project.py experiment --mode smoke
# Enable influx in configs/project.yaml → influx.enabled: true
python -c "from pathlib import Path; from src.storage.influx_writer import write_event_metrics; write_event_metrics(Path('results/runs/latest'), Path('configs/project.yaml'))"
open http://localhost:3000
```

**Expected:** Grafana datasource `InfluxDB-SWaT`; panels update after Influx write.

## Path C — Synthetic drift (if live stream shows no alarm)

The smoke window uses **verified SWaT label boundary** (not synthetic).  
For synthetic feature drift:

```bash
python -c "
from src.drift.adwin_u import ADWINU
d = ADWINU(delta=0.01, buffer_size=64, min_instances_between_drifts=5)
for i in range(400):
    x = {f'f{k}': (5.0 if i > 150 else 0.0) for k in range(10)}
    if d.update(x, i): print('SYNTHETIC drift at', i)
"
```

Label clearly as **synthetic** in presentation slides.

## Recovery

```bash
docker compose down
rm -rf results/runs/latest   # if symlink broken, remove target run dir only
```

## Troubleshooting

| Issue | Action |
|-------|--------|
| `ModuleNotFoundError: src` | Run from repo root; use `pytest.ini` pythonpath |
| Docker socket error | Start Docker Desktop |
| Java missing | Use local `experiment` mode; skip Spark |
| All-zero attack metrics | Increase `smoke_start_index` / include post-1387098 rows |
