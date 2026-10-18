from pathlib import Path

import yaml

from src.data.preprocess import fit_preprocess_state, iter_swat_events, vectorize_event


def test_no_label_in_features():
    cfg = yaml.safe_load(Path("configs/project.yaml").read_text())
    csv_path = Path(cfg["data"]["archive_dir"]) / cfg["data"]["primary_file"]
    events = iter_swat_events(csv_path, cfg, max_events=10)
    for ev in events:
        assert "Normal/Attack" not in ev["features"]
        assert "Timestamp" not in ev["features"]


def test_preprocess_fit_causal():
    cfg = yaml.safe_load(Path("configs/project.yaml").read_text())
    csv_path = Path(cfg["data"]["archive_dir"]) / cfg["data"]["primary_file"]
    n_fit = cfg["data"]["initial_train_size"]
    state = fit_preprocess_state(iter_swat_events(csv_path, cfg, max_events=n_fit), cfg)
    assert state.fitted
    ev = next(iter_swat_events(csv_path, cfg, max_events=1))
    vec = vectorize_event(ev["features"], state)
    assert len(vec) == len(state.feature_order)
