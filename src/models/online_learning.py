"""Factory helpers for River online classifiers."""

from __future__ import annotations

from typing import Any

from river import forest, tree


def build_classifier(name: str, seed: int = 42, **kwargs: Any):
    name = name.lower()
    kwargs.pop("seed", None)
    if name in {"hoeffding_tree", "ht"}:
        return tree.HoeffdingTreeClassifier(
            grace_period=kwargs.get("grace_period", 200),
            delta=kwargs.get("delta", 1e-7),
        )
    if name in {"adaptive_random_forest", "arf"}:
        return forest.ARFClassifier(
            n_models=kwargs.get("n_models", 10),
            seed=seed,
        )
    raise ValueError(f"Unsupported classifier: {name}")
