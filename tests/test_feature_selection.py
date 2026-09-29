import numpy as np
import pandas as pd


def test_greedy_selection_returns_selected_features(synthetic_data, fake_settings):
    from src.training.feature_selection import greedy_forward_selection

    X, y = synthetic_data
    baseline_params = dict(fake_settings.models["baseline"].fixed)

    result = greedy_forward_selection(
        X=X, y=y,
        baseline_params=baseline_params,
        cv_folds=2,
        scoring="pr_auc",
        max_features=3,
        min_improvement=0.0,
        verbose=False,
    )

    assert "selected" in result
    assert "history" in result
    assert 1 <= len(result["selected"]) <= 3
    assert all(f in X.columns for f in result["selected"])


def test_greedy_selection_stops_when_no_improvement(synthetic_data, fake_settings):
    from src.training.feature_selection import greedy_forward_selection

    X, y = synthetic_data
    baseline_params = dict(fake_settings.models["baseline"].fixed)

    result = greedy_forward_selection(
        X=X, y=y,
        baseline_params=baseline_params,
        cv_folds=2,
        scoring="pr_auc",
        min_improvement=0.5,
        verbose=False,
    )

    assert len(result["selected"]) < len(X.columns)


def test_save_and_load_selection_roundtrip(tmp_path):
    from src.training.feature_selection import save_selection, load_selection

    result = {
        "selected": ["v1", "v2", "amount"],
        "best_score": 0.78,
        "scoring": "pr_auc",
        "cv_folds": 3,
        "history": [{"feature": "v1", "score": 0.5, "n_features": 1}],
    }
    path = tmp_path / "selection.json"
    save_selection(result, path)
    loaded = load_selection(path)

    assert loaded == result