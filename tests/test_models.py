import numpy as np
import pytest


def test_baseline_train_predict_proba_returns_1d(train_test_split_data):
    from src.models.baseline import BaselineModel

    X_train, X_test, y_train, y_test = train_test_split_data
    model = BaselineModel(params={"C": 1.0, "max_iter": 200, "class_weight": "balanced"})
    model.train(X_train, y_train)

    proba = model.predict_proba(X_test)
    assert proba.ndim == 1
    assert len(proba) == len(X_test)
    assert np.all((proba >= 0) & (proba <= 1))


def test_baseline_evaluate_returns_expected_metrics(train_test_split_data):
    from src.models.baseline import BaselineModel

    X_train, X_test, y_train, y_test = train_test_split_data
    model = BaselineModel(params={"C": 1.0, "max_iter": 200, "class_weight": "balanced"})
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    expected_keys = {"accuracy", "precision", "recall", "f1", "f2", "pr_auc", "roc_auc"}
    assert expected_keys.issubset(metrics.keys())
    for v in metrics.values():
        assert 0.0 <= v <= 1.0


def test_baseline_predict_uses_threshold(train_test_split_data):
    from src.models.baseline import BaselineModel

    X_train, X_test, y_train, y_test = train_test_split_data
    model = BaselineModel(params={"C": 1.0, "max_iter": 200, "threshold": 0.9})
    model.train(X_train, y_train)

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert preds.sum() <= (probas >= 0.5).sum()


def test_xgb_train_predict_proba_returns_1d(train_test_split_data):
    from src.models.xgboost import XGBModel

    X_train, X_test, y_train, y_test = train_test_split_data
    model = XGBModel(params={
        "n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1,
    })
    model.train(X_train, y_train)

    proba = model.predict_proba(X_test)
    assert proba.ndim == 1
    assert len(proba) == len(X_test)


def test_xgb_threshold_extracted_from_params(train_test_split_data):
    from src.models.xgboost import XGBModel

    X_train, _, y_train, _ = train_test_split_data
    model = XGBModel(params={
        "n_estimators": 5, "max_depth": 3, "threshold": 0.7, "n_jobs": 1,
    })
    assert model.threshold == 0.7
    assert "threshold" not in model.model_params