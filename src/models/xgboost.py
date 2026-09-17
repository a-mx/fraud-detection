from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class XGBModel:
    def __init__(self, params: dict[str, Any], scale_pos_weight: float | None = None):
        params = dict(params)
        self.threshold = float(params.pop("threshold", 0.5))

        if scale_pos_weight is not None:
            params.setdefault("scale_pos_weight", scale_pos_weight)

        params.setdefault("eval_metric", "auc")
        params.setdefault("n_jobs", -1)
        params.setdefault("random_state", 42)

        self.model_params = params

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(**params)),
        ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def evaluate(self, X_test, y_test) -> dict[str, float]:
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba >= self.threshold).astype(int)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "f2": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
            "pr_auc": average_precision_score(y_test, y_proba),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }