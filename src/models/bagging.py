from typing import Any, Callable

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BaggingModel:

    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        n_estimators: int = 20,
        max_samples: float = 0.8,
        max_features: float = 0.8,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
        model_params: dict | None = None,
    ):
        self.estimator_factory = estimator_factory
        self.model_params = model_params or {}
        self.training_params = {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "bootstrap_features": bootstrap_features,
            "random_state": random_state,
        }

        self.model = BaggingClassifier(
            estimator=estimator_factory(),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def train(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test) -> dict[str, float]:
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }