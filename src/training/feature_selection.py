import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    recall_score,
)

from src.models.baseline import BaselineModel

logger = logging.getLogger(__name__)

SCORERS = {
    "pr_auc": lambda y, p: average_precision_score(y, p),
    "recall": lambda y, p: recall_score(y, (p >= 0.5).astype(int), zero_division=0),
    "f2": lambda y, p: fbeta_score(y, (p >= 0.5).astype(int), beta=2, zero_division=0),
}


def _make_scoring_model(baseline_params: dict) -> BaselineModel:
    return BaselineModel(params=dict(baseline_params))


def _cv_score(
    X,
    y,
    features: list[str],
    baseline_params: dict,
    cv_folds: int,
    scoring: str,
    random_state: int = 42,
) -> float:
    scorer = SCORERS[scoring]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    X_subset = X[features]
    scores = []
    for train_idx, val_idx in cv.split(X_subset, y):
        X_tr = X_subset.iloc[train_idx]
        X_val = X_subset.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model = _make_scoring_model(baseline_params)
        model.train(X_tr, y_tr)
        proba = model.predict_proba(X_val)
        scores.append(scorer(y_val, proba))

    return float(np.mean(scores))


def greedy_forward_selection(
    X,
    y,
    baseline_params: dict,
    candidate_features: list[str] | None = None,
    cv_folds: int = 3,
    scoring: str = "pr_auc",
    min_improvement: float = 1e-4,
    max_features: int | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    candidates = list(candidate_features or X.columns)
    selected: list[str] = []
    history: list[dict] = []
    best_score = 0.0

    if verbose:
        logger.info("Starting greedy forward selection on %d candidates", len(candidates))

    while True:
        if max_features is not None and len(selected) >= max_features:
            logger.info("Reached max_features=%d, stopping", max_features)
            break
        if not candidates:
            logger.info("No candidates left, stopping")
            break

        best_candidate = None
        best_candidate_score = best_score

        for feat in candidates:
            trial = selected + [feat]
            score = _cv_score(
                X, y, trial,
                baseline_params=baseline_params,
                cv_folds=cv_folds,
                scoring=scoring,
                random_state=random_state,
            )
            history.append({"feature": feat, "score": score, "n_features": len(trial)})

            if verbose:
                logger.info(
                    "  try %-20s | n=%2d | %s=%.4f",
                    feat, len(trial), scoring, score,
                )

            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = feat

        improvement = best_candidate_score - best_score

        if best_candidate is None or improvement < min_improvement:
            logger.info(
                "Stop: best improvement %.5f < %.5f (best_score=%.4f)",
                improvement, min_improvement, best_score,
            )
            break

        selected.append(best_candidate)
        candidates.remove(best_candidate)
        best_score = best_candidate_score

        logger.info(
            "Added %-20s | n=%2d | %s=%.4f (+%.4f)",
            best_candidate, len(selected), scoring, best_score, improvement,
        )

    return {
        "selected": selected,
        "history": history,
        "best_score": best_score,
        "scoring": scoring,
        "cv_folds": cv_folds,
    }


def save_selection(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Saved selection to %s", path)


def load_selection(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))