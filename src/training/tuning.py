import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, recall_score, fbeta_score


SCORERS = {
    "pr_auc": lambda y, p: average_precision_score(y, p),
    "recall": lambda y, p: recall_score(y, (p >= 0.5).astype(int), zero_division=0),
    "f2": lambda y, p: fbeta_score(y, (p >= 0.5).astype(int), beta=2, zero_division=0),
}


def _suggest(trial: optuna.Trial, name: str, spec: dict):

    kind = spec["type"]

    if kind == "int":
        return trial.suggest_int(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step", 1),
            log=spec.get("log", False),
        )
    if kind == "float":
        return trial.suggest_float(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step"),
            log=spec.get("log", False),
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown search spec type: {kind}")


def tune(
    model_factory,
    X, y,
    search_space: dict,
    fixed_params: dict,
    n_trials: int = 30,
    cv_folds: int = 5,
    scoring: str = "pr_auc",
    direction: str = "maximize",
    random_state: int = 42,
    timeout: int | None = None,
) -> optuna.Study:
    scorer = SCORERS[scoring]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed_params}
        for name, spec in search_space.items():
            params[name] = _suggest(trial, name, spec)

        scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = model_factory(params)
            model.train(X_tr, y_tr)
            proba = model.predict_proba(X_val)
            scores.append(scorer(y_val, proba))

        return float(np.mean(scores))

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study