import argparse
import logging

import mlflow

from src.config.settings import Settings
from src.data.build_dataset import build_dataset
from src.training.feature_selection import (
    greedy_forward_selection,
    save_selection,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Greedy feature selection. Parameters come from config.yaml; "
                    "flags below override them for one-off runs."
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--scoring", choices=["pr_auc", "recall", "f2"], default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--min-improvement", type=float, default=None)
    args = parser.parse_args()


    settings = Settings.from_yaml()
    fs = settings.feature_selection

    if not fs.enabled:
        logger.warning("feature_selection.enabled is false — exiting")
        return

    output = args.output or fs.output
    cv_folds = args.cv_folds if args.cv_folds is not None else fs.cv_folds
    scoring = args.scoring or fs.scoring
    max_features = args.max_features if args.max_features is not None else fs.max_features
    min_improvement = (
        args.min_improvement if args.min_improvement is not None else fs.min_improvement
    )

    baseline_spec = settings.models["baseline"]
    baseline_params = dict(baseline_spec.fixed)

    X_train, _, y_train, _ = build_dataset()

    mlflow.set_experiment("fraud-detection-feature-selection")
    with mlflow.start_run(run_name=f"greedy-{scoring}"):
        mlflow.log_params({
            "cv_folds": cv_folds,
            "scoring": scoring,
            "min_improvement": min_improvement,
            "max_features": max_features,
            "method": fs.method,
            **{f"baseline_{k}": v for k, v in baseline_params.items()},
        })

        result = greedy_forward_selection(
            X=X_train,
            y=y_train,
            baseline_params=baseline_params,
            cv_folds=cv_folds,
            scoring=scoring,
            min_improvement=min_improvement,
            max_features=max_features,
            verbose=True,
        )

        mlflow.log_metric("best_score", result["best_score"])
        mlflow.log_metric("n_selected", len(result["selected"]))
        mlflow.log_text(", ".join(result["selected"]), "selected_features.txt")

        save_selection(result, output)
        logger.info(
            "Selected %d features: %s", len(result["selected"]), result["selected"]
        )


if __name__ == "__main__":
    main()