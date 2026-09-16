import argparse
import logging
from datetime import datetime

import mlflow

from src.data.build_dataset import build_dataset
from src.config.settings import Settings
from src.training.registry import MODEL_NAME, set_production_alias
from src.training.trainers import TRAINERS

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud detection model training")
    parser.add_argument(
        "--model", type=str, default="baseline",
        choices=list(TRAINERS.keys()),
    )
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    settings = Settings.from_yaml()
    X_train, X_test, y_train, y_test = build_dataset()

    experiment_name = getattr(settings, "experiment_name", "fraud-detection")
    mlflow.set_experiment(experiment_name)

    run_name = f"{args.model}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        train_fn = TRAINERS[args.model]
        version = train_fn(X_train, X_test, y_train, y_test, settings)

        if args.promote:
            set_production_alias(
                version=version,
                description=f"run={run.info.run_id} model={args.model}",
            )
            print(f"Alias 'production' -> {MODEL_NAME} v{version}")


if __name__ == "__main__":
    main()