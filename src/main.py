import argparse
import torch
from torch import nn, optim
from datetime import datetime

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from src.data.build_dataset import build_dataset
from src.models.baseline import BaselineModel
from src.models.mlp import MLP
from src.models.xgboost import XGBModel
from src.training.train import Trainer
from src.training.dataloader import make_loaders
from src.config.settings import Settings

MODEL_NAME = "fraud-detection"
ALIAS = "production"


def set_production_alias(model_name: str, version: int, description: str | None = None):
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias=ALIAS,
        version=version,
    )
    if description:
        client.update_model_version(
            name=model_name, version=version, description=description
        )
    return version


class ScalerTorchWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, scaler, torch_model):
        self.scaler = scaler
        self.torch_model = torch_model

    def predict(self, context, model_input):
        X = self.scaler.transform(model_input)
        self.torch_model.eval()
        with torch.no_grad():
            out = self.torch_model(torch.tensor(X, dtype=torch.float32))
        return out.numpy().ravel()


def main():
    parser = argparse.ArgumentParser(description="Fraud detection model training")
    parser.add_argument(
        "--model", type=str, default="baseline",
        choices=["baseline", "mlp", "xgb"],
    )
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    settings = Settings.from_yaml()
    X_train, X_test, y_train, y_test = build_dataset()
    mlflow.set_experiment("fraud-detection")
    run_name = f"{args.model}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model", args.model)
        registered_version = None

        if args.model == "baseline":
            model = BaselineModel(params=settings.models.baseline)
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            mlflow.log_params(model.model_params)
            mlflow.log_metrics(metrics)

            info = mlflow.sklearn.log_model(
                model.model,
                name="model",
                registered_model_name=MODEL_NAME,
            )
            registered_version = info.registered_model_version

        elif args.model == "xgb":
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            model = XGBModel(
                params=settings.models.xgb,
                scale_pos_weight=scale_pos_weight,
            )
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            mlflow.log_params(model.model_params)
            mlflow.log_metrics(metrics)

            info = mlflow.sklearn.log_model(
                model.model,
                name="model",
                skops_trusted_types=[
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                ],
                registered_model_name=MODEL_NAME,
            )
            registered_version = info.registered_model_version

        elif args.model == "mlp":
            input_size = X_train.shape[1]
            model = MLP(input_size=input_size, params=settings.models.mlp)

            train_loader, test_loader, scaler = make_loaders(
                X_train, X_test, y_train, y_test,
                batch_size=settings.models.mlp.batch_size,
                use_scaler=True,
            )

            loss_fn = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=settings.models.mlp.lr)
            trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)

            for epoch in range(settings.models.mlp.epochs):
                print(f"\nEpoch {epoch + 1}/{settings.models.mlp.epochs}")
                trainer.train_loop(train_loader, model)
                metrics = trainer.test_loop(test_loader, model)
                mlflow.log_metrics(metrics=metrics, step=epoch)

            mlflow.log_params({**model.model_params, **model.training_params})

            wrapped = ScalerTorchWrapper(scaler=scaler, torch_model=model)
            info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=wrapped,
                registered_model_name=MODEL_NAME,
                pip_requirements=[
                    "torch", "scikit-learn", "numpy",
                ],
            )
            registered_version = info.registered_model_version

        if args.promote:
            if registered_version is None:
                raise RuntimeError("No registered version, promotion is canceled")
            v = set_production_alias(
                MODEL_NAME,
                version=int(registered_version),
                description=f"run={run.info.run_id} model={args.model}",
            )
            print(f"Alias '{ALIAS}' -> {MODEL_NAME} v{v}")


if __name__ == "__main__":
    main()