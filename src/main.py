import argparse
import torch
from torch import nn, optim
from src.data.build_dataset import build_dataset
from src.models.baseline import BaselineModel
from src.models.mlp import MLP
from src.models.xgboost import XGBModel
from src.training.train import Trainer
from src.training.dataloader import make_loaders
from src.config.settings import Settings
import mlflow
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Fraud detection model training")
    parser.add_argument(
        "--model", 
        type=str, 
        default="baseline",
        choices=["baseline", "mlp", "xgb"],
        help="Which model to train"
    )
    
    args = parser.parse_args()
    settings = Settings.from_yaml()
    X_train, X_test, y_train, y_test = build_dataset()
    mlflow.set_experiment("fraud-detection")
    run_name = f"{args.model}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("model", args.model)

        if args.model == "baseline":

            model = BaselineModel(
                params=settings.models.baseline
            )
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)

            mlflow.log_params(model.model_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model.model, name="model")

        elif args.model == "xgb":

            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

            model = XGBModel(
                params=settings.models.xgb,
                scale_pos_weight=scale_pos_weight
            )
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            
            mlflow.log_params(model.model_params)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model.model, name="model")
        
        elif args.model == "mlp":

            input_size = X_train.shape[1]
            model = MLP(
                input_size=input_size,
                params=settings.models.mlp
            )
            
            train_loader, test_loader = make_loaders(
                X_train, X_test, y_train, y_test, 
                batch_size=settings.models.mlp.batch_size
            )
            
            loss_fn = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=settings.models.mlp.lr)
            trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)
            
            for epoch in range(settings.models.mlp.epochs):
                print(f"\nEpoch {epoch + 1}/{settings.models.mlp.epochs}")
                trainer.train_loop(train_loader, model)
                metrics = trainer.test_loop(test_loader, model)
                mlflow.log_metrics(
                    metrics=metrics,
                    step=epoch,
                )

            mlflow.log_params({**model.model_params, **model.training_params})
            mlflow.pytorch.log_model(
                model,
                name="model",
                serialization_format="pickle"
            )

if __name__ == "__main__":
    main()