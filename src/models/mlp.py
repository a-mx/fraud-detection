from typing import Any

import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from src.training.pytorch.dataloader import make_loaders
from src.training.pytorch.train import Trainer


class MLP(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class MLPModel:

    def __init__(self, input_size: int, params: dict[str, Any]):
        self.input_size = input_size
        self.params = dict(params)

        self.threshold = float(self.params.pop("threshold", 0.5))
        self.lr = float(self.params.pop("lr", 1e-3))
        self.batch_size = int(self.params.pop("batch_size", 256))
        self.epochs = int(self.params.pop("epochs", 5))
        self.weight_decay = float(self.params.pop("weight_decay", 0.0))
        self.optimizer_name = self.params.pop("optimizer", "adam")
        self.use_scaler = bool(self.params.pop("use_scaler", True))
        self.verbose = bool(self.params.pop("verbose", False))
        
        self.mlp = MLP(
            input_size=input_size,
            hidden_size=int(self.params.get("hidden_size", 64)),
            output_size=int(self.params.get("output_size", 1)),
            dropout=float(self.params.get("dropout", 0.2)),
        )

        self.scaler: StandardScaler | None = None

        self.model_params = {
            "input_size": input_size,
            "hidden_size": self.mlp.hidden_size,
            "output_size": self.mlp.output_size,
            "dropout": self.mlp.dropout,
        }
        self.training_params = {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer_name,
            "threshold": self.threshold,
        }

    def _make_optimizer(self):
        if self.optimizer_name == "adam":
            return optim.Adam(
                self.mlp.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        if self.optimizer_name == "sgd":
            return optim.SGD(
                self.mlp.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def train(self, X_train, y_train, log_metrics_callback=None) -> "MLPModel":
        train_loader, _, self.scaler = make_loaders(
            X_train,
            X_train,
            y_train,
            y_train,
            batch_size=self.batch_size,
            use_scaler=self.use_scaler,
        )

        loss_fn = nn.BCELoss()
        optimizer = self._make_optimizer()
        trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)

        for epoch in range(self.epochs):
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs}")

            trainer.train_loop(train_loader, self.mlp)

            if log_metrics_callback is not None:
                metrics = trainer.test_loop(train_loader, self.mlp, self.threshold)
                log_metrics_callback(metrics, epoch)

        return self

    def predict_proba(self, X) -> np.ndarray:
        X_arr = self.scaler.transform(X) if self.scaler is not None else np.asarray(X)
        self.mlp.eval()
        with torch.no_grad():
            tensor = torch.tensor(X_arr, dtype=torch.float32)
            out = self.mlp(tensor)
        return out.numpy().ravel()

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