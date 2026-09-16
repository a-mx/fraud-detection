from torch import nn, optim
import mlflow
from src.models.mlp import MLP
from src.training.pytorch.train import Trainer
from src.training.pytorch.dataloader import make_loaders
from src.training.logging import log_metrics, log_and_register_pyfunc
from src.training.wrappers import TorchProbaWrapper
from src.training.registry import MODEL_NAME


def train(X_train, X_test, y_train, y_test, settings) -> int:
    model = MLP(input_size=X_train.shape[1], params=settings.models.mlp)

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
        log_metrics(metrics, step=epoch)

    mlflow.log_param("model", "mlp")
    mlflow.log_params({**model.model_params, **model.training_params})

    wrapped = TorchProbaWrapper(scaler=scaler, torch_model=model)
    return log_and_register_pyfunc(
        python_model=wrapped,
        X_sample=X_train,
        y_pred_sample=wrapped.predict(None, X_train),
        model_name=MODEL_NAME,
        pip_requirements=["torch", "scikit-learn", "numpy", "pandas"],
    )