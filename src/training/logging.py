from typing import Any
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)


def log_and_register_pyfunc(
    python_model,
    X_sample: pd.DataFrame,
    y_pred_sample,
    model_name: str,
    pip_requirements: list[str],
) -> int:
    signature = infer_signature(X_sample, np.asarray(y_pred_sample).ravel())
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=python_model,
        signature=signature,
        input_example=X_sample.head(3),
        registered_model_name=model_name,
        pip_requirements=pip_requirements,
    )
    return int(info.registered_model_version)