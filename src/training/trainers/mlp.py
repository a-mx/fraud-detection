import logging

import mlflow

from src.models.mlp import MLPModel
from src.training.logging import log_metrics, log_and_register_pyfunc
from src.training.registry import MODEL_NAME
from src.training.tuning import tune
from src.training.wrappers import TorchProbaWrapper

logger = logging.getLogger(__name__)


def train(X_train, X_test, y_train, y_test, settings, tune_hparams: bool = True) -> int:
    spec = settings.models["mlp"]
    input_size = X_train.shape[1]

    if tune_hparams and spec.search:
        def model_factory(params: dict) -> MLPModel:
            return MLPModel(input_size=input_size, params=params)

        tuning_overrides = spec.search.get("_tuning", {})
        study = tune(
            model_factory=model_factory,
            X=X_train,
            y=y_train,
            search_space=spec.search,
            fixed_params=spec.fixed,
            n_trials=settings.tuning.n_trials,
            cv_folds=settings.training.cv_folds,
            scoring=settings.training.scoring,
            direction=settings.tuning.direction,
            random_state=settings.training.random_state,
            timeout=settings.tuning.timeout,
        )

        best_params = study.best_params
        mlflow.log_params({f"tuned_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_score", study.best_value)
        mlflow.log_param("cv_scoring", settings.training.scoring)
        mlflow.log_param("cv_folds", settings.training.cv_folds)
        logger.info("Best CV score: %.4f", study.best_value)
    else:
        best_params = {}

    final_params = {**spec.fixed, **best_params}
    model = MLPModel(input_size=input_size, params=final_params)

    def on_epoch(metrics: dict, step: int) -> None:
        log_metrics(metrics, step=step)

    model.train(X_train, y_train, log_metrics_callback=on_epoch)

    holdout_metrics = model.evaluate(X_test, y_test)
    log_metrics(holdout_metrics)

    mlflow.log_param("model", "mlp")
    mlflow.log_params(model.model_params)
    mlflow.log_params(model.training_params)

    wrapped = TorchProbaWrapper(scaler=model.scaler, torch_model=model.mlp)
    return log_and_register_pyfunc(
        python_model=wrapped,
        X_sample=X_train,
        y_pred_sample=model.predict_proba(X_train),
        model_name=MODEL_NAME,
        pip_requirements=["torch", "scikit-learn", "numpy", "pandas"],
    )