import mlflow
from src.models.randomforest import RandomForestModel
from src.training.tuning import tune
from src.training.logging import log_metrics, log_and_register_pyfunc
from src.training.wrappers import SklearnProbaWrapper
from src.training.registry import MODEL_NAME


def train(X_train, X_test, y_train, y_test, settings, tune_hparams: bool = True) -> int:
    spec = settings.models["random_forest"]

    if tune_hparams and spec.search:
        def model_factory(params: dict) -> RandomForestModel:
            return RandomForestModel(
                params=params,
            )

        study = tune(
            model_factory=model_factory,
            X=X_train, y=y_train,
            search_space=spec.search,
            fixed_params={**spec.fixed},
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
    else:
        best_params = {}

    final_params = {**spec.fixed, **best_params}
    final_model = RandomForestModel(params=final_params)
    final_model.train(X_train, y_train)
    metrics = final_model.evaluate(X_test, y_test)

    mlflow.log_param("model", "random_forest")
    mlflow.log_params(final_params)
    log_metrics(metrics)

    wrapped = SklearnProbaWrapper(final_model.model)
    return log_and_register_pyfunc(
        python_model=wrapped,
        X_sample=X_train,
        y_pred_sample=wrapped.predict(None, X_train),
        model_name=MODEL_NAME,
        pip_requirements=["scikit-learn", "xgboost", "numpy", "pandas"],
    )