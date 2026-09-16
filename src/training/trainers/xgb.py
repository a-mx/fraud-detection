import mlflow
from src.models.xgboost import XGBModel
from src.training.logging import log_metrics, log_and_register_pyfunc
from src.training.wrappers import SklearnProbaWrapper
from src.training.registry import MODEL_NAME


def train(X_train, X_test, y_train, y_test, settings) -> int:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBModel(
        params=settings.models.xgb,
        scale_pos_weight=scale_pos_weight,
    )
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    mlflow.log_param("model", "xgb")
    mlflow.log_params(model.model_params)
    log_metrics(metrics)

    wrapped = SklearnProbaWrapper(model.model)
    return log_and_register_pyfunc(
        python_model=wrapped,
        X_sample=X_train,
        y_pred_sample=wrapped.predict(None, X_train),
        model_name=MODEL_NAME,
        pip_requirements=["scikit-learn", "xgboost", "numpy", "pandas"],
    )