import mlflow
from src.models.baseline import BaselineModel
from src.training.logging import log_metrics, log_and_register_pyfunc
from src.training.wrappers import SklearnProbaWrapper
from src.training.registry import MODEL_NAME


def train(X_train, X_test, y_train, y_test, settings) -> int:
    model = BaselineModel(params=settings.models.baseline)
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    mlflow.log_param("model", "baseline")
    mlflow.log_params(model.model_params)
    log_metrics(metrics)

    wrapped = SklearnProbaWrapper(model.model)
    return log_and_register_pyfunc(
        python_model=wrapped,
        X_sample=X_train,
        y_pred_sample=wrapped.predict(None, X_train),
        model_name=MODEL_NAME,
        pip_requirements=["scikit-learn", "numpy", "pandas"],
    )