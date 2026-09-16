import numpy as np
import pandas as pd
import mlflow.pyfunc


class SklearnProbaWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input: pd.DataFrame) -> pd.Series:
        if hasattr(self.pipeline, "predict_proba"):
            return pd.Series(
                self.pipeline.predict_proba(model_input)[:, 1], name="score"
            )
        return pd.Series(self.pipeline.predict(model_input), name="score")


class TorchProbaWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, scaler, torch_model):
        self.scaler = scaler
        self.torch_model = torch_model

    def predict(self, context, model_input: pd.DataFrame) -> pd.Series:
        import torch

        X = self.scaler.transform(model_input)
        self.torch_model.eval()
        with torch.no_grad():
            out = self.torch_model(torch.tensor(X, dtype=torch.float32))
        return pd.Series(out.numpy().ravel(), name="score")