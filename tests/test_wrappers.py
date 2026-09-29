import numpy as np
import pandas as pd
import pytest


def test_sklearn_wrapper_returns_series(synthetic_data):
    from src.training.wrappers import SklearnProbaWrapper

    X, _ = synthetic_data

    class FakePipeline:
        def predict_proba(self, X):
            return np.column_stack([1 - np.full(len(X), 0.3), np.full(len(X), 0.3)])

    wrapper = SklearnProbaWrapper(FakePipeline())
    result = wrapper.predict(X)

    assert isinstance(result, pd.Series)
    assert result.name == "score"
    assert len(result) == len(X)
    assert np.allclose(result.values, 0.3)


def test_sklearn_wrapper_handles_1d_predict_proba(synthetic_data):
    from src.training.wrappers import SklearnProbaWrapper

    X, _ = synthetic_data

    class FakePipeline:
        def predict_proba(self, X):
            return np.full(len(X), 0.75)

    wrapper = SklearnProbaWrapper(FakePipeline())
    result = wrapper.predict(X)
    assert np.allclose(result.values, 0.75)


def test_torch_wrapper_applies_scaler(synthetic_data):
    import torch
    from src.training.wrappers import TorchProbaWrapper

    X, _ = synthetic_data
    X_small = X.iloc[:5]

    class FakeScaler:
        def transform(self, X):
            return np.asarray(X) * 2.0

    class FakeTorchModel(torch.nn.Module):
        def forward(self, x):
            return torch.sigmoid(x[:, 0:1])

    wrapper = TorchProbaWrapper(scaler=FakeScaler(), torch_model=FakeTorchModel())
    result = wrapper.predict(X_small)

    assert isinstance(result, pd.Series)
    assert result.name == "score"
    assert len(result) == 5
    assert np.all((result.values >= 0) & (result.values <= 1))