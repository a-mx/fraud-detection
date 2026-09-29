import pytest
from pydantic import ValidationError

from src.api.schemas import PredictRequest, PredictResponse


def test_predict_request_accepts_valid_features():
    req = PredictRequest(features={"amount": 100.0, "v1": -0.5})
    assert req.features == {"amount": 100.0, "v1": -0.5}


def test_predict_request_rejects_empty_features():
    with pytest.raises(ValidationError):
        PredictRequest(features={})


def test_predict_request_rejects_extra_fields():
    with pytest.raises(ValidationError):
        PredictRequest(features={"amount": 1.0}, unexpected="x")


def test_predict_response_allows_none_score():
    resp = PredictResponse(prediction=0, score=None)
    assert resp.score is None


def test_predict_response_rejects_non_int_prediction():
    with pytest.raises(ValidationError):
        PredictResponse(prediction="fraud", score=0.5)