import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RELOAD_TOKEN", "test-token")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://test:5000")
    monkeypatch.setenv("MLFLOW_MODEL_URI", "models:/test-model@production")

    def fake_load():
        class FakeModel:
            def predict(self, X):
                return np.full(len(X), 0.87)
        return FakeModel()

    import src.api.app as app_module
    import importlib
    importlib.reload(app_module)

    monkeypatch.setattr(app_module, "load_model", fake_load)

    app = app_module.app
    with TestClient(app) as client:
        yield client


def test_health_returns_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_score(client):
    r = client.post(
        "/predict",
        json={"features": {"v1": 0.5, "amount": 100.0}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 1
    assert body["score"] == pytest.approx(0.87)


def test_predict_rejects_empty_features(client):
    r = client.post("/predict", json={"features": {}})
    assert r.status_code == 422


def test_predict_rejects_extra_fields(client):
    r = client.post(
        "/predict",
        json={"features": {"v1": 1.0}, "extra": "field"},
    )
    assert r.status_code == 422


def test_reload_requires_token(client):
    r = client.post("/reload")
    assert r.status_code == 403


def test_reload_rejects_wrong_token(client):
    r = client.post("/reload", headers={"X-Reload-Token": "wrong"})
    assert r.status_code == 403


def test_reload_succeeds_with_correct_token(client):
    r = client.post("/reload", headers={"X-Reload-Token": "test-token"})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setenv("RELOAD_TOKEN", "test-token")

    import src.api.app as app_module
    import importlib

    def raise_load():
        raise RuntimeError("model not found")

    monkeypatch.setattr(app_module, "load_model", raise_load)
    importlib.reload(app_module)

    with TestClient(app_module.app) as client:
        r = client.post("/predict", json={"features": {"v1": 1.0}})
        assert r.status_code == 503