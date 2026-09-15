import logging
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/fraud-detection@production")

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(min_length=1)

class PredictResponse(BaseModel):
    prediction: int
    score: float | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    app.state.model = None
    app.state.model_uri = MLFLOW_MODEL_URI

    try:
        app.state.model = mlflow.pyfunc.load_model(
            MLFLOW_MODEL_URI,
        )
        logger.info("Loaded model from %s", MLFLOW_MODEL_URI)
    except Exception:
        logger.exception(
            "Model not found"
        )

    yield

    app.state.model = None


app = FastAPI(lifespan=lifespan)


def get_loaded_model():
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": getattr(app.state, "model", None) is not None,
        "model_uri": getattr(app.state, "model_uri", None),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model=Depends(get_loaded_model)):
    try:
        df = pd.DataFrame([req.features])

        score = None
        try:
            proba = model.predict(df, params={"predict_method": "predict_proba"})
            if hasattr(proba, "shape") and proba.ndim == 2:
                score = float(proba[0][1]) 
        except Exception:
            raw = model.predict(df)
            value = raw[0]
            if hasattr(value, "item"):
                value = value.item()
            score = float(value) if isinstance(value, (int, float)) else None

        if score is not None:
            prediction = int(score >= 0.5)
        else:
            prediction = int(model.predict(df)[0])

        return PredictResponse(prediction=prediction, score=score)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc