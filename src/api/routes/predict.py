import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_loaded_model
from src.api.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model=Depends(get_loaded_model)):
    try:
        df = pd.DataFrame([req.features])
        raw = np.asarray(model.predict(df)).ravel()
        score = float(raw[0])
        return PredictResponse(prediction=int(score >= 0.5), score=score)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc