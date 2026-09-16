from fastapi import APIRouter, Request

from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    return HealthResponse(
        status="ok",
        model_loaded=getattr(request.app.state, "model", None) is not None,
        model_uri=getattr(request.app.state, "model_uri", None),
    )