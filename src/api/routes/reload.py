import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.deps import verify_reload_token
from src.api.schemas import ReloadResponse
from src.api.model import reload_model_into

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin"])


@router.post(
    "/reload",
    response_model=ReloadResponse,
    dependencies=[Depends(verify_reload_token)],
)
def reload_model(request: Request):
    try:
        reload_model_into(request.app.state)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Reload failed")
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc

    return ReloadResponse(status="ok", model_uri=request.app.state.model_uri)