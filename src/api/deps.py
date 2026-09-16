from fastapi import Header, HTTPException, Request

from src.config.settings import Settings
settings = Settings.from_yaml()

def get_loaded_model(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model


def verify_reload_token(x_reload_token: str = Header(default="")) -> None:
    if not settings.env.reload_token or x_reload_token != settings.env.reload_token:
        raise HTTPException(status_code=403, detail="Forbidden")