import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config.settings import Settings
from src.api.model import load_model
from src.api.routes import reload as reload_route, health, predict
settings = Settings.from_yaml()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_uri = settings.env.mlflow_model_uri

    try:
        app.state.model = load_model()
        logger.info("Loaded model from %s", settings.env.mlflow_model_uri)
    except Exception:
        logger.exception("Model not found")

    yield

    app.state.model = None


app = FastAPI(lifespan=lifespan)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(reload_route.router)