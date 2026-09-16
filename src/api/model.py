import logging
import threading
from typing import Any

import mlflow
import mlflow.pyfunc

from src.config.settings import Settings

logger = logging.getLogger(__name__)

_reload_lock = threading.Lock()
settings = Settings.from_yaml()

def load_model() -> Any:
    mlflow.set_tracking_uri(settings.env.mlflow_tracking_uri)
    return mlflow.pyfunc.load_model(settings.env.mlflow_model_uri)


def reload_model_into(state) -> None:
    if not _reload_lock.acquire(blocking=False):
        raise RuntimeError("Reload already in progress")
    try:
        model = load_model()
        state.model = model
        state.model_uri = settings.env.mlflow_model_uri
        logger.info("Reloaded model from %s", settings.env.mlflow_model_uri)
    finally:
        _reload_lock.release()