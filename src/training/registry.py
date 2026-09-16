import logging
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MODEL_NAME = "fraud-detection"
ALIAS = "production"


def set_production_alias(
    version: int,
    model_name: str = MODEL_NAME,
    alias: str = ALIAS,
    description: str | None = None,
) -> int:
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name, alias=alias, version=version,
    )
    if description:
        client.update_model_version(
            name=model_name, version=version, description=description
        )
    logger.info("Alias %s -> %s v%s", alias, model_name, version)
    return version