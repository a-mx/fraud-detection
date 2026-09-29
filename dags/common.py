import os
from datetime import datetime, timedelta

from airflow.sdk import Asset
from docker.types import Mount
WORKDIR = "/app"
DATA_DIR = "/app/data"
APP_IMAGE = "fraud-detection-api:latest"
DOCKER_NETWORK = "fraud-detection_default"

APP_ENV_KEYS = [
    "MLFLOW_TRACKING_URI",
    "MLFLOW_MODEL_URI",
    "DB_HOST",
    "DB_PORT",
    "DB_USER",
    "DB_PASSWORD",
    "DB_NAME",
    "KAGGLE_API_TOKEN",
    "RELOAD_TOKEN",
]

APP_ENV = {k: os.environ[k] for k in APP_ENV_KEYS if k in os.environ}

COMMON_DOCKER_KWARGS = dict(
    image=APP_IMAGE,
    docker_url="unix://var/run/docker.sock",
    network_mode=DOCKER_NETWORK,
    auto_remove="force",
    mount_tmp_dir=False,
    environment=APP_ENV,
    working_dir=WORKDIR,
    force_pull=False,
    mounts=[
        Mount(
            source="fraud-detection_app_data",
            target="/app/data",
            type="volume",
        ),
    ],
)

RAW_DATA = Asset(f"file://{DATA_DIR}/raw/creditcard.csv")
SELECTED_FEATURES = Asset(f"file://{DATA_DIR}/selected_features.json")
TRAINED_MODELS = Asset(f"file://{DATA_DIR}/trained_models.json")

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}

START_DATE = datetime(2026, 9, 29)
MIN_PR_AUC = 0.75
MIN_RECALL = 0.80