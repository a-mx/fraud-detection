from datetime import datetime, timedelta


WORKDIR = "/opt/airflow"
SRC_DIR = f"{WORKDIR}/src"
DATA_DIR = f"{WORKDIR}/data"

CANDIDATE_MODELS = ["baseline", "xgb", "mlp"]


MIN_PR_AUC = 0.75
MIN_RECALL = 0.80

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}

START_DATE = datetime(2026, 9, 29)