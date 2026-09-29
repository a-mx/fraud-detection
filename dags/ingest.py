from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator

from common import (
    COMMON_DOCKER_KWARGS,
    DATA_DIR,
    DEFAULT_ARGS,
    RAW_DATA,
    START_DATE,
)






with DAG(
    dag_id="ingest_data",
    default_args=DEFAULT_ARGS,
    description="Download and validate fraud detection dataset",
    schedule="@weekly",
    start_date=START_DATE,
    catchup=False,
    tags=["data", "ingest"],
) as dag:

    check_env = DockerOperator(
        task_id="check_environment",
        command="bash -c \"ls -la /app/src/ && python --version && "
                "(pip list | grep -E 'kaggle|mlflow|pandas|scikit' || echo 'some missing')\"",
        **COMMON_DOCKER_KWARGS,
    )

    ingest = DockerOperator(
        task_id="ingest_dataset",
        command="python -m src.data.ingest",
        **COMMON_DOCKER_KWARGS,
    )

    validate = DockerOperator(
        task_id="validate_dataset",
        command=(
            "python -c \""
            "import pandas as pd; from pathlib import Path; "
            "p = Path('/app/data/raw/creditcard.csv'); "
            "assert p.exists(), f'Missing: {p}'; "
            "df = pd.read_csv(p); "
            "assert df.shape[1] == 31, f'Expected 31 cols, got {df.shape[1]}'; "
            "assert not df.isna().any().any(), 'NaN found'; "
            "fr = df['Class'].mean(); "
            "assert 0.001 < fr < 0.01, f'Bad fraud rate: {fr}'; "
            "print(f'OK: {len(df)} rows, fraud_rate={fr:.4f}')"
            "\""
        ),
        **COMMON_DOCKER_KWARGS,
)
    publish = DockerOperator(
        task_id="publish_dataset",
        command="echo 'Dataset ready for downstream DAGs'",
        outlets=[RAW_DATA],
        **COMMON_DOCKER_KWARGS,
    )

    check_env >> ingest >> validate >> publish