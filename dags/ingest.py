from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sdk import Asset

from common import DEFAULT_ARGS, DATA_DIR, START_DATE, SRC_DIR, WORKDIR
RAW_DATA = Asset(f"file://{DATA_DIR}/raw/creditcard.csv")


def _validate_data():
    import pandas as pd
    from pathlib import Path

    path = Path(f"{DATA_DIR}/raw/creditcard.csv")
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    expected_cols = 31

    if df.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, got {df.shape[1]}")

    if df.isna().any().any():
        raise ValueError("Dataset contains NaN values")

with DAG(
    dag_id="ingest_data",
    default_args=DEFAULT_ARGS,
    description="Download and validate fraud detection dataset",
    schedule="@weekly",
    start_date=START_DATE,
    catchup=False,
    tags=["data", "ingest"],
) as dag:

    check_env = BashOperator(
        task_id="check_environment",
        bash_command=f"ls -la {SRC_DIR}/ && python --version && pip list | grep -E 'mlflow|pandas|scikit'",
    )

    ingest = BashOperator(
        task_id="ingest_dataset",
        bash_command=f"cd {WORKDIR} && python -m src.data.ingest",
    )

    validate = PythonOperator(
        task_id="validate_dataset",
        python_callable=_validate_data,
    )

    publish = BashOperator(
        task_id="publish_dataset",
        bash_command="echo 'Dataset ready for downstream DAGs'",
        outlets=[RAW_DATA],
    )

    check_env >> ingest >> validate >> publish