from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

from common import (
    COMMON_DOCKER_KWARGS,
    DEFAULT_ARGS,
    RAW_DATA,
    SELECTED_FEATURES,
    START_DATE,
)


with DAG(
    dag_id="select_features",
    default_args=DEFAULT_ARGS,
    description="Greedy forward selection of features",
    schedule=[RAW_DATA],
    start_date=START_DATE,
    catchup=False,
    tags=["features", "selection"],
) as dag:

    select = DockerOperator(
        task_id="greedy_selection",
        command="python -m src.select_features",
        **COMMON_DOCKER_KWARGS,
    )

    publish = DockerOperator(
        task_id="publish_features",
        command="echo 'Feature selection done'",
        outlets=[SELECTED_FEATURES],
        **COMMON_DOCKER_KWARGS,
    )

    select >> publish