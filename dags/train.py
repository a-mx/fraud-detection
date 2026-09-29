from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.task_group import TaskGroup

from common import (
    COMMON_DOCKER_KWARGS,
    DEFAULT_ARGS,
    SELECTED_FEATURES,
    START_DATE,
    TRAINED_MODELS,
)

CANDIDATE_MODELS = ["baseline", "xgb", "mlp"]


with DAG(
    dag_id="train_models",
    default_args=DEFAULT_ARGS,
    description="Train candidate models in parallel",
    schedule=[SELECTED_FEATURES],
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    tags=["training"],
) as dag:

    with TaskGroup("train") as train_group:
        for model in CANDIDATE_MODELS:
            DockerOperator(
                task_id=f"train_{model}",
                command=f"python -m src.main --model {model} --no-tune",
                **COMMON_DOCKER_KWARGS,
            )

    publish = DockerOperator(
        task_id="publish_trained_models",
        command="echo 'All candidate models trained'",
        outlets=[TRAINED_MODELS],
        **COMMON_DOCKER_KWARGS,
    )

    trigger_promote = TriggerDagRunOperator(
        task_id="trigger_promote",
        trigger_dag_id="promote_best_model",
        wait_for_completion=False,
    )

    train_group >> publish >> trigger_promote