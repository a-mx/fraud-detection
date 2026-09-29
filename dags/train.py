from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sdk import Asset
from airflow.utils.task_group import TaskGroup
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from common import (
    CANDIDATE_MODELS, DATA_DIR, DEFAULT_ARGS, START_DATE, WORKDIR,
)

SELECTED_FEATURES = Asset(f"file://{DATA_DIR}/selected_features.json")
TRAINED_MODELS = Asset(f"file://{DATA_DIR}/trained_models.json")


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
        for model_name in CANDIDATE_MODELS:
            BashOperator(
                task_id=f"train_{model_name}",
                bash_command=(
                    f"cd {WORKDIR} && "
                    f"python -m src.main --model {model_name} "
                    f"  --no-tune "
                ),
            )

    train_bagging = BashOperator(
        task_id="train_bagging_xgb",
        bash_command=(
            f"cd {WORKDIR} && "
            f"python -m src.main --model bagging_xgb --no-tune"
        ),
    )

    publish = BashOperator(
        task_id="publish_trained_models",
        bash_command="echo 'All candidate models trained'",
        outlets=[TRAINED_MODELS],
    )

    train_group >> train_bagging >> publish

    trigger_promote = TriggerDagRunOperator(
        task_id="trigger_promote",
        trigger_dag_id="promote_best_model",
        wait_for_completion=False,
    )

    publish >> trigger_promote