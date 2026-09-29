from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from common import DEFAULT_ARGS, START_DATE


with DAG(
    dag_id="full_ml_pipeline",
    default_args=DEFAULT_ARGS,
    description="Orchestrates: ingest → select → train → promote",
    schedule="@weekly",
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    tags=["pipeline", "orchestration"],
) as dag:

    ingest = TriggerDagRunOperator(
        task_id="trigger_ingest",
        trigger_dag_id="ingest_data",
        wait_for_completion=True,
        poke_interval=30,
    )

    select = TriggerDagRunOperator(
        task_id="trigger_feature_selection",
        trigger_dag_id="select_features",
        wait_for_completion=True,
        poke_interval=30,
    )

    train = TriggerDagRunOperator(
        task_id="trigger_train",
        trigger_dag_id="train_models",
        wait_for_completion=True,
        poke_interval=30,
    )

    ingest >> select >> train