from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sdk import Asset

from common import DATA_DIR, DEFAULT_ARGS, START_DATE, WORKDIR

RAW_DATA = Asset(f"file://{DATA_DIR}/raw/creditcard.csv")
SELECTED_FEATURES = Asset(f"file://{DATA_DIR}/selected_features.json")


with DAG(
    dag_id="select_features",
    default_args=DEFAULT_ARGS,
    description="Greedy forward selection of features",
    schedule=[RAW_DATA],
    start_date=START_DATE,
    catchup=False,
    tags=["features", "selection"],
) as dag:

    select_baseline = BashOperator(
        task_id="greedy_selection_baseline",
        bash_command=(
            f"cd {WORKDIR} && "
            f"python -m src.select_features "
            f"  --model baseline "
            f"  --scoring pr_auc "
            f"  --cv-folds 3 "
            f"  --max-features 20 "
            f"  --output {DATA_DIR}/selected_features.json"
        ),
    )

    publish = BashOperator(
        task_id="publish_features",
        bash_command="echo 'Feature selection done'",
        outlets=[SELECTED_FEATURES],
    )

    select_baseline >> publish