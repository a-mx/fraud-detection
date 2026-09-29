from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

from common import (
    CANDIDATE_MODELS, DATA_DIR, DEFAULT_ARGS, MIN_PR_AUC, MIN_RECALL,
    START_DATE, WORKDIR,
)


def _pick_best_model(**context):
    """Czyta metryki z MLflow i zwraca nazwę najlepszego modelu, o ile spełnia progi."""
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name("fraud-detection")
    if experiment is None:
        raise ValueError("Experiment 'fraud-detection' not found")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=200,
    )

    best = {"model": None, "pr_auc": 0.0, "recall": 0.0, "run_id": None}
    seen = set()

    for run in runs:
        model = run.data.params.get("model")
        if model in seen or model not in CANDIDATE_MODELS:
            continue
        seen.add(model)

        pr_auc = run.data.metrics.get("pr_auc", 0.0)
        recall = run.data.metrics.get("recall", 0.0)

        if pr_auc > best["pr_auc"]:
            best = {
                "model": model,
                "pr_auc": pr_auc,
                "recall": recall,
                "run_id": run.info.run_id,
            }

    print(f"Best candidate: {best}")

    if best["pr_auc"] < MIN_PR_AUC or best["recall"] < MIN_RECALL:
        return "skip_promotion"

    context["ti"].xcom_push(key="best_model", value=best["model"])
    context["ti"].xcom_push(key="pr_auc", value=best["pr_auc"])
    return "promote_best"


with DAG(
    dag_id="promote_best_model",
    default_args=DEFAULT_ARGS,
    description="Pick best model by PR-AUC and promote to production",
    schedule=None,
    start_date=START_DATE,
    catchup=False,
    tags=["promotion"],
) as dag:

    pick = BranchPythonOperator(
        task_id="pick_best_model",
        python_callable=_pick_best_model,
    )

    promote = BashOperator(
        task_id="promote_best",
        bash_command=(
            f"cd {WORKDIR} && "
            f'python -m src.main --model {{{{ ti.xcom_pull(task_ids="pick_best_model", key="best_model") }}}} '
            f"  --promote"
        ),
    )

    reload_api = BashOperator(
        task_id="reload_api",
        bash_command=(
            'curl -fsS -X POST http://api:8000/reload '
            '-H "X-Reload-Token: {{ var.value.reload_token }}"'
        ),
    )

    notify = BashOperator(
        task_id="notify_success",
        bash_command=(
            'echo "Promoted {{ ti.xcom_pull(task_ids=\'pick_best_model\', key=\'best_model\') }}" '
            '"with PR-AUC={{ ti.xcom_pull(task_ids=\'pick_best_model\', key=\'pr_auc\') }}"'
        ),
    )

    skip = BashOperator(
        task_id="skip_promotion",
        bash_command='echo "No candidate met quality thresholds — skipping"',
        trigger_rule="none_failed_min_one_success",
    )

    pick >> [promote, skip]
    promote >> reload_api >> notify