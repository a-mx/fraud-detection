from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.providers.docker.operators.docker import DockerOperator

from common import (
    COMMON_DOCKER_KWARGS,
    DEFAULT_ARGS,
    MIN_PR_AUC,
    MIN_RECALL,
    START_DATE,
)

CANDIDATE_MODELS = ["baseline", "xgb", "mlp", "bagging_xgb"]


def _pick_best_model(**context):
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name("fraud-detection")
    if experiment is None:
        return "skip_promotion"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=200,
    )

    best = {"model": None, "pr_auc": 0.0, "recall": 0.0}
    seen = set()

    for run in runs:
        model = run.data.params.get("model")
        if not model or model in seen or model not in CANDIDATE_MODELS:
            continue
        seen.add(model)

        pr_auc = run.data.metrics.get("pr_auc", 0.0)
        recall = run.data.metrics.get("recall", 0.0)

        if pr_auc > best["pr_auc"]:
            best = {"model": model, "pr_auc": pr_auc, "recall": recall}

    if best["model"] is None:
        return "skip_promotion"

    if best["pr_auc"] < MIN_PR_AUC or best["recall"] < MIN_RECALL:
        print(f"Best {best['model']} below thresholds: {best}")
        return "skip_promotion"

    context["ti"].xcom_push(key="best_model", value=best["model"])
    context["ti"].xcom_push(key="pr_auc", value=best["pr_auc"])
    print(f"Best: {best}")
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

    promote = DockerOperator(
        task_id="promote_best",
        command=(
            'python -m src.main '
            '--model {{ ti.xcom_pull(task_ids="pick_best_model", key="best_model") }} '
            '--promote'
        ),
        **COMMON_DOCKER_KWARGS,
    )

    reload_api = DockerOperator(
        task_id="reload_api",
        command=(
            'bash -c "curl -fsS -X POST http://api:8000/reload '
            '-H \\"X-Reload-Token: $RELOAD_TOKEN\\""'
        ),
        **COMMON_DOCKER_KWARGS,
    )

    skip = DockerOperator(
        task_id="skip_promotion",
        command="echo 'No candidate met quality thresholds — skipping'",
        trigger_rule="none_failed_min_one_success",
        **COMMON_DOCKER_KWARGS,
    )

    pick >> [promote, skip]
    promote >> reload_api