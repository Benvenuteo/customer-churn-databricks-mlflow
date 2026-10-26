"""Optional Databricks AutoML with explicit features and the frozen development split."""

import json
from typing import Any

import mlflow
import mlflow.sklearn
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from churn_study.config import Settings, feature_columns
from churn_study.io import timestamp, write_json
from churn_study.metrics import choose_threshold, classification_metrics, regression_metrics


def automl_input(silver: DataFrame, task: str) -> DataFrame:
    """Whitelist model features; IDs, source, fold, hashes and the opposite target never enter AutoML."""
    target = "churn" if task == "classification" else "total_spend"
    return silver.filter("source = 'development'").select(
        *feature_columns(task), F.col(target).alias("label"), "split"
    )


def benchmark(
    silver: DataFrame, settings: Settings, task: str, timeout_minutes: int = 5, api: Any = None
) -> dict[str, Any]:
    """Execute only in a compatible authenticated Databricks ML runtime; never simulate a cloud run."""
    if timeout_minutes < 5:
        raise ValueError("Databricks AutoML requires a timeout of at least five minutes")
    if mlflow.active_run() is not None:
        raise ValueError("Call AutoML outside an active MLflow run")
    if api is None:
        try:
            from databricks import automl as api
        except ImportError as exc:
            raise RuntimeError(
                "This optional comparison requires Databricks AutoML on a compatible ML runtime"
            ) from exc
    method = api.classify if task == "classification" else api.regress
    summary = method(
        dataset=automl_input(silver, task),
        target_col="label",
        split_col="split",
        primary_metric="roc_auc" if task == "classification" else "rmse",
        timeout_minutes=timeout_minutes,
    )
    trial = summary.best_trial
    # Databricks generated tabular pipelines expose the sklearn flavor. Reject a
    # different interface explicitly rather than treating predicted labels as probabilities.
    flavors = mlflow.models.Model.load(trial.model_path).flavors
    if "sklearn" not in flavors:
        raise ValueError(
            "The best trial lacks the supported sklearn pipeline flavor; export its trial notebook to inspect its probability interface"
        )
    model = mlflow.sklearn.load_model(trial.model_path)
    features = list(feature_columns(task))
    target = "churn" if task == "classification" else "total_spend"
    frames = {}
    for split in ("validate", "test", "external"):
        frame = (
            silver.filter(F.col("split") == split)
            .select(*features, F.col(target).alias("label"), "row_key")
            .orderBy("row_key")
            .limit(200001)
            .toPandas()
        )
        if len(frame) > 200000 or frame.empty:
            raise ValueError("AutoML evaluation must contain 1–200000 rows per partition")
        if task == "classification":
            if not hasattr(model, "predict_proba"):
                raise ValueError("Classification pipeline must expose predict_proba")
            positive = list(model.classes_).index(1)
            frame["score"] = model.predict_proba(frame[features])[:, positive]
        else:
            frame["score"] = model.predict(frame[features])
        frames[split] = frame
    threshold = (
        choose_threshold(
            frames["validate"]["label"].to_numpy(), frames["validate"]["score"].to_numpy()
        )
        if task == "classification"
        else 0.5
    )
    evaluations = {}
    for split, frame in frames.items():
        y, score = frame["label"].to_numpy(), frame["score"].to_numpy()
        evaluations[split] = (
            classification_metrics(y, score, threshold, settings.seed)
            if task == "classification"
            else regression_metrics(y, score)
        )
    result = {
        "task": task,
        "executed_at": timestamp(),
        "runtime": "Databricks AutoML",
        "timeout_minutes": timeout_minutes,
        "features": features,
        "split_column": "split",
        "best_trial_run_id": trial.mlflow_run_id,
        "best_trial_model_uri": trial.model_path,
        "actual_trial_metric_keys": sorted(trial.metrics),
        "trial_metrics": {key: float(value) for key, value in trial.metrics.items()},
        "threshold": threshold,
        "evaluations": evaluations,
        "budget_note": "AutoML uses its configured time budget; manual CV uses the recorded training sample. Optimization budgets differ.",
    }
    write_json(settings.output / f"automl_{task}.json", result)
    with mlflow.start_run(run_id=trial.mlflow_run_id):
        mlflow.log_dict(result, "matched_holdout_evaluation.json")
    manifest_path = settings.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stages"]["automl_" + task] = {"status": "complete", "run_id": trial.mlflow_run_id}
    write_json(manifest_path, manifest)
    return result
