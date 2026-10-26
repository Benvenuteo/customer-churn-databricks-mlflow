"""MLflow experiment lineage, raw-input signatures, aliases and load-back verification."""

import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.spark
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from churn_study.config import Settings, feature_columns


def configure(settings: Settings) -> str:
    """Use SQLite locally or the caller's authenticated tracking server in Databricks."""
    uri = settings.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        uri = "sqlite:///" + str((settings.output / "mlflow.db").resolve())
    mlflow.set_tracking_uri(uri)
    registry = os.environ.get("MLFLOW_REGISTRY_URI")
    mlflow.set_registry_uri(registry or ("databricks-uc" if uri == "databricks" else uri))
    experiment = mlflow.get_experiment_by_name(settings.experiment_name)
    if experiment is None:
        location = (
            (settings.output / "mlflow-artifacts").resolve().as_uri()
            if uri.startswith("sqlite:")
            else None
        )
        experiment_id = mlflow.create_experiment(
            settings.experiment_name, artifact_location=location
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def log_and_verify(
    model: PipelineModel, train: DataFrame, task: str, settings: Settings
) -> dict[str, Any]:
    """Log a complete pipeline, verify native Spark predictions, then assign its registry alias."""
    columns = list(feature_columns(task))
    example_spark = train.select(*columns).limit(5)
    example = example_spark.toPandas()
    output = model.transform(example_spark).select("prediction").toPandas()
    signature = infer_signature(example, output)
    info = mlflow.spark.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=example,
        pip_requirements=[f"pyspark=={train.sparkSession.version}"],
    )
    uri = info.model_uri
    restored = mlflow.spark.load_model(uri)
    expected = output["prediction"].to_numpy()
    actual = (
        restored.transform(example_spark).select("prediction").toPandas()["prediction"].to_numpy()
    )
    if not np.allclose(expected, actual, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Predictions changed after MLflow save/load")
    native_path = settings.output / "models" / task
    model.write().save(str(native_path))
    name = settings.registry_prefix + "_" + task
    version = mlflow.register_model(uri, name)
    MlflowClient().set_registered_model_alias(name, "candidate", version.version)
    return {
        "run_id": mlflow.active_run().info.run_id,
        "model_uri": uri,
        "registered_model": name,
        "version": str(version.version),
        "alias": "candidate",
        "native_path": str(Path("models") / task),
        "roundtrip_predictions_equal": True,
        "signature_input_columns": columns,
        "promotion_policy": "Candidate is not automatically approved for real customer decisions",
    }
