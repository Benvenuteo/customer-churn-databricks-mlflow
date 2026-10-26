"""Restartable data preparation and isolated, reproducible learning tasks."""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from churn_study.config import COLUMNS, Settings, feature_columns
from churn_study.data import (
    assign_splits,
    clean,
    quality_summary,
    read_csv,
    shift_audit,
    validate_identity,
)
from churn_study.io import environment, fingerprint, timestamp, write_json
from churn_study.metrics import (
    bootstrap_intervals,
    choose_threshold,
    classification_metrics,
    regression_metrics,
    segment_errors,
)
from churn_study.modeling import fit_full, model_details, prediction_frame, select_candidate
from churn_study.tracking import configure, log_and_verify

LOGGER = logging.getLogger(__name__)


def create_spark(settings: Settings) -> SparkSession:
    """Local entry point only; Databricks notebooks pass their existing session directly."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    spark = (
        SparkSession.builder.master(settings.master)
        .appName("CustomerStudy-" + settings.run_id)
        .config("spark.sql.shuffle.partitions", settings.partitions)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.csv.parser.columnPruning.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_table(spark: SparkSession, settings: Settings, name: str) -> DataFrame:
    return spark.read.format(settings.table_format).load(str(settings.output / name))


def prepare(
    spark: SparkSession, training: Path, external: Path, settings: Settings
) -> dict[str, Any]:
    """Persist raw/silver/quarantine tables and freeze splits before any model selection."""
    root = settings.output
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Output directory is not empty: {root}; use a new run directory")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = {
        "run_id": settings.run_id,
        "started_at": timestamp(),
        "settings": settings.to_dict(),
        "inputs": {"development": fingerprint(training), "external": fingerprint(external)},
        "environment": environment(),
        "execution": {
            "master": spark.sparkContext.master,
            "driver_memory": spark.sparkContext.getConf().get("spark.driver.memory", "1g"),
            "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
        },
        "stages": {},
    }
    write_json(root / "manifest.json", manifest)
    try:
        raw = read_csv(spark, training, "development").unionByName(
            read_csv(spark, external, "external")
        )
        raw.write.format(settings.table_format).mode("error").save(str(root / "bronze"))
        classified = clean(read_table(spark, settings, "bronze")).persist(StorageLevel.DISK_ONLY)
        try:
            silver = classified.filter("valid").select(
                *COLUMNS, "source", "row_key", "feature_hash"
            )
            validate_identity(silver)
            split = assign_splits(silver, settings).persist(StorageLevel.DISK_ONLY)
            try:
                split.repartition(settings.partitions).write.format(settings.table_format).mode(
                    "error"
                ).save(str(root / "silver"))
                classified.filter("NOT valid").write.format(settings.table_format).mode(
                    "error"
                ).save(str(root / "quarantine"))
                quality = quality_summary(classified, split)
                audit = shift_audit(split)
            finally:
                split.unpersist()
        finally:
            classified.unpersist()
        write_json(root / "quality.json", quality)
        write_json(root / "shift_audit.json", audit)
        manifest["stages"]["prepare"] = {
            "status": "complete",
            "seconds": time.perf_counter() - started,
            "sources": quality["sources"],
        }
        write_json(root / "manifest.json", manifest)
        LOGGER.info("Prepared sources: %s", quality["sources"])
        return manifest
    except Exception as exc:
        manifest["stages"]["prepare"] = {"status": "failed", "error_type": type(exc).__name__}
        write_json(root / "manifest.json", manifest)
        raise


def train_task(
    spark: SparkSession, settings: Settings, task: str, smoke: bool = False
) -> dict[str, Any]:
    """CV-select, refit, freeze a threshold, evaluate once, and preserve model lineage."""
    root = settings.output
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["stages"].get("prepare", {}).get("status") != "complete":
        raise ValueError("A completed preparation stage is required")
    if (root / f"{task}.json").exists() or (root / "models" / task).exists():
        raise ValueError(f"Existing {task} results will not be overwritten")
    started = time.perf_counter()
    target = "churn" if task == "classification" else "total_spend"
    data = read_table(spark, settings, "silver").withColumn("label", F.col(target))
    train = data.filter("split = 'train'").persist(StorageLevel.MEMORY_AND_DISK)
    try:
        training_count = train.count()
        if training_count < 50:
            raise ValueError("At least 50 development-training rows are required")
        configure(settings)
        with mlflow.start_run(run_name=f"{settings.run_id}-{task}") as active:
            mlflow.set_tags(
                {
                    "task": task,
                    "study_run_id": settings.run_id,
                    "external_used_for_selection": "false",
                    "smoke_profile": str(smoke),
                }
            )
            mlflow.log_params(
                {
                    "seed": settings.seed,
                    "folds": settings.folds,
                    "cv_fraction": settings.cv_fraction,
                    "training_rows": training_count,
                    "features": json.dumps(feature_columns(task)),
                    "target": target,
                }
            )
            mlflow.log_dict(manifest["inputs"], "input_fingerprints.json")
            winner, scores, cv_rows = select_candidate(train, task, settings, smoke)
            for record in scores:
                with mlflow.start_run(run_name="cv-" + record["name"], nested=True):
                    mlflow.log_params(
                        {
                            key: value
                            for key, value in record.items()
                            if key not in {"fold_scores", "mean", "std"}
                        }
                    )
                    mlflow.log_metrics({"cv_mean": record["mean"], "cv_std": record["std"]})
                    mlflow.log_dict({"scores": record["fold_scores"]}, "fold_scores.json")
            LOGGER.info(
                "Refitting %s winner on all %d training rows: %s", task, training_count, winner.name
            )
            model = fit_full(train, task, winner, settings)
            validation = prediction_frame(model, data.filter("split = 'validate'"), task)
            threshold = (
                choose_threshold(validation["label"].to_numpy(), validation["score"].to_numpy())
                if task == "classification"
                else 0.5
            )
            baseline_value = float(train.agg(F.avg("label")).first()[0])
            evaluations = {}
            prediction_directory = root / "predictions" / task
            prediction_directory.mkdir(parents=True, exist_ok=True)
            for split in ("validate", "test", "external"):
                frame = (
                    validation
                    if split == "validate"
                    else prediction_frame(model, data.filter(F.col("split") == split), task)
                )
                frame.to_parquet(prediction_directory / f"{split}.parquet", index=False)
                y = frame["label"].to_numpy()
                predicted = frame["score"].to_numpy()
                baseline = np.full(len(y), baseline_value)
                metrics = (
                    classification_metrics(y, predicted, threshold, settings.seed)
                    if task == "classification"
                    else regression_metrics(y, predicted)
                )
                base_metrics = (
                    classification_metrics(y, baseline, 0.5, settings.seed)
                    if task == "classification"
                    else regression_metrics(y, baseline)
                )
                intervals = (
                    bootstrap_intervals(
                        y, predicted, task, settings.bootstrap_repeats, settings.seed
                    )
                    if split != "validate"
                    else {"repeats": 0, "intervals": {}}
                )
                evaluations[split] = {
                    "model": metrics,
                    "baseline": base_metrics,
                    "bootstrap": intervals,
                    "segments": segment_errors(frame, task, threshold),
                }
                mlflow.log_metrics(
                    {
                        f"{split}_{name}": value
                        for name, value in metrics.items()
                        if isinstance(value, float | int) and value is not None
                    }
                )
            details = model_details(model, train)
            if task == "regression":
                from churn_study.regularization import coefficient_paths

                paths = coefficient_paths(train, model, settings, smoke)
                write_json(root / "regularization.json", paths)
                mlflow.log_dict(paths, "regularization_paths.json")
            lineage = log_and_verify(model, train, task, settings)
            result = {
                "task": task,
                "target": target,
                "features": feature_columns(task),
                "cv_selection_metric": "ROC-AUC" if task == "classification" else "RMSE",
                "cv_sample_rows": cv_rows,
                "cv_fraction_requested": settings.cv_fraction,
                "folds": settings.folds,
                "candidates": scores,
                "winner": winner.name,
                "full_training_rows": training_count,
                "threshold": threshold,
                "threshold_selection": "validation F1" if task == "classification" else None,
                "baseline_training_mean": baseline_value,
                "evaluations": evaluations,
                "model_details": details,
                "lineage": lineage,
                "mlflow_run_id": active.info.run_id,
                "seconds": time.perf_counter() - started,
            }
            mlflow.log_dict(result, "evaluation.json")
            write_json(root / f"{task}.json", result)
        manifest["stages"][task] = {
            "status": "complete",
            "seconds": time.perf_counter() - started,
            "mlflow_run_id": result["mlflow_run_id"],
        }
        write_json(root / "manifest.json", manifest)
        return result
    except Exception as exc:
        manifest["stages"][task] = {"status": "failed", "error_type": type(exc).__name__}
        write_json(root / "manifest.json", manifest)
        raise
    finally:
        train.unpersist()
