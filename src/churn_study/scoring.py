"""Validate new raw records and apply a saved pipeline without manual preprocessing."""

import csv
import json
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from churn_study.config import CATEGORICAL, COLUMNS, RAW_COLUMNS, Settings, feature_columns
from churn_study.engine import create_spark


def score_csv(
    input_path: Path, model_path: Path, output: Path, task: str, threshold: float | None = None
) -> None:
    """Accept original or normalized headers, optional targets, missing predictors and new categories."""
    if output.exists():
        raise FileExistsError(f"Scoring output already exists: {output}")
    if task == "classification" and threshold is None:
        metadata = model_path.parent.parent / "classification.json"
        if not metadata.exists():
            raise ValueError("Provide --threshold or use a model from a complete run directory")
        threshold = float(json.loads(metadata.read_text())["threshold"])
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    mapping = dict(zip(RAW_COLUMNS, COLUMNS, strict=True))
    normalized = [mapping.get(name, name) for name in header]
    required = feature_columns(task)
    if len(set(normalized)) != len(normalized) or not set(required).issubset(normalized):
        raise ValueError("Scoring CSV requires unique columns and every task feature")
    spark = create_spark(Settings(output, run_id="batch-scoring"))
    try:
        schema = StructType([StructField(name, StringType(), True) for name in header])
        data = (
            spark.read.schema(schema)
            .option("header", True)
            .option("mode", "FAILFAST")
            .csv(str(input_path.resolve()))
        )
        for old, new in zip(header, normalized, strict=True):
            data = data.withColumnRenamed(old, new)
        for name in required:
            if name in CATEGORICAL:
                data = data.withColumn(name, F.when(F.trim(name) != "", F.trim(name)))
            else:
                supplied = F.col(name).isNotNull() & (F.trim(name) != "")
                value = F.expr(f"try_cast(`{name}` AS DOUBLE)")
                invalid = supplied & (
                    value.isNull() | F.isnan(value) | (F.abs(value) >= 1e308) | (value < 0)
                )
                if name == "age":
                    invalid = invalid | (value > 120)
                if data.filter(invalid).limit(1).count():
                    raise ValueError(f"Invalid supplied numeric predictor: {name}")
                data = data.withColumn(name, value)
        predicted = PipelineModel.load(str(model_path.resolve())).transform(data)
        identifiers = ["customer_id"] if "customer_id" in data.columns else []
        if task == "classification":
            result = predicted.withColumn("churn_probability", vector_to_array("probability")[1])
            result = result.withColumn(
                "churn_decision", (F.col("churn_probability") >= threshold).cast("int")
            )
            result = result.select(*identifiers, "churn_probability", "churn_decision")
        else:
            result = predicted.select(
                *identifiers, F.col("prediction").alias("predicted_total_spend")
            )
        result.write.mode("error").parquet(str(output.resolve()))
    finally:
        spark.stop()
