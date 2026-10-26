"""Validate source records, preserve dispositions and create reproducible stratified splits."""

import csv
from pathlib import Path
from typing import Any

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from churn_study.config import CATEGORICAL, COLUMNS, RAW_COLUMNS, Settings

RAW_SCHEMA = StructType(
    [StructField(name, StringType(), True) for name in RAW_COLUMNS]
    + [StructField("_corrupt_record", StringType(), True)]
)


def read_csv(spark: SparkSession, path: Path, source: str) -> DataFrame:
    """Validate column order before applying the positional raw schema."""
    if source not in {"development", "external"}:
        raise ValueError("source must be development or external")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if next(csv.reader(stream), []) != list(RAW_COLUMNS):
            raise ValueError(f"Unexpected CSV header: {path.name}")
    return (
        spark.read.schema(RAW_SCHEMA)
        .option("header", True)
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .csv(str(path.resolve()))
        .withColumn("source", F.lit(source))
    )


def _finite(column: Column) -> Column:
    return column.isNotNull() & ~F.isnan(column) & (F.abs(column) < 1e308)


def clean(raw: DataFrame) -> DataFrame:
    """Retain missing predictors for train-only imputation; quarantine invalid labels/domains."""
    result = raw
    for old, new in zip(RAW_COLUMNS, COLUMNS, strict=True):
        result = result.withColumnRenamed(old, new)
    all_empty = F.lit(True)
    for name in COLUMNS:
        all_empty = all_empty & (F.col(name).isNull() | (F.trim(name) == ""))
    result = result.withColumn("empty_record", all_empty)
    conditions: dict[str, Column] = {
        "empty_record": F.col("empty_record"),
        "csv_malformed": F.col("_corrupt_record").isNotNull(),
    }
    for name in (column for column in COLUMNS if column not in CATEGORICAL):
        result = result.withColumnRenamed(name, name + "_raw")
        result = result.withColumn(name, F.expr(f"try_cast(`{name}_raw` AS DOUBLE)"))
        supplied = F.col(name + "_raw").isNotNull() & (F.trim(name + "_raw") != "")
        condition = supplied & (~_finite(F.col(name)) | (F.col(name) < 0))
        if name in {"customer_id", "churn", "total_spend"}:
            condition = condition | ~_finite(F.col(name))
        if name == "churn":
            condition = condition | ~F.col(name).isin(0.0, 1.0)
        if name == "customer_id":
            condition = condition | (F.col(name) != F.floor(name)) | (F.col(name) <= 0)
        if name == "age":
            condition = condition | (F.col(name) > 120)
        conditions["invalid_" + name] = F.coalesce(condition, F.lit(False))
    for name in CATEGORICAL:
        result = result.withColumn(name, F.when(F.trim(name) != "", F.trim(name)))
    result = result.withColumn(
        "reasons",
        F.filter(
            F.array(
                *[F.when(condition, F.lit(reason)) for reason, condition in conditions.items()]
            ),
            lambda item: item.isNotNull(),
        ),
    )
    result = (
        result.withColumn("primary_reason", F.element_at("reasons", 1))
        .withColumn("valid", F.size("reasons") == 0)
        .withColumn("row_key", F.concat_ws(":", F.col("source"), F.col("customer_id").cast("long")))
        .withColumn(
            "feature_hash",
            F.sha2(
                F.to_json(
                    F.struct(*[name for name in COLUMNS if name not in {"customer_id", "churn"}])
                ),
                256,
            ),
        )
    )
    return result


def validate_identity(silver: DataFrame) -> None:
    """Fail on ambiguous within-file identity instead of silently leaking duplicates across splits."""
    if silver.groupBy("source", "customer_id").count().filter("count > 1").limit(1).count():
        raise ValueError("Repeated customer_id within a source; identity policy must be resolved")
    if silver.groupBy("source", "feature_hash").count().filter("count > 1").limit(1).count():
        raise ValueError(
            "Repeated feature rows within a source; grouped splitting must be resolved"
        )
    for source in ("development", "external"):
        if silver.filter(F.col("source") == source).select("churn").distinct().count() != 2:
            raise ValueError(f"Both churn classes are required in {source}")


def assign_splits(silver: DataFrame, settings: Settings) -> DataFrame:
    """Exactly stratify development data; external records never enter tuning or preprocessing."""
    development = silver.filter("source = 'development'")
    order = Window.partitionBy("churn").orderBy(
        F.xxhash64("row_key", F.lit(settings.seed)), "row_key"
    )
    group = Window.partitionBy("churn")
    ranked = development.withColumn("row_number", F.row_number().over(order)).withColumn(
        "class_size", F.count("*").over(group)
    )
    ranked = (
        ranked.withColumn(
            "split",
            F.when(F.col("row_number") <= F.floor(F.col("class_size") * 0.7), "train")
            .when(F.col("row_number") <= F.floor(F.col("class_size") * 0.85), "validate")
            .otherwise("test"),
        )
        .withColumn("fold", F.pmod(F.col("row_number") - 1, F.lit(settings.folds)).cast("int"))
        .drop("row_number", "class_size")
    )
    return ranked.unionByName(
        silver.filter("source = 'external'")
        .withColumn("split", F.lit("external"))
        .withColumn("fold", F.lit(-1))
    )


def quality_summary(classified: DataFrame, split: DataFrame) -> dict[str, Any]:
    """Reconcile input dispositions and report overlapping invalidity reasons separately."""
    dispositions = [
        row.asDict()
        for row in classified.groupBy("source", "valid", "primary_reason")
        .count()
        .orderBy("source", "valid", "primary_reason")
        .collect()
    ]
    split_counts = [
        row.asDict()
        for row in split.groupBy("source", "split", "churn")
        .count()
        .orderBy("source", "split", "churn")
        .collect()
    ]
    source_counts = {}
    for source in ("development", "external"):
        records = [item for item in dispositions if item["source"] == source]
        source_counts[source] = {
            "input_rows": sum(item["count"] for item in records),
            "valid_rows": sum(item["count"] for item in records if item["valid"]),
            "invalid_rows": sum(item["count"] for item in records if not item["valid"]),
        }
    return {
        "sources": source_counts,
        "dispositions": dispositions,
        "split_counts": split_counts,
        "reconciled": all(
            value["input_rows"] == value["valid_rows"] + value["invalid_rows"]
            for value in source_counts.values()
        ),
    }


def shift_audit(silver: DataFrame) -> dict[str, Any]:
    """Describe observed differences without claiming a timestamped drift or a causal effect."""
    numeric = [name for name in COLUMNS if name not in CATEGORICAL]
    summary = []
    for source in ("development", "external"):
        data = silver.filter(F.col("source") == source)
        row = (
            data.agg(
                F.count("*").alias("rows"),
                *[F.avg(name).alias("mean_" + name) for name in numeric],
                F.corr("customer_id", "churn").alias("id_churn_pearson"),
            )
            .first()
            .asDict()
        )
        row["source"] = source
        summary.append(row)
    support = [
        row.asDict()
        for row in silver.filter("support_calls >= 6")
        .groupBy("source")
        .agg(F.count("*").alias("rows"), F.avg("churn").alias("churn_rate"))
        .orderBy("source")
        .collect()
    ]
    a = silver.filter("source = 'development'").alias("a")
    b = silver.filter("source = 'external'").alias("b")
    overlap = (
        a.join(b, "customer_id")
        .agg(
            F.count("*").alias("shared_id_values"),
            F.sum(
                (
                    (F.col("a.feature_hash") == F.col("b.feature_hash"))
                    & (F.col("a.churn") == F.col("b.churn"))
                ).cast("int")
            ).alias("identical_rows_for_shared_id"),
        )
        .first()
        .asDict()
    )
    categories = [
        row.asDict()
        for name in CATEGORICAL
        for row in silver.groupBy("source", name)
        .count()
        .withColumn("feature", F.lit(name))
        .withColumnRenamed(name, "value")
        .select("source", "feature", "value", "count")
        .collect()
    ]
    numeric_bins = []
    for name, width in (
        ("age", 5),
        ("tenure", 5),
        ("support_calls", 1),
        ("payment_delay", 5),
        ("total_spend", 100),
    ):
        rows = (
            silver.select("source", (F.floor(F.col(name) / width) * width).alias("lower_bound"))
            .groupBy("source", "lower_bound")
            .count()
            .orderBy("source", "lower_bound")
            .limit(2001)
            .collect()
        )
        if len(rows) > 2000:
            raise ValueError(f"Too many diagnostic histogram bins for {name}")
        numeric_bins += [{"feature": name, "width": width, **row.asDict()} for row in rows]
    return {
        "source_summary": summary,
        "numeric_bins": numeric_bins,
        "support_calls_ge6": support,
        "identity_overlap": overlap,
        "category_counts": categories,
        "interpretation": "Observed between-file distribution/conditional differences; no temporal or causal claim",
    }
