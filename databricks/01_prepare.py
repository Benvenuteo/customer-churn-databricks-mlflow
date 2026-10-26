# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare the customer study
# MAGIC Install the project wheel on a compatible dedicated ML cluster. Use the
# MAGIC platform Spark installation. Set the existing Volume directories and the
# MAGIC three-level registry prefix to locations you can access.

# COMMAND ----------
# ruff: noqa: F821
from pathlib import Path

from churn_study.config import Settings
from churn_study.engine import prepare

dbutils.widgets.text("input_directory", "")
dbutils.widgets.text("output_directory", "")
dbutils.widgets.text("run_id", "customer-study-v1")
dbutils.widgets.text("experiment_path", "/Shared/customer-churn-spending")
dbutils.widgets.text("registry_prefix", "")

input_directory = Path(dbutils.widgets.get("input_directory"))
output_directory = Path(dbutils.widgets.get("output_directory"))
registry_prefix = dbutils.widgets.get("registry_prefix")
if not str(input_directory).startswith("/Volumes/") or not str(output_directory).startswith(
    "/Volumes/"
):
    raise ValueError("Set input_directory and output_directory to authorized /Volumes/... paths")
if registry_prefix.count(".") != 2:
    raise ValueError("registry_prefix must be catalog.schema.model_prefix")

settings = Settings(
    output=output_directory / dbutils.widgets.get("run_id"),
    run_id=dbutils.widgets.get("run_id"),
    tracking_uri="databricks",
    experiment_name=dbutils.widgets.get("experiment_path"),
    registry_prefix=registry_prefix,
    table_format="delta",
)
manifest = prepare(
    spark,
    input_directory / "customer_churn_dataset-training-master.csv",
    input_directory / "customer_churn_dataset-testing-master.csv",
    settings,
)
display(
    spark.read.format("delta")
    .load(str(settings.output / "silver"))
    .groupBy("source", "split", "churn")
    .count()
)
