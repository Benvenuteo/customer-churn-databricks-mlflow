# Databricks notebook source
# MAGIC %md
# MAGIC # Optional matched AutoML comparison
# MAGIC This notebook starts actual Databricks AutoML trials and uses cluster time.
# MAGIC It is deliberately separate from the default job. It requires AutoML,
# MAGIC access to the study output and a compatible ML runtime supporting split_col.
# MAGIC Local results must never be presented as evidence that this notebook ran.

# COMMAND ----------
# ruff: noqa: F821
import json
from pathlib import Path

import mlflow

from churn_study.automl import benchmark
from churn_study.config import Settings
from churn_study.engine import read_table
from churn_study.report import render_report

dbutils.widgets.text("run_directory", "")
dbutils.widgets.text("timeout_minutes", "5")
root = Path(dbutils.widgets.get("run_directory"))
values = json.loads((root / "manifest.json").read_text())["settings"]
values.update(output=root, tracking_uri="databricks")
settings = Settings(**values)
mlflow.set_tracking_uri("databricks")
silver = read_table(spark, settings, "silver")
for task in ("classification", "regression"):
    result = benchmark(silver, settings, task, int(dbutils.widgets.get("timeout_minutes")))
    display(result["evaluations"])
displayHTML(render_report(root).read_text())
