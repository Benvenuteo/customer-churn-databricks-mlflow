# Databricks notebook source
# MAGIC %md
# MAGIC # Train a complete customer-study pipeline
# MAGIC Candidate selection uses training-only CV. Every winning pipeline is refit
# MAGIC on all development-training records, logged to MLflow and verified after reload.
# MAGIC The registered alias is `candidate`, not an automatic production promotion.

# COMMAND ----------
# ruff: noqa: F821
import json
from pathlib import Path

from churn_study.config import Settings
from churn_study.engine import train_task

dbutils.widgets.text("run_directory", "")
dbutils.widgets.dropdown("task", "classification", ["classification", "regression"])
root = Path(dbutils.widgets.get("run_directory"))
values = json.loads((root / "manifest.json").read_text())["settings"]
values.update(output=root, tracking_uri="databricks")
settings = Settings(**values)
result = train_task(spark, settings, dbutils.widgets.get("task"))
display(result["evaluations"])
