# Databricks notebook source
# MAGIC %md
# MAGIC # Export the report
# MAGIC The HTML embeds its plots, so downloading one file preserves the figures.

# COMMAND ----------
# ruff: noqa: F821
from pathlib import Path

from churn_study.report import render_report

dbutils.widgets.text("run_directory", "")
report_path = render_report(Path(dbutils.widgets.get("run_directory")))
displayHTML(report_path.read_text())
print("Standalone report:", report_path)
