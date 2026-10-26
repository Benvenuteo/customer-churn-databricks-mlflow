# Databricks workspace run

The original university export contains Azure Databricks experiment output. The rewritten version has been measured locally; this document does not claim a new cloud run occurred.

## Requirements

- An authenticated Databricks workspace and an authorized dedicated/classic ML cluster with compatible Spark 3.5 and Python 3.11/3.12.
- For optional AutoML, a runtime supporting the tabular AutoML API and manual `split_col` (15.3 ML or newer), plus workspace entitlement to that feature.
- Existing Unity Catalog catalog/schema and a Volume with read/write permissions for input/output files.
- Permission to create an MLflow experiment and register candidate models in the chosen catalog/schema.

Use the runtime's Spark installation. Build the project wheel locally with `python -m build --wheel`, then install that wheel on the cluster. Do not install `requirements.txt` or the `local` extra on Databricks because they pin a standalone Spark installation. If installing manually into a compatible preconfigured ML runtime, install the wheel with `--no-deps` and confirm its declared core dependency ranges are available.

## Manual notebook workflow

1. Upload the two source CSVs into an authorized Volume directory.
2. Import `databricks/01_prepare.py` as a Python notebook. Set `input_directory`, `output_directory`, `run_id`, `experiment_path` and `registry_prefix` to your existing resources. The registry prefix is `catalog.schema.model_prefix`.
3. Run preparation. Inspect `quality.json`, `shift_audit.json` and split counts before training.
4. Run `02_train.py` for `classification`, then `regression`, using the same `run_directory`. Sequential execution preserves the shared manifest.
5. Run `03_report.py` and download its self-contained HTML report.
6. Optionally run `04_automl_comparison.py`. This starts real AutoML trials with the configured time budget and updates the report only after measured results exist.

MLflow logs complete raw-input Spark pipelines, signatures, dataset fingerprints, candidate CV results and held-out metrics. A model version receives the alias `candidate`; no automated production promotion is performed. Registered Unity Catalog models require signatures and appropriate access rights.

## Bundle workflow

After installing the Databricks CLI and completing its supported login flow, supply deployment variables through your CLI configuration or a local ignored variable file. Required variable names are `cluster_id`, `input_directory`, `output_directory`, and `registry_prefix`; `run_id` and `experiment_path` can be overridden as well.

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run customer_study -t dev
```

The parameterized bundle intentionally has no embedded account host, credentials, cluster identifier or storage account key. Validation and deployment need the required variable values; Python/YAML checks alone do not establish remote compatibility. Use a fresh run ID for each full experiment.

## Evidence needed after a real cloud run

Export the new manifest, run/job ID, runtime version, experiment references, matched AutoML evaluation and report. Keep actual cloud measurements separate from the committed local results. Account-specific paths and identities should be reviewed before publishing screenshots or logs.

Official references:

- https://learn.microsoft.com/en-us/azure/databricks/machine-learning/automl/automl-api-reference
- https://learn.microsoft.com/en-us/azure/databricks/machine-learning/manage-model-lifecycle/
- https://learn.microsoft.com/en-us/azure/databricks/dbfs/
- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/bundles/jobs-tutorial
