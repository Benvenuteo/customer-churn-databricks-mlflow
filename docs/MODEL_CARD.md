# Candidate model card

## Intended use

Portfolio/research demonstration of reproducible tabular ML and external generalization checks. The classification target is the supplied churn label; the regression target is recorded total spending. No dated future prediction horizon is established.

## Training and evaluation

- Source: Muhammad Shahid Azeem, Customer Churn Dataset, version 1; input SHA-256 hashes are in the run manifest.
- Full development training: 308,582 valid records.
- Training-only CV: 77,490 sampled records, three folds, train-only preprocessing.
- Validation: 66,125 rows, used for the classification threshold.
- Internal test: 66,125 rows. External file: 64,374 rows.
- Churn winner: depth-5 GBT, 20 boosting iterations. Spending winner: depth-3 GBT, 20 iterations.
- Exact scores, bootstrap intervals, feature diagnostics and run IDs: `reports/provided-v1`.

## Known limitations

The churn model's external ROC-AUC is approximately 0.616 and its top-decile lift is approximately 0.891. Probability calibration deteriorates substantially on the external file. These measurements do not support presenting it as a validated retention decision system. The registry alias is `candidate`.

Conditional target relationships differ markedly between the source files. Shared numeric IDs do not establish global customer identity. Observation dates, source sampling, feature availability before churn, currency and the real/synthetic generation process are not established by the data card. Very high internal scores do not resolve these questions.

Bootstrap intervals describe fixed-model evaluation uncertainty only. One declared split seed and one declared CV sampling scheme were used; robustness over many retraining seeds is not established. Tree importance is descriptive, not causal.

## Deployment interface

Native Spark pipelines accept the declared raw feature columns, including supported unknown categories and missing predictors. The scoring CLI validates supplied numeric values and applies the saved validation threshold. Model save/load predictions were checked for equality. The handoff ZIP includes native candidate model exports and threshold metadata.

The rewritten implementation was exercised locally with Spark 3.5.7, Python 3.11 and Java 17. Databricks/Unity Catalog/AutoML notebooks require an authenticated compatible workspace and a separate recorded cloud run. The original university notebook's historical cloud output is not reused as evidence for the rewritten code.
