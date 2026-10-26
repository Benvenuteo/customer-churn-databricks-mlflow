# Audit of the university notebook

Input: a 21-cell Databricks DBC export containing regression, classification, MLflow registration and AutoML output. The source archive stores cell positions separately from serialization order; the maintained audit uses display positions. The original ROC cell correctly appears after logistic training in that order.

Material corrections:

| Original behavior | Maintained implementation |
| --- | --- |
| Shared mutable `df_training_final` / `label` variables reused for two targets | Independent task functions and explicit feature contracts |
| Nominal `StringIndexer` outputs fed directly to linear models | One-hot categorical encoding inside each fold's pipeline |
| Training ROC with an external-test AUC legend | Curves and legend metrics use the same held-out population |
| One regularization value, no OLS/mean baseline | Predeclared penalty grid, OLS/mean comparisons, coefficient paths |
| Manual models receive hand-selected features; AutoML receives ID and, for regression, Churn | Feature whitelists match; ID excluded, opposite regression target excluded |
| AutoML's internal holdout compared with hand-built models on the supplied external file | Explicit shared development split and matched final holdouts |
| `metrics.get('rmse')` / `metrics.get('roc_auc')` prints `Not available` | Actual metric keys recorded, independent holdout metrics computed |
| Only fitted estimators logged | Complete preprocessing/model pipelines, raw-input signatures and load-back checks |
| Generic model names and unconditional legacy `Staging` transitions | Configured registry prefix and `candidate` aliases |
| Broad conclusions that regularization and AutoML improve generalization | Claims follow measured comparisons and documented limitations |

The original preprocessing was fitted on the training file and then applied to testing. That alone is not evidence of train/test preprocessing leakage. The AutoML input mismatch, different evaluation populations and ID associations are actual issues to correct; perfect historical AutoML scores are not asserted to prove a particular leakage mechanism.

The raw export contains historical account/workspace metadata and remains in the user's original supplied materials rather than being republished as a maintained notebook. This project retains its academic provenance through this audit.
