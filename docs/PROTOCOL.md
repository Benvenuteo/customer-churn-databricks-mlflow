# Study protocol and interpretation

## Population and information set

The records are the two supplied version-1 customer CSVs. A customer is the row unit; the author provides no verified dated observation window, future prediction horizon, currency or sampling description. The targets are the supplied churn label and recorded aggregate spending. The study measures associations/prediction within this dataset, not causality or future customer value.

Classification predictors are age, tenure, usage frequency, support calls, payment delay, total spending, last interaction, gender, subscription type and contract length. Spending regression excludes spending from inputs and also excludes churn. IDs, hashes, source tags, partition labels and fold indices are never predictive features.

Availability of aggregates before a real future churn event is not established. A deployed forecasting use case would require a timestamped feature/label design and a proper outcome horizon. Unknown categories are supported; missing predictors use training medians. Static numeric domain checks identify invalid values without trimming legitimate extremes.

## Splits and selection

After quality validation, within-file identity is checked. Development rows are stratified by churn using a seeded hash ordering and exact 70/15/15 counts. IDs have source-local scope. No chronological or global-customer grouping is claimed. The entire separate external file remains external.

Approximately 25% of the training partition is selected with a second independent seeded hash for the bounded CV search. Fold assignment is approximately balanced within the original churn strata. Every fold learns its own preprocessing; candidate estimators reuse only that fold's fitted preprocessing. Mean CV ROC-AUC (classification) or RMSE (regression) selects one candidate. Fold SD is reported as descriptive variability.

The winner is refitted on the entire training partition. Validation F1 selects a threshold from 0.05 to 0.95; ties prefer a value closest to 0.5. Internal test and external data are scored using the same frozen model/threshold. No further candidates are fitted because an external result is disappointing.

The external file's distributions and conditional label patterns were already inspected during the data audit. It is therefore described as an external transfer stress test; it is not claimed to be a previously unseen business deployment. Its labels are not used for preprocessing, hyperparameters or threshold selection.

## Uncertainty and errors

The classifier reports ROC-AUC, average precision, thresholded metrics, Brier score, log loss and a confusion matrix. The ranking budget is the highest-scored 10%, with seeded tie-breaking to avoid accidental ID-order bias. AP, rather than trapezoidal integration, is the stated precision-recall summary.

The regressor reports MAE, RMSE and R². R² compares against the evaluated partition's own mean. The deployable baseline uses the training mean; these references differ under distribution shift. Negative external R² does not itself establish that the model loses to the training-mean baseline.

Two hundred percentile bootstrap replicates estimate uncertainty conditional on fixed held-out predictions. Classification preserves class counts while resampling within classes. These intervals do not quantify repeated-training, hyperparameter-search, split-seed, sampling-process or deployment uncertainty. Gender/contract segment summaries are descriptive error diagnostics, not a claim that fairness has been established.

## Regularization

The regression grid includes OLS, Ridge, Lasso, Elastic Net and a tree candidate. Additional training-only linear refits show coefficient paths per one training standard deviation, keeping target units unchanged. They are diagnostic and cannot change the frozen candidate selection. Sparse unseen-category columns and the nominal encoding can make OLS covariance singular; MLlib's fallback solver handles prediction, but individual OLS coefficients should not be presented as uniquely identified causal effects.

## Measured conclusion

The selected churn GBT has approximately 0.997 internal ROC-AUC and 0.616 external ROC-AUC. External top-decile lift is below one. The recorded specification does not establish useful transfer for customer prioritization in the second file. The spending GBT has negative external R² and only a modest improvement against the training-mean baseline. The dataset's between-file relationships and provenance require investigation before any stronger operational claim.

The next empirical step is a documented independent dataset or a trustworthy timestamped sampling process. Repeated adjustment to the already inspected external file would answer a different question and needs another final holdout.
