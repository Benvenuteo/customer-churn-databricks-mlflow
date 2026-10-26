"""Whole-pipeline serialization, unseen categories and feature-whitelist controls."""

import pytest
from pyspark.ml import PipelineModel

from churn_study.automl import automl_input
from churn_study.config import COLUMNS, feature_columns
from churn_study.data import assign_splits, clean
from churn_study.modeling import Candidate, fit_full


def test_unknown_category_and_missing_predictor_survive_serialization(raw_frame, raw_row, settings):
    train = clean(
        raw_frame(
            [
                raw_row(
                    CustomerID=str(index + 1),
                    Churn=str(index % 2),
                    Age=str(20 + index % 30),
                    Gender="Female" if index % 2 else "Male",
                    **{"Total Spend": str(100 + index)},
                )
                for index in range(80)
            ]
        )
    ).withColumnRenamed("churn", "label")
    model = fit_full(train, "classification", Candidate("lr", "logistic"), settings)
    test = clean(
        raw_frame([raw_row(Age=None, Gender="Unknown", **{"Subscription Type": "NewPlan"})])
    )
    expected = model.transform(test).first().prediction
    path = settings.output / "pipeline"
    model.write().save(str(path))
    actual = PipelineModel.load(str(path)).transform(test).first()
    assert actual.prediction == pytest.approx(expected)
    assert actual.probability[1] >= 0
    assert actual.age_imputed is not None


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_automl_whitelist_and_split_match(raw_frame, raw_row, settings, task):
    rows = [
        raw_row(
            CustomerID=str(index + 1),
            Churn=str(index % 2),
            source="development" if index < 40 else "external",
            **{"Total Spend": str(100 + index)},
        )
        for index in range(60)
    ]
    silver = clean(raw_frame(rows)).select(*COLUMNS, "source", "row_key", "feature_hash")
    split = assign_splits(silver, settings)
    data = automl_input(split, task)
    assert set(data.columns) == set(feature_columns(task)) | {"label", "split"}
    assert "customer_id" not in data.columns
    assert data.filter("split = 'external'").count() == 0
    assert data.count() == 40
