"""Adverse records, explicit schemas and reproducible partition contracts."""

import csv

import pytest
from pyspark.sql import functions as F

from churn_study.config import COLUMNS, RAW_COLUMNS, feature_columns
from churn_study.data import assign_splits, clean, read_csv, validate_identity


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"Age": "-1"}, "invalid_age"),
        ({"Age": "121"}, "invalid_age"),
        ({"Support Calls": "broken"}, "invalid_support_calls"),
        ({"Total Spend": "NaN"}, "invalid_total_spend"),
        ({"Total Spend": "Infinity"}, "invalid_total_spend"),
        ({"CustomerID": "1.5"}, "invalid_customer_id"),
        ({"Churn": "3"}, "invalid_churn"),
        ({"Churn": None}, "invalid_churn"),
        ({"_corrupt_record": "bad"}, "csv_malformed"),
    ],
)
def test_invalid_records_have_reasons(raw_frame, raw_row, changes, reason):
    result = clean(raw_frame([raw_row(**changes)])).first()
    assert not result.valid
    assert reason in result.reasons


def test_missing_predictor_is_retained(raw_frame, raw_row):
    result = clean(raw_frame([raw_row(Age=None, Gender="Unseen")])).first()
    assert result.valid
    assert result.age is None
    assert result.gender == "Unseen"


def test_empty_record_has_exclusive_primary_reason(raw_frame, raw_row):
    result = clean(raw_frame([raw_row(**dict.fromkeys(RAW_COLUMNS))])).first()
    assert not result.valid
    assert result.primary_reason == "empty_record"


def test_header_order_is_validated(spark, tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text(",".join(reversed(RAW_COLUMNS)))
    with pytest.raises(ValueError, match="header"):
        read_csv(spark, path, "development")


def test_missing_spending_does_not_become_zero(spark, tmp_path, raw_row):
    path = tmp_path / "source.csv"
    row = raw_row(**{"Total Spend": ""})
    row.pop("source")
    row.pop("_corrupt_record")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    result = clean(read_csv(spark, path, "development")).first()
    assert not result.valid
    assert result.total_spend is None


def test_splits_are_stratified_deterministic_and_keep_external_separate(
    raw_frame, raw_row, settings
):
    rows = [
        raw_row(
            CustomerID=str(index + 1),
            Age=str(20 + index % 40),
            Churn=str(index % 2),
            **{"Total Spend": str(100 + index)},
        )
        for index in range(200)
    ]
    rows += [
        raw_row(
            CustomerID=str(index + 1),
            source="external",
            Churn=str(index % 2),
            **{"Total Spend": str(900 + index)},
        )
        for index in range(20)
    ]
    data = clean(raw_frame(rows)).select(*COLUMNS, "source", "row_key", "feature_hash")
    validate_identity(data)
    split = assign_splits(data, settings)
    assert split.filter("source = 'external' AND split != 'external'").count() == 0
    counts = {
        (row.split, row.churn): row["count"]
        for row in split.groupBy("split", "churn").count().collect()
    }
    assert counts[("train", 0.0)] == counts[("train", 1.0)] == 70
    assert counts[("validate", 0.0)] == counts[("test", 0.0)] == 15
    first = {row.row_key: (row.split, row.fold) for row in split.collect()}
    second = {
        row.row_key: (row.split, row.fold)
        for row in assign_splits(data.repartition(3), settings).collect()
    }
    assert first == second
    assert (
        split.groupBy("row_key").agg(F.countDistinct("split").alias("n")).filter("n > 1").count()
        == 0
    )


def test_predictive_features_exclude_id_and_opposite_regression_target():
    assert "customer_id" not in feature_columns("classification")
    assert "churn" not in feature_columns("regression")
    assert "total_spend" not in feature_columns("regression")


def test_ambiguous_source_identity_is_rejected(raw_frame, raw_row):
    data = clean(raw_frame([raw_row(), raw_row(Age="31")]))
    with pytest.raises(ValueError, match="Repeated customer_id"):
        validate_identity(data)
