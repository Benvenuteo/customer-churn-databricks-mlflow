"""Small deterministic fixtures without a Databricks account or full source data."""

import os

import pytest
from pyspark.sql.types import StringType, StructField, StructType

from churn_study.config import RAW_COLUMNS, Settings
from churn_study.data import RAW_SCHEMA
from churn_study.engine import create_spark


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 1g pyspark-shell")
    session = create_spark(Settings(tmp_path_factory.mktemp("spark"), run_id="tests"))
    yield session
    session.stop()


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path / "run", run_id="tests", cv_fraction=1.0, folds=2, bootstrap_repeats=5)


@pytest.fixture
def raw_row():
    def build(**overrides):
        values = dict(
            zip(
                RAW_COLUMNS,
                ("1", "30", "Female", "12", "10", "2", "0", "Basic", "Monthly", "500", "10", "0"),
                strict=True,
            )
        )
        values.update({"_corrupt_record": None, "source": "development"})
        values.update(overrides)
        return values

    return build


@pytest.fixture
def raw_frame(spark):
    schema = StructType([*RAW_SCHEMA.fields, StructField("source", StringType(), True)])
    return lambda rows: spark.createDataFrame(rows, schema)
