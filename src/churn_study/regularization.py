"""Training-only coefficient paths supplement, but never replace, CV selection evidence."""

from typing import Any

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.stat import Summarizer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from churn_study.config import Settings
from churn_study.modeling import Candidate, estimator


def coefficient_paths(
    train: DataFrame, pipeline: PipelineModel, settings: Settings, smoke: bool = False
) -> dict[str, Any]:
    """Refit linear diagnostics on train, expressing each coefficient per one training SD."""
    transformer = PipelineModel(stages=pipeline.stages[:-1])
    prepared = transformer.transform(train).persist(StorageLevel.MEMORY_AND_DISK)
    try:
        metadata = prepared.schema["features"].metadata["ml_attr"]
        attributes = sorted(
            [value for group in metadata["attrs"].values() for value in group],
            key=lambda value: value["idx"],
        )
        names = [value["name"] for value in attributes]
        scale = prepared.select(Summarizer.std(F.col("features"))).first()[0].toArray()
        rows = []
        for family, elastic in (("ridge", 0.0), ("lasso", 1.0)):
            for parameter in (0.1,) if smoke else (0.001, 0.1, 10.0):
                candidate = Candidate(f"{family}_{parameter}", "linear", parameter, elastic)
                model = estimator(candidate, "regression", settings).fit(prepared)
                coefficients = model.coefficients.toArray()
                rows.append(
                    {
                        "family": family,
                        "reg_param": parameter,
                        "zero_coefficients": int((abs(coefficients) < 1e-8).sum()),
                        "coefficients": [
                            {
                                "name": name,
                                "raw": float(value),
                                "per_training_sd": float(value * deviation),
                            }
                            for name, value, deviation in zip(
                                names, coefficients, scale, strict=True
                            )
                        ],
                    }
                )
        return {
            "fit_population": "development train only",
            "used_for_model_selection": False,
            "units": "Spending units per one training SD of a transformed feature",
            "interpretation": "Associations conditional on the linear specification; not causal effects",
            "paths": rows,
        }
    finally:
        prepared.unpersist()
