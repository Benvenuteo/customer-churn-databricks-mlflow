"""Train-only preprocessing, bounded cross-validation and full-training refits in MLlib."""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from churn_study.config import CATEGORICAL, Settings, feature_columns
from churn_study.metrics import classification_metrics, regression_metrics

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """A small predeclared candidate grid, not a search adapted to test performance."""

    name: str
    family: str
    reg_param: float = 0.1
    elastic_net: float = 0.0
    depth: int = 3


def candidates(task: str, smoke: bool = False) -> list[Candidate]:
    if task == "classification":
        result = [
            Candidate(f"logistic_l2_{value:g}", "logistic", value)
            for value in ((0.1,) if smoke else (0.001, 0.1, 1.0))
        ]
        return result + [
            Candidate(f"gbt_depth{depth}", "gbt", depth=depth)
            for depth in ((3,) if smoke else (3, 5))
        ]
    if task != "regression":
        raise ValueError(f"Unknown task: {task}")
    result = [Candidate("ols", "linear", 0.0)]
    for family, elastic in (("ridge", 0.0), ("lasso", 1.0)):
        result += [
            Candidate(f"{family}_{value:g}", "linear", value, elastic)
            for value in ((0.1,) if smoke else (0.001, 0.1, 10.0))
        ]
    if not smoke:
        result.append(Candidate("elastic_net_0.1", "linear", 0.1, 0.5))
    return result + [Candidate("gbt_depth3", "gbt")]


def preprocessing(task: str) -> Pipeline:
    """Keep learned category frequencies and imputation within each fold's training subset."""
    numeric = [name for name in feature_columns(task) if name not in CATEGORICAL]
    indexers = [
        StringIndexer(inputCol=name, outputCol=name + "_index", handleInvalid="keep")
        for name in CATEGORICAL
    ]
    return Pipeline(
        stages=[
            *indexers,
            OneHotEncoder(
                inputCols=[name + "_index" for name in CATEGORICAL],
                outputCols=[name + "_encoded" for name in CATEGORICAL],
                handleInvalid="keep",
            ),
            Imputer(
                inputCols=numeric,
                outputCols=[name + "_imputed" for name in numeric],
                strategy="median",
            ),
            VectorAssembler(
                inputCols=[name + "_imputed" for name in numeric]
                + [name + "_encoded" for name in CATEGORICAL],
                outputCol="features",
            ),
        ]
    )


def estimator(candidate: Candidate, task: str, settings: Settings) -> Any:
    """Build estimators with explicit randomness and bounded training budgets."""
    if candidate.family == "gbt":
        constructor = GBTClassifier if task == "classification" else GBTRegressor
        return constructor(
            labelCol="label",
            seed=settings.seed,
            maxIter=settings.tree_iterations,
            maxDepth=candidate.depth,
            maxBins=32,
        )
    if task == "classification":
        return LogisticRegression(
            labelCol="label", regParam=candidate.reg_param, maxIter=80, standardization=True
        )
    return LinearRegression(
        labelCol="label",
        regParam=candidate.reg_param,
        elasticNetParam=candidate.elastic_net,
        maxIter=100,
        standardization=True,
    )


def prediction_frame(model: Any, data: DataFrame, task: str, limit: int = 200_000) -> pd.DataFrame:
    """Collect only a checked number of prediction/segment columns, never a raw training table."""
    predicted = model.transform(data)
    score = vector_to_array("probability")[1] if task == "classification" else F.col("prediction")
    result = (
        predicted.select("row_key", "label", "gender", "contract_length", score.alias("score"))
        .orderBy("row_key")
        .limit(limit + 1)
        .toPandas()
    )
    if len(result) > limit:
        raise ValueError(f"Evaluation exceeds the bounded {limit}-row driver budget")
    if result.empty:
        raise ValueError("Cannot evaluate an empty partition")
    return result


def select_candidate(
    train: DataFrame, task: str, settings: Settings, smoke: bool = False
) -> tuple[Candidate, list[dict[str, Any]], int]:
    """Select by stratified CV on a declared training-only sample, never external/test scores."""
    grid = candidates(task, smoke)
    threshold = int(settings.cv_fraction * 1_000_000)
    sampled = train.filter(
        F.pmod(F.xxhash64("row_key", F.lit(settings.seed + 99)), F.lit(1_000_000)) < threshold
    ).persist(StorageLevel.MEMORY_AND_DISK)
    try:
        sample_count = sampled.count()
        if sample_count < max(30, settings.folds * 10):
            raise ValueError("CV sample is too small; increase cv_fraction or provide more rows")
        scores: dict[str, list[float]] = {candidate.name: [] for candidate in grid}
        for fold in range(settings.folds):
            fit = sampled.filter(F.col("fold") != fold)
            holdout = sampled.filter(F.col("fold") == fold)
            if task == "classification" and fit.select("label").distinct().count() != 2:
                raise ValueError("Each classification CV training fold must contain both classes")
            transformer = preprocessing(task).fit(fit)
            fitted = transformer.transform(fit).persist(StorageLevel.MEMORY_AND_DISK)
            validation = transformer.transform(holdout).persist(StorageLevel.MEMORY_AND_DISK)
            try:
                for candidate in grid:
                    LOGGER.info(
                        "CV %s fold %d/%d: %s", task, fold + 1, settings.folds, candidate.name
                    )
                    model = estimator(candidate, task, settings).fit(fitted)
                    frame = prediction_frame(model, validation, task)
                    y, probability = frame["label"].to_numpy(), frame["score"].to_numpy()
                    metrics = (
                        classification_metrics(y, probability)
                        if task == "classification"
                        else regression_metrics(y, probability)
                    )
                    scores[candidate.name].append(
                        metrics["roc_auc" if task == "classification" else "rmse"]
                    )
            finally:
                fitted.unpersist()
                validation.unpersist()
        records = [
            {
                **asdict(candidate),
                "fold_scores": scores[candidate.name],
                "mean": float(np.mean(scores[candidate.name])),
                "std": float(np.std(scores[candidate.name], ddof=1)),
            }
            for candidate in grid
        ]
        records.sort(
            key=lambda record: (
                (-record["mean"] if task == "classification" else record["mean"]),
                record["name"],
            )
        )
        winner = next(candidate for candidate in grid if candidate.name == records[0]["name"])
        return winner, records, sample_count
    finally:
        sampled.unpersist()


def fit_full(
    train: DataFrame, task: str, candidate: Candidate, settings: Settings
) -> PipelineModel:
    """Refit the selected complete pipeline on every development-training row."""
    stages = preprocessing(task).getStages() + [estimator(candidate, task, settings)]
    return Pipeline(stages=stages).fit(train)


def model_details(model: PipelineModel, train: DataFrame) -> dict[str, Any]:
    """Recover transformed feature names to interpret coefficients or tree importance."""
    metadata = model.transform(train.limit(1)).schema["features"].metadata["ml_attr"]
    attributes = sorted(
        [value for group in metadata.get("attrs", {}).values() for value in group],
        key=lambda value: value["idx"],
    )
    names = [attribute["name"] for attribute in attributes]
    learner = model.stages[-1]
    if hasattr(learner, "coefficients"):
        values = learner.coefficients.toArray().tolist()
        return {
            "kind": "coefficients_in_original_feature_units",
            "intercept": float(learner.intercept),
            "features": [
                {"name": name, "value": float(value)}
                for name, value in zip(names, values, strict=True)
            ],
        }
    values = learner.featureImportances.toArray().tolist()
    return {
        "kind": "tree_split_importance_not_causal",
        "features": [
            {"name": name, "value": float(value)} for name, value in zip(names, values, strict=True)
        ],
    }
