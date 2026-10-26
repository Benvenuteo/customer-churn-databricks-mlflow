"""Metric definitions, threshold choice and bootstrap behavior on known cases."""

import numpy as np
import pytest

from churn_study.metrics import (
    bootstrap_intervals,
    choose_threshold,
    classification_metrics,
    regression_metrics,
)


def test_perfect_classification():
    result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert result["roc_auc"] == result["average_precision"] == 1
    assert result["confusion_matrix_tn_fp_fn_tp"] == [2, 0, 0, 2]


def test_constant_prior_has_auc_half():
    result = classification_metrics(np.array([0, 1, 0, 1]), np.full(4, 0.5))
    assert result["roc_auc"] == 0.5
    assert result["average_precision"] == 0.5


def test_invalid_probability_rejected():
    with pytest.raises(ValueError, match="Probabilities"):
        classification_metrics(np.array([0, 1]), np.array([0.2, 1.1]))


def test_regression_units_and_rmse():
    result = regression_metrics(np.array([10.0, 20.0]), np.array([8.0, 24.0]))
    assert result["mae"] == 3
    assert result["rmse"] == pytest.approx(10**0.5)
    assert result["r2"] == pytest.approx(0.6)


def test_threshold_tie_break_and_repeatable_bootstrap():
    y = np.array([0, 0, 1, 1] * 20)
    score = np.array([0.1, 0.2, 0.8, 0.9] * 20)
    assert choose_threshold(y, score) == pytest.approx(0.5)
    first = bootstrap_intervals(y, score, "classification", 20, 42)
    assert first == bootstrap_intervals(y, score, "classification", 20, 42)
    assert first["intervals"]["roc_auc"] == {"low": 1.0, "high": 1.0}
