"""
tests/test_metrics.py
======================
Unit tests for src/evaluation/metrics.py
"""

import pytest
import numpy as np
from src.evaluation.metrics import compute_metrics


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = compute_metrics(y, y, "perfect")
    assert m["MAE"]  == pytest.approx(0.0)
    assert m["MSE"]  == pytest.approx(0.0)
    assert m["RMSE"] == pytest.approx(0.0)
    assert m["R2"]   == pytest.approx(1.0, abs=1e-6)


def test_directional_accuracy_all_correct():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # same direction always
    m = compute_metrics(y_true, y_pred, "da_test")
    assert m["DA"] == pytest.approx(100.0)


def test_directional_accuracy_all_wrong():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # opposite direction always
    m = compute_metrics(y_true, y_pred, "da_wrong")
    assert m["DA"] == pytest.approx(0.0)


def test_model_name_in_output():
    y = np.ones(5)
    m = compute_metrics(y, y, "MyModel")
    assert m["Model"] == "MyModel"


def test_mape_ignores_zero_targets():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.0, 1.0, 2.0])
    m = compute_metrics(y_true, y_pred, "mape_zero")
    # MAPE should be 0 for non-zero elements
    assert m["MAPE"] == pytest.approx(0.0)


def test_single_sample_da_is_zero():
    m = compute_metrics(np.array([1.0]), np.array([1.0]), "single")
    assert m["DA"] == 0.0
