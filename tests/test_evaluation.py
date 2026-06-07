import numpy as np
import pytest

from src.evaluation import calculate_metrics, calculate_predict_metrics


def test_calculate_metrics_includes_auc_precision_and_fnr():
    y_true = np.array([1, 1, 0, 0])
    y_probs = np.array([0.9, 0.6, 0.4, 0.1])

    metrics = calculate_metrics(y_true, y_probs, threshold=0.5, model_name="Test Model")

    assert metrics["model_name"] == "Test Model"
    assert metrics["threshold"] == 0.5
    assert "auc" in metrics
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["fnr"] == pytest.approx(0.0)


def test_calculate_predict_metrics_from_hard_predictions():
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    metrics = calculate_predict_metrics(y_true, y_pred, model_name="SVM")

    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["fnr"] == pytest.approx(0.5)
