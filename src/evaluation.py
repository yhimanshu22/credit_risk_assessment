import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _compute_metrics(y_true, y_pred, y_probs=None, threshold=None, model_name=None):
    cm = confusion_matrix(y_true, y_pred)

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0

    metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": cm,
    }

    if y_probs is not None:
        metrics["auc"] = roc_auc_score(y_true, y_probs)

    return metrics


def calculate_metrics(y_true, y_probs, threshold, model_name=None):
    """
    Calculate classification metrics for a given probability threshold.
    """
    y_pred = np.where(y_probs > threshold, 1, 0)
    return _compute_metrics(
        y_true, y_pred, y_probs=y_probs, threshold=threshold, model_name=model_name
    )


def calculate_predict_metrics(y_true, y_pred, y_probs=None, model_name=None):
    """
    Calculate classification metrics from hard predictions.
    """
    return _compute_metrics(y_true, y_pred, y_probs=y_probs, model_name=model_name)


def print_performance_summary(metrics):
    """
    Print the performance metrics in a readable format.
    """
    title = metrics.get("model_name") or "Model"
    threshold = metrics.get("threshold")
    if threshold is not None:
        print(f"--- {title} | Threshold {threshold} ---")
    else:
        print(f"--- {title} ---")

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"TPR:       {metrics['tpr']:.4f}")
    print(f"FPR:       {metrics['fpr']:.4f}")
    print(f"FNR:       {metrics['fnr']:.4f}")
    if "auc" in metrics:
        print(f"AUC:       {metrics['auc']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("-" * 40)


def print_model_comparison(results):
    """
    Print a side-by-side comparison of model metrics.
    """
    print("\n=== Model Comparison ===")
    header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'FNR':>8} {'AUC':>8}"
    print(header)
    print("-" * len(header))

    for result in results:
        auc = f"{result['auc']:.4f}" if "auc" in result else "N/A"
        name = result.get("model_name", "Model")
        if result.get("threshold") is not None:
            name = f"{name} (t={result['threshold']})"
        print(
            f"{name:<22} "
            f"{result['accuracy']:>9.4f} "
            f"{result['precision']:>10.4f} "
            f"{result['recall']:>8.4f} "
            f"{result['fnr']:>8.4f} "
            f"{auc:>8}"
        )

    best = max(results, key=lambda item: item["accuracy"])
    print(
        f"\nBest accuracy: {best.get('model_name', 'Model')} "
        f"({best['accuracy']:.4f})"
    )
