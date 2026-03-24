import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_metrics(y_true, y_probs, threshold):
    """
    Calculate Accuracy, TPR, and FPR for a given threshold.
    """
    y_pred = np.where(y_probs > threshold, 1, 0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle cases where confusion matrix might not have all 4 quadrants (unlikely with this dataset)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback if only one class is present in prediction (e.g. threshold extreme)
        # This is a safety measure.
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        "threshold": threshold,
        "tpr": tpr,
        "fpr": fpr,
        "accuracy": acc,
        "confusion_matrix": cm
    }

def print_performance_summary(metrics):
    """
    Print the performance metrics in a readable format.
    """
    print(f"--- Performance for Threshold {metrics['threshold']} ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"True Positive Rate (TPR): {metrics['tpr']:.4f}")
    print(f"False Positive Rate (FPR): {metrics['fpr']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("-" * 40)
