from src.config import DEFAULT_THRESHOLD, THRESHOLDS
from src.data_loader import load_credit_data
from src.preprocessing import preprocess_data, split_credit_data
from src.model import get_all_trainers, get_prediction_probabilities
from src.evaluation import (
    calculate_metrics,
    calculate_predict_metrics,
    print_model_comparison,
    print_performance_summary,
)


def run_pipeline():
    """
    Execute the full machine learning pipeline.
    """
    print("Starting Credit Risk Assessment Pipeline...")

    print("Loading data...")
    df = load_credit_data()

    print("Preprocessing data...")
    df = preprocess_data(df)

    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_credit_data(df)

    comparison_results = []

    print("\n=== Logistic Regression Threshold Analysis ===")
    for name, train_fn in get_all_trainers():
        if name != "Logistic Regression":
            continue

        print(f"Training {name} model...")
        model = train_fn(X_train, y_train)
        y_probs = get_prediction_probabilities(model, X_test)

        for threshold in THRESHOLDS:
            metrics = calculate_metrics(
                y_test, y_probs, threshold, model_name=name
            )
            print_performance_summary(metrics)

            if threshold == DEFAULT_THRESHOLD:
                comparison_results.append(metrics)

    print("\n=== Additional Model Training & Comparison ===")
    print(
        "Tuning Decision Tree, Bagging, and Random Forest with "
        "GridSearchCV (5-fold cross-validation)..."
    )
    for name, train_fn in get_all_trainers():
        if name == "Logistic Regression":
            continue

        print(f"\nTraining {name} model...")
        model = train_fn(X_train, y_train)
        y_probs = get_prediction_probabilities(model, X_test)
        y_pred = model.predict(X_test)

        metrics = calculate_predict_metrics(
            y_test, y_pred, y_probs=y_probs, model_name=name
        )
        print_performance_summary(metrics)
        comparison_results.append(metrics)

    print_model_comparison(comparison_results)
    print("\nPipeline execution complete.")


if __name__ == "__main__":
    run_pipeline()
