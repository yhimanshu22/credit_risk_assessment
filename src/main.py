from src.config import THRESHOLDS
from src.data_loader import load_credit_data
from src.preprocessing import preprocess_data, split_credit_data
from src.model import train_logistic_regression, get_prediction_probabilities
from src.evaluation import calculate_metrics, print_performance_summary

def run_pipeline():
    """
    Execute the full machine learning pipeline.
    """
    print("Starting Credit Risk Assessment Pipeline...")
    
    # 1. Load Data
    print("Loading data...")
    df = load_credit_data()
    
    # 2. Preprocess Data
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    # 3. Split Data
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_credit_data(df)
    
    # 4. Train Model
    print("Training Logistic Regression model...")
    model = train_logistic_regression(X_train, y_train)
    
    # 5. Get Probabilities
    print("Generating prediction probabilities...")
    y_probs = get_prediction_probabilities(model, X_test)
    
    # 6. Evaluate at various thresholds
    print("Evaluating model performance...")
    for t in THRESHOLDS:
        metrics = calculate_metrics(y_test, y_probs, t)
        print_performance_summary(metrics)
    
    print("Pipeline execution complete.")

if __name__ == "__main__":
    run_pipeline()
