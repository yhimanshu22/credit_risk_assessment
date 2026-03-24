from sklearn.linear_model import LogisticRegression
from src.config import MAX_ITER

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model with the specified max_iter.
    """
    model = LogisticRegression(max_iter=MAX_ITER)
    model.fit(X_train, y_train)
    return model

def get_prediction_probabilities(model, X_test):
    """
    Get the probability estimates for the positive class (Good risk).
    """
    return model.predict_proba(X_test)[:, 1]
