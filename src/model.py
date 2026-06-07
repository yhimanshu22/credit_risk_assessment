from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from src.config import (
    BAGGING_PARAM_GRID,
    CV_FOLDS,
    DECISION_TREE_PARAM_GRID,
    GRID_SEARCH_SCORING,
    MAX_ITER,
    RANDOM_FOREST_PARAM_GRID,
)


def _train_with_grid_search(estimator, param_grid, X_train, y_train, model_name):
    """
    Run GridSearchCV, print best params/score, and return the tuned estimator.
    """
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring=GRID_SEARCH_SCORING,
    )
    grid_search.fit(X_train, y_train)
    print(f"{model_name} best params: {grid_search.best_params_}")
    print(f"{model_name} best CV accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model with the specified max_iter.
    """
    model = LogisticRegression(max_iter=MAX_ITER)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree classifier using GridSearchCV hyperparameter tuning.
    """
    return _train_with_grid_search(
        DecisionTreeClassifier(),
        DECISION_TREE_PARAM_GRID,
        X_train,
        y_train,
        "Decision Tree",
    )


def train_bagging(X_train, y_train):
    """
    Train a Bagging classifier using GridSearchCV hyperparameter tuning.
    """
    return _train_with_grid_search(
        BaggingClassifier(),
        BAGGING_PARAM_GRID,
        X_train,
        y_train,
        "Bagging",
    )


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier using GridSearchCV hyperparameter tuning.
    """
    return _train_with_grid_search(
        RandomForestClassifier(),
        RANDOM_FOREST_PARAM_GRID,
        X_train,
        y_train,
        "Random Forest",
    )


def train_svm(X_train, y_train):
    """
    Train an SVM classifier with an RBF kernel.
    """
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)
    return model


def get_prediction_probabilities(model, X_test):
    """
    Get the probability estimates for the positive class (Good risk).
    """
    return model.predict_proba(X_test)[:, 1]


def get_all_trainers():
    """
    Return ordered model trainers for comparison against logistic regression.
    """
    return [
        ("Logistic Regression", train_logistic_regression),
        ("Decision Tree", train_decision_tree),
        ("Bagging", train_bagging),
        ("Random Forest", train_random_forest),
        ("SVM (RBF)", train_svm),
    ]
