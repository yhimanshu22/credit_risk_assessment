import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "Credit.csv")

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.3
MAX_ITER = int(1e8)

# Evaluation Parameters
THRESHOLDS = [0.2, 0.35, 0.5]
DEFAULT_THRESHOLD = 0.5

# GridSearchCV Parameters
CV_FOLDS = 5
GRID_SEARCH_SCORING = "accuracy"

DECISION_TREE_PARAM_GRID = {
    "max_depth": [3, 4, 5, 6, 7, 8, 10, 20],
}

BAGGING_PARAM_GRID = {
    "n_estimators": [100, 150, 200],
    "max_features": [0.5, 0.7, 1.0],
}

RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [100, 150, 200],
    "max_features": [0.5, 0.7, 1.0],
}
