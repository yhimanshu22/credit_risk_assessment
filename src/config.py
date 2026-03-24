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
