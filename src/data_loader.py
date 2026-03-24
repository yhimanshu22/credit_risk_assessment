import pandas as pd
from src.config import DATA_PATH

def load_credit_data(path=DATA_PATH):
    """
    Load the credit data from a CSV file.
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {path}")
    except Exception as e:
        raise Exception(f"An error occurred while loading data: {e}")
