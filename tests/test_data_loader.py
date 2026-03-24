import pytest
from src.data_loader import load_credit_data
from src.config import DATA_PATH
import os

def test_load_credit_data_exists():
    if os.path.exists(DATA_PATH):
        df = load_credit_data()
        assert not df.empty
    else:
        pytest.skip("Data file not found, skipping integration test.")
