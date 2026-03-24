import pandas as pd
import numpy as np
import pytest
from src.preprocessing import preprocess_data

def test_preprocess_data_encoding():
    # Mock data
    data = {'Class': ['Good', 'Bad', 'Good'], 'Other': [1, 2, 3]}
    df = pd.DataFrame(data)
    
    processed_df = preprocess_data(df)
    
    assert processed_df['Class'].tolist() == [1, 0, 1]
    assert processed_df.columns[-1] == 'Class'

def test_preprocess_data_no_change_if_already_numeric():
    data = {'Class': [1, 0, 1], 'Other': [1, 2, 3]}
    df = pd.DataFrame(data)
    
    processed_df = preprocess_data(df)
    
    assert processed_df['Class'].tolist() == [1, 0, 1]
