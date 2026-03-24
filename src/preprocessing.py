import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE, RANDOM_STATE

def preprocess_data(df):
    """
    Perform label encoding for the target variable and reorder columns.
    """
    df = df.copy()
    
    # Map 'Class' from 'Good'/'Bad' to 1/0
    if 'Class' in df.columns:
        # Use replace for robust encoding of strings to integers
        df['Class'] = df['Class'].replace({'Good': 1, 'Bad': 0, 'Good ': 1, 'Bad ': 0, ' Good': 1, ' Bad': 0})
            
    # Reorder columns to ensure 'Class' is the last column
    cols = [col for col in df.columns if col != 'Class']
    cols.append('Class')
    df = df[cols]
    
    # Ensure Class is integer
    try:
        df['Class'] = df['Class'].astype(int)
    except Exception as e:
        print(f"Warning: Could not convert Class to int: {e}")
        print(f"Unique values: {df['Class'].unique()}")
    
    return df

def split_credit_data(df):
    """
    Split the data into training and testing sets.
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    return X_train, X_test, y_train, y_test
