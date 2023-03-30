import pandas as pd
import numpy as np

def clean_data(df):
    # Drop columns with missing values
    df = df.dropna(axis=1)
    
    # Remove outliers in target variable MEDV
    df = df[df['MEDV'] <= 50]
    
    # Create new feature AGE_BINS from AGE feature
    df['AGE_BINS'] = pd.cut(df['AGE'], bins=[0, 25, 50, 75, 100], labels=['0-25', '25-50', '50-75', '75-100'])
    
    # One-hot encode categorical feature AGE_BINS
    df = pd.get_dummies(df, columns=['AGE_BINS'])
    
    # Normalize continuous features
    continuous_features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    for feature in continuous_features:
        df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
    
    return df
