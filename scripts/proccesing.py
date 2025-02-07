# Import necessary libraries
import pandas as pd
import numpy as np

def handle_missing_values(df):
    """
    Handle missing values in the dataframe.
    Args:
        df (pd.DataFrame): Input dataframe.
    Returns:
        pd.DataFrame: Dataframe with missing values handled.
    """
    # Check for missing values
    print("Missing values before handling:\n", df.isnull().sum())

    # Impute missing values
    for column in df.columns:
        if df[column].dtype == 'object':  # Categorical columns
            df[column].fillna('Unknown', inplace=True)
        elif df[column].dtype in ['int64', 'float64']:  # Numerical columns
            df[column].fillna(df[column].median(), inplace=True)

    print("Missing values after handling:\n", df.isnull().sum())
    return df

def clean_data(df):
    """
    Clean the dataframe by removing duplicates and ensuring correct data types.
    Args:
        df (pd.DataFrame): Input dataframe.
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Convert timestamps to datetime format
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')

    # Ensure class column is binary
    if 'class' in df.columns:
        df['class'] = df['class'].astype('int')
    if 'Class' in df.columns:
        df['Class'] = df['Class'].astype('int')

    return df