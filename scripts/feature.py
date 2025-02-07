import pandas as pd
def feature_engineering(df):
    """
    Perform feature engineering on the dataframe.
    Args:
        df (pd.DataFrame): Input dataframe.
    Returns:
        pd.DataFrame: Dataframe with engineered features.
    """
    # Transaction frequency and velocity
    df['transaction_frequency'] = df.groupby('user_id')['purchase_time'].transform('count')
    df['transaction_velocity'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / df['transaction_frequency']

    # Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek

    return df

from sklearn.preprocessing import StandardScaler, OneHotEncoder

def normalize_and_encode(df):
    """
    Normalize numerical features and encode categorical features.
    Args:
        df (pd.DataFrame): Input dataframe.
    Returns:
        pd.DataFrame: Dataframe with normalized and encoded features.
    """
    # Normalize numerical features
    scaler = StandardScaler()
    num_cols = ['purchase_value', 'transaction_velocity']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Encode categorical features
    cat_cols = ['source', 'browser', 'sex', 'country']
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cat_cols))

    # Combine original and encoded features
    df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

    return df