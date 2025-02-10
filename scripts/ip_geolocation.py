import pandas as pd
import numpy as np

# Function to convert float IP addresses to integer (if necessary)
def convert_ip_to_int(ip):
    """
    Convert float IP address to an integer format.
    :param ip: float - IP address in floating point format.
    :return: int - IP address as an integer.
    """
    return int(np.floor(ip))

# Function to merge fraud_data with ip_data based on IP ranges
def merge_datasets(fraud_data, ip_data):
    # Convert ip_address in fraud_data to integer
    fraud_data['ip_address'] = fraud_data['ip_address'].apply(convert_ip_to_int)

    # Perform a range-based merge (find where fraud_data.ip_address falls in the range of IPs in ip_data)
    merged_data = pd.merge_asof(
        fraud_data.sort_values('ip_address'),
        ip_data.sort_values('lower_bound_ip_address'),
        left_on='ip_address',
        right_on='lower_bound_ip_address',
        direction='backward'
    )

    # Filter rows where ip_address falls between lower and upper bounds
    merged_data = merged_data[(merged_data['ip_address'] >= merged_data['lower_bound_ip_address']) &
                              (merged_data['ip_address'] <= merged_data['upper_bound_ip_address'])]

    # Drop unnecessary columns after merging
    merged_data.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)

    return merged_data

# Function to display the merged data
def display_merged_data(merged_data):
    print(merged_data.head())

# Main function
def merge(fraud_df,ip_df):
    # Merge the datasets and perform geolocation analysis
    merged_data = merge_datasets(fraud_df, ip_df)

    # Display the first few rows of the merged dataset
    display_merged_data(merged_data)
    return merged_data
