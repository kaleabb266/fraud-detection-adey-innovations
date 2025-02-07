import pandas as pd

def ip_to_int(ip):
    """
    Convert IP address to integer.
    Args:
        ip (str): IP address in string format.
    Returns:
        int: Integer representation of the IP address.
    """
    try:
        parts = list(map(int, ip.split('.')))
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except:
        return None  # Handle invalid IPs gracefully

def merge_with_geolocation(fraud_data, ip_data):
    """
    Merge fraud data with geolocation data based on IP ranges.
    Args:
        fraud_data (pd.DataFrame): Fraud dataset.
        ip_data (pd.DataFrame): IP address to country mapping dataset.
    Returns:
        pd.DataFrame: Merged dataframe with geolocation information.
    """
    # Step 1: Convert IP addresses in fraud_data to integers
    fraud_data['ip_int'] = fraud_data['ip_address'].apply(ip_to_int)

    # Step 2: Ensure IP ranges in ip_data are integers
    ip_data['lower_bound_ip_int'] = ip_data['lower_bound_ip_address'].astype(int)
    ip_data['upper_bound_ip_int'] = ip_data['upper_bound_ip_address'].astype(int)

    # Step 3: Perform IP range matching manually
    merged_data = []
    for _, row in fraud_data.iterrows():
        ip_int = row['ip_int']
        if ip_int is not None:  # Skip invalid IPs
            # Find matching country for the IP range
            matching_country = ip_data[
                (ip_data['lower_bound_ip_int'] <= ip_int) & 
                (ip_data['upper_bound_ip_int'] >= ip_int)
            ][['country']].values
            if len(matching_country) > 0:
                row['country'] = matching_country[0][0]
            else:
                row['country'] = 'Unknown'  # Assign 'Unknown' if no match found
        else:
            row['country'] = 'Invalid IP'
        merged_data.append(row)

    # Step 4: Create the final DataFrame
    merged_df = pd.DataFrame(merged_data)

    return merged_df