import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

def analyze_blockchain_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze blockchain transaction data to derive insights.
    
    Args:
        df: DataFrame containing blockchain transaction data
    
    Returns:
        DataFrame with analysis results
    """
    # Create a copy to avoid modifying the original dataframe
    analysis_df = df.copy()
    
    # Calculate transaction metrics
    if 'value' in df.columns:
        analysis_df['transaction_size'] = pd.cut(
            df['value'], 
            bins=[0, 0.1, 1, 10, 100, float('inf')],
            labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
        )
    
    # Calculate temporal patterns if timestamp is available
    if 'timestamp' in df.columns:
        analysis_df['timestamp'] = pd.to_datetime(df['timestamp'])
        analysis_df['hour'] = analysis_df['timestamp'].dt.hour
        analysis_df['day'] = analysis_df['timestamp'].dt.day_name()
        
        # Identify time-based patterns
        hour_counts = analysis_df.groupby('hour').size()
        peak_hours = hour_counts[hour_counts > hour_counts.mean()].index.tolist()
        analysis_df['is_peak_hour'] = analysis_df['hour'].isin(peak_hours)
    
    # Analyze network patterns if from/to addresses are available
    if 'from_address' in df.columns and 'to_address' in df.columns:
        # Calculate address activity
        from_counts = df['from_address'].value_counts()
        to_counts = df['to_address'].value_counts()
        
        # Identify high-activity addresses
        high_activity_threshold = np.percentile(
            np.concatenate([from_counts.values, to_counts.values]), 95
        )
        high_activity_senders = from_counts[from_counts > high_activity_threshold].index.tolist()
        high_activity_receivers = to_counts[to_counts > high_activity_threshold].index.tolist()
        
        analysis_df['high_activity_sender'] = analysis_df['from_address'].isin(high_activity_senders)
        analysis_df['high_activity_receiver'] = analysis_df['to_address'].isin(high_activity_receivers)
        
        # Calculate transaction velocities (for addresses with multiple transactions)
        address_transactions = {}
        for _, row in df.iterrows():
            if 'timestamp' in df.columns:
                sender = row['from_address']
                if sender not in address_transactions:
                    address_transactions[sender] = []
                address_transactions[sender].append(row['timestamp'])
    
    # Return the enriched dataframe with analysis
    return analysis_df

def identify_risks(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Identify potential risks in blockchain transactions.
    
    Args:
        df: DataFrame containing blockchain transaction data
        threshold: Risk score threshold (0.0 to 1.0)
    
    Returns:
        DataFrame with risk assessment results
    """
    risk_df = df.copy()
    
    # Initialize risk scores
    risk_df['risk_score'] = 0.0
    risk_df['risk_factors'] = ""
    
    # Risk factor 1: Unusual transaction amounts
    if 'value' in df.columns:
        # Calculate statistics for transaction values
        mean_value = df['value'].mean()
        std_value = df['value'].std()
        
        # Identify outliers based on z-score
        z_scores = (df['value'] - mean_value) / std_value
        risk_df.loc[abs(z_scores) > 3, 'risk_score'] += 0.2
        risk_df.loc[abs(z_scores) > 3, 'risk_factors'] += "Unusual transaction amount; "
    
    # Risk factor 2: Suspicious patterns in transactions
    if 'from_address' in df.columns and 'to_address' in df.columns:
        # Look for circular transactions (A -> B -> A)
        address_pairs = list(zip(df['from_address'], df['to_address']))
        address_pairs_reversed = list(zip(df['to_address'], df['from_address']))
        
        for i, (sender, receiver) in enumerate(address_pairs):
            if (receiver, sender) in address_pairs:
                risk_df.loc[i, 'risk_score'] += 0.15
                risk_df.loc[i, 'risk_factors'] += "Potential circular transaction; "
    
    # Risk factor 3: High frequency transactions
    if 'timestamp' in df.columns and 'from_address' in df.columns:
        risk_df['timestamp'] = pd.to_datetime(risk_df['timestamp'])
        
        # Group by sender and time window
        risk_df['time_window'] = risk_df['timestamp'].dt.floor('1H')
        transaction_counts = risk_df.groupby(['from_address', 'time_window']).size().reset_index(name='count')
        
        # Find high frequency senders
        high_freq_threshold = np.percentile(transaction_counts['count'], 95)
        high_freq_groups = transaction_counts[transaction_counts['count'] > high_freq_threshold]
        
        for _, row in high_freq_groups.iterrows():
            mask = (risk_df['from_address'] == row['from_address']) & (risk_df['time_window'] == row['time_window'])
            risk_df.loc[mask, 'risk_score'] += 0.25
            risk_df.loc[mask, 'risk_factors'] += "High transaction frequency; "
    
    # Risk factor 4: New addresses with high value transactions
    if 'from_address' in df.columns and 'value' in df.columns and 'timestamp' in df.columns:
        # Sort by timestamp
        sorted_df = df.sort_values('timestamp')
        
        # Get first appearance of each address
        first_appearance = sorted_df.groupby('from_address')['timestamp'].first().reset_index()
        first_appearance['timestamp'] = pd.to_datetime(first_appearance['timestamp'])
        
        # Identify new addresses with high value transactions
        for idx, row in risk_df.iterrows():
            address_first_tx = first_appearance[first_appearance['from_address'] == row['from_address']]['timestamp'].iloc[0]
            current_tx_time = pd.to_datetime(row['timestamp'])
            
            # If address is new (less than 24 hours old) and transaction value is high
            if (current_tx_time - address_first_tx).total_seconds() < 86400 and row['value'] > mean_value + 2*std_value:
                risk_df.loc[idx, 'risk_score'] += 0.3
                risk_df.loc[idx, 'risk_factors'] += "New address with high value transaction; "
    
    # Filter transactions based on risk threshold
    risk_df = risk_df[['from_address', 'to_address', 'value', 'timestamp', 'risk_score', 'risk_factors']]
    
    # Cap risk score at 1.0
    risk_df['risk_score'] = risk_df['risk_score'].clip(upper=1.0)
    
    # Add risk category
    risk_df['risk_category'] = pd.cut(
        risk_df['risk_score'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    return risk_df
