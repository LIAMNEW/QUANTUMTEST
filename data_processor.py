import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import networkx as nx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_blockchain_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate blockchain transaction DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check for empty DataFrame
    if df.empty:
        validation_results['valid'] = False
        validation_results['errors'].append("DataFrame is empty")
        return validation_results
    
    # Check for essential columns
    required_columns = ['from_address', 'to_address']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        validation_results['warnings'].append(
            f"Missing columns (will try to infer): {missing_cols}"
        )
    
    # Check for value/amount column
    value_columns = ['value', 'amount', 'transaction_value', 'tx_value']
    has_value = any(col in df.columns for col in value_columns)
    if not has_value:
        validation_results['warnings'].append(
            "No value/amount column found - will use default values"
        )
    
    # Check for null values in critical columns
    if 'from_address' in df.columns:
        null_count = df['from_address'].isnull().sum()
        if null_count > 0:
            validation_results['warnings'].append(
                f"Found {null_count} null values in from_address column"
            )
    
    if 'to_address' in df.columns:
        null_count = df['to_address'].isnull().sum()
        if null_count > 0:
            validation_results['warnings'].append(
                f"Found {null_count} null values in to_address column"
            )
    
    logger.info(f"Validation complete: {len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings")
    return validation_results

def clean_blockchain_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize blockchain transaction data.
    
    Args:
        df: DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove duplicates
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed = initial_len - len(df_clean)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")
    
    # Strip whitespace from string columns
    string_columns = df_clean.select_dtypes(include=['object']).columns
    for col in string_columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.strip()
    
    # Handle missing addresses
    if 'from_address' in df_clean.columns:
        df_clean['from_address'] = df_clean['from_address'].fillna('unknown')
    if 'to_address' in df_clean.columns:
        df_clean['to_address'] = df_clean['to_address'].fillna('unknown')
    
    logger.info(f"Data cleaning complete: {len(df_clean)} rows remaining")
    return df_clean

def preprocess_blockchain_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess blockchain transaction data with enhanced error handling.
    
    Args:
        df: DataFrame containing raw blockchain data
    
    Returns:
        Preprocessed DataFrame
    """
    try:
        processed_df = df.copy()
        
        # Log diagnostic information
        logger.info(f"Preprocessing data: {len(processed_df)} rows, columns: {processed_df.columns.tolist()}")
        
        # Set index to None to avoid the "not in index" error
        processed_df = processed_df.reset_index(drop=True)
        
        # Check if essential columns exist
        required_columns = ['from_address', 'to_address']
        for col in required_columns:
            if col not in processed_df.columns:
                # Try to infer column names if they're named differently
                if col == 'from_address' and 'sender' in processed_df.columns:
                    processed_df['from_address'] = processed_df['sender']
                elif col == 'to_address' and 'receiver' in processed_df.columns:
                    processed_df['to_address'] = processed_df['receiver']
                else:
                    # Create empty column if it can't be inferred
                    processed_df[col] = np.nan
        
        # Handle timestamp - IMPORTANT FIX FOR "TIMESTAMP NOT IN INDEX" ERROR
        if 'timestamp' in processed_df.columns:
            # Convert to datetime
            try:
                processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
            except Exception as e:
                logger.warning(f"Error converting timestamp to datetime: {str(e)}")
                # Try unix timestamp conversion
                try:
                    processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'], unit='s')
                except Exception as e:
                    logger.warning(f"Error converting as unix timestamp: {str(e)}")
                    # Don't drop the column, just leave it as is
                    logger.info("Keeping timestamp column as-is without conversion")
        else:
            # CRITICAL FIX: If timestamp is missing, create a dummy timestamp column
            logger.warning("No timestamp column found. Creating a dummy timestamp column.")
            processed_df['timestamp'] = pd.to_datetime('2025-01-01')  # Use a default date
        
        # Handle transaction value
        if 'value' in processed_df.columns:
            # Convert to numeric
            processed_df['value'] = pd.to_numeric(processed_df['value'], errors='coerce')
            # Fill missing values with median (avoiding the deprecated inplace method)
            median_value = processed_df['value'].median()
            processed_df['value'] = processed_df['value'].fillna(median_value)
        else:
            # Try to find value column with different name
            value_columns = ['amount', 'transaction_value', 'tx_value']
            for col in value_columns:
                if col in processed_df.columns:
                    processed_df['value'] = pd.to_numeric(processed_df[col], errors='coerce')
                    break
            
            # Create a dummy value column if none exists
            if 'value' not in processed_df.columns:
                processed_df['value'] = 1.0
        
        # Handle categorical data
        if 'status' in processed_df.columns:
            # Convert to lowercase for consistency
            processed_df['status'] = processed_df['status'].str.lower()
        
        # Handle missing values
        processed_df.fillna({
            'from_address': 'unknown',
            'to_address': 'unknown'
        }, inplace=True)
        
        logger.info("Preprocessing completed successfully")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error preprocessing blockchain data: {str(e)}")
        raise

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from preprocessed blockchain data for ML models.
    
    Args:
        df: Preprocessed DataFrame
    
    Returns:
        DataFrame with extracted features
    """
    features = pd.DataFrame()
    
    # Transaction value features
    if 'value' in df.columns:
        features['transaction_value'] = df['value']
        # Z-score normalization for transaction values
        features['transaction_value_z'] = (df['value'] - df['value'].mean()) / df['value'].std()
    
    # Network features
    if 'from_address' in df.columns and 'to_address' in df.columns:
        # Create a graph from transactions
        G = nx.from_pandas_edgelist(df, 'from_address', 'to_address', create_using=nx.DiGraph())
        
        # Calculate sender activity (out-degree)
        out_degree = dict(G.out_degree())
        features['sender_activity'] = df['from_address'].map(lambda x: out_degree.get(x, 0))
        
        # Calculate receiver activity (in-degree)
        in_degree = dict(G.in_degree())
        features['receiver_activity'] = df['to_address'].map(lambda x: in_degree.get(x, 0))
        
        # Normalize degree centrality
        if len(G) > 1:  # Only if we have at least 2 nodes
            # Centrality measures
            centrality = nx.degree_centrality(G)
            features['network_centrality'] = df['from_address'].map(lambda x: centrality.get(x, 0))
    
    # Temporal features
    if 'timestamp' in df.columns:
        # Hour of day
        if pd.api.types.is_datetime64_dtype(df['timestamp']):
            features['hour_of_day'] = df['timestamp'].dt.hour
            features['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Encode hour of day using sine and cosine to capture cyclical nature
            # This preserves the circular relationship of hours (23 is close to 0)
            features['hour_sin'] = np.sin(2 * np.pi * features['hour_of_day'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour_of_day'] / 24)
            
            # Similarly encode day of week
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    
    # Transaction pattern features
    # Count transactions between each pair of addresses
    if 'from_address' in df.columns and 'to_address' in df.columns:
        address_pairs = df.groupby(['from_address', 'to_address']).size().reset_index(name='tx_count')
        pair_dict = dict(zip(zip(address_pairs['from_address'], address_pairs['to_address']), address_pairs['tx_count']))
        features['pair_frequency'] = df.apply(lambda x: pair_dict.get((x['from_address'], x['to_address']), 0), axis=1)
    
    # Normalize all features to prevent any one feature from dominating
    numeric_features = features.select_dtypes(include=[np.number])
    features[numeric_features.columns] = (numeric_features - numeric_features.mean()) / numeric_features.std()
    
    # Replace infinities and NaNs
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    
    return features

def calculate_network_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate various network metrics from blockchain transaction data.
    
    Args:
        df: Preprocessed DataFrame
    
    Returns:
        Dictionary of network metrics
    """
    # Create a graph from transactions
    G = nx.from_pandas_edgelist(df, 'from_address', 'to_address', create_using=nx.DiGraph())
    
    metrics = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
    }
    
    # Calculate metrics only if we have enough nodes
    if G.number_of_nodes() > 1:
        # Average degree
        degrees = [d for n, d in G.degree()]
        metrics['avg_degree'] = sum(degrees) / len(degrees)
        
        # Identify most connected nodes
        sorted_degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        metrics['top_addresses'] = sorted_degrees[:5] if len(sorted_degrees) >= 5 else sorted_degrees
        
        # Clustering coefficient
        try:
            metrics['clustering'] = nx.average_clustering(G.to_undirected())
        except:
            metrics['clustering'] = 0
            
        # Connected components
        undirected_G = G.to_undirected()
        metrics['connected_components'] = nx.number_connected_components(undirected_G)
        
        # Largest component size
        largest_cc = max(nx.connected_components(undirected_G), key=len)
        metrics['largest_component_size'] = len(largest_cc)
    
    return metrics
