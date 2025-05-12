import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Any, Dict

def train_anomaly_detection(features: pd.DataFrame) -> Any:
    """
    Train an anomaly detection model using Isolation Forest.
    
    Args:
        features: DataFrame containing extracted features
    
    Returns:
        Trained anomaly detection model
    """
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Initialize and train Isolation Forest model
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.05,  # Expected proportion of anomalies
        random_state=42
    )
    
    model.fit(scaled_features)
    
    # Store scaler with the model for future use
    model.scaler = scaler
    
    return model

def detect_anomalies(model: Any, features: pd.DataFrame, sensitivity: float = 0.8) -> List[int]:
    """
    Detect anomalies in blockchain transaction data.
    
    Args:
        model: Trained anomaly detection model
        features: DataFrame containing extracted features
        sensitivity: Threshold adjustment for anomaly detection (0.0-1.0)
    
    Returns:
        List of indices corresponding to anomalous transactions
    """
    # Scale features using the same scaler
    scaled_features = model.scaler.transform(features)
    
    # Predict anomaly scores (-1 for anomalies, 1 for normal)
    anomaly_scores = model.decision_function(scaled_features)
    
    # Adjust threshold based on sensitivity
    # Lower values = more sensitive (more anomalies detected)
    base_threshold = -0.2  # Base threshold for anomaly scores
    adjusted_threshold = base_threshold - (sensitivity * 0.5)  # Scale with sensitivity
    
    # Find anomalies based on adjusted threshold
    anomalies = np.where(anomaly_scores < adjusted_threshold)[0].tolist()
    
    return anomalies

def cluster_transactions(features: pd.DataFrame) -> np.ndarray:
    """
    Cluster similar transactions using DBSCAN.
    
    Args:
        features: DataFrame containing extracted features
    
    Returns:
        Array of cluster labels for each transaction
    """
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Use DBSCAN for clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_features)
    
    return clusters

def calculate_transaction_risk(transaction_features: pd.Series) -> float:
    """
    Calculate risk score for a single transaction based on its features.
    
    Args:
        transaction_features: Series with transaction features
    
    Returns:
        Risk score between 0.0 and 1.0
    """
    # This is a simplified risk scoring model
    # In a real application, this would be more sophisticated
    
    risk_score = 0.0
    risk_weights = {
        'transaction_value_z': 0.3,
        'sender_activity': 0.2,
        'receiver_activity': 0.2,
        'temporal_pattern': 0.15,
        'network_centrality': 0.15
    }
    
    # Apply weights to normalized feature values
    for feature, weight in risk_weights.items():
        if feature in transaction_features:
            # Normalize the feature value to 0-1 range
            normalized_value = min(abs(transaction_features[feature]), 10) / 10
            risk_score += normalized_value * weight
    
    return min(risk_score, 1.0)  # Cap at 1.0
