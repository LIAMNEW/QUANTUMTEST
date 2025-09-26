import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Any, Dict

# Import enhanced anomaly detection system
try:
    from enhanced_anomaly_detection import (
        enhanced_anomaly_detector,
        train_enhanced_anomaly_detection,
        detect_enhanced_anomalies
    )
    HAS_ENHANCED_DETECTION = True
except ImportError:
    HAS_ENHANCED_DETECTION = False

def train_anomaly_detection(features: pd.DataFrame) -> Any:
    """
    Train an enhanced anomaly detection model using advanced ML algorithms.
    Falls back to traditional Isolation Forest if enhanced models unavailable.
    
    Args:
        features: DataFrame containing extracted features
    
    Returns:
        Trained anomaly detection model
    """
    if HAS_ENHANCED_DETECTION:
        print("🚀 Training Enhanced Anomaly Detection System with advanced ML models...")
        try:
            # Use enhanced system with LSTM autoencoders, VAE, and ensemble methods
            enhanced_anomaly_detector.fit(features)
            print("✅ Enhanced anomaly detection training completed")
            return enhanced_anomaly_detector
        except Exception as e:
            print(f"⚠️ Enhanced training failed: {e}, falling back to traditional method")
    
    # Traditional Isolation Forest fallback
    print("📊 Training traditional Isolation Forest model...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.05,  # Expected proportion of anomalies
        random_state=42
    )
    
    model.fit(scaled_features)
    model.scaler = scaler
    
    print("✅ Traditional anomaly detection training completed")
    return model

def detect_anomalies(model: Any, features: pd.DataFrame, sensitivity: float = 0.8) -> List[int]:
    """
    Detect anomalies using enhanced ML models or traditional methods.
    
    Args:
        model: Trained anomaly detection model (enhanced or traditional)
        features: DataFrame containing extracted features  
        sensitivity: Threshold adjustment for anomaly detection (0.0-1.0)
    
    Returns:
        List of indices corresponding to anomalous transactions
    """
    # Check if this is an enhanced detection system
    if HAS_ENHANCED_DETECTION and hasattr(model, 'detect_anomalies'):
        print("🔍 Using Enhanced Anomaly Detection with advanced ML models...")
        try:
            anomalies, anomaly_indices, detection_results = model.detect_anomalies(features)
            print(f"✅ Enhanced detection completed: {len(anomaly_indices)} anomalies found")
            
            # Store detection results for analysis
            if hasattr(model, 'last_detection_results'):
                model.last_detection_results = detection_results
                
            return anomaly_indices
            
        except Exception as e:
            print(f"⚠️ Enhanced detection failed: {e}, falling back to traditional method")
    
    # Traditional Isolation Forest method
    print("📊 Using traditional Isolation Forest detection...")
    scaled_features = model.scaler.transform(features)
    
    # Predict anomaly scores (-1 for anomalies, 1 for normal)
    anomaly_scores = model.decision_function(scaled_features)
    
    # Adjust threshold based on sensitivity
    base_threshold = -0.2
    adjusted_threshold = base_threshold - (sensitivity * 0.5)
    
    # Find anomalies based on adjusted threshold
    anomalies = np.where(anomaly_scores < adjusted_threshold)[0].tolist()
    
    print(f"✅ Traditional detection completed: {len(anomalies)} anomalies found")
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
