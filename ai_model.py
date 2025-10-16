import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Tuple, Optional
import logging
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionModel:
    """
    Machine Learning model for detecting fraudulent blockchain transactions.
    Uses Isolation Forest algorithm for anomaly detection.
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize the fraud detection model.
        
        Args:
            contamination: The proportion of outliers in the dataset (default 0.1 = 10%)
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1  # Use all CPU cores
        )
        self.feature_columns = None
        self.is_fitted = False
        logger.info(f"Initialized FraudDetectionModel with contamination={contamination}")
    
    def train(self, X: pd.DataFrame) -> None:
        """
        Train the fraud detection model.
        
        Args:
            X: DataFrame with feature columns for training
        """
        try:
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Handle missing values
            X_clean = X.fillna(X.median())
            
            # Train model
            logger.info(f"Training model with {len(X_clean)} samples and {len(self.feature_columns)} features...")
            self.model.fit(X_clean)
            
            self.is_fitted = True
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies and return predictions and risk scores.
        
        Args:
            X: DataFrame with feature columns
        
        Returns:
            Tuple of (predictions, risk_scores)
            - predictions: Array of -1 (anomaly) or 1 (normal)
            - risk_scores: Array of risk scores (0-100, higher is riskier)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction. Call train() first.")
        
        try:
            # Ensure same features as training
            if self.feature_columns:
                missing_cols = set(self.feature_columns) - set(X.columns)
                if missing_cols:
                    logger.warning(f"Missing features in prediction data: {missing_cols}")
                    for col in missing_cols:
                        X[col] = 0  # Add missing columns with default value
                X = X[self.feature_columns]
            
            # Handle missing values
            X_clean = X.fillna(X.median())
            
            # Get predictions (-1 for anomalies, 1 for normal)
            predictions = self.model.predict(X_clean)
            
            # Get anomaly scores (lower is more anomalous)
            scores = self.model.score_samples(X_clean)
            
            # Convert to 0-100 risk score (higher is riskier)
            risk_scores = self._convert_to_risk_score(scores)
            
            logger.info(f"Prediction complete: {(predictions == -1).sum()} anomalies detected out of {len(predictions)} transactions")
            
            return predictions, risk_scores
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def _convert_to_risk_score(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert anomaly scores to 0-100 risk scale.
        
        Args:
            scores: Raw anomaly scores from Isolation Forest
        
        Returns:
            Risk scores normalized to 0-100 range
        """
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            # All scores are the same, return moderate risk
            return np.full_like(scores, 50.0)
        
        # Invert so higher score = higher risk
        # Isolation Forest returns lower scores for anomalies
        normalized = 100 * (max_score - scores) / (max_score - min_score)
        
        return normalized
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        try:
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'is_fitted': self.is_fitted
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to the saved model file
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.is_fitted = model_data['is_fitted']
            
            logger.info(f"Model loaded successfully from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores (not directly available in Isolation Forest,
        but we can estimate based on the contamination of each feature).
        
        Returns:
            DataFrame with feature importance scores, or None if model not fitted
        """
        if not self.is_fitted or self.feature_columns is None:
            logger.warning("Model not fitted or no feature columns available")
            return None
        
        try:
            # For Isolation Forest, we can't get direct feature importance
            # But we can return the feature names for reference
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'note': ['Feature used in model'] * len(self.feature_columns)
            })
            return importance_df
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None

    def evaluate_transaction(self, transaction_features: pd.DataFrame) -> dict:
        """
        Evaluate a single transaction and return detailed risk assessment.
        
        Args:
            transaction_features: DataFrame with single transaction features
        
        Returns:
            Dictionary with risk assessment details
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            predictions, risk_scores = self.predict(transaction_features)
            
            is_anomaly = predictions[0] == -1
            risk_score = risk_scores[0]
            
            # Determine risk level
            if risk_score >= 80:
                risk_level = "Critical"
            elif risk_score >= 60:
                risk_level = "High"
            elif risk_score >= 40:
                risk_level = "Medium"
            elif risk_score >= 20:
                risk_level = "Low"
            else:
                risk_level = "Very Low"
            
            return {
                'is_anomaly': bool(is_anomaly),
                'risk_score': float(risk_score),
                'risk_level': risk_level,
                'recommendation': 'Flag for review' if is_anomaly else 'Normal transaction'
            }
            
        except Exception as e:
            logger.error(f"Error evaluating transaction: {str(e)}")
            raise
