"""
Enhanced Anomaly Detection System for QuantumGuard AI
Integrating advanced ML models with ensemble methods and online learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
import pickle
import os

# Import advanced ML models
from advanced_ml_models import (
    LSTMAutoencoder, VariationalAutoencoder, GraphNeuralNetwork,
    EnsembleAnomalyDetector, OnlineLearningDetector, FeedbackLearningSystem,
    AdvancedMLModelFactory
)

# Traditional ML models
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report


class EnhancedAnomalyDetectionSystem:
    """
    Enhanced anomaly detection system combining traditional and advanced ML models
    """
    
    def __init__(self):
        self.traditional_models = {}
        self.advanced_models = {}
        self.ensemble_detector = None
        self.online_learner = None
        self.feedback_system = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_performance = {}
        
    def initialize_models(self):
        """Initialize all anomaly detection models"""
        print("🚀 Initializing Enhanced Anomaly Detection System...")
        
        # Traditional models
        self.traditional_models = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'dbscan_outliers': DBSCAN(eps=0.5, min_samples=5)
        }
        
        # Advanced models
        self.advanced_models = {
            'lstm_autoencoder': AdvancedMLModelFactory.create_lstm_autoencoder(),
            'vae': AdvancedMLModelFactory.create_vae(),
            'gnn': AdvancedMLModelFactory.create_gnn()
        }
        
        # Ensemble detector
        self.ensemble_detector = AdvancedMLModelFactory.create_ensemble_detector()
        
        # Online learning system
        self.online_learner = AdvancedMLModelFactory.create_online_learning_system()
        
        # Feedback system
        self.feedback_system = AdvancedMLModelFactory.create_feedback_system()
        
        print("✅ All models initialized successfully")
        
    def prepare_features(self, df):
        """Prepare features for anomaly detection"""
        if df is None or df.empty:
            return None
            
        features = []
        
        # Basic transaction features
        if 'amount' in df.columns:
            features.append('amount')
        if 'gas_price' in df.columns:
            features.append('gas_price')
        if 'gas_used' in df.columns:
            features.append('gas_used')
            
        # Time-based features
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            features.extend(['hour', 'day_of_week'])
            
        # Address-based features (simplified)
        if 'from_address' in df.columns and 'to_address' in df.columns:
            df['same_address'] = (df['from_address'] == df['to_address']).astype(int)
            features.append('same_address')
            
        # Create additional engineered features
        if 'amount' in df.columns:
            df['log_amount'] = np.log1p(df['amount'])
            features.append('log_amount')
            
        # Select available features
        available_features = [f for f in features if f in df.columns]
        
        if not available_features:
            # Create minimal feature set if none available
            df['transaction_index'] = range(len(df))
            df['constant_feature'] = 1
            available_features = ['transaction_index', 'constant_feature']
            
        return df[available_features].fillna(0)
    
    def train_traditional_models(self, X):
        """Train traditional anomaly detection models"""
        results = {}
        
        for name, model in self.traditional_models.items():
            try:
                if name == 'isolation_forest':
                    model.fit(X)
                    results[name] = {"status": "success", "model_type": "supervised"}
                elif name == 'dbscan_outliers':
                    # DBSCAN doesn't have explicit anomaly detection, use clustering
                    clusters = model.fit_predict(X)
                    # Consider outliers as points in cluster -1
                    results[name] = {
                        "status": "success", 
                        "model_type": "clustering",
                        "outlier_ratio": np.sum(clusters == -1) / len(clusters)
                    }
                
                print(f"✅ {name} trained successfully")
                
            except Exception as e:
                print(f"⚠️ {name} training failed: {e}")
                results[name] = {"status": "failed", "error": str(e)}
                
        return results
    
    def train_advanced_models(self, X):
        """Train advanced ML models"""
        results = {}
        
        for name, model in self.advanced_models.items():
            try:
                if hasattr(model, 'fit'):
                    model_result = model.fit(X)
                    results[name] = {
                        "status": "success",
                        "model_type": "advanced",
                        "training_info": model_result
                    }
                    print(f"✅ {name} trained successfully")
                else:
                    results[name] = {"status": "skipped", "reason": "no fit method"}
                    
            except Exception as e:
                print(f"⚠️ {name} training failed: {e}")
                results[name] = {"status": "failed", "error": str(e)}
                
        return results
    
    def fit(self, df):
        """Train the complete enhanced anomaly detection system"""
        if not hasattr(self, 'traditional_models') or not self.traditional_models:
            self.initialize_models()
            
        # Prepare features
        X = self.prepare_features(df)
        if X is None:
            raise ValueError("Cannot prepare features from the provided data")
            
        print(f"📊 Training on {len(X)} transactions with {X.shape[1]} features...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train traditional models
        traditional_results = self.train_traditional_models(X_scaled)
        
        # Train advanced models
        advanced_results = self.train_advanced_models(X_scaled)
        
        # Train ensemble detector
        ensemble_result = {}
        try:
            ensemble_result = self.ensemble_detector.fit(X_scaled)
            print("✅ Ensemble detector trained successfully")
        except Exception as e:
            print(f"⚠️ Ensemble training failed: {e}")
            ensemble_result = {"status": "failed", "error": str(e)}
        
        # Initialize online learner
        online_result = {}
        try:
            self.online_learner.initialize()
            online_result = {"status": "initialized"}
            print("✅ Online learner initialized")
        except Exception as e:
            print(f"⚠️ Online learner initialization failed: {e}")
            online_result = {"status": "failed", "error": str(e)}
        
        self.is_trained = True
        
        # Store training results
        self.model_performance = {
            'traditional_models': traditional_results,
            'advanced_models': advanced_results,
            'ensemble_detector': ensemble_result,
            'online_learner': online_result,
            'training_timestamp': datetime.now().isoformat(),
            'training_data_shape': X_scaled.shape
        }
        
        print("🎉 Enhanced Anomaly Detection System training completed!")
        return self.model_performance
    
    def detect_anomalies(self, df):
        """Detect anomalies using the ensemble of models"""
        if not self.is_trained:
            raise ValueError("System must be trained first")
            
        # Prepare features
        X = self.prepare_features(df)
        if X is None:
            return None, None, {}
            
        X_scaled = self.scaler.transform(X)
        
        results = {
            'traditional_predictions': {},
            'advanced_predictions': {},
            'ensemble_prediction': None,
            'confidence_scores': {},
            'model_agreements': {}
        }
        
        # Traditional model predictions
        for name, model in self.traditional_models.items():
            try:
                if name == 'isolation_forest':
                    pred = model.predict(X_scaled)
                    anomalies = pred == -1  # Isolation Forest uses -1 for anomalies
                    scores = model.decision_function(X_scaled)
                    results['traditional_predictions'][name] = {
                        'anomalies': anomalies,
                        'scores': scores
                    }
                elif name == 'dbscan_outliers':
                    clusters = model.fit_predict(X_scaled)
                    anomalies = clusters == -1
                    results['traditional_predictions'][name] = {
                        'anomalies': anomalies,
                        'clusters': clusters
                    }
            except Exception as e:
                print(f"⚠️ {name} prediction failed: {e}")
                
        # Advanced model predictions
        for name, model in self.advanced_models.items():
            try:
                if hasattr(model, 'predict_anomalies'):
                    anomalies, scores, _ = model.predict_anomalies(X_scaled)
                    results['advanced_predictions'][name] = {
                        'anomalies': anomalies,
                        'scores': scores
                    }
            except Exception as e:
                print(f"⚠️ {name} prediction failed: {e}")
        
        # Ensemble prediction
        try:
            ensemble_anomalies, ensemble_scores, individual_preds = self.ensemble_detector.predict_anomalies(X_scaled)
            results['ensemble_prediction'] = {
                'anomalies': ensemble_anomalies,
                'scores': ensemble_scores,
                'individual_predictions': individual_preds
            }
        except Exception as e:
            print(f"⚠️ Ensemble prediction failed: {e}")
            # Fallback to simple majority voting
            all_predictions = []
            for trad_preds in results['traditional_predictions'].values():
                if 'anomalies' in trad_preds:
                    all_predictions.append(trad_preds['anomalies'])
            for adv_preds in results['advanced_predictions'].values():
                if 'anomalies' in adv_preds:
                    all_predictions.append(adv_preds['anomalies'])
                    
            if all_predictions:
                ensemble_anomalies = np.sum(all_predictions, axis=0) > len(all_predictions) // 2
                results['ensemble_prediction'] = {
                    'anomalies': ensemble_anomalies,
                    'method': 'majority_voting'
                }
        
        # Calculate model agreement
        self._calculate_model_agreement(results)
        
        # Return final anomalies and results
        if results['ensemble_prediction']:
            final_anomalies = results['ensemble_prediction']['anomalies']
        else:
            final_anomalies = np.zeros(len(X_scaled), dtype=bool)
            
        return final_anomalies, np.where(final_anomalies)[0].tolist(), results
    
    def _calculate_model_agreement(self, results):
        """Calculate agreement between different models"""
        predictions = []
        model_names = []
        
        # Collect all predictions
        for name, pred in results['traditional_predictions'].items():
            if 'anomalies' in pred:
                predictions.append(pred['anomalies'])
                model_names.append(name)
                
        for name, pred in results['advanced_predictions'].items():
            if 'anomalies' in pred:
                predictions.append(pred['anomalies'])
                model_names.append(name)
        
        if len(predictions) >= 2:
            # Calculate pairwise agreement
            agreements = {}
            for i in range(len(predictions)):
                for j in range(i+1, len(predictions)):
                    agreement = np.mean(predictions[i] == predictions[j])
                    agreements[f"{model_names[i]}_vs_{model_names[j]}"] = agreement
            
            results['model_agreements'] = agreements
            results['average_agreement'] = np.mean(list(agreements.values()))
        
    def update_with_feedback(self, transaction_indices, predictions, user_feedback):
        """Update models with user feedback"""
        if not hasattr(self, 'feedback_system') or self.feedback_system is None:
            return {}
            
        feedback_results = []
        
        for i, (pred, feedback) in enumerate(zip(predictions, user_feedback)):
            result = self.feedback_system.collect_feedback(
                transaction_id=transaction_indices[i] if i < len(transaction_indices) else i,
                prediction=pred,
                user_feedback=feedback
            )
            feedback_results.append(result)
        
        # Calculate updated performance
        performance = self.feedback_system.calculate_model_performance()
        suggestions = self.feedback_system.suggest_model_updates()
        
        return {
            'feedback_collected': len(feedback_results),
            'performance': performance,
            'suggestions': suggestions,
            'total_feedback': len(self.feedback_system.feedback_data)
        }
    
    def continuous_learning_update(self, new_df):
        """Update models with new data using online learning"""
        if not hasattr(self, 'online_learner') or self.online_learner is None:
            return {"status": "online_learner_not_available"}
            
        # Prepare features for new data
        X_new = self.prepare_features(new_df)
        if X_new is None:
            return {"status": "failed", "reason": "cannot_prepare_features"}
            
        X_new_scaled = self.scaler.transform(X_new)
        
        # Update online learner
        drift_detected = self.online_learner.update(X_new_scaled)
        
        return {
            "status": "updated",
            "drift_detected": drift_detected,
            "new_data_points": len(X_new_scaled)
        }
    
    def get_model_summary(self):
        """Get summary of all models and their performance"""
        if not self.is_trained:
            return {"status": "not_trained"}
            
        summary = {
            "system_status": "trained",
            "training_performance": self.model_performance,
            "available_models": {
                "traditional": list(self.traditional_models.keys()),
                "advanced": list(self.advanced_models.keys()),
                "ensemble": "available" if self.ensemble_detector else "not_available",
                "online_learning": "available" if self.online_learner else "not_available",
                "feedback_system": "available" if self.feedback_system else "not_available"
            }
        }
        
        # Add feedback system performance if available
        if self.feedback_system and self.feedback_system.feedback_data:
            summary["feedback_performance"] = self.feedback_system.calculate_model_performance()
            
        return summary
    
    def save_models(self, filepath):
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("System must be trained first")
            
        model_data = {
            'scaler': self.scaler,
            'traditional_models': self.traditional_models,
            'model_performance': self.model_performance,
            'is_trained': self.is_trained,
            'feedback_data': self.feedback_system.feedback_data if self.feedback_system else []
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath):
        """Load trained models from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.scaler = model_data['scaler']
        self.traditional_models = model_data['traditional_models']
        self.model_performance = model_data['model_performance']
        self.is_trained = model_data['is_trained']
        
        # Reinitialize advanced models and systems
        self.initialize_models()
        
        # Restore feedback data if available
        if 'feedback_data' in model_data and model_data['feedback_data']:
            self.feedback_system.feedback_data = model_data['feedback_data']
            
        print(f"✅ Models loaded from {filepath}")


# Global instance for easy access
enhanced_anomaly_detector = EnhancedAnomalyDetectionSystem()


# Integration functions for backward compatibility
def train_enhanced_anomaly_detection(df):
    """Train the enhanced anomaly detection system"""
    return enhanced_anomaly_detector.fit(df)


def detect_enhanced_anomalies(df):
    """Detect anomalies using enhanced system"""
    return enhanced_anomaly_detector.detect_anomalies(df)


def update_models_with_feedback(transaction_indices, predictions, user_feedback):
    """Update models with user feedback"""
    return enhanced_anomaly_detector.update_with_feedback(transaction_indices, predictions, user_feedback)


def continuous_learning_update(new_df):
    """Update models with continuous learning"""
    return enhanced_anomaly_detector.continuous_learning_update(new_df)


# Export main classes
__all__ = [
    'EnhancedAnomalyDetectionSystem',
    'enhanced_anomaly_detector',
    'train_enhanced_anomaly_detection',
    'detect_enhanced_anomalies',
    'update_models_with_feedback',
    'continuous_learning_update'
]