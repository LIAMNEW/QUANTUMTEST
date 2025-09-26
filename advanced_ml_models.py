"""
Advanced Machine Learning Models for QuantumGuard AI
Enhanced anomaly detection, Graph Neural Networks, and ensemble methods
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import json

# Core ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score

# Advanced ML libraries (will be installed separately)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Online learning library
try:
    from river import anomaly, compose, preprocessing
    from river.drift import ADWIN
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


class LSTMAutoencoder:
    """
    LSTM Autoencoder for advanced anomaly detection in transaction sequences
    """
    
    def __init__(self, sequence_length=10, n_features=8, encoding_dim=32):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.encoding_dim = encoding_dim
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def build_model(self):
        """Build the LSTM Autoencoder architecture"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for LSTM Autoencoder")
            
        # Encoder
        input_layer = keras.Input(shape=(self.sequence_length, self.n_features))
        encoded = layers.LSTM(self.encoding_dim, return_sequences=True)(input_layer)
        encoded = layers.LSTM(self.encoding_dim//2, return_sequences=False)(encoded)
        
        # Decoder
        decoded = layers.RepeatVector(self.sequence_length)(encoded)
        decoded = layers.LSTM(self.encoding_dim//2, return_sequences=True)(decoded)
        decoded = layers.LSTM(self.encoding_dim, return_sequences=True)(decoded)
        decoded = layers.TimeDistributed(layers.Dense(self.n_features))(decoded)
        
        # Compile model
        self.model = keras.Model(input_layer, decoded)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return self.model
    
    def prepare_sequences(self, data):
        """Convert transaction data into sequences for LSTM"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
    
    def fit(self, X, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM Autoencoder"""
        if not HAS_TENSORFLOW:
            print("⚠️ TensorFlow not available, using simplified implementation")
            return self._fit_simplified(X)
            
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences
        X_sequences = self.prepare_sequences(X_scaled)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Train the model
        history = self.model.fit(
            X_sequences, X_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            verbose=0
        )
        
        self.is_trained = True
        return history
    
    def _fit_simplified(self, X):
        """Simplified version when TensorFlow is not available"""
        self.scaler.fit(X)
        self.threshold = np.percentile(np.sum((X - np.mean(X, axis=0))**2, axis=1), 95)
        self.is_trained = True
        return {"simplified": True}
    
    def predict_anomalies(self, X, threshold_percentile=95):
        """Predict anomalies using reconstruction error"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        if not HAS_TENSORFLOW:
            return self._predict_simplified(X)
            
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Prepare sequences
        X_sequences = self.prepare_sequences(X_scaled)
        
        # Get reconstructions
        reconstructions = self.model.predict(X_sequences, verbose=0)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(X_sequences - reconstructions), axis=(1, 2))
        
        # Determine threshold
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        
        # Identify anomalies
        anomalies = reconstruction_errors > threshold
        
        return anomalies, reconstruction_errors, threshold
    
    def _predict_simplified(self, X):
        """Simplified prediction when TensorFlow is not available"""
        X_scaled = self.scaler.transform(X)
        errors = np.sum((X_scaled - np.mean(X_scaled, axis=0))**2, axis=1)
        anomalies = errors > self.threshold
        return anomalies, errors, self.threshold


class VariationalAutoencoder:
    """
    Variational Autoencoder for probabilistic anomaly detection
    """
    
    def __init__(self, input_dim=8, latent_dim=4, hidden_dims=[16, 8]):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.model = None
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def build_model(self):
        """Build the VAE architecture"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for VAE")
            
        # Encoder
        encoder_inputs = keras.Input(shape=(self.input_dim,))
        x = encoder_inputs
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
        
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
        
        # Create encoder model
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        decoder_inputs = keras.Input(shape=(self.latent_dim,), name='z_sampling')
        x = decoder_inputs
        for dim in reversed(self.hidden_dims):
            x = layers.Dense(dim, activation='relu')(x)
        decoder_outputs = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        # Create decoder model
        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
        
        # Full VAE model
        outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.model = keras.Model(encoder_inputs, outputs, name='vae')
        
        # VAE loss function
        def vae_loss(x, x_decoded):
            reconstruction_loss = keras.losses.binary_crossentropy(x, x_decoded)
            reconstruction_loss *= self.input_dim
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
            return total_loss
        
        self.model.compile(optimizer='adam', loss=vae_loss)
        
        return self.model
    
    def fit(self, X, epochs=100, batch_size=32, validation_split=0.2):
        """Train the VAE"""
        if not HAS_TENSORFLOW:
            print("⚠️ TensorFlow not available, using simplified implementation")
            return self._fit_simplified(X)
            
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Train the model
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        self.is_trained = True
        return history
    
    def _fit_simplified(self, X):
        """Simplified version when TensorFlow is not available"""
        self.scaler.fit(X)
        self.threshold = np.percentile(np.sum((X - np.mean(X, axis=0))**2, axis=1), 95)
        self.is_trained = True
        return {"simplified": True}
    
    def predict_anomalies(self, X, threshold_percentile=95):
        """Predict anomalies using reconstruction probability"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        if not HAS_TENSORFLOW:
            return self._predict_simplified(X)
            
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Get reconstructions
        reconstructions = self.model.predict(X_scaled, verbose=0)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Determine threshold
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        
        # Identify anomalies
        anomalies = reconstruction_errors > threshold
        
        return anomalies, reconstruction_errors, threshold
    
    def _predict_simplified(self, X):
        """Simplified prediction when TensorFlow is not available"""
        X_scaled = self.scaler.transform(X)
        errors = np.sum((X_scaled - np.mean(X_scaled, axis=0))**2, axis=1)
        anomalies = errors > self.threshold
        return anomalies, errors, self.threshold


class GraphNeuralNetwork:
    """
    Graph Neural Network for transaction network analysis
    """
    
    def __init__(self, hidden_dim=64, output_dim=32, num_layers=2):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.model = None
        self.is_trained = False
        
    def build_simplified_gnn(self, num_nodes, num_features):
        """Build a simplified GNN when PyTorch Geometric is not available"""
        # Use a simple feedforward network as approximation
        if HAS_TENSORFLOW:
            model = keras.Sequential([
                layers.Dense(self.hidden_dim, activation='relu', input_shape=(num_features,)),
                layers.Dense(self.hidden_dim, activation='relu'),
                layers.Dense(self.output_dim, activation='relu'),
                layers.Dense(1, activation='sigmoid')  # Anomaly probability
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        else:
            return None
    
    def create_transaction_graph(self, df):
        """Create graph representation from transaction data"""
        # Create adjacency matrix based on transaction relationships
        unique_addresses = pd.concat([df['from_address'], df['to_address']]).unique()
        address_to_idx = {addr: idx for idx, addr in enumerate(unique_addresses)}
        
        # Create adjacency matrix
        num_nodes = len(unique_addresses)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        
        # Node features (aggregated transaction statistics per address)
        node_features = np.zeros((num_nodes, 6))  # 6 features per node
        
        for addr_idx, address in enumerate(unique_addresses):
            # Outgoing transactions
            out_txs = df[df['from_address'] == address]
            # Incoming transactions
            in_txs = df[df['to_address'] == address]
            
            # Calculate node features
            node_features[addr_idx] = [
                len(out_txs),  # Number of outgoing transactions
                len(in_txs),   # Number of incoming transactions
                float(out_txs['amount'].sum()) if len(out_txs) > 0 else 0.0,  # Total sent
                float(in_txs['amount'].sum()) if len(in_txs) > 0 else 0.0,    # Total received
                float(out_txs['amount'].mean()) if len(out_txs) > 0 else 0.0, # Average sent
                float(in_txs['amount'].mean()) if len(in_txs) > 0 else 0.0    # Average received
            ]
            
            # Create edges in adjacency matrix
            for _, tx in out_txs.iterrows():
                if tx['to_address'] in address_to_idx:
                    to_idx = address_to_idx[tx['to_address']]
                    adjacency_matrix[addr_idx, to_idx] = 1.0
        
        return adjacency_matrix, node_features, address_to_idx
    
    def fit(self, df, anomaly_labels=None, epochs=50):
        """Train the GNN"""
        adj_matrix, node_features, addr_mapping = self.create_transaction_graph(df)
        
        # Build simplified model
        self.model = self.build_simplified_gnn(len(addr_mapping), node_features.shape[1])
        
        if self.model is None:
            print("⚠️ Neural network libraries not available, using graph-based heuristics")
            self.is_trained = True
            self.adj_matrix = adj_matrix
            self.node_features = node_features
            self.addr_mapping = addr_mapping
            return {"simplified": True}
        
        # If no labels provided, create pseudo-labels based on statistical outliers
        if anomaly_labels is None:
            # Use node degree and transaction volume as anomaly indicators
            degrees = np.sum(adj_matrix, axis=1) + np.sum(adj_matrix, axis=0)
            volumes = node_features[:, 2] + node_features[:, 3]  # Total sent + received
            
            # Identify outliers as potential anomalies
            degree_threshold = np.percentile(degrees, 95)
            volume_threshold = np.percentile(volumes, 95)
            
            anomaly_labels = ((degrees > degree_threshold) | (volumes > volume_threshold)).astype(int)
        
        # Train the model
        history = self.model.fit(
            node_features, anomaly_labels,
            epochs=epochs,
            validation_split=0.2,
            verbose=0
        )
        
        self.is_trained = True
        self.addr_mapping = addr_mapping
        return history
    
    def predict_node_anomalies(self, df):
        """Predict anomalies for nodes in the transaction graph"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        adj_matrix, node_features, addr_mapping = self.create_transaction_graph(df)
        
        if self.model is None:
            # Use simplified heuristic-based approach
            degrees = np.sum(adj_matrix, axis=1) + np.sum(adj_matrix, axis=0)
            volumes = node_features[:, 2] + node_features[:, 3]
            
            # Normalize features
            degree_scores = (degrees - np.mean(degrees)) / (np.std(degrees) + 1e-8)
            volume_scores = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-8)
            
            # Combine scores
            anomaly_scores = (degree_scores + volume_scores) / 2
            threshold = np.percentile(anomaly_scores, 95)
            
            anomalies = anomaly_scores > threshold
            return anomalies, anomaly_scores, addr_mapping
        
        # Use trained neural network
        predictions = self.model.predict(node_features, verbose=0)
        anomaly_scores = predictions.flatten()
        threshold = 0.5  # Standard binary classification threshold
        
        anomalies = anomaly_scores > threshold
        return anomalies, anomaly_scores, addr_mapping


class EnsembleAnomalyDetector:
    """
    Ensemble method combining multiple anomaly detection algorithms
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
    def add_model(self, name, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def fit(self, X, y=None):
        """Train all models in the ensemble"""
        results = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'fit'):
                    if name in ['lstm_autoencoder', 'vae']:
                        results[name] = model.fit(X)
                    else:
                        results[name] = model.fit(X)
                    print(f"✅ {name} trained successfully")
            except Exception as e:
                print(f"⚠️ {name} training failed: {e}")
                results[name] = {"error": str(e)}
        
        self.is_trained = True
        return results
    
    def predict_anomalies(self, X):
        """Predict anomalies using ensemble voting"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained first")
        
        predictions = {}
        scores = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_anomalies'):
                    anomalies, anomaly_scores, _ = model.predict_anomalies(X)
                elif hasattr(model, 'predict'):
                    anomaly_scores = model.decision_function(X)
                    anomalies = anomaly_scores < 0  # Isolation Forest convention
                else:
                    continue
                
                predictions[name] = anomalies
                scores[name] = anomaly_scores
                
            except Exception as e:
                print(f"⚠️ {name} prediction failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Weighted ensemble voting
        ensemble_scores = np.zeros(len(X))
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = self.weights[name]
            # Convert boolean predictions to scores
            if pred.dtype == bool:
                pred_scores = pred.astype(float)
            else:
                pred_scores = pred
            
            ensemble_scores += weight * pred_scores
            total_weight += weight
        
        ensemble_scores /= total_weight
        
        # Determine final anomalies
        threshold = np.percentile(ensemble_scores, 95)
        final_anomalies = ensemble_scores > threshold
        
        return final_anomalies, ensemble_scores, predictions


class OnlineLearningDetector:
    """
    Online learning anomaly detector with concept drift detection
    """
    
    def __init__(self):
        self.drift_detector = None
        self.online_model = None
        self.is_initialized = False
        self.performance_history = []
        
    def initialize(self):
        """Initialize online learning components"""
        if HAS_RIVER:
            # Use River for online learning
            self.online_model = compose.Pipeline(
                preprocessing.StandardScaler(),
                anomaly.HalfSpaceTrees(n_trees=10, height=8)
            )
            self.drift_detector = ADWIN()
            self.is_initialized = True
        else:
            print("⚠️ River not available, using simplified online learning")
            self.is_initialized = True
            self.buffer = []
            self.buffer_size = 1000
    
    def update(self, X, y_true=None):
        """Update the model with new data"""
        if not self.is_initialized:
            self.initialize()
        
        if HAS_RIVER and self.online_model:
            for i, x in enumerate(X):
                x_dict = {f'feature_{j}': float(x[j]) for j in range(len(x))}
                
                # Get prediction before learning
                score = self.online_model.score_one(x_dict)
                
                # Learn from the example
                self.online_model.learn_one(x_dict)
                
                # Update drift detector if we have true labels
                if y_true is not None and i < len(y_true):
                    error = abs(score - y_true[i])
                    self.drift_detector.update(error)
                    
                    if self.drift_detector.drift_detected:
                        print("🚨 Concept drift detected! Adapting model...")
                        return True  # Indicate drift was detected
        else:
            # Simplified approach without River
            for x in X:
                self.buffer.append(x)
                if len(self.buffer) > self.buffer_size:
                    self.buffer.pop(0)
        
        return False  # No drift detected
    
    def predict_anomaly(self, x):
        """Predict anomaly for a single instance"""
        if not self.is_initialized:
            self.initialize()
        
        if HAS_RIVER and self.online_model:
            x_dict = {f'feature_{j}': float(x[j]) for j in range(len(x))}
            return self.online_model.score_one(x_dict)
        else:
            # Simplified prediction
            if not self.buffer:
                return 0.5
            
            buffer_array = np.array(self.buffer)
            distances = np.linalg.norm(buffer_array - x, axis=1)
            return np.percentile(distances, 95)


class FeedbackLearningSystem:
    """
    System for incorporating user feedback to improve model accuracy
    """
    
    def __init__(self):
        self.feedback_data = []
        self.model_performance = {}
        self.confidence_scores = {}
        
    def collect_feedback(self, transaction_id, prediction, user_feedback, confidence=None):
        """Collect user feedback on predictions"""
        feedback_entry = {
            'transaction_id': transaction_id,
            'prediction': prediction,
            'user_feedback': user_feedback,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'correct': prediction == user_feedback
        }
        
        self.feedback_data.append(feedback_entry)
        
        return feedback_entry
    
    def calculate_model_performance(self):
        """Calculate model performance based on user feedback"""
        if not self.feedback_data:
            return {}
        
        df_feedback = pd.DataFrame(self.feedback_data)
        
        performance = {
            'accuracy': df_feedback['correct'].mean(),
            'total_feedback': len(df_feedback),
            'true_positives': len(df_feedback[(df_feedback['prediction'] == True) & (df_feedback['user_feedback'] == True)]),
            'false_positives': len(df_feedback[(df_feedback['prediction'] == True) & (df_feedback['user_feedback'] == False)]),
            'true_negatives': len(df_feedback[(df_feedback['prediction'] == False) & (df_feedback['user_feedback'] == False)]),
            'false_negatives': len(df_feedback[(df_feedback['prediction'] == False) & (df_feedback['user_feedback'] == True)])
        }
        
        # Calculate precision, recall, F1
        if performance['true_positives'] + performance['false_positives'] > 0:
            performance['precision'] = performance['true_positives'] / (performance['true_positives'] + performance['false_positives'])
        else:
            performance['precision'] = 0
            
        if performance['true_positives'] + performance['false_negatives'] > 0:
            performance['recall'] = performance['true_positives'] / (performance['true_positives'] + performance['false_negatives'])
        else:
            performance['recall'] = 0
            
        if performance['precision'] + performance['recall'] > 0:
            performance['f1_score'] = 2 * (performance['precision'] * performance['recall']) / (performance['precision'] + performance['recall'])
        else:
            performance['f1_score'] = 0
        
        self.model_performance = performance
        return performance
    
    def get_training_data_from_feedback(self):
        """Extract training data from user feedback"""
        if not self.feedback_data:
            return None, None
        
        df_feedback = pd.DataFrame(self.feedback_data)
        
        # Extract features (would need to be customized based on your feature set)
        X = []  # Features would be extracted from transaction data
        y = df_feedback['user_feedback'].values
        
        return X, y
    
    def suggest_model_updates(self):
        """Suggest model updates based on feedback patterns"""
        if not self.feedback_data:
            return []
        
        performance = self.calculate_model_performance()
        suggestions = []
        
        if performance['accuracy'] < 0.8:
            suggestions.append("Model accuracy is below 80%. Consider retraining with feedback data.")
        
        if performance['false_positives'] > performance['true_positives']:
            suggestions.append("High false positive rate. Consider adjusting anomaly threshold.")
        
        if performance['false_negatives'] > performance['true_positives']:
            suggestions.append("High false negative rate. Model may be too conservative.")
        
        if len(self.feedback_data) > 100:
            suggestions.append("Sufficient feedback collected for model retraining.")
        
        return suggestions


# Factory class for creating and managing advanced ML models
class AdvancedMLModelFactory:
    """Factory for creating and managing advanced ML models"""
    
    @staticmethod
    def create_lstm_autoencoder(sequence_length=10, n_features=8):
        """Create LSTM Autoencoder model"""
        return LSTMAutoencoder(sequence_length=sequence_length, n_features=n_features)
    
    @staticmethod
    def create_vae(input_dim=8, latent_dim=4):
        """Create Variational Autoencoder model"""
        return VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    
    @staticmethod
    def create_gnn(hidden_dim=64):
        """Create Graph Neural Network model"""
        return GraphNeuralNetwork(hidden_dim=hidden_dim)
    
    @staticmethod
    def create_ensemble_detector():
        """Create ensemble anomaly detector with multiple models"""
        ensemble = EnsembleAnomalyDetector()
        
        # Add traditional models
        ensemble.add_model('isolation_forest', IsolationForest(contamination=0.1), weight=1.0)
        
        # Add advanced models
        ensemble.add_model('lstm_autoencoder', AdvancedMLModelFactory.create_lstm_autoencoder(), weight=1.5)
        ensemble.add_model('vae', AdvancedMLModelFactory.create_vae(), weight=1.5)
        
        return ensemble
    
    @staticmethod
    def create_online_learning_system():
        """Create online learning system"""
        return OnlineLearningDetector()
    
    @staticmethod
    def create_feedback_system():
        """Create feedback learning system"""
        return FeedbackLearningSystem()


# Export main classes and factory
__all__ = [
    'LSTMAutoencoder',
    'VariationalAutoencoder', 
    'GraphNeuralNetwork',
    'EnsembleAnomalyDetector',
    'OnlineLearningDetector',
    'FeedbackLearningSystem',
    'AdvancedMLModelFactory'
]