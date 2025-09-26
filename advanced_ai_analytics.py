import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import openai
from openai import OpenAI
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class AdvancedAnalytics:
    """Advanced AI analytics for blockchain and financial transaction analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.predictive_model = None
        self.clustering_model = None
        self.anomaly_model = None
        
    def multimodal_analysis(self, df, risk_data=None, network_metrics=None):
        """
        Perform comprehensive multimodal analysis combining multiple AI approaches
        
        Args:
            df: Transaction dataframe
            risk_data: Risk assessment data
            network_metrics: Network analysis metrics
            
        Returns:
            Dictionary containing multimodal analysis results
        """
        results = {
            'transaction_clustering': self._perform_transaction_clustering(df),
            'behavioral_patterns': self._analyze_behavioral_patterns(df),
            'risk_correlation': self._analyze_risk_correlations(df, risk_data),
            'network_insights': self._analyze_network_patterns(network_metrics),
            'temporal_patterns': self._analyze_temporal_patterns(df),
            'value_distribution': self._analyze_value_distribution(df),
            'ai_insights': self._generate_ai_insights(df, risk_data, network_metrics)
        }
        
        return results
    
    def predictive_analysis(self, df, prediction_horizon=30):
        """
        Perform predictive analysis to forecast transaction trends and risks
        
        Args:
            df: Transaction dataframe
            prediction_horizon: Number of days to predict ahead
            
        Returns:
            Dictionary containing predictive analysis results
        """
        results = {
            'volume_forecast': self._predict_transaction_volume(df, prediction_horizon),
            'value_forecast': self._predict_transaction_values(df, prediction_horizon),
            'risk_forecast': self._predict_risk_trends(df, prediction_horizon),
            'anomaly_likelihood': self._predict_anomaly_likelihood(df, prediction_horizon),
            'pattern_evolution': self._predict_pattern_evolution(df, prediction_horizon),
            'recommendations': self._generate_predictive_recommendations(df)
        }
        
        return results
    
    def _perform_transaction_clustering(self, df):
        """Cluster transactions based on multiple features"""
        try:
            # Create features for clustering
            features = []
            
            if 'value' in df.columns:
                features.append(np.log1p(df['value']))  # Log-transformed values
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                features.append(df['timestamp'].dt.hour)  # Hour of day
                features.append(df['timestamp'].dt.dayofweek)  # Day of week
            
            # Address frequency features
            if 'from_address' in df.columns:
                from_counts = df['from_address'].value_counts()
                features.append(df['from_address'].map(from_counts))
                
            if 'to_address' in df.columns:
                to_counts = df['to_address'].value_counts()
                features.append(df['to_address'].map(to_counts))
            
            if not features:
                return {'error': 'Insufficient features for clustering'}
            
            # Combine features
            feature_matrix = np.column_stack(features)
            feature_matrix = self.scaler.fit_transform(feature_matrix)
            
            # Perform DBSCAN clustering
            self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
            clusters = self.clustering_model.fit_predict(feature_matrix)
            
            # Analyze clusters
            unique_clusters = np.unique(clusters)
            cluster_analysis = {}
            
            for cluster_id in unique_clusters:
                mask = clusters == cluster_id
                cluster_data = df[mask]
                
                if cluster_id == -1:
                    cluster_name = "Outliers"
                else:
                    cluster_name = f"Cluster_{cluster_id}"
                
                cluster_analysis[cluster_name] = {
                    'size': int(mask.sum()),
                    'avg_value': float(cluster_data['value'].mean()) if 'value' in cluster_data else 0,
                    'pattern_description': self._describe_cluster_pattern(cluster_data)
                }
            
            return {
                'clusters': cluster_analysis,
                'total_clusters': len(unique_clusters) - (1 if -1 in unique_clusters else 0),
                'outlier_percentage': float((clusters == -1).mean() * 100)
            }
            
        except Exception as e:
            return {'error': f'Clustering analysis failed: {str(e)}'}
    
    def _analyze_behavioral_patterns(self, df):
        """Analyze behavioral patterns in transactions"""
        try:
            patterns = {}
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Time-based patterns
                patterns['hourly_activity'] = df.groupby(df['timestamp'].dt.hour)['value'].count().to_dict()
                patterns['daily_activity'] = df.groupby(df['timestamp'].dt.dayofweek)['value'].count().to_dict()
                
                # Transaction frequency patterns
                patterns['peak_hour'] = int(df['timestamp'].dt.hour.mode().iloc[0])
                patterns['peak_day'] = int(df['timestamp'].dt.dayofweek.mode().iloc[0])
            
            if 'value' in df.columns:
                # Value patterns
                patterns['avg_transaction_value'] = float(df['value'].mean())
                patterns['value_volatility'] = float(df['value'].std())
                patterns['large_transaction_threshold'] = float(df['value'].quantile(0.95))
                
            # Address patterns
            if 'from_address' in df.columns:
                patterns['unique_senders'] = int(df['from_address'].nunique())
                patterns['most_active_sender'] = df['from_address'].value_counts().index[0]
                
            if 'to_address' in df.columns:
                patterns['unique_receivers'] = int(df['to_address'].nunique())
                patterns['most_popular_receiver'] = df['to_address'].value_counts().index[0]
            
            return patterns
            
        except Exception as e:
            return {'error': f'Behavioral analysis failed: {str(e)}'}
    
    def _analyze_risk_correlations(self, df, risk_data):
        """Analyze correlations between transaction features and risk"""
        try:
            if risk_data is None or risk_data.empty:
                return {'error': 'No risk data available for correlation analysis'}
            
            correlations = {}
            
            # Merge transaction and risk data
            if 'value' in df.columns and 'risk_score' in risk_data.columns:
                # Value vs Risk correlation
                correlation_value = np.corrcoef(df['value'][:len(risk_data)], risk_data['risk_score'])[0, 1]
                correlations['value_risk_correlation'] = float(correlation_value)
            
            # Time-based risk patterns
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df_with_risk = df[:len(risk_data)].copy()
                df_with_risk['risk_score'] = risk_data['risk_score'].values
                
                hourly_risk = df_with_risk.groupby(df_with_risk['timestamp'].dt.hour)['risk_score'].mean()
                correlations['high_risk_hours'] = hourly_risk.nlargest(3).to_dict()
                
                daily_risk = df_with_risk.groupby(df_with_risk['timestamp'].dt.dayofweek)['risk_score'].mean()
                correlations['high_risk_days'] = daily_risk.nlargest(3).to_dict()
            
            return correlations
            
        except Exception as e:
            return {'error': f'Risk correlation analysis failed: {str(e)}'}
    
    def _analyze_network_patterns(self, network_metrics):
        """Analyze network-level patterns and insights"""
        try:
            if not network_metrics:
                return {'error': 'No network metrics available'}
            
            insights = {}
            
            if 'total_nodes' in network_metrics:
                insights['network_size'] = network_metrics['total_nodes']
                insights['network_density'] = network_metrics.get('total_edges', 0) / max(network_metrics['total_nodes'], 1)
            
            if 'avg_degree' in network_metrics:
                insights['connectivity_level'] = 'High' if network_metrics['avg_degree'] > 5 else 'Moderate' if network_metrics['avg_degree'] > 2 else 'Low'
            
            if 'clustering' in network_metrics:
                insights['clustering_coefficient'] = network_metrics['clustering']
                insights['network_structure'] = 'Highly clustered' if network_metrics['clustering'] > 0.3 else 'Loosely connected'
            
            if 'top_addresses' in network_metrics:
                insights['hub_nodes'] = len([addr for addr, degree in network_metrics['top_addresses'] if degree > 10])
            
            return insights
            
        except Exception as e:
            return {'error': f'Network analysis failed: {str(e)}'}
    
    def _analyze_temporal_patterns(self, df):
        """Analyze temporal patterns in transactions"""
        try:
            if 'timestamp' not in df.columns:
                return {'error': 'No timestamp data available'}
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            patterns = {}
            
            # Transaction frequency over time
            daily_counts = df.set_index('timestamp').resample('D').size()
            patterns['avg_daily_transactions'] = float(daily_counts.mean())
            patterns['peak_transaction_day'] = daily_counts.idxmax().strftime('%Y-%m-%d')
            
            # Value trends over time
            if 'value' in df.columns:
                daily_values = df.set_index('timestamp').resample('D')['value'].sum()
                patterns['avg_daily_value'] = float(daily_values.mean())
                patterns['trend_direction'] = 'Increasing' if daily_values.iloc[-1] > daily_values.iloc[0] else 'Decreasing'
            
            return patterns
            
        except Exception as e:
            return {'error': f'Temporal analysis failed: {str(e)}'}
    
    def _analyze_value_distribution(self, df):
        """Analyze the distribution of transaction values"""
        try:
            if 'value' not in df.columns:
                return {'error': 'No value data available'}
            
            distribution = {}
            
            # Basic statistics
            distribution['mean'] = float(df['value'].mean())
            distribution['median'] = float(df['value'].median())
            distribution['std'] = float(df['value'].std())
            distribution['skewness'] = float(df['value'].skew())
            
            # Percentiles
            distribution['percentiles'] = {
                '25th': float(df['value'].quantile(0.25)),
                '75th': float(df['value'].quantile(0.75)),
                '90th': float(df['value'].quantile(0.90)),
                '95th': float(df['value'].quantile(0.95)),
                '99th': float(df['value'].quantile(0.99))
            }
            
            # Value categories
            total_transactions = len(df)
            distribution['value_categories'] = {
                'micro_transactions': int((df['value'] < distribution['percentiles']['25th']).sum()),
                'small_transactions': int(((df['value'] >= distribution['percentiles']['25th']) & 
                                         (df['value'] < distribution['percentiles']['75th'])).sum()),
                'large_transactions': int((df['value'] >= distribution['percentiles']['75th']).sum()),
                'whale_transactions': int((df['value'] >= distribution['percentiles']['95th']).sum())
            }
            
            return distribution
            
        except Exception as e:
            return {'error': f'Value distribution analysis failed: {str(e)}'}
    
    def _generate_ai_insights(self, df, risk_data, network_metrics):
        """Generate AI-powered insights using OpenAI"""
        try:
            # Prepare context for AI analysis
            context = self._prepare_analysis_context(df, risk_data, network_metrics)
            
            # The newest OpenAI model is "gpt-5" which was released August 7, 2025.
            # do not change this unless explicitly requested by the user
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are an expert financial transaction analyst. Provide concise, actionable insights based on the transaction data analysis."},
                    {"role": "user", "content": f"Based on this transaction analysis data, provide key insights and recommendations:\n\n{context}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"AI insights generation failed: {str(e)}"
    
    def _predict_transaction_volume(self, df, horizon):
        """Predict future transaction volume"""
        try:
            if 'timestamp' not in df.columns:
                return {'error': 'No timestamp data for prediction'}
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            daily_counts = df.set_index('timestamp').resample('D').size()
            
            # Simple trend-based prediction
            recent_trend = daily_counts.tail(7).mean()
            overall_trend = daily_counts.mean()
            
            # Predict based on recent trend
            prediction = recent_trend * (1 + (recent_trend - overall_trend) / overall_trend * 0.1)
            
            return {
                'predicted_daily_volume': float(prediction),
                'confidence': 'Medium' if abs(recent_trend - overall_trend) / overall_trend < 0.2 else 'Low',
                'trend': 'Increasing' if recent_trend > overall_trend else 'Decreasing'
            }
            
        except Exception as e:
            return {'error': f'Volume prediction failed: {str(e)}'}
    
    def _predict_transaction_values(self, df, horizon):
        """Predict future transaction values"""
        try:
            if 'value' not in df.columns or 'timestamp' not in df.columns:
                return {'error': 'Insufficient data for value prediction'}
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            daily_values = df.set_index('timestamp').resample('D')['value'].mean()
            
            # Simple moving average prediction
            recent_avg = daily_values.tail(7).mean()
            overall_avg = daily_values.mean()
            
            predicted_value = recent_avg * (1 + (recent_avg - overall_avg) / overall_avg * 0.05)
            
            return {
                'predicted_avg_value': float(predicted_value),
                'value_trend': 'Increasing' if recent_avg > overall_avg else 'Decreasing',
                'volatility_forecast': 'High' if daily_values.std() > overall_avg * 0.5 else 'Moderate'
            }
            
        except Exception as e:
            return {'error': f'Value prediction failed: {str(e)}'}
    
    def _predict_risk_trends(self, df, horizon):
        """Predict future risk trends"""
        try:
            # Simplified risk trend prediction based on transaction patterns
            if 'value' not in df.columns:
                return {'error': 'Insufficient data for risk prediction'}
            
            # Calculate risk indicators
            high_value_ratio = (df['value'] > df['value'].quantile(0.9)).mean()
            unusual_pattern_score = high_value_ratio * 0.7
            
            return {
                'risk_level_forecast': 'High' if unusual_pattern_score > 0.3 else 'Moderate' if unusual_pattern_score > 0.1 else 'Low',
                'monitoring_recommendation': 'Increase monitoring' if unusual_pattern_score > 0.2 else 'Standard monitoring',
                'risk_score': float(unusual_pattern_score)
            }
            
        except Exception as e:
            return {'error': f'Risk prediction failed: {str(e)}'}
    
    def _predict_anomaly_likelihood(self, df, horizon):
        """Predict likelihood of future anomalies"""
        try:
            if 'value' not in df.columns:
                return {'error': 'Insufficient data for anomaly prediction'}
            
            # Calculate anomaly indicators
            value_volatility = df['value'].std() / df['value'].mean()
            outlier_ratio = ((df['value'] > df['value'].quantile(0.95)) | 
                           (df['value'] < df['value'].quantile(0.05))).mean()
            
            anomaly_likelihood = (value_volatility + outlier_ratio) / 2
            
            return {
                'anomaly_likelihood': 'High' if anomaly_likelihood > 0.3 else 'Medium' if anomaly_likelihood > 0.15 else 'Low',
                'volatility_score': float(value_volatility),
                'outlier_frequency': float(outlier_ratio),
                'recommendation': 'Enhanced monitoring recommended' if anomaly_likelihood > 0.25 else 'Standard monitoring sufficient'
            }
            
        except Exception as e:
            return {'error': f'Anomaly prediction failed: {str(e)}'}
    
    def _predict_pattern_evolution(self, df, horizon):
        """Predict how transaction patterns might evolve"""
        try:
            evolution = {}
            
            if 'timestamp' in df.columns and 'value' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Analyze recent vs historical patterns
                midpoint = len(df) // 2
                recent_data = df.iloc[midpoint:]
                historical_data = df.iloc[:midpoint]
                
                recent_avg = recent_data['value'].mean()
                historical_avg = historical_data['value'].mean()
                
                evolution['value_pattern_change'] = float((recent_avg - historical_avg) / historical_avg * 100)
                evolution['pattern_stability'] = 'Stable' if abs(evolution['value_pattern_change']) < 10 else 'Changing'
                
                # Transaction frequency evolution
                recent_freq = len(recent_data) / (recent_data['timestamp'].max() - recent_data['timestamp'].min()).days
                historical_freq = len(historical_data) / (historical_data['timestamp'].max() - historical_data['timestamp'].min()).days
                
                evolution['frequency_change'] = float((recent_freq - historical_freq) / historical_freq * 100)
            
            return evolution
            
        except Exception as e:
            return {'error': f'Pattern evolution prediction failed: {str(e)}'}
    
    def _generate_predictive_recommendations(self, df):
        """Generate actionable recommendations based on predictive analysis"""
        try:
            recommendations = []
            
            if 'value' in df.columns:
                high_value_ratio = (df['value'] > df['value'].quantile(0.9)).mean()
                
                if high_value_ratio > 0.2:
                    recommendations.append("High frequency of large transactions detected - consider enhanced monitoring")
                
                value_volatility = df['value'].std() / df['value'].mean()
                if value_volatility > 0.5:
                    recommendations.append("High value volatility - implement dynamic risk thresholds")
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                night_transactions = df[df['timestamp'].dt.hour.isin([0, 1, 2, 3, 4, 5])]['value'].count()
                total_transactions = len(df)
                
                if night_transactions / total_transactions > 0.2:
                    recommendations.append("Significant off-hours activity - review for unusual patterns")
            
            if not recommendations:
                recommendations.append("Transaction patterns appear normal - maintain standard monitoring")
            
            return recommendations
            
        except Exception as e:
            return [f"Recommendation generation failed: {str(e)}"]
    
    def _describe_cluster_pattern(self, cluster_data):
        """Describe the pattern characteristics of a transaction cluster"""
        try:
            if 'value' in cluster_data.columns:
                avg_value = cluster_data['value'].mean()
                if avg_value > cluster_data['value'].quantile(0.8):
                    return "High-value transactions"
                elif avg_value < cluster_data['value'].quantile(0.2):
                    return "Low-value transactions"
                else:
                    return "Medium-value transactions"
            return "Standard transaction pattern"
        except:
            return "Unknown pattern"
    
    def _prepare_analysis_context(self, df, risk_data, network_metrics):
        """Prepare context for AI analysis"""
        context = f"Transaction Dataset Analysis:\n"
        context += f"- Total transactions: {len(df)}\n"
        
        if 'value' in df.columns:
            context += f"- Average transaction value: ${df['value'].mean():.2f}\n"
            context += f"- Value range: ${df['value'].min():.2f} to ${df['value'].max():.2f}\n"
        
        if risk_data is not None and not risk_data.empty:
            context += f"- High risk transactions: {(risk_data['risk_score'] > 0.7).sum()}\n"
        
        if network_metrics:
            context += f"- Network nodes: {network_metrics.get('total_nodes', 'N/A')}\n"
            context += f"- Network connections: {network_metrics.get('total_edges', 'N/A')}\n"
        
        return context