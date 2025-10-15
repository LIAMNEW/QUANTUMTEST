"""
Enhanced Bank Transaction Risk Analyzer for QuantumGuard AI
Specialized detection for credit card and bank transaction fraud patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, time
import re
from collections import Counter

class BankTransactionRiskAnalyzer:
    """
    Specialized risk analyzer for bank and credit card transactions
    Focuses on real-world fraud indicators
    """
    
    def __init__(self):
        self.name = "Bank Transaction Risk Analyzer"
        
        # Suspicious merchant patterns
        self.suspicious_merchant_patterns = [
            r'.*\bincorporated\b.*\b(uber|lyft|cafe|restaurant)\b.*',  # Odd legal structures
            r'\b[A-Z]{2,}\d{4,}\b',  # Random capital letters + numbers
            r'.*\b(offshore|international|global)\b.*\b(holdings|ventures|corp)\b.*',
            r'.*\btransfer\b.*\bservice\b.*',  # Generic transfer services
            r'.*\b(crypto|bitcoin|btc|eth|blockchain)\b.*',  # Crypto-related
            r'.*\b(casino|gaming|bet|poker)\b.*',  # Gambling
            r'.*\b(shell|nominee|proxy)\b.*',  # Shell company indicators
            r'^[A-Z]{1}\s[A-Z]{1}\s.*',  # Single letter initials (e.g., "A B Smith Inc")
            r'.*\d{3,}.*incorporated.*',  # Numbers in incorporated names
            r'.*\b(pty|ltd|llc|inc)\b.*\d+.*',  # Numbers with business entities
        ]
        
        # High-risk countries (expanded AUSTRAC list)
        self.high_risk_countries = [
            # FATF Blacklist
            "Iran", "North Korea", "Myanmar",
            # FATF Greylist
            "Pakistan", "Jordan", "Mali", "Morocco", "Nigeria", 
            "Philippines", "Senegal", "South Africa", "Syria", 
            "Turkey", "Uganda", "United Arab Emirates", "Yemen",
            # Sanctions
            "Russia", "Belarus", "Cuba", "Sudan", "Venezuela",
            # Tax Havens
            "Cayman Islands", "British Virgin Islands", "Bermuda", 
            "Panama", "Monaco", "Andorra", "Bahamas", "Vanuatu",
            "Seychelles", "Belize", "Malta", "Cyprus",
            # Additional Risk
            "Somalia", "Afghanistan", "Iraq", "Libya", "Lebanon"
        ]
        
        # Suspicious transaction times (late night / early morning)
        self.suspicious_hours = {
            'high_risk': list(range(0, 5)),  # 12am-5am
            'medium_risk': list(range(22, 24)),  # 10pm-12am
        }
        
        # Amount thresholds for structuring detection
        self.structuring_thresholds = {
            'cash_threshold': 10000,  # AUD $10,000 AUSTRAC threshold
            'structuring_buffer': 0.2,  # 20% below threshold is suspicious
            'repetitive_count': 3,  # 3+ similar amounts = structuring
            'repetitive_tolerance': 0.1  # 10% variance
        }
        
        # Velocity thresholds
        self.velocity_thresholds = {
            'transactions_per_hour': 5,
            'transactions_per_day': 20,
            'amount_per_hour': 25000,
            'amount_per_day': 100000
        }
        
    def analyze_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single transaction for risk indicators
        
        Args:
            transaction: Dict with keys: amount, merchant/description, timestamp, 
                        country, card_type, location, etc.
        
        Returns:
            Risk analysis with score, flags, and reasons
        """
        risk_score = 0.0
        risk_flags = []
        risk_details = []
        
        # Extract transaction details
        amount = float(transaction.get('amount', 0))
        merchant = str(transaction.get('merchant', transaction.get('description', ''))).lower()
        timestamp = transaction.get('timestamp', transaction.get('date', datetime.now().isoformat()))
        country = transaction.get('country', transaction.get('location', 'Australia'))
        
        # 1. TIME-BASED RISK ANALYSIS
        time_risk = self._analyze_transaction_time(timestamp)
        if time_risk['is_suspicious']:
            risk_score += time_risk['risk_points']
            risk_flags.append(time_risk['flag'])
            risk_details.append(time_risk['detail'])
        
        # 2. MERCHANT NAME ANALYSIS
        merchant_risk = self._analyze_merchant_name(merchant)
        if merchant_risk['is_suspicious']:
            risk_score += merchant_risk['risk_points']
            risk_flags.append(merchant_risk['flag'])
            risk_details.append(merchant_risk['detail'])
        
        # 3. GEOGRAPHIC RISK ANALYSIS
        geo_risk = self._analyze_geographic_risk(country)
        if geo_risk['is_risky']:
            risk_score += geo_risk['risk_points']
            risk_flags.append(geo_risk['flag'])
            risk_details.append(geo_risk['detail'])
        
        # 4. AMOUNT-BASED RISK (Structuring detection)
        amount_risk = self._analyze_transaction_amount(amount)
        if amount_risk['is_suspicious']:
            risk_score += amount_risk['risk_points']
            risk_flags.append(amount_risk['flag'])
            risk_details.append(amount_risk['detail'])
        
        # Calculate final risk level
        risk_level = self._calculate_risk_level(risk_score)
        
        return {
            'transaction_id': transaction.get('transaction_id', 'N/A'),
            'risk_score': min(risk_score, 100.0),
            'risk_level': risk_level,
            'risk_flags': risk_flags,
            'risk_details': risk_details,
            'flagged_for_review': risk_score >= 40.0,
            'austrac_reporting_required': risk_score >= 60.0
        }
    
    def _analyze_transaction_time(self, timestamp: str) -> Dict:
        """Analyze transaction time for suspicious patterns"""
        try:
            if isinstance(timestamp, str):
                dt = pd.to_datetime(timestamp)
            else:
                dt = timestamp
            
            hour = dt.hour
            
            # High risk hours (12am-5am)
            if hour in self.suspicious_hours['high_risk']:
                return {
                    'is_suspicious': True,
                    'risk_points': 25.0,
                    'flag': 'LATE_NIGHT_TRANSACTION',
                    'detail': f'Transaction at {hour:02d}:00 (High-risk hours: 12am-5am)'
                }
            
            # Medium risk hours (10pm-12am)
            elif hour in self.suspicious_hours['medium_risk']:
                return {
                    'is_suspicious': True,
                    'risk_points': 15.0,
                    'flag': 'UNUSUAL_HOURS',
                    'detail': f'Transaction at {hour:02d}:00 (Late evening hours)'
                }
            
        except Exception as e:
            pass
        
        return {'is_suspicious': False, 'risk_points': 0}
    
    def _analyze_merchant_name(self, merchant: str) -> Dict:
        """Analyze merchant name for suspicious patterns"""
        
        # Check against suspicious patterns
        for pattern in self.suspicious_merchant_patterns:
            if re.search(pattern, merchant, re.IGNORECASE):
                # Determine risk level based on pattern type
                if 'crypto' in pattern or 'offshore' in pattern or 'shell' in pattern:
                    risk_points = 30.0
                    severity = 'HIGH'
                elif 'casino' in pattern or 'gaming' in pattern:
                    risk_points = 25.0
                    severity = 'HIGH'
                else:
                    risk_points = 20.0
                    severity = 'MEDIUM'
                
                return {
                    'is_suspicious': True,
                    'risk_points': risk_points,
                    'flag': 'SUSPICIOUS_MERCHANT',
                    'detail': f'Suspicious merchant pattern detected: "{merchant}" (Severity: {severity})'
                }
        
        # Check for excessively long merchant names (often fraudulent)
        if len(merchant) > 50:
            return {
                'is_suspicious': True,
                'risk_points': 15.0,
                'flag': 'UNUSUAL_MERCHANT_NAME',
                'detail': f'Unusually long merchant name ({len(merchant)} characters)'
            }
        
        # Check for repeated characters (e.g., "AAAA CORP")
        char_counts = Counter(merchant.replace(' ', ''))
        max_repeat = max(char_counts.values()) if char_counts else 0
        if max_repeat > 5:
            return {
                'is_suspicious': True,
                'risk_points': 20.0,
                'flag': 'SUSPICIOUS_MERCHANT',
                'detail': f'Merchant name has suspicious character repetition'
            }
        
        return {'is_suspicious': False, 'risk_points': 0}
    
    def _analyze_geographic_risk(self, country: str) -> Dict:
        """Analyze geographic origin for risk"""
        
        # Check if country is in high-risk list
        for risk_country in self.high_risk_countries:
            if risk_country.lower() in country.lower():
                # Determine risk level based on category
                if risk_country in ["Iran", "North Korea", "Myanmar"]:
                    risk_points = 40.0
                    category = "FATF Blacklist"
                elif risk_country in ["Russia", "Belarus", "Cuba", "Sudan", "Venezuela"]:
                    risk_points = 35.0
                    category = "Sanctioned"
                elif "Cayman" in risk_country or "Virgin" in risk_country or risk_country == "Panama":
                    risk_points = 30.0
                    category = "Tax Haven"
                else:
                    risk_points = 25.0
                    category = "High Risk Jurisdiction"
                
                return {
                    'is_risky': True,
                    'risk_points': risk_points,
                    'flag': 'HIGH_RISK_COUNTRY',
                    'detail': f'Transaction from/to {country} ({category})'
                }
        
        # International transactions (non-Australia) get minor risk
        if country.lower() not in ['australia', 'aus', 'au', '']:
            return {
                'is_risky': True,
                'risk_points': 10.0,
                'flag': 'INTERNATIONAL_TRANSACTION',
                'detail': f'International transaction: {country}'
            }
        
        return {'is_risky': False, 'risk_points': 0}
    
    def _analyze_transaction_amount(self, amount: float) -> Dict:
        """Analyze transaction amount for structuring patterns"""
        
        cash_threshold = self.structuring_thresholds['cash_threshold']
        buffer_threshold = cash_threshold * (1 - self.structuring_thresholds['structuring_buffer'])
        
        # Structuring detection - amounts just below reporting threshold
        if buffer_threshold <= amount < cash_threshold:
            proximity = ((cash_threshold - amount) / cash_threshold) * 100
            return {
                'is_suspicious': True,
                'risk_points': 30.0,
                'flag': 'POTENTIAL_STRUCTURING',
                'detail': f'Amount ${amount:,.2f} is {proximity:.1f}% below AUSTRAC threshold (${cash_threshold:,})'
            }
        
        # Very large transactions
        if amount >= 100000:
            return {
                'is_suspicious': True,
                'risk_points': 25.0,
                'flag': 'LARGE_TRANSACTION',
                'detail': f'Large transaction: ${amount:,.2f}'
            }
        
        # Round number detection (often associated with fraud)
        if amount >= 1000 and amount % 1000 == 0:
            return {
                'is_suspicious': True,
                'risk_points': 15.0,
                'flag': 'ROUND_AMOUNT',
                'detail': f'Suspiciously round amount: ${amount:,.2f}'
            }
        
        return {'is_suspicious': False, 'risk_points': 0}
    
    def _calculate_risk_level(self, risk_score: float) -> str:
        """Calculate risk level from score"""
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "VERY_HIGH"
        elif risk_score >= 40:
            return "HIGH"
        elif risk_score >= 20:
            return "MEDIUM"
        else:
            return "LOW"
    
    def analyze_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze entire dataset for risk patterns
        
        Args:
            df: DataFrame with transaction data
        
        Returns:
            DataFrame with risk analysis added
        """
        # Preprocess: Handle separate Date/Time columns
        df = df.copy()
        
        # Check for separate Date and Time columns (case insensitive)
        date_col = None
        time_col = None
        
        for col in df.columns:
            if col.lower() == 'date':
                date_col = col
            elif col.lower() == 'time':
                time_col = col
        
        # If we have separate Date and Time, combine them
        if date_col and time_col:
            df['timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))
        
        # Normalize column names for easier detection
        column_mapping = {}
        for col in df.columns:
            lower_col = col.lower()
            if lower_col in ['amount', 'merchant', 'country', 'description', 'timestamp', 'date', 'time']:
                column_mapping[col] = lower_col
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        results = []
        
        for idx, row in df.iterrows():
            transaction = row.to_dict()
            transaction['transaction_id'] = transaction.get('transaction_id', f'TX_{idx+1:06d}')
            
            risk_analysis = self.analyze_transaction(transaction)
            results.append(risk_analysis)
        
        # Create results dataframe
        risk_df = pd.DataFrame(results)
        
        # Combine with original data
        combined_df = pd.concat([df.reset_index(drop=True), risk_df], axis=1)
        
        return combined_df
    
    def detect_repetitive_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect repetitive small amounts (structuring/smurfing)
        
        Args:
            df: DataFrame with transaction data
        
        Returns:
            List of detected patterns
        """
        patterns = []
        
        if 'amount' not in df.columns:
            return patterns
        
        # Group similar amounts
        amounts = df['amount'].values
        tolerance = self.structuring_thresholds['repetitive_tolerance']
        
        # Find clusters of similar amounts
        amount_groups = {}
        for idx, amount in enumerate(amounts):
            found_group = False
            for group_amount, indices in amount_groups.items():
                if abs(amount - group_amount) / max(group_amount, amount) <= tolerance:
                    indices.append(idx)
                    found_group = True
                    break
            
            if not found_group:
                amount_groups[amount] = [idx]
        
        # Check for repetitive patterns
        min_count = self.structuring_thresholds['repetitive_count']
        
        for amount, indices in amount_groups.items():
            if len(indices) >= min_count:
                patterns.append({
                    'pattern_type': 'REPETITIVE_AMOUNTS',
                    'amount': amount,
                    'occurrences': len(indices),
                    'transaction_indices': indices,
                    'risk_score': min(len(indices) * 10, 50),
                    'detail': f'{len(indices)} transactions of similar amount ${amount:,.2f} detected (possible structuring)'
                })
        
        return patterns
    
    def detect_velocity_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect high-velocity transaction patterns
        
        Args:
            df: DataFrame with transaction data
        
        Returns:
            List of velocity anomalies
        """
        anomalies = []
        
        # Preprocess: Handle separate Date/Time columns
        df = df.copy()
        
        # Check for separate Date and Time columns (case insensitive)
        date_col = None
        time_col = None
        
        for col in df.columns:
            if col.lower() == 'date':
                date_col = col
            elif col.lower() == 'time':
                time_col = col
        
        # If we have separate Date and Time, combine them
        if date_col and time_col and 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))
        
        if 'timestamp' not in df.columns and 'date' not in df.columns:
            return anomalies
        
        # Parse timestamps
        timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        df['dt'] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values('dt')
        
        # Check hourly velocity
        df['hour_group'] = df['dt'].dt.floor('h')
        hourly_counts = df.groupby('hour_group').size()
        hourly_amounts = df.groupby('hour_group')['amount'].sum() if 'amount' in df.columns else None
        
        for hour, count in hourly_counts.items():
            if count >= self.velocity_thresholds['transactions_per_hour']:
                anomalies.append({
                    'anomaly_type': 'HIGH_FREQUENCY',
                    'timeframe': f'{hour}',
                    'transaction_count': count,
                    'risk_score': min(count * 5, 40),
                    'detail': f'{count} transactions in 1 hour (threshold: {self.velocity_thresholds["transactions_per_hour"]})'
                })
        
        if hourly_amounts is not None:
            for hour, amount in hourly_amounts.items():
                if amount >= self.velocity_thresholds['amount_per_hour']:
                    anomalies.append({
                        'anomaly_type': 'HIGH_VALUE_VELOCITY',
                        'timeframe': f'{hour}',
                        'total_amount': amount,
                        'risk_score': 35,
                        'detail': f'${amount:,.2f} total in 1 hour (threshold: ${self.velocity_thresholds["amount_per_hour"]:,})'
                    })
        
        return anomalies
    
    def generate_risk_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive risk report for dataset
        
        Args:
            df: DataFrame with transaction data
        
        Returns:
            Comprehensive risk report
        """
        # Analyze individual transactions
        analyzed_df = self.analyze_dataset(df)
        
        # Detect patterns
        repetitive_patterns = self.detect_repetitive_patterns(df)
        velocity_anomalies = self.detect_velocity_anomalies(df)
        
        # Calculate summary statistics
        total_transactions = len(analyzed_df)
        flagged_count = analyzed_df['flagged_for_review'].sum() if 'flagged_for_review' in analyzed_df.columns else 0
        reporting_required = analyzed_df['austrac_reporting_required'].sum() if 'austrac_reporting_required' in analyzed_df.columns else 0
        
        risk_distribution = analyzed_df['risk_level'].value_counts().to_dict() if 'risk_level' in analyzed_df.columns else {}
        
        # Top risk flags
        all_flags = []
        if 'risk_flags' in analyzed_df.columns:
            for flags in analyzed_df['risk_flags']:
                all_flags.extend(flags)
        flag_counts = Counter(all_flags)
        
        return {
            'summary': {
                'total_transactions': total_transactions,
                'flagged_for_review': int(flagged_count),
                'austrac_reporting_required': int(reporting_required),
                'risk_percentage': (flagged_count / total_transactions * 100) if total_transactions > 0 else 0
            },
            'risk_distribution': risk_distribution,
            'top_risk_flags': dict(flag_counts.most_common(10)),
            'repetitive_patterns': repetitive_patterns,
            'velocity_anomalies': velocity_anomalies,
            'analyzed_data': analyzed_df,
            'high_risk_transactions': analyzed_df[analyzed_df['risk_score'] >= 60].to_dict('records') if 'risk_score' in analyzed_df.columns else []
        }


# Global instance
bank_risk_analyzer = BankTransactionRiskAnalyzer()


# Convenience functions
def analyze_bank_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze bank transaction dataset"""
    return bank_risk_analyzer.analyze_dataset(df)


def generate_bank_risk_report(df: pd.DataFrame) -> Dict:
    """Generate comprehensive risk report"""
    return bank_risk_analyzer.generate_risk_report(df)
