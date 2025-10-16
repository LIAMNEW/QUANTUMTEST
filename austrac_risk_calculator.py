"""
AUSTRAC Risk Calculator for QuantumGuard AI Main Dashboard
Calculates user-friendly risk percentage based on AUSTRAC compliance requirements
"""

import pandas as pd
import numpy as np
from austrac_classifier import AUSTRACClassifier, AUSTRACRiskLevel
from typing import Dict, List, Tuple
from datetime import datetime

def calculate_austrac_risk_score(df: pd.DataFrame) -> Dict:
    """
    Calculate overall AUSTRAC compliance risk score for dataset
    
    Returns a user-friendly risk percentage and assessment details
    """
    
    # Import bank transaction risk analyzer for enhanced detection
    try:
        from bank_transaction_risk_analyzer import bank_risk_analyzer
        use_enhanced_detection = True
    except:
        use_enhanced_detection = False
    
    classifier = AUSTRACClassifier()
    
    # Sample transactions for risk assessment
    sample_size = min(100, len(df))
    risk_scores = []
    high_risk_count = 0
    critical_count = 0
    reporting_required_count = 0
    
    # Normalize column names for case-insensitive matching
    df_normalized = df.copy()
    col_mapping = {col: col.lower() for col in df.columns}
    df_normalized.columns = [col.lower() for col in df.columns]
    
    # Process sample transactions
    for i in range(sample_size):
        row = df_normalized.iloc[i]
        
        # Extract actual data from row with case-insensitive column matching
        amount = float(row.get('value', row.get('amount', 1000)))
        
        # Handle both combined datetime and separate Date/Time columns
        if 'timestamp' in row:
            timestamp = row.get('timestamp')
        elif 'date' in row and 'time' in row:
            # Combine separate Date and Time columns
            date_str = str(row.get('date', ''))
            time_str = str(row.get('time', ''))
            if date_str and time_str:
                timestamp = f"{date_str} {time_str}"
            else:
                timestamp = row.get('date', datetime.now().isoformat())
        else:
            timestamp = row.get('date', datetime.now().isoformat())
        
        merchant = row.get('merchant', row.get('description', ''))
        country = row.get('country', row.get('location', 'Australia'))
        
        # Create transaction record for AUSTRAC classification
        transaction = {
            "transaction_id": f"TX_{i+1:06d}",
            "amount": amount,
            "currency": "AUD",
            "from_country": country if country != 'Australia' else "Australia",
            "to_country": country if country != 'Australia' else "Australia",
            "customer_name": f"Customer_{i+1}",
            "customer_id": f"CUST_{i+1:06d}",
            "verification_status": "Verified",
            "timestamp": timestamp,
            "high_frequency_flag": False,
            "complexity_score": 1,
            "fraud_indicators": False,
            "tax_haven_flag": False,
            "velocity_flag": False
        }
        
        # Use enhanced bank transaction analyzer if available
        if use_enhanced_detection:
            bank_analysis = bank_risk_analyzer.analyze_transaction({
                'amount': amount,
                'merchant': merchant,
                'timestamp': timestamp,
                'country': country
            })
            
            # Combine bank analysis with AUSTRAC classification
            transaction["fraud_indicators"] = bank_analysis['risk_score'] >= 40
            transaction["velocity_flag"] = 'HIGH_FREQUENCY' in bank_analysis.get('risk_flags', [])
            transaction["tax_haven_flag"] = 'HIGH_RISK_COUNTRY' in bank_analysis.get('risk_flags', [])
            transaction["high_frequency_flag"] = 'LATE_NIGHT_TRANSACTION' in bank_analysis.get('risk_flags', [])
            
            if 'HIGH_RISK_COUNTRY' in bank_analysis.get('risk_flags', []):
                # Extract country from detail if it's a tax haven
                for detail in bank_analysis.get('risk_details', []):
                    if 'Tax Haven' in detail:
                        transaction["from_country"] = country
                        transaction["to_country"] = country
        
        # Classify transaction
        classification = classifier.classify_transaction(transaction)
        
        # Collect risk data
        risk_scores.append(classification["risk_score"])
        
        if classification["risk_level"] in [AUSTRACRiskLevel.HIGH, AUSTRACRiskLevel.VERY_HIGH]:
            high_risk_count += 1
        
        if classification["risk_level"] == AUSTRACRiskLevel.CRITICAL:
            critical_count += 1
            
        if classification["reporting_required"]:
            reporting_required_count += 1
    
    # Calculate overall risk metrics
    avg_risk_score = np.mean(risk_scores) if risk_scores else 0
    max_risk_score = np.max(risk_scores) if risk_scores else 0
    
    # Calculate percentage-based risk score (0-100%)
    # Enhanced formula to properly reflect AUSTRAC compliance risks
    base_risk = avg_risk_score * 0.8  # Base risk from individual scores
    high_risk_penalty = (high_risk_count / sample_size) * 50  # High-risk transactions are serious
    critical_penalty = (critical_count / sample_size) * 60    # Critical = severe compliance risk
    reporting_penalty = (reporting_required_count / sample_size) * 40  # AUSTRAC reporting is a major red flag
    
    # Special penalty: if >20% require reporting, add extra weight
    if reporting_required_count / sample_size > 0.20:
        reporting_penalty += 15  # Systemic compliance issue
    
    # Combine all factors
    overall_risk_percentage = min(float(base_risk + high_risk_penalty + critical_penalty + reporting_penalty), 100.0)
    
    # Determine risk level and color
    if overall_risk_percentage >= 80:
        risk_level = "Critical"
        risk_color = "red"
        risk_status = "🚨 CRITICAL RISK"
    elif overall_risk_percentage >= 60:
        risk_level = "Very High"
        risk_color = "darkred" 
        risk_status = "⚠️ VERY HIGH RISK"
    elif overall_risk_percentage >= 40:
        risk_level = "High"
        risk_color = "orange"
        risk_status = "🔶 HIGH RISK"
    elif overall_risk_percentage >= 20:
        risk_level = "Medium"
        risk_color = "yellow"
        risk_status = "🟡 MEDIUM RISK"
    else:
        risk_level = "Low"
        risk_color = "green"
        risk_status = "✅ LOW RISK"
    
    # Generate user-friendly summary
    summary_message = f"""
    AUSTRAC Compliance Assessment Summary:
    
    • Overall Risk Level: {risk_level}
    • Transactions Analyzed: {sample_size:,}
    • Reporting Required: {reporting_required_count} transactions
    • High Risk Transactions: {high_risk_count}
    • Critical Risk Transactions: {critical_count}
    
    Regulatory Compliance Status:
    """
    
    if overall_risk_percentage >= 60:
        summary_message += "\n• 🔍 Enhanced due diligence recommended\n• 📋 AUSTRAC reporting likely required\n• 👥 Senior management review suggested"
    elif overall_risk_percentage >= 40:
        summary_message += "\n• 📊 Standard monitoring procedures apply\n• 📋 Some transactions may require reporting\n• ✅ Normal compliance processes sufficient"
    else:
        summary_message += "\n• ✅ Low regulatory risk profile\n• 📊 Standard monitoring sufficient\n• 🛡️ Good compliance standing"
    
    return {
        "risk_percentage": round(overall_risk_percentage, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_status": risk_status,
        "transactions_analyzed": sample_size,
        "high_risk_count": high_risk_count,
        "critical_count": critical_count,
        "reporting_required": reporting_required_count,
        "avg_individual_risk": round(avg_risk_score, 1),
        "max_individual_risk": round(max_risk_score, 1),
        "summary_message": summary_message,
        "compliance_recommendations": generate_compliance_recommendations(
            overall_risk_percentage, 
            reporting_required_count, 
            critical_count
        )
    }

def generate_compliance_recommendations(risk_percentage: float, reporting_count: int, critical_count: int) -> List[str]:
    """Generate specific compliance recommendations based on risk assessment"""
    
    recommendations = []
    
    if risk_percentage >= 80:
        recommendations.extend([
            "🚨 Immediate compliance team review required",
            "📞 Contact AUSTRAC for guidance on high-risk transactions",
            "🛑 Consider implementing transaction holds for critical cases",
            "📋 Prepare Suspicious Matter Reports (SMR) for critical transactions"
        ])
    elif risk_percentage >= 60:
        recommendations.extend([
            "🔍 Enhanced customer due diligence procedures required",
            "📅 Schedule compliance review within 48 hours",
            "📋 Prepare threshold transaction reports as needed",
            "👥 Senior management notification recommended"
        ])
    elif risk_percentage >= 40:
        recommendations.extend([
            "📊 Implement enhanced monitoring for high-risk transactions",
            "📋 Ensure timely filing of required reports",
            "🔍 Review customer verification status",
            "📅 Regular compliance check scheduled"
        ])
    else:
        recommendations.extend([
            "✅ Maintain standard monitoring procedures",
            "📊 Continue regular compliance processes",
            "🛡️ Good compliance profile - no immediate actions required"
        ])
    
    if reporting_count > 0:
        recommendations.append(f"📋 {reporting_count} transactions require AUSTRAC reporting within 3 business days")
    
    if critical_count > 0:
        recommendations.append(f"🚨 {critical_count} critical transactions need immediate attention")
    
    return recommendations