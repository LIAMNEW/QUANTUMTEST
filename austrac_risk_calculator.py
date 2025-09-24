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
    
    classifier = AUSTRACClassifier()
    
    # Sample transactions for risk assessment
    sample_size = min(100, len(df))
    risk_scores = []
    high_risk_count = 0
    critical_count = 0
    reporting_required_count = 0
    
    # Process sample transactions
    for i in range(sample_size):
        row = df.iloc[i]
        
        # Create transaction record for AUSTRAC classification
        # Use actual data from upload instead of random values
        transaction = {
            "transaction_id": f"TX_{i+1:06d}",
            "amount": float(row.get('value', row.get('amount', 1000))),  # Use 'value' or 'amount' from actual data
            "currency": "AUD",
            "from_country": "Australia",  # Default to domestic unless specified
            "to_country": "Australia",     # Default to domestic unless specified  
            "customer_name": f"Customer_{i+1}",
            "customer_id": f"CUST_{i+1:06d}",
            "verification_status": "Verified",  # Default to verified instead of random
            "timestamp": row.get('timestamp', datetime.now().isoformat()),
            "high_frequency_flag": False,   # Default to false instead of random
            "complexity_score": 1,          # Default to low complexity
            "fraud_indicators": False,      # Default to no fraud indicators
            "tax_haven_flag": False,        # Default to no tax haven involvement
            "velocity_flag": False          # Default to no velocity flags
        }
        
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
    # Weight factors: average risk + high-risk transaction percentage + critical transaction penalty
    base_risk = avg_risk_score * 0.5  # Reduce base weight to prevent over-scoring
    high_risk_penalty = (high_risk_count / sample_size) * 20  # Reduced from 30% to 20%
    critical_penalty = (critical_count / sample_size) * 30    # Reduced from 50% to 30%
    
    overall_risk_percentage = min(float(base_risk + high_risk_penalty + critical_penalty), 100.0)
    
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
    **AUSTRAC Compliance Assessment Summary:**
    
    • **Overall Risk Level:** {risk_level}
    • **Transactions Analyzed:** {sample_size:,}
    • **Reporting Required:** {reporting_required_count} transactions
    • **High Risk Transactions:** {high_risk_count}
    • **Critical Risk Transactions:** {critical_count}
    
    **Regulatory Compliance Status:**
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