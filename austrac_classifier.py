"""
AUSTRAC-Compliant Classification System for QuantumGuard AI
Australian Transaction Reports and Analysis Centre (AUSTRAC) compliance requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import json

class AUSTRACRiskLevel(Enum):
    """AUSTRAC Risk Level Classifications"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"
    CRITICAL = "Critical"

class AUSTRACTransactionType(Enum):
    """AUSTRAC Transaction Type Classifications"""
    THRESHOLD_TRANSACTION = "Threshold Transaction Report (TTR)"
    SUSPICIOUS_MATTER = "Suspicious Matter Report (SMR)"
    INTERNATIONAL_FUNDS_TRANSFER = "International Funds Transfer Instruction (IFTI)"
    SIGNIFICANT_CASH_TRANSACTION = "Significant Cash Transaction Report (SCTR)"
    CROSS_BORDER_MOVEMENT = "Cross-border Movement of Physical Currency (CBM)"
    COMPLIANCE_ASSESSMENT = "Compliance Assessment Report (CAR)"

class AUSTRACViolationType(Enum):
    """AUSTRAC Violation/Suspicion Categories"""
    MONEY_LAUNDERING = "Money Laundering"
    TERRORISM_FINANCING = "Terrorism Financing"
    TAX_EVASION = "Tax Evasion"
    FRAUD = "Fraud"
    STRUCTURING = "Structuring/Smurfing"
    UNUSUAL_TRANSACTION_PATTERN = "Unusual Transaction Pattern"
    IDENTITY_VERIFICATION_FAILURE = "Identity Verification Failure"
    SANCTIONS_EVASION = "Sanctions Evasion"
    PROCEEDS_OF_CRIME = "Proceeds of Crime"
    BENEFICIAL_OWNERSHIP_CONCEALMENT = "Beneficial Ownership Concealment"

class AUSTRACClassifier:
    """
    AUSTRAC-compliant transaction classification system for Australian financial institutions
    Implements reporting obligations under the Anti-Money Laundering and Counter-Terrorism 
    Financing Act 2006 (AML/CTF Act)
    """
    
    def __init__(self):
        self.name = "AUSTRAC Compliance Classifier"
        
        # AUSTRAC reporting thresholds (in AUD)
        self.thresholds = {
            "cash_transaction": 10000,  # $10,000 AUD for cash transactions
            "international_transfer": 1000,  # $1,000 AUD for international transfers
            "suspicious_amount": 5000,  # Lower threshold for suspicious activity
            "high_risk_amount": 50000,  # $50,000 AUD for enhanced due diligence
            "critical_amount": 100000  # $100,000 AUD for critical monitoring
        }
        
        # High-risk jurisdictions as per AUSTRAC guidance
        self.high_risk_jurisdictions = {
            "FATF_BLACKLIST": ["Iran", "North Korea", "Myanmar"],
            "FATF_GREYLIST": ["Pakistan", "Jordan", "Mali", "Morocco", "Nigeria", 
                            "Philippines", "Senegal", "South Africa", "Syria", 
                            "Turkey", "Uganda", "United Arab Emirates", "Yemen"],
            "SANCTIONS": ["Russia", "Belarus", "Cuba", "Sudan"],
            "TAX_HAVENS": ["Cayman Islands", "British Virgin Islands", "Bermuda", 
                         "Panama", "Monaco", "Andorra"]
        }
        
        # Suspicious transaction patterns
        self.suspicious_patterns = {
            "rapid_succession": {"max_time_minutes": 60, "min_transactions": 5},
            "round_amounts": {"threshold_percentage": 0.8},
            "unusual_times": {"start_hour": 22, "end_hour": 6},
            "cross_border_frequency": {"max_per_day": 10},
            "velocity_threshold": {"max_amount_per_hour": 25000}
        }
    
    def classify_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single transaction according to AUSTRAC requirements
        
        Args:
            transaction: Dictionary containing transaction details
            
        Returns:
            Dictionary containing AUSTRAC classification results
        """
        classification = {
            "transaction_id": transaction.get("transaction_id", ""),
            "timestamp": datetime.now().isoformat(),
            "risk_level": AUSTRACRiskLevel.LOW,
            "transaction_types": [],
            "violation_types": [],
            "reporting_required": False,
            "compliance_flags": [],
            "risk_score": 0.0,
            "recommendations": []
        }
        
        amount = float(transaction.get("amount", 0))
        currency = transaction.get("currency", "AUD")
        from_country = transaction.get("from_country", "Australia")
        to_country = transaction.get("to_country", "Australia")
        
        # Convert to AUD if necessary (simplified conversion)
        if currency != "AUD":
            amount_aud = self._convert_to_aud(amount, currency)
        else:
            amount_aud = amount
        
        # Risk scoring
        risk_score = 0.0
        
        # 1. Amount-based classification
        if amount_aud >= self.thresholds["critical_amount"]:
            risk_score += 40
            classification["risk_level"] = AUSTRACRiskLevel.CRITICAL
            classification["compliance_flags"].append("Critical Amount Threshold")
            classification["reporting_required"] = True
            classification["transaction_types"].append(AUSTRACTransactionType.THRESHOLD_TRANSACTION)
            
        elif amount_aud >= self.thresholds["high_risk_amount"]:
            risk_score += 30
            classification["risk_level"] = AUSTRACRiskLevel.VERY_HIGH
            classification["compliance_flags"].append("High Risk Amount")
            classification["reporting_required"] = True
            
        elif amount_aud >= self.thresholds["cash_transaction"]:
            risk_score += 20
            classification["risk_level"] = AUSTRACRiskLevel.HIGH
            classification["compliance_flags"].append("Cash Transaction Threshold")
            classification["reporting_required"] = True
            classification["transaction_types"].append(AUSTRACTransactionType.SIGNIFICANT_CASH_TRANSACTION)
        
        # 2. International transfer classification
        if from_country != "Australia" or to_country != "Australia":
            if amount_aud >= self.thresholds["international_transfer"]:
                risk_score += 15
                classification["transaction_types"].append(AUSTRACTransactionType.INTERNATIONAL_FUNDS_TRANSFER)
                classification["compliance_flags"].append("International Transfer")
                classification["reporting_required"] = True
        
        # 3. High-risk jurisdiction analysis
        jurisdiction_risk = self._assess_jurisdiction_risk(from_country, to_country)
        if jurisdiction_risk > 0:
            risk_score += jurisdiction_risk
            classification["compliance_flags"].append("High-Risk Jurisdiction")
            if jurisdiction_risk >= 30:
                classification["violation_types"].append(AUSTRACViolationType.SANCTIONS_EVASION)
        
        # 4. Pattern analysis
        pattern_risk = self._analyze_transaction_patterns(transaction)
        risk_score += pattern_risk
        
        # 5. Identity and verification checks
        identity_risk = self._assess_identity_verification(transaction)
        if identity_risk > 0:
            risk_score += identity_risk
            classification["violation_types"].append(AUSTRACViolationType.IDENTITY_VERIFICATION_FAILURE)
        
        # 6. Suspicious activity indicators
        suspicious_indicators = self._detect_suspicious_indicators(transaction)
        if suspicious_indicators:
            risk_score += 25
            classification["violation_types"].extend(suspicious_indicators)
            classification["transaction_types"].append(AUSTRACTransactionType.SUSPICIOUS_MATTER)
            classification["reporting_required"] = True
        
        # Final risk level determination
        classification["risk_score"] = min(risk_score, 100.0)
        
        if risk_score >= 80:
            classification["risk_level"] = AUSTRACRiskLevel.CRITICAL
        elif risk_score >= 60:
            classification["risk_level"] = AUSTRACRiskLevel.VERY_HIGH
        elif risk_score >= 40:
            classification["risk_level"] = AUSTRACRiskLevel.HIGH
        elif risk_score >= 20:
            classification["risk_level"] = AUSTRACRiskLevel.MEDIUM
        else:
            classification["risk_level"] = AUSTRACRiskLevel.LOW
        
        # Generate recommendations
        classification["recommendations"] = self._generate_recommendations(classification)
        
        return classification
    
    def _convert_to_aud(self, amount: float, currency: str) -> float:
        """Convert amount to AUD (simplified conversion)"""
        # In production, use real-time exchange rates
        conversion_rates = {
            "USD": 1.50, "EUR": 1.65, "GBP": 1.85, "JPY": 0.011,
            "CNY": 0.21, "CAD": 1.10, "NZD": 0.92, "CHF": 1.67,
            "BTC": 75000, "ETH": 4500, "XRP": 2.5
        }
        return amount * conversion_rates.get(currency, 1.0)
    
    def _assess_jurisdiction_risk(self, from_country: str, to_country: str) -> float:
        """Assess risk based on countries involved"""
        risk_score = 0.0
        
        countries = [from_country, to_country]
        
        for country in countries:
            if country in self.high_risk_jurisdictions["FATF_BLACKLIST"]:
                risk_score += 40
            elif country in self.high_risk_jurisdictions["SANCTIONS"]:
                risk_score += 35
            elif country in self.high_risk_jurisdictions["FATF_GREYLIST"]:
                risk_score += 25
            elif country in self.high_risk_jurisdictions["TAX_HAVENS"]:
                risk_score += 20
        
        return min(risk_score, 50.0)
    
    def _analyze_transaction_patterns(self, transaction: Dict[str, Any]) -> float:
        """Analyze transaction for suspicious patterns"""
        risk_score = 0.0
        
        # Round amount detection
        amount = float(transaction.get("amount", 0))
        if amount > 0 and amount == round(amount) and amount >= 1000:
            # Check if it's a very round number (multiples of 1000, 5000, 10000)
            if amount % 10000 == 0 or amount % 5000 == 0:
                risk_score += 15
        
        # Time-based analysis
        timestamp = transaction.get("timestamp")
        if timestamp:
            try:
                tx_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour = tx_time.hour
                if (hour >= self.suspicious_patterns["unusual_times"]["start_hour"] or 
                    hour <= self.suspicious_patterns["unusual_times"]["end_hour"]):
                    risk_score += 10
            except:
                pass
        
        # Velocity indicators (would need transaction history)
        velocity_flag = transaction.get("velocity_flag", False)
        if velocity_flag:
            risk_score += 20
        
        return min(risk_score, 30.0)
    
    def _assess_identity_verification(self, transaction: Dict[str, Any]) -> float:
        """Assess identity verification completeness"""
        risk_score = 0.0
        
        # Check for missing or incomplete identity information
        required_fields = ["customer_name", "customer_id", "verification_status"]
        missing_fields = [field for field in required_fields 
                         if not transaction.get(field)]
        
        if missing_fields:
            risk_score += len(missing_fields) * 10
        
        # Check verification status
        verification_status = transaction.get("verification_status", "")
        if verification_status.lower() in ["pending", "failed", "incomplete"]:
            risk_score += 25
        
        return min(risk_score, 40.0)
    
    def _detect_suspicious_indicators(self, transaction: Dict[str, Any]) -> List[AUSTRACViolationType]:
        """Detect specific suspicious activity indicators"""
        indicators = []
        
        amount = float(transaction.get("amount", 0))
        
        # Structuring detection
        if (amount > 0 and amount < self.thresholds["cash_transaction"] and 
            amount > self.thresholds["cash_transaction"] * 0.8):
            indicators.append(AUSTRACViolationType.STRUCTURING)
        
        # High-frequency transactions
        frequency_flag = transaction.get("high_frequency_flag", False)
        if frequency_flag:
            indicators.append(AUSTRACViolationType.UNUSUAL_TRANSACTION_PATTERN)
        
        # Complex transaction structures
        complexity_score = transaction.get("complexity_score", 0)
        if complexity_score > 7:
            indicators.append(AUSTRACViolationType.BENEFICIAL_OWNERSHIP_CONCEALMENT)
        
        # Fraud indicators
        fraud_flag = transaction.get("fraud_indicators", False)
        if fraud_flag:
            indicators.append(AUSTRACViolationType.FRAUD)
        
        # Tax evasion patterns
        tax_haven_involved = transaction.get("tax_haven_flag", False)
        if tax_haven_involved:
            indicators.append(AUSTRACViolationType.TAX_EVASION)
        
        return indicators
    
    def _generate_recommendations(self, classification: Dict[str, Any]) -> List[str]:
        """Generate AUSTRAC compliance recommendations"""
        recommendations = []
        
        risk_level = classification["risk_level"]
        reporting_required = classification["reporting_required"]
        
        if reporting_required:
            recommendations.append("AUSTRAC reporting required within 3 business days")
        
        if risk_level in [AUSTRACRiskLevel.CRITICAL, AUSTRACRiskLevel.VERY_HIGH]:
            recommendations.append("Enhanced due diligence required")
            recommendations.append("Senior management notification recommended")
            recommendations.append("Consider transaction monitoring hold")
        
        if risk_level == AUSTRACRiskLevel.CRITICAL:
            recommendations.append("Immediate compliance team review required")
            recommendations.append("Consider filing Suspicious Matter Report (SMR)")
        
        if AUSTRACViolationType.SANCTIONS_EVASION in classification["violation_types"]:
            recommendations.append("Check against DFAT sanctions lists")
            recommendations.append("Consider transaction block pending review")
        
        if AUSTRACViolationType.IDENTITY_VERIFICATION_FAILURE in classification["violation_types"]:
            recommendations.append("Complete customer identification procedures")
            recommendations.append("Update customer verification documents")
        
        if AUSTRACViolationType.STRUCTURING in classification["violation_types"]:
            recommendations.append("Review customer transaction history for patterns")
            recommendations.append("Consider Suspicious Matter Report for structuring")
        
        return recommendations
    
    def generate_austrac_report(self, classifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive AUSTRAC compliance report"""
        report = {
            "report_id": f"AUSTRAC_RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generation_time": datetime.now().isoformat(),
            "total_transactions": len(classifications),
            "summary": {
                "reporting_required": 0,
                "risk_distribution": {level.value: 0 for level in AUSTRACRiskLevel},
                "violation_types": {vtype.value: 0 for vtype in AUSTRACViolationType},
                "transaction_types": {ttype.value: 0 for ttype in AUSTRACTransactionType}
            },
            "high_priority_transactions": [],
            "compliance_alerts": [],
            "regulatory_deadlines": []
        }
        
        for classification in classifications:
            # Count reporting requirements
            if classification["reporting_required"]:
                report["summary"]["reporting_required"] += 1
            
            # Risk distribution
            risk_level = classification["risk_level"].value
            report["summary"]["risk_distribution"][risk_level] += 1
            
            # Violation types
            for violation in classification["violation_types"]:
                report["summary"]["violation_types"][violation.value] += 1
            
            # Transaction types
            for tx_type in classification["transaction_types"]:
                report["summary"]["transaction_types"][tx_type.value] += 1
            
            # High priority transactions
            if classification["risk_level"] in [AUSTRACRiskLevel.CRITICAL, AUSTRACRiskLevel.VERY_HIGH]:
                report["high_priority_transactions"].append({
                    "transaction_id": classification["transaction_id"],
                    "risk_level": classification["risk_level"].value,
                    "risk_score": classification["risk_score"],
                    "violations": [v.value for v in classification["violation_types"]],
                    "recommendations": classification["recommendations"]
                })
        
        # Generate compliance alerts
        if report["summary"]["reporting_required"] > 0:
            report["compliance_alerts"].append({
                "alert_type": "REPORTING_DEADLINE",
                "severity": "HIGH",
                "message": f"{report['summary']['reporting_required']} transactions require AUSTRAC reporting within 3 business days",
                "deadline": (datetime.now() + timedelta(days=3)).isoformat()
            })
        
        critical_count = report["summary"]["risk_distribution"]["Critical"]
        if critical_count > 0:
            report["compliance_alerts"].append({
                "alert_type": "CRITICAL_TRANSACTIONS",
                "severity": "CRITICAL",
                "message": f"{critical_count} critical risk transactions require immediate review",
                "deadline": (datetime.now() + timedelta(hours=24)).isoformat()
            })
        
        return report

def create_austrac_compliance_dashboard():
    """Create sample AUSTRAC compliance data for demonstration"""
    return {
        "dashboard_title": "AUSTRAC Compliance Dashboard - QuantumGuard AI",
        "compliance_metrics": {
            "reporting_compliance_rate": "98.5%",
            "average_risk_score": "23.4",
            "high_risk_transactions_today": 12,
            "pending_reports": 3,
            "overdue_reports": 0
        },
        "regulatory_status": "COMPLIANT",
        "last_audit_date": "2024-11-15",
        "next_audit_due": "2025-05-15"
    }