"""
AUSTRAC Compliance Dashboard for QuantumGuard AI
Interactive dashboard for Australian regulatory compliance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from austrac_classifier import (
    AUSTRACClassifier, 
    AUSTRACRiskLevel, 
    AUSTRACTransactionType,
    AUSTRACViolationType,
    create_austrac_compliance_dashboard
)

def create_austrac_dashboard_page():
    """Create the AUSTRAC compliance dashboard page"""
    
    st.header("🇦🇺 AUSTRAC Compliance Dashboard")
    st.markdown("**Australian Transaction Reports and Analysis Centre (AUSTRAC) Regulatory Compliance**")
    
    # Initialize classifier
    classifier = AUSTRACClassifier()
    
    # Compliance overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Compliance Status", "✅ COMPLIANT", delta="Active")
    
    with col2:
        st.metric("Reports Due", "3", delta="-2 from yesterday")
    
    with col3:
        st.metric("High Risk Transactions", "12", delta="+4 today")
    
    with col4:
        st.metric("Risk Score Average", "23.4", delta="-1.2 improvement")
    
    # AUSTRAC reporting requirements section
    st.subheader("📋 AUSTRAC Reporting Requirements")
    
    reporting_info = st.expander("View AUSTRAC Reporting Thresholds & Requirements")
    with reporting_info:
        st.markdown("""
        **Threshold Transaction Reports (TTR)**
        - Cash transactions ≥ AUD $10,000
        - Must be reported within 3 business days
        
        **International Funds Transfer Instructions (IFTI)**
        - International transfers ≥ AUD $1,000
        - Includes details of both sending and receiving parties
        
        **Suspicious Matter Reports (SMR)**
        - Transactions suspected of money laundering or terrorism financing
        - No minimum threshold - based on suspicion
        - Must be reported as soon as practicable
        
        **Significant Cash Transaction Reports (SCTR)**
        - Cash transactions ≥ AUD $10,000
        - Additional enhanced due diligence required
        """)
    
    # Transaction classification section
    st.subheader("🔍 Transaction Classification")
    
    if st.session_state.get('df') is not None:
        df = st.session_state.df
        
        # Sample transaction for demonstration
        if len(df) > 0:
            sample_transaction = {
                "transaction_id": "TX_" + str(df.iloc[0].get('transaction_id', 'DEMO_001')),
                "amount": float(df.iloc[0].get('amount', 15000)),
                "currency": "AUD",
                "from_country": "Australia",
                "to_country": df.iloc[0].get('to_country', 'Singapore'),
                "customer_name": "Demo Customer",
                "customer_id": "CUST_001",
                "verification_status": "Verified",
                "timestamp": datetime.now().isoformat(),
                "high_frequency_flag": False,
                "complexity_score": 3,
                "fraud_indicators": False,
                "tax_haven_flag": False,
                "velocity_flag": False
            }
            
            # Classify the transaction
            classification = classifier.classify_transaction(sample_transaction)
            
            # Display classification results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Transaction Details**")
                st.json({
                    "ID": sample_transaction["transaction_id"],
                    "Amount": f"${sample_transaction['amount']:,.2f} {sample_transaction['currency']}",
                    "Route": f"{sample_transaction['from_country']} → {sample_transaction['to_country']}",
                    "Customer": sample_transaction["customer_name"]
                })
            
            with col2:
                st.markdown("**AUSTRAC Classification**")
                
                # Risk level with color coding
                risk_color = {
                    "Low": "green",
                    "Medium": "orange", 
                    "High": "red",
                    "Very High": "darkred",
                    "Critical": "purple"
                }
                
                risk_level = classification["risk_level"].value
                st.markdown(f"**Risk Level:** :{risk_color.get(risk_level, 'gray')}[{risk_level}]")
                st.markdown(f"**Risk Score:** {classification['risk_score']:.1f}/100")
                st.markdown(f"**Reporting Required:** {'✅ Yes' if classification['reporting_required'] else '❌ No'}")
        
        # Batch classification
        st.subheader("📊 Batch Transaction Analysis")
        
        if st.button("Analyze All Transactions for AUSTRAC Compliance"):
            with st.spinner("Analyzing transactions for AUSTRAC compliance..."):
                # Process a sample of transactions
                sample_size = min(100, len(df))
                classifications = []
                
                progress_bar = st.progress(0)
                
                for i in range(sample_size):
                    row = df.iloc[i]
                    
                    # Create transaction record
                    transaction = {
                        "transaction_id": f"TX_{i+1:06d}",
                        "amount": float(row.get('amount', np.random.uniform(1000, 50000))),
                        "currency": "AUD",
                        "from_country": "Australia",
                        "to_country": np.random.choice(["Singapore", "USA", "UK", "China", "Japan", "New Zealand"]),
                        "customer_name": f"Customer_{i+1}",
                        "customer_id": f"CUST_{i+1:06d}",
                        "verification_status": np.random.choice(["Verified", "Pending", "Incomplete"], p=[0.8, 0.15, 0.05]),
                        "timestamp": datetime.now().isoformat(),
                        "high_frequency_flag": np.random.choice([True, False], p=[0.1, 0.9]),
                        "complexity_score": np.random.randint(1, 10),
                        "fraud_indicators": np.random.choice([True, False], p=[0.05, 0.95]),
                        "tax_haven_flag": np.random.choice([True, False], p=[0.03, 0.97]),
                        "velocity_flag": np.random.choice([True, False], p=[0.08, 0.92])
                    }
                    
                    # Classify transaction
                    classification = classifier.classify_transaction(transaction)
                    classifications.append(classification)
                    
                    progress_bar.progress((i + 1) / sample_size)
                
                # Generate AUSTRAC report
                austrac_report = classifier.generate_austrac_report(classifications)
                
                # Display results
                st.success(f"Analyzed {sample_size} transactions for AUSTRAC compliance")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Reporting Required", 
                        austrac_report["summary"]["reporting_required"],
                        delta=f"{(austrac_report['summary']['reporting_required']/sample_size)*100:.1f}%"
                    )
                
                with col2:
                    critical_count = austrac_report["summary"]["risk_distribution"]["Critical"]
                    st.metric("Critical Risk", critical_count, delta="Immediate action needed" if critical_count > 0 else "None")
                
                with col3:
                    high_count = austrac_report["summary"]["risk_distribution"]["Very High"]
                    st.metric("Very High Risk", high_count, delta="Enhanced DD required" if high_count > 0 else "None")
                
                with col4:
                    smr_count = austrac_report["summary"]["transaction_types"].get("Suspicious Matter Report (SMR)", 0)
                    st.metric("SMR Required", smr_count, delta="File within 24h" if smr_count > 0 else "None")
                
                # Risk distribution chart
                st.subheader("Risk Level Distribution")
                
                risk_data = pd.DataFrame([
                    {"Risk Level": level, "Count": count} 
                    for level, count in austrac_report["summary"]["risk_distribution"].items()
                    if count > 0
                ])
                
                if not risk_data.empty:
                    fig = px.bar(
                        risk_data, 
                        x="Risk Level", 
                        y="Count",
                        color="Risk Level",
                        color_discrete_map={
                            "Low": "green",
                            "Medium": "orange",
                            "High": "red", 
                            "Very High": "darkred",
                            "Critical": "purple"
                        },
                        title="AUSTRAC Risk Level Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Violation types chart
                violation_data = pd.DataFrame([
                    {"Violation Type": vtype.replace("_", " ").title(), "Count": count}
                    for vtype, count in austrac_report["summary"]["violation_types"].items()
                    if count > 0
                ])
                
                if not violation_data.empty:
                    st.subheader("Detected Violation Types")
                    fig = px.pie(
                        violation_data,
                        values="Count",
                        names="Violation Type", 
                        title="AUSTRAC Violation Type Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # High priority transactions
                if austrac_report["high_priority_transactions"]:
                    st.subheader("🚨 High Priority Transactions Requiring Immediate Action")
                    
                    priority_df = pd.DataFrame(austrac_report["high_priority_transactions"])
                    st.dataframe(
                        priority_df[["transaction_id", "risk_level", "risk_score", "violations"]],
                        use_container_width=True
                    )
                
                # Compliance alerts
                if austrac_report["compliance_alerts"]:
                    st.subheader("⚠️ Compliance Alerts")
                    
                    for alert in austrac_report["compliance_alerts"]:
                        severity_color = {
                            "CRITICAL": "error",
                            "HIGH": "warning", 
                            "MEDIUM": "info",
                            "LOW": "success"
                        }
                        
                        if alert['severity'] == "CRITICAL":
                            st.error(f"**{alert['alert_type']}:** {alert['message']}\n\n**Deadline:** {alert['deadline']}")
                        else:
                            st.warning(f"**{alert['alert_type']}:** {alert['message']}\n\n**Deadline:** {alert['deadline']}")
                
                # Export report
                st.subheader("📄 Export AUSTRAC Report")
                
                if st.button("Generate Downloadable AUSTRAC Report"):
                    report_json = json.dumps(austrac_report, indent=2, default=str)
                    
                    st.download_button(
                        label="Download AUSTRAC Compliance Report (JSON)",
                        data=report_json,
                        file_name=f"austrac_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    else:
        st.info("Upload transaction data to perform AUSTRAC compliance analysis")
        
        # Show sample classification
        st.subheader("Sample AUSTRAC Classification")
        
        sample_transaction = {
            "transaction_id": "TX_SAMPLE_001",
            "amount": 25000.0,
            "currency": "AUD", 
            "from_country": "Australia",
            "to_country": "Singapore",
            "customer_name": "Sample Customer",
            "customer_id": "CUST_SAMPLE_001",
            "verification_status": "Verified",
            "timestamp": datetime.now().isoformat(),
            "high_frequency_flag": False,
            "complexity_score": 4,
            "fraud_indicators": False,
            "tax_haven_flag": False,
            "velocity_flag": False
        }
        
        classification = classifier.classify_transaction(sample_transaction)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sample Transaction**")
            st.json({
                "ID": sample_transaction["transaction_id"],
                "Amount": f"${sample_transaction['amount']:,.2f} {sample_transaction['currency']}",
                "Route": f"{sample_transaction['from_country']} → {sample_transaction['to_country']}",
                "Type": "International Transfer"
            })
        
        with col2:
            st.markdown("**AUSTRAC Classification**")
            st.json({
                "Risk Level": classification["risk_level"].value,
                "Risk Score": f"{classification['risk_score']:.1f}/100",
                "Reporting Required": classification["reporting_required"],
                "Transaction Types": [t.value for t in classification["transaction_types"]],
                "Compliance Flags": classification["compliance_flags"]
            })
    
    # Regulatory guidance section
    st.subheader("📚 AUSTRAC Regulatory Guidance")
    
    guidance_tabs = st.tabs(["Reporting Obligations", "Risk Assessment", "Customer Due Diligence", "Record Keeping"])
    
    with guidance_tabs[0]:
        st.markdown("""
        **Key AUSTRAC Reporting Obligations:**
        
        1. **Threshold Transaction Reports (TTR)**
           - Cash transactions ≥ AUD $10,000
           - Report within 3 business days
           
        2. **International Funds Transfer Instructions (IFTI)**
           - International transfers ≥ AUD $1,000
           - Include originator and beneficiary details
           
        3. **Suspicious Matter Reports (SMR)**
           - Report suspicious transactions regardless of amount
           - File as soon as practicable after forming suspicion
           
        4. **Compliance Reports**
           - Annual compliance reports for reporting entities
           - Document compliance program effectiveness
        """)
    
    with guidance_tabs[1]:
        st.markdown("""
        **Risk Assessment Framework:**
        
        - **Customer Risk:** PEPs, sanctions lists, high-risk jurisdictions
        - **Product Risk:** Cash transactions, international transfers, complex structures
        - **Delivery Channel Risk:** Non-face-to-face, third-party introductions
        - **Geographic Risk:** High-risk countries, sanctions jurisdictions
        
        **Risk Mitigation:**
        - Enhanced due diligence for high-risk customers
        - Ongoing monitoring and transaction screening
        - Regular risk assessment updates
        """)
    
    with guidance_tabs[2]:
        st.markdown("""
        **Customer Due Diligence Requirements:**
        
        1. **Customer Identification**
           - Verify identity using reliable documents
           - Obtain beneficial ownership information
           
        2. **Ongoing Customer Due Diligence**
           - Monitor transactions for unusual patterns
           - Keep customer information current
           
        3. **Enhanced Due Diligence**
           - Higher risk customers require additional verification
           - Source of funds verification
           - Senior management approval
        """)
    
    with guidance_tabs[3]:
        st.markdown("""
        **Record Keeping Requirements:**
        
        - **Transaction Records:** Minimum 7 years
        - **Customer Identification:** Minimum 7 years after relationship ends
        - **AUSTRAC Reports:** Minimum 7 years
        - **Compliance Training:** Document all staff training
        
        **Digital Records:**
        - Must be easily accessible and searchable
        - Maintain data integrity and security
        - Regular backup and recovery procedures
        """)

if __name__ == "__main__":
    create_austrac_dashboard_page()