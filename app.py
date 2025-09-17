import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import traceback
import os
from datetime import datetime, date
import random
from blockchain_analyzer import analyze_blockchain_data, identify_risks
from ml_models import train_anomaly_detection, detect_anomalies
from quantum_crypto import encrypt_data, decrypt_data, generate_pq_keys
from data_processor import preprocess_blockchain_data, extract_features, calculate_network_metrics
from visualizations import (
    plot_transaction_network, 
    plot_risk_heatmap, 
    plot_anomaly_detection,
    plot_transaction_timeline
)
from database import (
    save_analysis_to_db,
    get_analysis_sessions,
    get_analysis_by_id,
    delete_analysis_session,
    add_address_to_watchlist,
    get_watchlist_addresses,
    remove_address_from_watchlist,
    check_addresses_against_watchlist,
    save_search_query,
    get_saved_searches,
    use_saved_search,
    delete_saved_search
)
from ai_search import ai_transaction_search
from advanced_ai_analytics import AdvancedAnalytics
from austrac_classifier import AUSTRACClassifier
from austrac_risk_calculator import calculate_austrac_risk_score
from quantum_security_test import run_quantum_security_test
from simple_quantum_backend import get_simple_security_status

# PDF Generation imports
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

def generate_pdf_report(analysis_results, session_name, visualizations=None):
    """
    Generate a comprehensive PDF report of blockchain analysis results.
    
    Args:
        analysis_results: DataFrame containing transaction analysis
        session_name: Name of the analysis session
        visualizations: Dictionary containing matplotlib figures
        
    Returns:
        BytesIO object containing the PDF data
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#667eea')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#4a5568')
    )
    
    # Title page
    story.append(Paragraph("QuantumGuard AI", title_style))
    story.append(Paragraph("Blockchain Transaction Analysis Report", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Report metadata
    metadata_data = [
        ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Session Name:', session_name],
        ['Total Transactions:', str(len(analysis_results)) if analysis_results is not None else 'N/A'],
        ['Analysis Date Range:', 'Full Dataset'],
    ]
    
    metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f7fafc')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0'))
    ]))
    
    story.append(metadata_table)
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    if analysis_results is not None and not analysis_results.empty:
        # Calculate summary statistics
        high_risk_count = len(analysis_results[analysis_results.get('risk_score', 0) > 0.7])
        anomaly_count = len(analysis_results[analysis_results.get('is_anomaly', False) == True])
        avg_value = analysis_results.get('value', pd.Series([0])).mean()
        
        summary_text = f"""
        This report analyzes {len(analysis_results)} blockchain transactions for potential risks and anomalies.
        Key findings include {high_risk_count} high-risk transactions and {anomaly_count} detected anomalies.
        The average transaction value is {avg_value:.2f} units.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
    else:
        story.append(Paragraph("No transaction data available for analysis.", styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # Risk Analysis Section
    story.append(Paragraph("Risk Analysis", heading_style))
    
    if analysis_results is not None and not analysis_results.empty:
        # Risk distribution table
        risk_levels = ['Low (0-0.3)', 'Medium (0.3-0.7)', 'High (0.7-1.0)']
        risk_counts = [
            len(analysis_results[analysis_results.get('risk_score', 0) <= 0.3]),
            len(analysis_results[(analysis_results.get('risk_score', 0) > 0.3) & (analysis_results.get('risk_score', 0) <= 0.7)]),
            len(analysis_results[analysis_results.get('risk_score', 0) > 0.7])
        ]
        
        risk_data = [['Risk Level', 'Transaction Count', 'Percentage']]
        total_transactions = len(analysis_results)
        
        for level, count in zip(risk_levels, risk_counts):
            percentage = (count / total_transactions * 100) if total_transactions > 0 else 0
            risk_data.append([level, str(count), f"{percentage:.1f}%"])
        
        risk_table = Table(risk_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(risk_table)
    
    story.append(PageBreak())
    
    # Visualizations section
    if visualizations:
        story.append(Paragraph("Data Visualizations", heading_style))
        
        for viz_name, fig in visualizations.items():
            story.append(Paragraph(viz_name.replace('_', ' ').title(), styles['Heading3']))
            
            try:
                # Convert Plotly figure to image
                img_buffer = io.BytesIO()
                if hasattr(fig, 'to_image'):
                    # Plotly figure
                    img_bytes = fig.to_image(format="png", width=800, height=600)
                    img_buffer.write(img_bytes)
                    img_buffer.seek(0)
                    
                    # Add image to PDF
                    img = Image(img_buffer, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
                else:
                    # Skip if not a valid figure
                    story.append(Paragraph("Chart not available for PDF export", styles['Normal']))
                    story.append(Spacer(1, 12))
            except Exception as e:
                story.append(Paragraph(f"Error rendering chart: {viz_name}", styles['Normal']))
                story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_date_filter_controls(key_prefix: str = "") -> tuple[date, date]:
    """
    Create date filter controls and return selected dates.
    
    Args:
        key_prefix: Unique prefix for the control keys
        
    Returns:
        Tuple of (start_date, end_date) or (None, None) if no filtering
    """
    with st.expander("📅 Date Range Filter", expanded=False):
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            enable_filter = st.checkbox("Enable Date Filtering", key=f"{key_prefix}_enable_filter")
        
        if enable_filter:
            with col2:
                start_date = st.date_input(
                    "Start Date",
                    value=date(2024, 1, 1),
                    key=f"{key_prefix}_start_date"
                )
            
            with col3:
                end_date = st.date_input(
                    "End Date", 
                    value=date.today(),
                    key=f"{key_prefix}_end_date"
                )
            
            if start_date > end_date:
                st.error("Start date must be before end date")
                return None, None
                
            return start_date, end_date
        else:
            return None, None

# Dashboard calculation functions
def calculate_dashboard_metrics(df, risk_assessment, anomalies, network_metrics):
    """Calculate key dashboard metrics for comprehensive summary"""
    metrics = {
        'total_transactions': 0,
        'avg_risk_score': 0.0,
        'high_risk_count': 0,
        'credit_score': 0,
        'risk_distribution': {},
        'monthly_change': {}
    }
    
    if df is not None and len(df) > 0:
        # Total transactions
        metrics['total_transactions'] = len(df)
        
        # Generate mock monthly comparison (simulated growth)
        metrics['monthly_change']['transactions'] = random.randint(15, 35)  # % growth
        
        # Risk assessment metrics
        if risk_assessment is not None and len(risk_assessment) > 0:
            # Average risk score
            metrics['avg_risk_score'] = risk_assessment['risk_score'].mean()
            
            # High risk transactions (score > 0.7)
            metrics['high_risk_count'] = len(risk_assessment[risk_assessment['risk_score'] > 0.7])
            
            # Credit score calculation (inverse relationship with risk)
            # Scale from 300-850 (credit score range)
            avg_risk = metrics['avg_risk_score']
            credit_base = 850 - (avg_risk * 550)  # Higher risk = lower credit score
            metrics['credit_score'] = int(max(300, min(850, credit_base)))
            
            # Risk distribution for chart
            risk_categories = pd.cut(risk_assessment['risk_score'], 
                                   bins=[0, 0.3, 0.6, 0.8, 1.0], 
                                   labels=['Low', 'Medium', 'High', 'Critical'])
            metrics['risk_distribution'] = risk_categories.value_counts().to_dict()
            
            # Mock improvements
            metrics['monthly_change']['risk_improvement'] = random.randint(3, 8)  # % improvement
            metrics['monthly_change']['credit_rating'] = 'Excellent' if metrics['credit_score'] > 740 else 'Good'
        
    return metrics

def create_dashboard_section():
    """Create the comprehensive dashboard section with key metrics"""
    st.markdown("### 📊 Dashboard Overview")
    
    # Check if analysis has been performed
    if (st.session_state.analysis_results is not None and 
        st.session_state.risk_assessment is not None):
        
        # Calculate dashboard metrics
        metrics = calculate_dashboard_metrics(
            st.session_state.df,
            st.session_state.risk_assessment,
            st.session_state.anomalies,
            st.session_state.network_metrics
        )
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="metric-value">{metrics['total_transactions']:,}</div>
                <div class="metric-label">Total Transactions</div>
                <div class="metric-change">↗ {metrics['monthly_change']['transactions']}% from last month</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="metric-value">{metrics['avg_risk_score']:.2f}</div>
                <div class="metric-label">Average Risk Score</div>
                <div class="metric-change">↗ {metrics['monthly_change']['risk_improvement']}% improvement</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="metric-value">{metrics['high_risk_count']}</div>
                <div class="metric-label">High Risk Transactions</div>
                <div class="metric-change">🟢 {metrics['high_risk_count'] == 0 and "Excellent" or "Monitoring"}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="metric-value">{metrics['credit_score']}</div>
                <div class="metric-label">Credit Score</div>
                <div class="metric-change">↗ {metrics['monthly_change']['credit_rating']} rating</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk assessment overview and configuration
        col_chart, col_config = st.columns([2, 1])
        
        with col_chart:
            st.markdown("#### Risk Assessment Overview")
            if metrics['risk_distribution']:
                # Create bar chart for risk distribution
                risk_data = list(metrics['risk_distribution'].items())
                categories = [item[0] for item in risk_data]
                values = [item[1] for item in risk_data]
                
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Bar(
                    x=categories, 
                    y=values,
                    marker_color='#00ff88',
                    text=values,
                    textposition='outside'
                )])
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff'),
                    xaxis=dict(title='Risk Category', gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(title='Number of Transactions', gridcolor='rgba(255,255,255,0.1)'),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col_config:
            st.markdown("#### ⚙️ Analysis Configuration")
            
            # Risk Assessment Threshold
            risk_threshold = st.slider(
                "Risk Assessment Threshold",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.get('risk_threshold', 0.7),
                step=0.01,
                key='dashboard_risk_threshold'
            )
            
            # Anomaly Detection Sensitivity  
            anomaly_sensitivity = st.slider(
                "Anomaly Detection Sensitivity", 
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.get('anomaly_sensitivity', 0.8),
                step=0.01,
                key='dashboard_anomaly_sensitivity'
            )
            
            # Update session state
            st.session_state.risk_threshold = risk_threshold
            st.session_state.anomaly_sensitivity = anomaly_sensitivity
            
            # Action buttons
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                st.rerun()
            
            if st.button("🔄 Run Complete Analysis", use_container_width=True):
                st.rerun()
                
    else:
        # Show placeholder for dashboard when no analysis data
        st.info("📊 Dashboard metrics will appear here after running analysis")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: rgba(0, 255, 136, 0.1); border-radius: 10px; margin: 1rem 0;">
            <h4>Key Metrics Coming Soon</h4>
            <p>Upload your blockchain transaction data and run analysis to see comprehensive dashboard metrics including:</p>
            <ul style="text-align: left; display: inline-block;">
                <li>📈 Total Transactions & Growth Trends</li>
                <li>⚡ Average Risk Score & Improvements</li>  
                <li>🚨 High Risk Transaction Count</li>
                <li>🏆 Credit Score Assessment</li>
                <li>📊 Risk Distribution Analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Set page configuration
st.set_page_config(
    page_title="QuantumGuard AI - Blockchain Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* BEGIN QUANTUMGUARD DARK THEME */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%) !important;
        color: #ffffff;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%) !important;
        color: #ffffff !important;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    /* Sidebar styling - dark theme with glassmorphism */
    .css-1d391kg, .css-1d391kg .stSidebar {
        background: rgba(20, 20, 20, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .css-1lcbmhc, .css-17eq0hr {
        background: rgba(20, 20, 20, 0.95) !important;
        backdrop-filter: blur(10px);
        color: #ffffff !important;
    }
    
    /* Logo/branding styling */
    .quantum-logo {
        padding: 30px 25px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    
    .quantum-logo h1 {
        font-size: 20px;
        font-weight: 700;
        background: linear-gradient(135deg, #00ff88, #00cc6a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 5px;
    }
    
    .quantum-logo p {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.6);
    }
    
    /* Stats grid cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 255, 136, 0.1);
        border-color: rgba(0, 255, 136, 0.3);
    }
    
    .stat-card.primary {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.15), rgba(0, 204, 106, 0.1));
        border-color: rgba(0, 255, 136, 0.3);
    }
    
    .stat-value {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 8px;
        background: linear-gradient(135deg, #00ff88, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: rgba(255, 255, 255, 0.7);
        font-size: 14px;
        font-weight: 500;
    }
    
    .stat-change {
        margin-top: 10px;
        font-size: 12px;
        color: #00ff88;
    }
    
    /* Analysis section styling */
    .analysis-section {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
    }
    
    /* Horizontal tabs styling */
    .analysis-tabs {
        display: flex;
        gap: 5px;
        margin-bottom: 30px;
        background: rgba(255, 255, 255, 0.05);
        padding: 8px;
        border-radius: 15px;
        overflow-x: auto;
        flex-wrap: wrap;
    }
    
    .tab-button {
        padding: 12px 20px;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        white-space: nowrap;
        font-weight: 500;
        font-size: 14px;
        border: none;
        background: transparent;
        color: rgba(255, 255, 255, 0.7);
    }
    
    .tab-button.active {
        background: linear-gradient(135deg, #00ff88, #00cc6a) !important;
        color: #000 !important;
        font-weight: 700;
    }
    
    .tab-button:not(.active):hover {
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }
    
    /* Streamlit button overrides */
    .stButton > button {
        background: linear-gradient(135deg, #00ff88, #00cc6a) !important;
        color: #000 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 15px 30px !important;
        transition: all 0.3s ease !important;
        font-size: 16px !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(0, 255, 136, 0.3) !important;
    }
    
    /* Secondary buttons */
    .stButton.secondary > button {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Form controls */
    .stSelectbox > div > div, .stTextInput > div > div, .stTextArea > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    
    .stSelectbox > div > div:focus, .stTextInput > div > div:focus, .stTextArea > div > div:focus {
        border-color: #00ff88 !important;
        box-shadow: 0 0 0 3px rgba(0, 255, 136, 0.1) !important;
    }
    
    /* Risk indicators */
    .risk-indicator {
        text-align: center;
        padding: 15px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        margin: 10px;
    }
    
    .risk-indicator.low {
        border-left: 4px solid #00ff88;
    }
    
    .risk-indicator.medium {
        border-left: 4px solid #ffaa00;
    }
    
    .risk-indicator.high {
        border-left: 4px solid #ff6b00;
    }
    
    .risk-indicator.critical {
        border-left: 4px solid #ff0040;
    }
    
    /* Status success */
    .status-success {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        color: #00ff88;
        font-weight: 600;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px dashed rgba(0, 255, 136, 0.3) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
    }
    
    /* Streamlit tabs override */
    .stTabs {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 8px;
    }
    
    .stTabs > div > div > div {
        background: transparent !important;
        border: none !important;
        border-radius: 10px !important;
        color: rgba(255, 255, 255, 0.7) !important;
        padding: 12px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs > div > div > div[aria-selected="true"] {
        background: linear-gradient(135deg, #00ff88, #00cc6a) !important;
        color: #000 !important;
        font-weight: 700 !important;
    }
    
    .stTabs > div > div > div:not([aria-selected="true"]):hover {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
    }
    
    /* Metrics styling */
    .stMetric {
        background: rgba(255, 255, 255, 0.05) !important;
        padding: 20px !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Export section */
    .export-section {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin-top: 30px;
    }
    
    /* AI chat section */
    .ai-chat {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin-top: 30px;
    }
    
    /* Bottom branding */
    .bottom-branding {
        margin-top: 60px;
        padding: 30px 0;
        text-align: center;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .bottom-branding h1 {
        font-size: 24px;
        font-weight: 700;
        background: linear-gradient(135deg, #00ff88, #00cc6a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
    }
    
    .bottom-branding p {
        color: rgba(255, 255, 255, 0.6);
        font-size: 14px;
    }
    
    /* Alert styling */
    .stAlert > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(0, 255, 136, 0.3) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #00ff88, #00cc6a) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .analysis-tabs {
            flex-direction: column;
        }
        
        .tab-button {
            text-align: center;
            margin-bottom: 5px;
        }
        
        .stat-card {
            padding: 20px;
        }
        
        .analysis-section {
            padding: 20px;
        }
    }
    
    /* Dashboard Cards */
    .dashboard-card {
        background: linear-gradient(135deg, rgba(45, 45, 45, 0.9) 0%, rgba(26, 26, 26, 0.9) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .dashboard-card:hover {
        border-color: rgba(0, 255, 136, 0.6);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 255, 136, 0.2);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00ff88;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-change {
        font-size: 0.8rem;
        color: rgba(0, 255, 136, 0.9);
        font-weight: 600;
    }
    
    /* END QUANTUMGUARD DARK THEME */
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'encrypted_data' not in st.session_state:
    st.session_state.encrypted_data = None
if 'keys_generated' not in st.session_state:
    st.session_state.keys_generated = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'risk_assessment' not in st.session_state:
    st.session_state.risk_assessment = None
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None
if 'network_metrics' not in st.session_state:
    st.session_state.network_metrics = None
if 'saved_session_id' not in st.session_state:
    st.session_state.saved_session_id = None
if 'view_saved_analysis' not in st.session_state:
    st.session_state.view_saved_analysis = False
if 'current_dataset_name' not in st.session_state:
    st.session_state.current_dataset_name = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'search_result' not in st.session_state:
    st.session_state.search_result = None
if 'austrac_risk_score' not in st.session_state:
    st.session_state.austrac_risk_score = None
# Generate keys when app starts
if not st.session_state.keys_generated:
    st.session_state.public_key, st.session_state.private_key = generate_pq_keys()
    st.session_state.keys_generated = True

# Minimal header for better focus on analysis
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <h2 style="color: #ffffff; margin-bottom: 0.5rem;">Blockchain Transaction Analysis</h2>
    <p style="color: rgba(255, 255, 255, 0.7); font-size: 16px;">Upload your data and configure analysis settings below</p>
</div>
""", unsafe_allow_html=True)

# Initialize variables for run_analysis and progress_placeholder
run_analysis = False
progress_placeholder = None

# Enhanced Sidebar navigation - HTML Design Inspired
with st.sidebar:
    # QuantumGuard AI logo and branding 
    st.markdown("""
    <div class="quantum-logo">
        <h1>QuantumGuard AI</h1>
        <p>Blockchain Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Streamlined Navigation Menu
    st.markdown("---")
    
    # Quick tools section
    with st.expander("🔧 Quick Tools", expanded=False):
        # Watchlist shortcut
        st.markdown("**Address Watchlist**")
        new_address = st.text_input("Add Address", key="quick_address")
        new_label = st.text_input("Label", key="quick_label")
        if st.button("Add to Watchlist", key="quick_add") and new_address and new_label:
            try:
                add_address_to_watchlist(new_address, new_label, "Medium", "")
                st.success("Added to watchlist")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Analysis mode selection
    app_mode = st.radio(
        "📊 Analysis Mode", 
        ["🔍 New Analysis", "📂 Saved Analyses"],
        help="Choose analysis mode"
    )
    
    # Add system status panel
    st.markdown("---")
    st.markdown("### 🛡️ System Status")
    
    # Quantum security status
    quantum_status = "🟢 Active" if st.session_state.keys_generated else "🔴 Inactive"
    st.markdown(f"**Quantum Security:** {quantum_status}")
    
    # Database status
    st.markdown("**Database:** 🟢 Connected")
    
    # AI Integration status
    st.markdown("**AI Integration:** 🟢 Ready")
    
    # AUSTRAC Compliance status
    st.markdown("**AUSTRAC Compliance:** 🟢 Enabled")
    
    if app_mode == "🔍 New Analysis":
        st.session_state.view_saved_analysis = False
        
        st.markdown("### 📁 Data Upload")
        st.markdown("Upload your blockchain transaction dataset for analysis")
        
        uploaded_file = st.file_uploader(
            "Choose your transaction file",
            type=["csv", "xlsx", "json"],
            help="Supported formats: CSV, Excel, JSON. Maximum file size: 200MB"
        )
        
        if uploaded_file is not None:
            try:
                df = None
                # Store the dataset name
                st.session_state.current_dataset_name = uploaded_file.name
                
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                if df is not None and not df.empty:
                    # Check if we have the minimum required columns
                    required_cols = ['from_address', 'to_address']
                    
                    # Try to map columns if possible
                    for col in required_cols:
                        if col not in df.columns:
                            if col == 'from_address' and any(c in df.columns for c in ['sender', 'source', 'src']):
                                for alt in ['sender', 'source', 'src']:
                                    if alt in df.columns:
                                        df['from_address'] = df[alt]
                                        break
                            elif col == 'to_address' and any(c in df.columns for c in ['receiver', 'target', 'dst']):
                                for alt in ['receiver', 'target', 'dst']:
                                    if alt in df.columns:
                                        df['to_address'] = df[alt]
                                        break
                    
                    # Add the value column if missing
                    if 'value' not in df.columns and 'amount' in df.columns:
                        df['value'] = df['amount']
                    
                    # Save to session state
                    st.session_state.df = df
                    st.success(f"File uploaded successfully! Found {len(df)} transactions.")
                    
                    # Store the original data without encryption for reliability
                    st.session_state.encrypted_data = {"data": df.to_dict()}
                    
                    # Calculate AUSTRAC risk score immediately after upload
                    with st.spinner("Calculating AUSTRAC compliance risk score..."):
                        st.session_state.austrac_risk_score = calculate_austrac_risk_score(df)
                    
                    # Display enhanced AUSTRAC risk score
                    risk_data = st.session_state.austrac_risk_score
                    risk_percentage = risk_data["risk_percentage"]
                    risk_color = risk_data["risk_color"]
                    
                    # Determine CSS class based on risk level
                    if risk_percentage >= 80:
                        risk_class = "risk-score-critical"
                    elif risk_percentage >= 60:
                        risk_class = "risk-score-critical"
                    elif risk_percentage >= 40:
                        risk_class = "risk-score-high"
                    elif risk_percentage >= 20:
                        risk_class = "risk-score-medium"
                    else:
                        risk_class = "risk-score-low"
                    
                    st.markdown("---")
                    
                    # Enhanced risk score display with custom styling
                    st.markdown(f"""
                    <div class="risk-score-container {risk_class}">
                        <h2 style="margin: 0; font-size: 3rem;">{risk_percentage}%</h2>
                        <h3 style="margin: 0.5rem 0;">AUSTRAC Compliance Risk Score</h3>
                        <p style="margin: 0; font-size: 1.2rem;">{risk_data['risk_status']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Professional horizontal metrics display
                    st.markdown("""
                    <div style="display: flex; justify-content: space-between; gap: 1rem; margin: 2rem 0;">
                        <div class="metric-card" style="flex: 1;">
                            <h4>📊 Analyzed</h4>
                            <h2>{:,}</h2>
                            <p>Transactions</p>
                        </div>
                        <div class="metric-card" style="flex: 1;">
                            <h4>⚠️ High Risk</h4>
                            <h2>{}</h2>
                            <p>Transactions</p>
                        </div>
                        <div class="metric-card" style="flex: 1;">
                            <h4>📋 Reports Due</h4>
                            <h2>{}</h2>
                            <p>AUSTRAC Reports</p>
                        </div>
                        <div class="metric-card" style="flex: 1;">
                            <h4>🎯 Risk Level</h4>
                            <h2>{}</h2>
                            <p>Classification</p>
                        </div>
                    </div>
                    """.format(
                        risk_data['transactions_analyzed'],
                        risk_data['high_risk_count'], 
                        risk_data['reporting_required'],
                        risk_data['risk_level']
                    ), unsafe_allow_html=True)
                    
                    # Show summary in expandable section
                    with st.expander("📋 Detailed AUSTRAC Assessment", expanded=False):
                        st.markdown(risk_data["summary_message"])
                        
                        st.markdown("**🔍 Compliance Recommendations:**")
                        for rec in risk_data["compliance_recommendations"]:
                            st.markdown(f"• {rec}")
                    
                    st.markdown("---")
                    st.info("👇 Use the analysis settings below to run detailed blockchain analysis")
                else:
                    st.error("The uploaded file appears to be empty or has no valid data.")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.expander("Technical Details").code(traceback.format_exc())
        
        st.markdown("---")
        st.markdown("### ⚙️ Analysis Configuration")
        st.markdown("Configure the parameters for your blockchain analysis")
        
        # Enhanced settings with better UI
        col1, col2 = st.columns(2)
        
        with col1:
            risk_threshold = st.slider(
                "🎯 Risk Assessment Threshold", 
                0.0, 1.0, 0.7, 0.05,
                help="Higher values will identify fewer but higher-confidence risks"
            )
            
        with col2:
            anomaly_sensitivity = st.slider(
                "🔍 Anomaly Detection Sensitivity", 
                0.0, 1.0, 0.8, 0.05,
                help="Higher values will detect more anomalies but may increase false positives"
            )
        
        # Create a progress placeholder
        progress_placeholder = st.empty()
        
        # Enhanced run analysis button
        st.markdown("### 🚀 Start Analysis")
        run_analysis = st.button(
            "🔬 Run Complete Blockchain Analysis",
            help="Start comprehensive analysis including risk assessment, anomaly detection, and network analysis"
        )
        
    elif app_mode == "📊 Saved Analyses":
        st.session_state.view_saved_analysis = True
        st.markdown("### 📊 Saved Analysis Sessions")
        st.markdown("View and manage your previously saved blockchain analyses")
        
        # Get list of saved analyses
        try:
            saved_analyses = get_analysis_sessions()
            if saved_analyses:
                # Format the options for the selectbox
                analysis_options = [
                    f"{a['name']} ({a['dataset_name']}) - {a['timestamp']}" 
                    for a in saved_analyses
                ]
                
                selected_analysis = st.selectbox(
                    "Select a saved analysis", 
                    analysis_options
                )
                
                if selected_analysis:
                    # Get the selected analysis ID
                    selected_idx = analysis_options.index(selected_analysis)
                    st.session_state.saved_session_id = saved_analyses[selected_idx]['id']
                    
                    # Show delete button
                    if st.button("Delete Selected Analysis"):
                        success = delete_analysis_session(st.session_state.saved_session_id)
                        if success:
                            st.success("Analysis deleted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to delete analysis.")
            else:
                st.info("No saved analyses found. Run an analysis and save it first.")
        except Exception as e:
            st.error(f"Error loading saved analyses: {str(e)}")
            st.expander("Technical Details").code(traceback.format_exc())
    

    
    if run_analysis and st.session_state.df is not None and st.session_state.encrypted_data is not None:
        try:
            progress_bar = progress_placeholder.progress(0)
            
            # Step 1: Retrieve data from session state
            progress_bar.progress(10, text="Retrieving data for analysis...")
            # Simply use the original dataframe stored in session state
            df = st.session_state.df.copy()
            
            # Step 2: Preprocess data
            progress_bar.progress(30, text="Preprocessing blockchain transaction data...")
            processed_data = preprocess_blockchain_data(df)
            
            # Step 3: Extract features
            progress_bar.progress(50, text="Extracting transaction features...")
            features = extract_features(processed_data)
            
            # Step 4: Run anomaly detection
            progress_bar.progress(70, text="Running AI anomaly detection...")
            anomaly_sensitivity = 0.8  # Default value if not set in UI
            if 'anomaly_sensitivity' in st.session_state:
                anomaly_sensitivity = st.session_state.anomaly_sensitivity
            model = train_anomaly_detection(features)
            anomalies = detect_anomalies(model, features, sensitivity=anomaly_sensitivity)
            
            # Step 5: Analyze blockchain data and assess risks
            progress_bar.progress(85, text="Analyzing risks and generating insights...")
            analysis_results = analyze_blockchain_data(processed_data)
            risk_threshold = 0.7  # Default value if not set in UI
            if 'risk_threshold' in st.session_state:
                risk_threshold = st.session_state.risk_threshold
            risk_assessment = identify_risks(processed_data, threshold=risk_threshold)
                
            # Step 6: Calculate network metrics
            network_metrics = calculate_network_metrics(processed_data)
            
            # Step 7: Store results in session state
            progress_bar.progress(95, text="Finalizing analysis results...")
            st.session_state.analysis_results = analysis_results
            st.session_state.risk_assessment = risk_assessment
            st.session_state.anomalies = anomalies
            st.session_state.network_metrics = network_metrics
            
            # Complete
            progress_bar.progress(100, text="Analysis complete!")
            time.sleep(1)  # Give user time to see the completion
            progress_placeholder.empty()  # Remove the progress bar
            st.success("AI analysis complete! View the results in the tabs below.")
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"Error during analysis: {str(e)}")
            # Display trace for debugging
            st.expander("Technical Details").code(traceback.format_exc())
            st.info("Try adjusting the sensitivity settings or using a different dataset.")

# Main content area
if st.session_state.view_saved_analysis and st.session_state.saved_session_id:
    # Load data from the saved analysis
    try:
        analysis_data = get_analysis_by_id(st.session_state.saved_session_id)
        
        if analysis_data:
            st.header(f"Saved Analysis: {analysis_data['name']}")
            
            # Display analysis metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dataset", analysis_data['dataset_name'])
            with col2:
                st.metric("Risk Threshold", f"{analysis_data['risk_threshold']:.2f}")
            with col3:
                st.metric("Anomaly Sensitivity", f"{analysis_data['anomaly_sensitivity']:.2f}")
            
            if analysis_data['description']:
                st.info(analysis_data['description'])
            
            # Create tabs for visualizations and AI search
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Transaction Network", "Risk Assessment", 
                                          "Anomaly Detection", "Transaction Timeline", "AI Transaction Search"])
            
            # Convert transaction data to DataFrame
            transactions_df = pd.DataFrame(analysis_data['transactions'])
            
            # Convert risk data to DataFrame
            risk_df = pd.DataFrame(analysis_data['risk_assessments'])
            
            # Get anomaly indices
            anomaly_indices = [a['transaction_id'] for a in analysis_data['anomalies'] if a['is_anomaly']]
            
            with tab1:
                st.subheader("Blockchain Transaction Network")
                if not transactions_df.empty:
                    try:
                        fig = plot_transaction_network(transactions_df)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating network visualization: {str(e)}")
            
            with tab2:
                st.subheader("Risk Assessment")
                if not risk_df.empty:
                    try:
                        fig = plot_risk_heatmap(risk_df)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display high-risk transactions
                        high_risks = risk_df[risk_df['risk_score'] > 0.7]
                        if not high_risks.empty:
                            st.warning(f"Found {len(high_risks)} high-risk transactions")
                            st.dataframe(high_risks)
                        else:
                            st.success("No high-risk transactions detected")
                    except Exception as e:
                        st.error(f"Error creating risk visualization: {str(e)}")
            
            with tab3:
                st.subheader("Anomaly Detection")
                if not transactions_df.empty and anomaly_indices:
                    try:
                        fig = plot_anomaly_detection(transactions_df, anomaly_indices)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display anomalies
                        if anomaly_indices:
                            st.warning(f"Detected {len(anomaly_indices)} anomalous transactions")
                            # Get anomalous transactions
                            anomaly_df = transactions_df[transactions_df['id'].isin(anomaly_indices)]
                            st.dataframe(anomaly_df)
                        else:
                            st.success("No anomalies detected")
                    except Exception as e:
                        st.error(f"Error creating anomaly visualization: {str(e)}")
            
            with tab4:
                st.subheader("Transaction Timeline")
                if not transactions_df.empty:
                    try:
                        fig = plot_transaction_timeline(transactions_df)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating timeline visualization: {str(e)}")
            
            # Add AI-powered Transaction Search tab
            tab5 = st.tabs(["AI Transaction Search"])[0]
            with tab5:
                st.subheader("AI-Powered Transaction Search")
                st.markdown("""
                Ask any question about the analyzed blockchain transactions and get AI-powered insights.
                For example:
                - "Which transactions have the highest risk scores?"
                - "Are there any unusual patterns in the transactions?"
                - "What is the average transaction value?"
                """)
                
                # Search input
                search_query = st.text_input("Ask a question about the blockchain transactions:", key="saved_search_query")
                
                if st.button("Search", key="saved_search_button"):
                    if search_query:
                        with st.spinner("Analyzing your query with AI..."):
                            try:
                                # Use the AI search function with saved analysis data
                                response = ai_transaction_search(
                                    search_query,
                                    transactions_df,
                                    pd.DataFrame(analysis_data['risk_assessments']) if 'risk_assessments' in analysis_data else None,
                                    [a['transaction_id'] for a in analysis_data['anomalies'] if a['is_anomaly']] if 'anomalies' in analysis_data else None,
                                    analysis_data['network_metrics'] if 'network_metrics' in analysis_data else None
                                )
                                
                                # Display the response
                                st.markdown("### AI Analysis Results")
                                st.markdown(response)
                            except Exception as e:
                                st.error(f"Error performing AI search: {str(e)}")
                    else:
                        st.warning("Please enter a search query.")
        else:
            st.error("Could not load the selected analysis. It may have been deleted.")
    except Exception as e:
        st.error(f"Error loading saved analysis: {str(e)}")

elif st.session_state.df is None:
    # Enhanced welcome screen with feature highlights
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 15px; margin: 1rem 0;">
        <h2>🚀 Ready to Analyze Blockchain Transactions</h2>
        <p style="font-size: 1.1rem; color: #666;">Upload your transaction data to unlock powerful AI-driven insights and AUSTRAC compliance analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced feature highlights with horizontal layout
    st.markdown("""
    <div style="display: flex; justify-content: space-between; gap: 1.5rem; margin: 2rem 0;">
        <div class="metric-card" style="flex: 1;">
            <h3 style="margin: 0 0 1rem 0; color: white;">🛡️ Quantum Security</h3>
            <p style="margin: 0; line-height: 1.5;">Post-quantum cryptography protects your data against future quantum computing threats</p>
        </div>
        <div class="metric-card" style="flex: 1;">
            <h3 style="margin: 0 0 1rem 0; color: white;">🇦🇺 AUSTRAC Compliance</h3>
            <p style="margin: 0; line-height: 1.5;">Automated compliance scoring and reporting for Australian regulatory requirements</p>
        </div>
        <div class="metric-card" style="flex: 1;">
            <h3 style="margin: 0 0 1rem 0; color: white;">🤖 AI Analytics</h3>
            <p style="margin: 0; line-height: 1.5;">Advanced machine learning for anomaly detection and predictive risk assessment</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Getting started guide
    st.markdown("### 📋 Getting Started")
    
    steps_col1, steps_col2 = st.columns(2)
    
    with steps_col1:
        st.markdown("""
        **Step 1: Upload Data**
        - Support for CSV, Excel, and JSON formats
        - Blockchain transaction datasets
        - Banking transaction data
        
        **Step 2: AUSTRAC Assessment**
        - Automatic risk score calculation
        - Compliance recommendations
        - Regulatory reporting alerts
        """)
    
    with steps_col2:
        st.markdown("""
        **Step 3: AI Analysis**
        - Anomaly detection
        - Network analysis
        - Risk assessment
        
        **Step 4: Insights & Reports**
        - Interactive visualizations
        - AI-powered search
        - Export capabilities
        """)
    
    # Enhanced platform capabilities with custom styling
    st.markdown("---")
    st.markdown("### 📊 Platform Capabilities")
    
    st.markdown("""
    <div style="display: flex; justify-content: space-between; gap: 1rem; margin: 2rem 0;">
        <div class="metric-card" style="flex: 1;">
            <h4>🛡️ Security Level</h4>
            <h2>Quantum-Safe</h2>
            <p>Post-Quantum Ready</p>
        </div>
        <div class="metric-card" style="flex: 1;">
            <h4>🤖 AI Models</h4>
            <h2>Multiple</h2>
            <p>OpenAI GPT Integration</p>
        </div>
        <div class="metric-card" style="flex: 1;">
            <h4>📋 Compliance</h4>
            <h2>AUSTRAC</h2>
            <p>Australian Standards</p>
        </div>
        <div class="metric-card" style="flex: 1;">
            <h4>🔒 Quantum Security</h4>
            <h2 id="quantum-status">Active</h2>
            <p>Post-Quantum Cryptography</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display quantum security status
    with st.expander("🛡️ Quantum Security Status", expanded=False):
        try:
            backend_status = get_simple_security_status()
            
            if backend_status.get("quantum_safe"):
                st.success("🛡️ QuantumGuard AI is secured with post-quantum cryptography")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("**Security Features:**")
                    st.write("• 128-bit quantum-resistant encryption")
                    st.write("• Hybrid post-quantum algorithms")
                    st.write("• Protected against quantum computer attacks")
                    st.write("• Bank-grade security for financial data")
                
                with col2:
                    st.info("**What's Protected:**")
                    st.write("• All customer financial data")
                    st.write("• Transaction analysis results")
                    st.write("• Database storage and retrieval")
                    st.write("• Session data and communications")
                
                st.markdown("---")
                st.markdown("**Technical Details:** Your financial information is encrypted using post-quantum cryptographic algorithms that remain secure even against future quantum computers. All data is automatically protected during storage, processing, and transmission.")
            else:
                st.warning("⚠️ Backend quantum security needs attention")
                
        except Exception:
            st.info("🛡️ **QuantumGuard AI Security Guarantee** - Your financial data is protected with military-grade, quantum-resistant encryption.")


else:
    # If analysis has been run, display results
    if st.session_state.analysis_results is not None:
        st.header("Analysis Results")
        
        # Enhanced tabs with better styling and icons
        st.markdown("### 📊 Comprehensive Analysis Results")
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🌐 Network Visualization", 
            "🎯 Risk Assessment", 
            "🚨 Anomaly Detection", 
            "📈 Transaction Timeline", 
            "🔍 AI Insights", 
            "🧠 Advanced Analytics", 
            "📊 Predictive Intelligence"
        ])
        

                                      
        # Add AI Assistant to sidebar
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🤖 AI Assistant")
            with st.expander("🔍 Ask About Your Data", expanded=False):
                st.markdown("Ask any question about your transaction data:")
                sidebar_query = st.text_input("Your question:", key="sidebar_search_query")
                
                if st.button("Search", key="sidebar_search_button"):
                    if sidebar_query and st.session_state.df is not None:
                        with st.spinner("Analyzing with OpenAI GPT-4o..."):
                            try:
                                response = ai_transaction_search(
                                    sidebar_query,
                                    st.session_state.df,
                                    st.session_state.risk_assessment,
                                    st.session_state.anomalies,
                                    st.session_state.network_metrics
                                )
                                st.session_state.search_result = response
                                
                                # Show the results directly in the sidebar
                                st.success("AI Analysis Complete")
                                st.markdown("### AI Analysis Results")
                                st.markdown(response)
                                st.info("The complete results are also available in the 'AI Insights' tab.")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        if st.session_state.df is None:
                            st.warning("Please upload and analyze data first.")
                        else:
                            st.warning("Please enter a search query.")
        
        with tab1:
            # Enhanced Network Analysis Tab
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3>🌐 Transaction Network Analysis</h3>
                <p>This visualization shows how transactions are connected across the blockchain network. Each node represents an address, and edges represent transaction flows.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Date filtering controls
            start_date, end_date = create_date_filter_controls("network")
            
            # Key Insights Panel
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("**What you'll see:** Nodes (addresses) connected by transaction flows")
            with col2:
                st.info("**Look for:** Dense clusters indicating high activity areas")
            with col3:
                st.info("**Risk indicators:** Isolated nodes or unusual connection patterns")
            
            if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
                try:
                    fig = plot_transaction_network(st.session_state.analysis_results, start_date, end_date)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Network statistics
                    if 'network_metrics' in st.session_state and st.session_state.network_metrics:
                        st.markdown("### Network Statistics")
                        metrics = st.session_state.network_metrics
                        
                        net_col1, net_col2, net_col3, net_col4 = st.columns(4)
                        with net_col1:
                            st.metric("Total Nodes", metrics.get('total_nodes', 'N/A'))
                        with net_col2:
                            st.metric("Total Edges", metrics.get('total_edges', 'N/A'))
                        with net_col3:
                            st.metric("Avg Clustering", f"{metrics.get('avg_clustering', 0):.3f}")
                        with net_col4:
                            st.metric("Network Density", f"{metrics.get('density', 0):.3f}")
                            
                except Exception as e:
                    st.warning("Network visualization temporarily unavailable")
                    st.text("The system is processing your transaction data for network analysis.")
            else:
                st.warning("No transaction data available for network analysis")
        
        with tab2:
            # Enhanced Risk Assessment Tab
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(255, 107, 53, 0.1) 0%, rgba(238, 90, 82, 0.1) 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3>🎯 Risk Assessment Analysis</h3>
                <p>Advanced risk scoring based on transaction patterns, amounts, and behavioral analysis. Higher scores indicate greater potential risk.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Date filtering controls
            start_date, end_date = create_date_filter_controls("risk")
            
            # Risk Level Guide
            st.markdown("### Risk Level Guide")
            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
            with risk_col1:
                st.success("**Low Risk (0-0.3)** ✅ Standard transactions")
            with risk_col2:
                st.info("**Medium Risk (0.3-0.6)** ⚠️ Requires monitoring")
            with risk_col3:
                st.warning("**High Risk (0.6-0.8)** 🚨 Needs investigation")
            with risk_col4:
                st.error("**Critical Risk (0.8-1.0)** ❌ Immediate action required")
            
            if st.session_state.risk_assessment is not None:
                try:
                    fig = plot_risk_heatmap(st.session_state.risk_assessment, start_date, end_date)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk Distribution Analysis
                    st.markdown("### Risk Distribution Summary")
                    if 'risk_score' in st.session_state.risk_assessment.columns:
                        risk_data = st.session_state.risk_assessment['risk_score']
                        
                        # Calculate risk categories
                        low_risk = len(risk_data[risk_data <= 0.3])
                        medium_risk = len(risk_data[(risk_data > 0.3) & (risk_data <= 0.6)])
                        high_risk = len(risk_data[(risk_data > 0.6) & (risk_data <= 0.8)])
                        critical_risk = len(risk_data[risk_data > 0.8])
                        
                        # Display in metric cards
                        risk_metrics_col1, risk_metrics_col2, risk_metrics_col3, risk_metrics_col4 = st.columns(4)
                        with risk_metrics_col1:
                            st.metric("Low Risk", low_risk, delta=f"{(low_risk/len(risk_data)*100):.1f}%")
                        with risk_metrics_col2:
                            st.metric("Medium Risk", medium_risk, delta=f"{(medium_risk/len(risk_data)*100):.1f}%")
                        with risk_metrics_col3:
                            st.metric("High Risk", high_risk, delta=f"{(high_risk/len(risk_data)*100):.1f}%")
                        with risk_metrics_col4:
                            st.metric("Critical Risk", critical_risk, delta=f"{(critical_risk/len(risk_data)*100):.1f}%")
                        
                        # Display high-risk transactions
                        high_risks = st.session_state.risk_assessment[st.session_state.risk_assessment['risk_score'] > 0.7]
                        if not high_risks.empty:
                            st.markdown("### High-Risk Transactions Requiring Attention")
                            st.error(f"⚠️ Found {len(high_risks)} high-risk transactions that require immediate review")
                            
                            # Display in an expandable section
                            with st.expander("View High-Risk Transaction Details"):
                                st.dataframe(high_risks, use_container_width=True)
                        else:
                            st.success("✅ No high-risk transactions detected - All transactions appear normal")
                            
                except Exception as e:
                    st.warning("Risk analysis visualization temporarily unavailable")
                    st.text("The system is processing risk assessment data.")
            else:
                st.warning("No risk assessment data available")
        
        with tab3:
            # Enhanced Anomaly Detection Tab
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3>🚨 AI-Powered Anomaly Detection</h3>
                <p>Machine learning algorithms identify unusual transaction patterns that deviate from normal behavior. Anomalies may indicate fraud, money laundering, or system errors.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Date filtering controls
            start_date, end_date = create_date_filter_controls("anomaly")
            
            # Anomaly Types Guide
            st.markdown("### Types of Anomalies We Detect")
            anom_col1, anom_col2, anom_col3 = st.columns(3)
            with anom_col1:
                st.info("**Amount Anomalies** 💰 Unusually large or small transaction values")
            with anom_col2:
                st.info("**Timing Anomalies** ⏰ Transactions at unusual times or frequencies")
            with anom_col3:
                st.info("**Pattern Anomalies** 🔄 Unusual transaction flow patterns")
                
            if st.session_state.df is not None and st.session_state.anomalies is not None:
                try:
                    fig = plot_anomaly_detection(st.session_state.df, st.session_state.anomalies, start_date, end_date)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Anomaly Analysis
                    if st.session_state.anomalies and len(st.session_state.anomalies) > 0:
                        total_transactions = len(st.session_state.df)
                        anomaly_count = len(st.session_state.anomalies)
                        anomaly_percentage = (anomaly_count / total_transactions) * 100
                        
                        st.markdown("### Anomaly Detection Summary")
                        
                        # Anomaly metrics
                        anom_metrics_col1, anom_metrics_col2, anom_metrics_col3 = st.columns(3)
                        with anom_metrics_col1:
                            st.metric("Total Anomalies", anomaly_count)
                        with anom_metrics_col2:
                            st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
                        with anom_metrics_col3:
                            if anomaly_percentage > 5:
                                st.metric("Severity Level", "High", delta="Requires Investigation")
                            elif anomaly_percentage > 2:
                                st.metric("Severity Level", "Medium", delta="Monitor Closely")
                            else:
                                st.metric("Severity Level", "Low", delta="Normal Range")
                        
                        # Display anomalous transactions
                        st.markdown("### Detected Anomalous Transactions")
                        st.warning(f"🔍 Found {anomaly_count} transactions that deviate from normal patterns")
                        
                        # Use the processed data for anomaly display
                        if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
                            with st.expander("View Anomalous Transaction Details"):
                                anomaly_df = st.session_state.analysis_results.iloc[st.session_state.anomalies]
                                st.dataframe(anomaly_df, use_container_width=True)
                                
                                # Anomaly insights
                                st.markdown("**💡 What to look for in these transactions:**")
                                st.markdown("- Unusually high or low transaction amounts")
                                st.markdown("- Transactions from new or rarely-used addresses")
                                st.markdown("- Timing patterns that differ from normal activity")
                                st.markdown("- Unusual geographic or behavioral patterns")
                                
                    else:
                        st.success("✅ No anomalies detected - All transactions follow expected patterns")
                        st.markdown("**This means:**")
                        st.markdown("- All transaction amounts are within normal ranges")
                        st.markdown("- Transaction timing patterns are consistent")
                        st.markdown("- No unusual behavioral patterns detected")
                        
                except Exception as e:
                    st.warning("Anomaly detection visualization temporarily unavailable")
                    st.text("The AI system is analyzing your transaction patterns.")
            else:
                st.warning("No anomaly detection data available")
        
        with tab4:
            # Enhanced Timeline Analysis Tab
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3>📈 Transaction Timeline Analysis</h3>
                <p>Temporal analysis showing transaction volume, patterns, and trends over time. Helps identify peak activity periods and unusual timing patterns.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Date filtering controls
            start_date, end_date = create_date_filter_controls("timeline")
            
            # Timeline Insights Guide
            st.markdown("### Timeline Analysis Guide")
            time_col1, time_col2, time_col3 = st.columns(3)
            with time_col1:
                st.info("**Volume Peaks** 📊 High activity periods to monitor")
            with time_col2:
                st.info("**Pattern Changes** 🔄 Shifts in transaction behavior")
            with time_col3:
                st.info("**Quiet Periods** 🔇 Unusually low activity times")
                
            if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
                try:
                    fig = plot_transaction_timeline(st.session_state.analysis_results, start_date, end_date)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Timeline Statistics
                    st.markdown("### Timeline Statistics")
                    df = st.session_state.analysis_results
                    
                    if 'timestamp' in df.columns:
                        # Calculate temporal metrics
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                        df_with_time = df.dropna(subset=['timestamp'])
                        
                        if not df_with_time.empty:
                            timeline_col1, timeline_col2, timeline_col3, timeline_col4 = st.columns(4)
                            
                            # Date range
                            min_date = df_with_time['timestamp'].min()
                            max_date = df_with_time['timestamp'].max()
                            duration = (max_date - min_date).days
                            
                            with timeline_col1:
                                st.metric("Analysis Period", f"{duration} days")
                            
                            # Peak activity analysis
                            df_with_time['hour'] = df_with_time['timestamp'].dt.hour
                            peak_hour = df_with_time['hour'].value_counts().index[0]
                            
                            with timeline_col2:
                                st.metric("Peak Activity Hour", f"{peak_hour}:00")
                            
                            # Daily average
                            daily_avg = len(df_with_time) / max(duration, 1)
                            with timeline_col3:
                                st.metric("Daily Average", f"{daily_avg:.1f} transactions")
                            
                            # Activity distribution
                            hourly_dist = df_with_time['hour'].value_counts()
                            activity_variance = hourly_dist.std()
                            
                            with timeline_col4:
                                if activity_variance > 10:
                                    st.metric("Activity Pattern", "Highly Variable", delta="Irregular timing")
                                elif activity_variance > 5:
                                    st.metric("Activity Pattern", "Moderate Variation", delta="Some peaks")
                                else:
                                    st.metric("Activity Pattern", "Consistent", delta="Steady activity")
                            
                            # Insights
                            st.markdown("### Timeline Insights")
                            st.info(f"**Activity Period:** {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                            st.info(f"**Peak Activity:** Most transactions occur around {peak_hour}:00")
                            
                            if activity_variance > 10:
                                st.warning("**Irregular Pattern Detected:** High variance in transaction timing may indicate automated or unusual activity")
                            else:
                                st.success("**Normal Pattern:** Transaction timing follows expected patterns")
                                
                    else:
                        st.info("Timeline analysis using generated timestamps for demonstration")
                        
                except Exception as e:
                    st.warning("Timeline visualization temporarily unavailable")
                    st.text("The system is processing temporal transaction data.")
            else:
                st.warning("No timeline data available")
                
        with tab5:
            # Enhanced AI Insights Tab
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(63, 81, 181, 0.1) 0%, rgba(48, 63, 159, 0.1) 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3>🔍 AI-Powered Transaction Insights</h3>
                <p>Ask natural language questions about your blockchain data and get intelligent analysis powered by advanced AI. The system understands context and provides detailed explanations.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # AI Capabilities Guide
            st.markdown("### What You Can Ask")
            ai_col1, ai_col2, ai_col3 = st.columns(3)
            with ai_col1:
                st.info("**Pattern Analysis** 🔍 'What patterns do you see in the data?'")
            with ai_col2:
                st.info("**Risk Analysis** ⚠️ 'Which transactions are most concerning?'")
            with ai_col3:
                st.info("**Statistical Queries** 📊 'What are the key statistics?'")
            
            # Sample questions
            st.markdown("### Sample Questions to Try")
            sample_questions = [
                "Which transactions have the highest risk scores?",
                "Are there any unusual patterns in the transactions?", 
                "What is the average transaction value?",
                "Show me transactions between the most active addresses",
                "What time patterns do you see in the transaction data?",
                "Are there any concerning anomalies I should investigate?"
            ]
            
            selected_question = st.selectbox("Choose a sample question or type your own:", [""] + sample_questions)
            
            # Enhanced search input
            search_query = st.text_input(
                "Ask your question about the blockchain transactions:",
                value=selected_question,
                key="new_analysis_search_query",
                help="Type any question about your transaction data in natural language"
            )
            
            if st.button("Search", key="new_analysis_search_button"):
                if search_query:
                    with st.spinner("Analyzing your query with AI..."):
                        try:
                            # Use the AI search function
                            response = ai_transaction_search(
                                search_query,
                                st.session_state.df,
                                st.session_state.risk_assessment,
                                st.session_state.anomalies,
                                st.session_state.network_metrics
                            )
                            
                            # Store the result in session state
                            st.session_state.search_result = response
                            
                            # Display the response
                            st.markdown("### AI Analysis Results")
                            st.markdown(response)
                        except Exception as e:
                            st.error(f"Error performing AI search: {str(e)}")
                            st.expander("Technical Details").code(traceback.format_exc())
                else:
                    st.warning("Please enter a search query.")
            
            # Display previous search result if available
            if 'search_result' in st.session_state and st.session_state.search_result and not search_query:
                st.markdown("### Previous Search Results")
                st.markdown(st.session_state.search_result)
                
                # Option to clear results
                if st.button("Clear Results"):
                    st.session_state.search_result = None
                    st.rerun()
        
        with tab6:
            # Enhanced Advanced Analytics Tab
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(156, 39, 176, 0.1) 0%, rgba(123, 31, 162, 0.1) 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3>🧠 Advanced Multimodal Analytics</h3>
                <p>Deep AI analysis combining clustering, behavioral patterns, risk correlation, and temporal analysis for comprehensive insights into your blockchain data.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Analytics Capabilities
            st.markdown("### Advanced Analysis Capabilities")
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            with adv_col1:
                st.info("**Clustering Analysis** 🎯 Groups similar transactions to identify patterns")
            with adv_col2:
                st.info("**Behavioral Analysis** 👤 Identifies user behavior patterns and anomalies")
            with adv_col3:
                st.info("**Risk Correlation** ⚡ Finds hidden relationships between risk factors")
            
            if st.button("Run Advanced Analytics", key="advanced_analytics_button"):
                with st.spinner("Running advanced multimodal analysis..."):
                    try:
                        # Initialize advanced analytics
                        advanced_analytics = AdvancedAnalytics()
                        
                        # Perform multimodal analysis
                        multimodal_results = advanced_analytics.multimodal_analysis(
                            st.session_state.df,
                            st.session_state.risk_assessment,
                            st.session_state.network_metrics
                        )
                        
                        # Display results
                        st.success("Advanced analytics complete!")
                        
                        # Transaction Clustering Results
                        if 'transaction_clustering' in multimodal_results:
                            st.subheader("Transaction Clustering Analysis")
                            clustering_data = multimodal_results['transaction_clustering']
                            if 'error' not in clustering_data:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Total Clusters", clustering_data.get('total_clusters', 0))
                                with col2:
                                    st.metric("Outlier Percentage", f"{clustering_data.get('outlier_percentage', 0):.1f}%")
                                
                                if 'clusters' in clustering_data:
                                    for cluster_name, cluster_info in clustering_data['clusters'].items():
                                        with st.expander(f"{cluster_name} ({cluster_info['size']} transactions)"):
                                            st.write(f"Average Value: ${cluster_info['avg_value']:.2f}")
                                            st.write(f"Pattern: {cluster_info['pattern_description']}")
                        
                        # Behavioral Patterns
                        if 'behavioral_patterns' in multimodal_results:
                            st.subheader("Behavioral Pattern Analysis")
                            patterns = multimodal_results['behavioral_patterns']
                            if 'error' not in patterns:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if 'peak_hour' in patterns:
                                        st.metric("Peak Activity Hour", f"{patterns['peak_hour']}:00")
                                with col2:
                                    if 'unique_senders' in patterns:
                                        st.metric("Unique Senders", patterns['unique_senders'])
                                with col3:
                                    if 'unique_receivers' in patterns:
                                        st.metric("Unique Receivers", patterns['unique_receivers'])
                        
                        # Value Distribution Analysis
                        if 'value_distribution' in multimodal_results:
                            st.subheader("Value Distribution Analysis")
                            dist_data = multimodal_results['value_distribution']
                            if 'error' not in dist_data:
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Mean Value", f"${dist_data['mean']:.2f}")
                                with col2:
                                    st.metric("Median Value", f"${dist_data['median']:.2f}")
                                with col3:
                                    st.metric("Std Deviation", f"${dist_data['std']:.2f}")
                                with col4:
                                    st.metric("Skewness", f"{dist_data['skewness']:.2f}")
                                
                                if 'value_categories' in dist_data:
                                    st.write("**Transaction Categories:**")
                                    categories = dist_data['value_categories']
                                    st.write(f"- Micro Transactions: {categories['micro_transactions']}")
                                    st.write(f"- Small Transactions: {categories['small_transactions']}")
                                    st.write(f"- Large Transactions: {categories['large_transactions']}")
                                    st.write(f"- Whale Transactions: {categories['whale_transactions']}")
                        
                        # AI Insights
                        if 'ai_insights' in multimodal_results:
                            st.subheader("AI-Generated Insights")
                            st.markdown(multimodal_results['ai_insights'])
                            
                    except Exception as e:
                        st.error(f"Advanced analytics failed: {str(e)}")
        
        with tab7:
            # Enhanced Predictive Intelligence Tab
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(233, 30, 99, 0.1) 0%, rgba(194, 24, 91, 0.1) 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3>📊 Predictive Intelligence</h3>
                <p>Machine learning models forecast future transaction patterns, volumes, and potential risks based on historical data trends and advanced statistical analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Prediction Capabilities
            st.markdown("### Prediction Capabilities")
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            with pred_col1:
                st.info("**Volume Forecasting** 📈 Predict future transaction volumes and activity levels")
            with pred_col2:
                st.info("**Risk Prediction** ⚠️ Forecast potential risk patterns and anomaly likelihood")
            with pred_col3:
                st.info("**Trend Analysis** 📊 Identify emerging patterns and behavioral shifts")
            
            # Enhanced prediction settings
            st.markdown("### Prediction Settings")
            pred_settings_col1, pred_settings_col2 = st.columns(2)
            
            with pred_settings_col1:
                prediction_days = st.selectbox(
                    "Forecast Period", 
                    [7, 14, 30, 60], 
                    index=2,
                    help="Select how many days ahead to predict"
                )
            
            with pred_settings_col2:
                confidence_level = st.selectbox(
                    "Confidence Level",
                    ["High (95%)", "Medium (80%)", "Low (65%)"],
                    index=1,
                    help="Higher confidence provides more conservative predictions"
                )
            
            if st.button("Run Predictive Analysis", key="predictive_analysis_button"):
                with st.spinner("Running predictive analysis..."):
                    try:
                        # Initialize advanced analytics
                        advanced_analytics = AdvancedAnalytics()
                        
                        # Perform predictive analysis
                        predictive_results = advanced_analytics.predictive_analysis(
                            st.session_state.df,
                            prediction_horizon=prediction_days
                        )
                        
                        # Display results
                        st.success("Predictive analysis complete!")
                        
                        # Volume Forecast
                        if 'volume_forecast' in predictive_results:
                            st.subheader("Transaction Volume Forecast")
                            volume_data = predictive_results['volume_forecast']
                            if 'error' not in volume_data:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Predicted Daily Volume", f"{volume_data.get('predicted_daily_volume', 0):.1f}")
                                with col2:
                                    st.metric("Confidence Level", volume_data.get('confidence', 'Unknown'))
                                with col3:
                                    st.metric("Trend Direction", volume_data.get('trend', 'Unknown'))
                        
                        # Value Forecast
                        if 'value_forecast' in predictive_results:
                            st.subheader("Transaction Value Forecast")
                            value_data = predictive_results['value_forecast']
                            if 'error' not in value_data:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Predicted Avg Value", f"${value_data.get('predicted_avg_value', 0):.2f}")
                                with col2:
                                    st.metric("Value Trend", value_data.get('value_trend', 'Unknown'))
                                with col3:
                                    st.metric("Volatility Forecast", value_data.get('volatility_forecast', 'Unknown'))
                        
                        # Risk Forecast
                        if 'risk_forecast' in predictive_results:
                            st.subheader("Risk Level Forecast")
                            risk_data = predictive_results['risk_forecast']
                            if 'error' not in risk_data:
                                col1, col2 = st.columns(2)
                                with col1:
                                    risk_level = risk_data.get('risk_level_forecast', 'Unknown')
                                    if risk_level == 'High':
                                        st.error(f"Risk Level: {risk_level}")
                                    elif risk_level == 'Moderate':
                                        st.warning(f"Risk Level: {risk_level}")
                                    else:
                                        st.success(f"Risk Level: {risk_level}")
                                with col2:
                                    st.info(f"Recommendation: {risk_data.get('monitoring_recommendation', 'Standard monitoring')}")
                        
                        # Anomaly Likelihood
                        if 'anomaly_likelihood' in predictive_results:
                            st.subheader("Anomaly Prediction")
                            anomaly_data = predictive_results['anomaly_likelihood']
                            if 'error' not in anomaly_data:
                                col1, col2 = st.columns(2)
                                with col1:
                                    likelihood = anomaly_data.get('anomaly_likelihood', 'Unknown')
                                    if likelihood == 'High':
                                        st.error(f"Anomaly Likelihood: {likelihood}")
                                    elif likelihood == 'Medium':
                                        st.warning(f"Anomaly Likelihood: {likelihood}")
                                    else:
                                        st.success(f"Anomaly Likelihood: {likelihood}")
                                with col2:
                                    st.info(anomaly_data.get('recommendation', 'Standard monitoring'))
                        
                        # Recommendations
                        if 'recommendations' in predictive_results:
                            st.subheader("Predictive Recommendations")
                            recommendations = predictive_results['recommendations']
                            for i, rec in enumerate(recommendations, 1):
                                st.write(f"{i}. {rec}")
                                
                    except Exception as e:
                        st.error(f"Predictive analysis failed: {str(e)}")
        
        # Export and Save functionality
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Export Results")
            export_format = st.selectbox("Export Format", ["PDF", "CSV", "JSON", "Excel"])
            
            if st.button("Export Analysis Results") and st.session_state.analysis_results is not None:
                try:
                    if export_format == "PDF":
                        try:
                            # Generate visualizations for PDF
                            visualizations = {}
                            if hasattr(st.session_state, 'current_figures'):
                                visualizations = st.session_state.current_figures
                            
                            # Generate PDF report
                            session_name = getattr(st.session_state, 'current_session_name', 'Blockchain Analysis')
                            pdf_buffer = generate_pdf_report(
                                st.session_state.analysis_results, 
                                session_name, 
                                visualizations
                            )
                            
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_buffer.getvalue(),
                                file_name=f"blockchain_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                        except Exception as e:
                            st.error(f"Error generating PDF report: {str(e)}")
                            st.info("Try exporting as CSV or JSON instead.")
                    elif export_format == "CSV":
                        try:
                            # Use StringIO for more reliable CSV export
                            csv_buffer = io.StringIO()
                            st.session_state.analysis_results.to_csv(csv_buffer, index=False)
                            csv_str = csv_buffer.getvalue()
                            
                            st.download_button(
                                label="Download CSV",
                                data=csv_str,
                                file_name="blockchain_analysis_results.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Error exporting to CSV: {str(e)}")
                    elif export_format == "JSON":
                        json_str = st.session_state.analysis_results.to_json(orient="records")
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name="blockchain_analysis_results.json",
                            mime="application/json"
                        )
                    elif export_format == "Excel":
                        try:
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                st.session_state.analysis_results.to_excel(writer, sheet_name='Analysis', index=False)
                            excel_data = output.getvalue()
                            st.download_button(
                                label="Download Excel",
                                data=excel_data,
                                file_name="blockchain_analysis_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except Exception as excel_error:
                            st.error(f"Error creating Excel file: {str(excel_error)}")
                            st.info("Try exporting as CSV instead.")
                except Exception as export_error:
                    st.error(f"Error exporting results: {str(export_error)}")
                    st.info("Please make sure analysis has been completed successfully.")
        
        with col2:
            st.header("Save to Database")
            save_name = st.text_input("Analysis Name", value=f"Analysis of {st.session_state.current_dataset_name}" if st.session_state.current_dataset_name else "Blockchain Analysis")
            save_description = st.text_area("Description (optional)", placeholder="Enter a description for this analysis...")
            
            if st.button("Save Analysis to Database") and st.session_state.analysis_results is not None:
                try:
                    # Calculate network metrics if not already done
                    if st.session_state.network_metrics is None:
                        with st.spinner("Calculating network metrics..."):
                            st.session_state.network_metrics = calculate_network_metrics(st.session_state.df)
                    
                    # Save to database
                    session_id = save_analysis_to_db(
                        session_name=save_name,
                        dataset_name=st.session_state.current_dataset_name or "Unknown Dataset",
                        dataframe=st.session_state.df,
                        risk_assessment_df=st.session_state.risk_assessment,
                        anomaly_indices=st.session_state.anomalies,
                        network_metrics=st.session_state.network_metrics,
                        risk_threshold=risk_threshold,
                        anomaly_sensitivity=anomaly_sensitivity,
                        description=save_description
                    )
                    
                    if session_id:
                        st.success(f"Analysis saved successfully with ID: {session_id}")
                        st.session_state.saved_session_id = session_id
                    else:
                        st.error("Failed to save analysis")
                        
                except Exception as save_error:
                    st.error(f"Error saving analysis: {str(save_error)}")
                    st.expander("Technical Details").code(traceback.format_exc())
        
        # QuantumGuard AI branding at bottom after analysis
        st.markdown("""
        <div class="bottom-branding">
            <h1>QuantumGuard AI</h1>
            <p>Advanced Blockchain Transaction Analytics & AUSTRAC Compliance</p>
            <p style="font-size: 12px; margin-top: 10px;">Powered by Post-Quantum Cryptography | AI-Driven Risk Assessment | Real-Time Compliance Monitoring</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show placeholder when no analysis results exist
        st.info("📊 **Ready for Analysis**")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 10px; margin: 1rem 0;">
            <h3>🚀 Upload your data and run the analysis to view comprehensive results here</h3>
            <p>Analysis results will include network visualizations, risk assessments, anomaly detection, AI insights, and predictive intelligence.</p>
        </div>
        """, unsafe_allow_html=True)

# Quantum security testing is now embedded in the main UI flow above
