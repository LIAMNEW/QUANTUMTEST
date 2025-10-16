import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import logging

# Import existing modules
from data_processor import preprocess_blockchain_data, extract_features, validate_blockchain_data, clean_blockchain_data
from ai_model import FraudDetectionModel

# Configure page
st.set_page_config(
    page_title="QuantumGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS with glassmorphism
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --dark-bg: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
    }
    
    /* Glass morphism effect */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }
    
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        background-size: 200% 200%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Metric cards with glow effect */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(99, 102, 241, 0.3);
        border-color: rgba(99, 102, 241, 0.6);
    }
    
    /* Animated number counter */
    @keyframes countUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: countUp 0.6s ease-out;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .status-high {
        background: rgba(245, 158, 11, 0.2);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .status-medium {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .status-low {
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px 0 rgba(99, 102, 241, 0.6);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        background: transparent;
        color: #94a3b8;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Chart container */
    .chart-container {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Success/Error messages */
    .stAlert {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border-left: 4px solid;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Data table styling */
    .dataframe {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 8px;
    }
    
    /* Risk indicator */
    .risk-indicator {
        width: 100%;
        height: 8px;
        background: rgba(71, 85, 105, 0.3);
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }
    
    .risk-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease-out;
        background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        'model': FraudDetectionModel(),
        'data': None,
        'predictions': None,
        'analysis_complete': False,
        'current_page': 'home',
        'uploaded_filename': None,
        'analysis_timestamp': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Header with logo and title
def render_header():
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <h1 style="
                    font-size: 3rem;
                    font-weight: 800;
                    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 0.5rem;
                ">
                    🛡️ QuantumGuard AI
                </h1>
                <p style="
                    color: #94a3b8;
                    font-size: 1.1rem;
                    font-weight: 500;
                ">
                    Advanced Blockchain Transaction Analytics & AUSTRAC Compliance
                </p>
            </div>
        """, unsafe_allow_html=True)

# Animated metric card
def metric_card(label, value, delta=None, icon="📊"):
    delta_html = ""
    if delta is not None:
        delta_color = "#10b981" if delta >= 0 else "#ef4444"
        delta_arrow = "↑" if delta >= 0 else "↓"
        delta_html = f"""
            <div style="color: {delta_color}; font-size: 0.9rem; margin-top: 0.5rem;">
                {delta_arrow} {abs(delta):.1f}%
            </div>
        """
    
    return f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <span style="color: #94a3b8; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em;">
                    {label}
                </span>
            </div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    """

# Home/Dashboard page
def render_home():
    render_header()
    
    # Quick stats if data is loaded
    if st.session_state.predictions is not None:
        df = st.session_state.predictions
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(metric_card(
                "Total Transactions",
                f"{len(df):,}",
                icon="📊"
            ), unsafe_allow_html=True)
        
        with col2:
            anomaly_count = (df['is_anomaly'] == 1).sum() if 'is_anomaly' in df.columns else 0
            anomaly_rate = (anomaly_count / len(df) * 100) if len(df) > 0 else 0
            st.markdown(metric_card(
                "Anomaly Rate",
                f"{anomaly_rate:.1f}%",
                delta=-2.3,
                icon="⚠️"
            ), unsafe_allow_html=True)
        
        with col3:
            high_risk = (df['risk_score'] > 70).sum() if 'risk_score' in df.columns else 0
            st.markdown(metric_card(
                "High Risk",
                f"{high_risk:,}",
                icon="🚨"
            ), unsafe_allow_html=True)
        
        with col4:
            total_volume = df['amount'].sum() if 'amount' in df.columns else 0
            st.markdown(metric_card(
                "Total Volume",
                f"${total_volume:,.0f}",
                delta=5.7,
                icon="💰"
            ), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Risk distribution chart
        st.markdown("""
            <div class="chart-container">
                <h3 style="color: white; margin-bottom: 1rem;">📈 Risk Distribution Overview</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if 'risk_score' in df.columns:
            fig = create_risk_distribution_chart(df)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Welcome state
        st.markdown("""
            <div style="
                text-align: center;
                padding: 4rem 2rem;
                background: rgba(30, 41, 59, 0.5);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.05);
                margin: 2rem 0;
            ">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🚀</div>
                <h2 style="color: white; margin-bottom: 1rem;">Ready to Analyze Blockchain Transactions</h2>
                <p style="color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem;">
                    Upload your transaction data to unlock powerful AI-driven insights and AUSTRAC compliance analysis
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        col1, col2, col3 = st.columns(3)
        
        features = [
            ("🤖", "AI-Powered Detection", "Machine learning algorithms identify unusual patterns"),
            ("🔐", "Quantum-Resilient", "Future-proofed with post-quantum cryptography"),
            ("📊", "Real-Time Monitoring", "Instant analysis of blockchain transactions")
        ]
        
        for col, (icon, title, desc) in zip([col1, col2, col3], features):
            with col:
                st.markdown(f"""
                    <div class="glass-card" style="text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
                        <h4 style="color: white; margin-bottom: 0.5rem;">{title}</h4>
                        <p style="color: #94a3b8; font-size: 0.9rem;">{desc}</p>
                    </div>
                """, unsafe_allow_html=True)

# Create enhanced risk distribution chart
def create_risk_distribution_chart(df):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Risk Score Distribution", "Anomaly Detection"),
        specs=[[{"type": "histogram"}, {"type": "pie"}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=df['risk_score'],
            nbinsx=50,
            marker=dict(
                color=df['risk_score'],
                colorscale='Viridis',
                showscale=True
            ),
            name="Risk Distribution"
        ),
        row=1, col=1
    )
    
    # Pie chart
    if 'is_anomaly' in df.columns:
        anomaly_counts = df['is_anomaly'].value_counts()
        labels = ['Normal', 'Anomaly']
        values = [anomaly_counts.get(0, 0), anomaly_counts.get(-1, 0) + anomaly_counts.get(1, 0)]
    else:
        labels = ['Data']
        values = [len(df)]
    
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=['#10b981', '#ef4444']),
            hole=0.4
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        showlegend=False
    )
    
    return fig

# Sidebar navigation
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
            <h2 style="color: white;">Navigation</h2>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "",
        ["🏠 Home", "📁 Upload Data", "🔍 Run Analysis", "📊 Results"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if st.session_state.uploaded_filename:
        st.markdown("""
            <div class="glass-card">
                <h4 style="color: white; margin-bottom: 0.5rem;">📄 Current Dataset</h4>
                <p style="color: #94a3b8; font-size: 0.9rem;">""" + str(st.session_state.uploaded_filename) + """</p>
            </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.analysis_timestamp:
        st.markdown("""
            <div class="glass-card">
                <h4 style="color: white; margin-bottom: 0.5rem;">⏰ Last Analysis</h4>
                <p style="color: #94a3b8; font-size: 0.9rem;">""" + str(st.session_state.analysis_timestamp.strftime("%Y-%m-%d %H:%M")) + """</p>
            </div>
        """, unsafe_allow_html=True)

# Main content routing
if page == "🏠 Home":
    render_home()

elif page == "📁 Upload Data":
    st.markdown("""
        <h2 style="color: white; margin-bottom: 2rem;">
            📁 Upload Transaction Data
        </h2>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="glass-card">
                <h3 style="color: white; margin-bottom: 1rem;">Upload Your Dataset</h3>
                <p style="color: #94a3b8; margin-bottom: 1rem;">
                    Supports CSV files with blockchain or banking transaction data
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "",
            type=['csv'],
            help="Upload a CSV file containing transaction data",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
            <div class="glass-card">
                <h4 style="color: white; margin-bottom: 1rem;">📋 Recommended Columns</h4>
                <ul style="color: #94a3b8; line-height: 1.8;">
                    <li>timestamp / date</li>
                    <li>amount / value</li>
                    <li>from_address</li>
                    <li>to_address</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            with st.spinner("🔄 Processing your data..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📂 Loading CSV file...")
                progress_bar.progress(20)
                time.sleep(0.3)
                
                df = pd.read_csv(uploaded_file)
                
                status_text.text("✓ Validating data structure...")
                progress_bar.progress(40)
                time.sleep(0.3)
                
                validation = validate_blockchain_data(df)
                
                if not validation['valid']:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ Data validation failed")
                    for error in validation['errors']:
                        st.error(error)
                    for warning in validation['warnings']:
                        st.warning(warning)
                else:
                    status_text.text("🧹 Cleaning data...")
                    progress_bar.progress(60)
                    time.sleep(0.3)
                    
                    df_clean = clean_blockchain_data(df)
                    
                    status_text.text("🔧 Extracting features...")
                    progress_bar.progress(80)
                    time.sleep(0.3)
                    
                    df_features = extract_features(df_clean)
                    
                    st.session_state.data = df_features
                    st.session_state.uploaded_filename = uploaded_file.name
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success(f"✅ Successfully processed {len(df_features):,} transactions!")
                    
                    st.markdown("""
                        <div class="glass-card">
                            <h3 style="color: white; margin-bottom: 1rem;">📊 Data Preview</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(df_features.head(10), use_container_width=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(metric_card("Rows", f"{len(df_features):,}", icon="📊"), unsafe_allow_html=True)
                    with col2:
                        st.markdown(metric_card("Columns", f"{len(df_features.columns)}", icon="📋"), unsafe_allow_html=True)
                    with col3:
                        unique_addresses = df_features['from_address'].nunique() if 'from_address' in df_features.columns else 0
                        st.markdown(metric_card("Unique Addresses", f"{unique_addresses:,}", icon="🏛️"), unsafe_allow_html=True)
                    with col4:
                        total_volume = df_features['amount'].sum() if 'amount' in df_features.columns else 0
                        st.markdown(metric_card("Total Volume", f"${total_volume:,.0f}", icon="💰"), unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            logging.error(f"File processing error: {str(e)}", exc_info=True)

elif page == "🔍 Run Analysis":
    st.markdown("""
        <h2 style="color: white; margin-bottom: 2rem;">
            🔍 Run Fraud Detection Analysis
        </h2>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        if st.button("Go to Upload Page", use_container_width=True):
            st.rerun()
    else:
        df = st.session_state.data
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
                <div class="glass-card">
                    <h3 style="color: white; margin-bottom: 1rem;">⚙️ Analysis Configuration</h3>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            contamination = st.slider(
                "Contamination Rate",
                min_value=0.01,
                max_value=0.3,
                value=0.1,
                step=0.01,
                help="Expected proportion of anomalies in the dataset"
            )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Run Complete Analysis", use_container_width=True, type="primary"):
                try:
                    with st.spinner("🤖 Training AI model and analyzing transactions..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("🔧 Preparing features...")
                        progress_bar.progress(20)
                        time.sleep(0.5)
                        
                        feature_cols = [col for col in df.columns if col not in 
                                      ['timestamp', 'from_address', 'to_address', 'date', 'time']]
                        X = df[feature_cols].fillna(0)
                        
                        status_text.text("🧠 Training AI model...")
                        progress_bar.progress(40)
                        
                        model = st.session_state.model
                        model.contamination = contamination
                        model.train(X)
                        
                        progress_bar.progress(60)
                        
                        status_text.text("🎯 Detecting anomalies...")
                        progress_bar.progress(80)
                        
                        predictions, risk_scores = model.predict(X)
                        
                        df['prediction'] = predictions
                        df['risk_score'] = risk_scores
                        df['is_anomaly'] = (predictions == -1).astype(int)
                        df['risk_level'] = pd.cut(
                            risk_scores,
                            bins=[0, 30, 60, 80, 100],
                            labels=['Low', 'Medium', 'High', 'Critical']
                        )
                        
                        st.session_state.predictions = df
                        st.session_state.analysis_complete = True
                        st.session_state.analysis_timestamp = datetime.now()
                        
                        progress_bar.progress(100)
                        time.sleep(0.5)
                        
                        status_text.empty()
                        progress_bar.empty()
                    
                    st.success("✅ Analysis complete!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logging.error(f"Analysis error: {str(e)}", exc_info=True)

elif page == "📊 Results":
    if st.session_state.predictions is None:
        st.warning("⚠️ No analysis results available. Please run analysis first.")
    else:
        df = st.session_state.predictions
        
        st.markdown("""
            <div class="glass-card">
                <h3 style="color: white; margin-bottom: 1rem;">📊 Detection Summary</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        n_anomalies = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
        anomaly_rate = n_anomalies / len(df) * 100 if len(df) > 0 else 0
        high_risk = (df['risk_score'] > 70).sum() if 'risk_score' in df.columns else 0
        suspicious_volume = df[df['is_anomaly'] == 1]['amount'].sum() if 'amount' in df.columns and 'is_anomaly' in df.columns else 0
        
        with col1:
            st.markdown(metric_card("Anomalies", f"{n_anomalies:,}", icon="⚠️"), unsafe_allow_html=True)
        with col2:
            st.markdown(metric_card("Anomaly Rate", f"{anomaly_rate:.2f}%", icon="📊"), unsafe_allow_html=True)
        with col3:
            st.markdown(metric_card("High Risk", f"{high_risk:,}", icon="🚨"), unsafe_allow_html=True)
        with col4:
            st.markdown(metric_card("Suspicious $", f"${suspicious_volume:,.0f}", icon="💰"), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Risk distribution visualization
        if 'risk_score' in df.columns:
            fig = px.histogram(
                df,
                x='risk_score',
                color='risk_level' if 'risk_level' in df.columns else None,
                title="Risk Score Distribution",
                labels={'risk_score': 'Risk Score', 'count': 'Number of Transactions'},
                color_discrete_map={
                    'Low': '#10b981',
                    'Medium': '#3b82f6',
                    'High': '#f59e0b',
                    'Critical': '#ef4444'
                } if 'risk_level' in df.columns else None
            )
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Transaction table
        st.markdown("""
            <div class="glass-card">
                <h4 style="color: white; margin-bottom: 1rem;">🔍 Transaction Details</h4>
            </div>
        """, unsafe_allow_html=True)
        
        show_anomalies = st.checkbox("Show Anomalies Only", value=False)
        
        display_df = df[df['is_anomaly'] == 1] if show_anomalies and 'is_anomaly' in df.columns else df
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Export button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results as CSV",
                csv,
                "fraud_analysis_results.csv",
                "text/csv",
                use_container_width=True
            )
