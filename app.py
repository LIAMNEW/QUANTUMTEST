import streamlit as st
import pandas as pd
import numpy as np
import io
from blockchain_analyzer import analyze_blockchain_data, identify_risks
from ml_models import train_anomaly_detection, detect_anomalies
from quantum_crypto import encrypt_data, decrypt_data, generate_pq_keys
from data_processor import preprocess_blockchain_data, extract_features
from visualizations import (
    plot_transaction_network, 
    plot_risk_heatmap, 
    plot_anomaly_detection,
    plot_transaction_timeline
)

# Set page configuration
st.set_page_config(
    page_title="Quantum-Secure Blockchain Analytics",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'encrypted_data' not in st.session_state:
    st.session_state.encrypted_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'risk_assessment' not in st.session_state:
    st.session_state.risk_assessment = None
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None
if 'public_key' not in st.session_state:
    st.session_state.public_key, st.session_state.private_key = generate_pq_keys()

# Header
st.title("Quantum-Secure Blockchain Transaction Analyzer")

# Sidebar for uploads and settings
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload blockchain transaction dataset", 
                                    type=["csv", "xlsx", "json"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            st.session_state.df = df
            st.success("File uploaded successfully!")
            
            # Automatically encrypt the data using post-quantum crypto
            encrypted_data = encrypt_data(df.to_json(), st.session_state.public_key)
            st.session_state.encrypted_data = encrypted_data
            st.info("Data encrypted with post-quantum cryptography")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    st.divider()
    st.header("Analysis Settings")
    
    risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.7, 0.05)
    anomaly_sensitivity = st.slider("Anomaly Detection Sensitivity", 0.0, 1.0, 0.8, 0.05)
    
    run_analysis = st.button("Run Analysis")
    
    if run_analysis and st.session_state.df is not None:
        with st.spinner("Running AI-powered analysis..."):
            # Decrypt data for processing
            json_data = decrypt_data(st.session_state.encrypted_data, st.session_state.private_key)
            df = pd.read_json(io.StringIO(json_data))
            
            # Preprocess data
            processed_data = preprocess_blockchain_data(df)
            features = extract_features(processed_data)
            
            # Run anomaly detection
            model = train_anomaly_detection(features)
            anomalies = detect_anomalies(model, features, sensitivity=anomaly_sensitivity)
            
            # Analyze blockchain data and assess risks
            analysis_results = analyze_blockchain_data(processed_data)
            risk_assessment = identify_risks(processed_data, threshold=risk_threshold)
            
            # Store results in session state
            st.session_state.analysis_results = analysis_results
            st.session_state.risk_assessment = risk_assessment
            st.session_state.anomalies = anomalies
            
            st.success("Analysis complete!")

# Main content area
if st.session_state.df is None:
    st.header("Welcome to Quantum-Secure Blockchain Analyzer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Upload your blockchain data to:
        - Analyze transaction patterns
        - Identify potential risks
        - Detect anomalies with AI
        - Visualize blockchain networks
        - Generate detailed insights
        """)
    
    with col2:
        st.image("https://pixabay.com/get/g344463069dc2cb1c2fe201943f121ae54347d44b450983461cba04da64a0f804ae2139e6014cfd8d76c546177feff3e4204cf4243d15c5ec292d4b15d2602ced_1280.jpg", 
                caption="Blockchain Transaction Analysis")
    
    st.image("https://pixabay.com/get/gb52e10f881ecac85f6560f93ebbb5325b1cda96ed6762ad4a700089852e001b2bb21ff55136c0e986457c9d7d9e229509c6a31cec9559a3f9fa9451b951a3a56_1280.jpg", 
            caption="Secure Blockchain Network Visualization")
    
    st.markdown("""
    ### Post-Quantum Security
    This application uses state-of-the-art post-quantum cryptography to protect your data,
    ensuring it remains secure even against quantum computing attacks.
    """)

else:
    # Display the data preview
    st.header("Data Preview")
    st.dataframe(st.session_state.df.head())
    
    # If analysis has been run, display results
    if st.session_state.analysis_results is not None:
        st.header("Analysis Results")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Transaction Network", "Risk Assessment", 
                                          "Anomaly Detection", "Transaction Timeline"])
        
        with tab1:
            st.subheader("Blockchain Transaction Network")
            st.image("https://pixabay.com/get/g6529d0db98955dfe8174b5acaa568d923927d28db0122b0db6ab44972011b793bd8ef5e798a55278d8faa81e2e0d1025132499258a8b64a914568805bc71007e_1280.jpg", 
                    caption="Network Visualization")
            fig = plot_transaction_network(st.session_state.df)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Risk Assessment")
            fig = plot_risk_heatmap(st.session_state.risk_assessment)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display high-risk transactions
            high_risks = st.session_state.risk_assessment[st.session_state.risk_assessment['risk_score'] > 0.7]
            if not high_risks.empty:
                st.warning(f"Found {len(high_risks)} high-risk transactions")
                st.dataframe(high_risks)
            else:
                st.success("No high-risk transactions detected")
        
        with tab3:
            st.subheader("Anomaly Detection")
            fig = plot_anomaly_detection(st.session_state.df, st.session_state.anomalies)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display anomalies
            if len(st.session_state.anomalies) > 0:
                st.warning(f"Detected {len(st.session_state.anomalies)} anomalous transactions")
                anomaly_df = st.session_state.df.iloc[st.session_state.anomalies]
                st.dataframe(anomaly_df)
            else:
                st.success("No anomalies detected")
        
        with tab4:
            st.subheader("Transaction Timeline")
            st.image("https://pixabay.com/get/ga135386cc8dfd2789f7b1bc9fe96ea3866bb276ebee4e580692a7ceef724f7ea6207f336bb8582491b52a24572b454cbf9b4ce2e774a8a82afa8f6d8130a8eea_1280.jpg", 
                    caption="Blockchain Timeline Analysis")
            fig = plot_transaction_timeline(st.session_state.df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        st.header("Export Results")
        export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
        
        if st.button("Export Analysis Results"):
            if export_format == "CSV":
                csv = st.session_state.analysis_results.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="blockchain_analysis_results.csv",
                    mime="text/csv"
                )
            elif export_format == "JSON":
                json_str = st.session_state.analysis_results.to_json(orient="records")
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="blockchain_analysis_results.json",
                    mime="application/json"
                )
            elif export_format == "Excel":
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
