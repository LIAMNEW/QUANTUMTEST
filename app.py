import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import traceback
import os
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
    delete_analysis_session
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
# Generate keys when app starts
if not st.session_state.keys_generated:
    st.session_state.public_key, st.session_state.private_key = generate_pq_keys()
    st.session_state.keys_generated = True

# Header
st.title("Quantum-Secure Blockchain Transaction Analyzer")

# Initialize variables for run_analysis and progress_placeholder
run_analysis = False
progress_placeholder = None

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    app_mode = st.radio("Select Mode", ["New Analysis", "Saved Analyses"])
    
    if app_mode == "New Analysis":
        st.session_state.view_saved_analysis = False
        
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload blockchain transaction dataset", 
                                        type=["csv", "xlsx", "json"])
        
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
                    st.info("Data prepared for analysis")
                    
                    # Show next steps guidance
                    st.info("👇 Now use the settings below to run the AI analysis")
                else:
                    st.error("The uploaded file appears to be empty or has no valid data.")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.expander("Technical Details").code(traceback.format_exc())
        
        st.divider()
        st.header("Analysis Settings")
        
        # Define risk threshold and anomaly sensitivity
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.7, 0.05)
        anomaly_sensitivity = st.slider("Anomaly Detection Sensitivity", 0.0, 1.0, 0.8, 0.05)
        
        # Create a progress placeholder
        progress_placeholder = st.empty()
        
        # Run analysis button
        run_analysis = st.button("Run Analysis")
        
    else:  # Saved Analyses mode
        st.session_state.view_saved_analysis = True
        st.header("Saved Analyses")
        
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
            # Show balloons to celebrate successful analysis
            st.balloons()
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
            
            # Create tabs for visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Transaction Network", "Risk Assessment", 
                                          "Anomaly Detection", "Transaction Timeline"])
            
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
        else:
            st.error("Could not load the selected analysis. It may have been deleted.")
    except Exception as e:
        st.error(f"Error loading saved analysis: {str(e)}")

elif st.session_state.df is None:
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
    
    ### Database Integration
    Analysis results are securely stored in a PostgreSQL database, allowing you to 
    save and retrieve your analysis at any time.
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
            # Use a more reliable image URL from a CDN
            st.image("https://cdn.pixabay.com/photo/2017/12/22/08/01/blockchain-3033200_1280.jpg", 
                    caption="Network Visualization")
            if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
                # Use the processed data for visualization
                fig = plot_transaction_network(st.session_state.analysis_results)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Risk Assessment")
            if st.session_state.risk_assessment is not None:
                fig = plot_risk_heatmap(st.session_state.risk_assessment)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display high-risk transactions
                if 'risk_score' in st.session_state.risk_assessment.columns:
                    high_risks = st.session_state.risk_assessment[st.session_state.risk_assessment['risk_score'] > 0.7]
                    if not high_risks.empty:
                        st.warning(f"Found {len(high_risks)} high-risk transactions")
                        st.dataframe(high_risks)
                    else:
                        st.success("No high-risk transactions detected")
        
        with tab3:
            st.subheader("Anomaly Detection")
            if st.session_state.df is not None and st.session_state.anomalies is not None:
                fig = plot_anomaly_detection(st.session_state.df, st.session_state.anomalies)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display anomalies
                if st.session_state.anomalies and len(st.session_state.anomalies) > 0:
                    st.warning(f"Detected {len(st.session_state.anomalies)} anomalous transactions")
                    # Use the processed data for anomaly display
                    if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
                        anomaly_df = st.session_state.analysis_results.iloc[st.session_state.anomalies]
                        st.dataframe(anomaly_df)
                else:
                    st.success("No anomalies detected")
        
        with tab4:
            st.subheader("Transaction Timeline")
            # Use a more reliable image URL from a CDN
            st.image("https://cdn.pixabay.com/photo/2021/05/25/02/51/crypto-6281601_1280.jpg", 
                    caption="Blockchain Timeline Analysis")
            if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
                # Use the processed data for visualization
                fig = plot_transaction_timeline(st.session_state.analysis_results)
                st.plotly_chart(fig, use_container_width=True)
        
        # Export and Save functionality
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Export Results")
            export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
            
            if st.button("Export Analysis Results") and st.session_state.analysis_results is not None:
                try:
                    if export_format == "CSV":
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
