import os
import pandas as pd
import openai
from openai import OpenAI
import json

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def prepare_transaction_context(df, risk_assessment=None, anomalies=None, network_metrics=None):
    """
    Prepare transaction data context for the AI model.
    
    Args:
        df: DataFrame with transaction data
        risk_assessment: DataFrame with risk assessment data (optional)
        anomalies: List of anomaly indices (optional)
        network_metrics: Dictionary of network metrics (optional)
    
    Returns:
        String containing transaction data context
    """
    # Maximum number of transactions to include in context
    max_transactions = 50
    
    # Format the transaction data
    if len(df) > max_transactions:
        # If too many transactions, sample a subset
        context_df = df.sample(max_transactions)
        transaction_info = f"Sample of {max_transactions} transactions from a total of {len(df)} transactions:\n"
    else:
        context_df = df
        transaction_info = f"All {len(df)} transactions:\n"
    
    # Format transaction data
    transaction_records = context_df.to_dict(orient='records')
    transaction_info += json.dumps(transaction_records, indent=2)
    
    # Add risk assessment data if available
    if risk_assessment is not None and not risk_assessment.empty:
        risk_info = "\n\nRisk Assessment Data:\n"
        high_risks = risk_assessment[risk_assessment['risk_score'] > 0.7]
        risk_info += f"Number of high-risk transactions: {len(high_risks)}\n"
        
        if not high_risks.empty:
            risk_info += "Top 5 highest risk transactions:\n"
            top_risks = high_risks.sort_values('risk_score', ascending=False).head(5)
            risk_info += json.dumps(top_risks.to_dict(orient='records'), indent=2)
        transaction_info += risk_info
    
    # Add anomaly information if available
    if anomalies is not None and len(anomalies) > 0:
        anomaly_info = f"\n\nAnomaly Detection Results:\n"
        anomaly_info += f"Number of anomalies detected: {len(anomalies)}\n"
        
        if df is not None:
            anomaly_df = df.iloc[anomalies] if len(anomalies) > 0 else pd.DataFrame()
            if not anomaly_df.empty:
                anomaly_info += "Anomalous transactions:\n"
                anomaly_info += json.dumps(anomaly_df.head(5).to_dict(orient='records'), indent=2)
        transaction_info += anomaly_info
    
    # Add network metrics if available
    if network_metrics is not None:
        network_info = "\n\nNetwork Metrics:\n"
        network_info += json.dumps(network_metrics, indent=2)
        transaction_info += network_info
    
    return transaction_info


def ai_transaction_search(query, df, risk_assessment=None, anomalies=None, network_metrics=None):
    """
    Perform an AI-powered search on blockchain transaction data.
    
    Args:
        query: User's query string
        df: DataFrame containing transaction data
        risk_assessment: DataFrame with risk assessment results (optional)
        anomalies: List of anomaly indices (optional)
        network_metrics: Dictionary of network metrics (optional)
    
    Returns:
        AI-generated response to the query
    """
    if df is None or df.empty:
        return "No transaction data available. Please upload and analyze transaction data first."
    
    # Prepare the context with transaction data
    transaction_context = prepare_transaction_context(df, risk_assessment, anomalies, network_metrics)
    
    try:
        # Build the prompt with system instructions and transaction context
        system_message = """You are an expert blockchain transaction analyst assistant. 
Your task is to analyze blockchain transaction data and provide insights based on user queries.
Be specific, concise, and provide evidence from the transaction data to support your answers.
Present quantitative insights when possible, and highlight any suspicious patterns or anomalies.
If you don't know the answer or there's insufficient data, say so rather than making up information."""

        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the specified model
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Here is the blockchain transaction data for analysis:\n\n{transaction_context}"},
                {"role": "user", "content": f"Based on this transaction data, please answer the following question: {query}"}
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=1000
        )
        
        # Extract and return the model's response
        return response.choices[0].message.content
    
    except Exception as e:
        # Handle errors gracefully
        error_message = f"Error performing AI search: {str(e)}"
        print(error_message)  # Log the error
        return f"I encountered a problem while analyzing your query: {str(e)}. Please try again or rephrase your question."