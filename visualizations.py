import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import List, Dict

def plot_transaction_network(df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive network visualization of blockchain transactions.
    
    Args:
        df: DataFrame containing transaction data
    
    Returns:
        Plotly Figure object containing the network visualization
    """
    # Create a graph from transactions
    G = nx.from_pandas_edgelist(
        df, 'from_address', 'to_address', 
        edge_attr=['value'] if 'value' in df.columns else None,
        create_using=nx.DiGraph()
    )
    
    # Calculate node positions with more spread for less overlap
    pos = nx.spring_layout(G, seed=42, k=0.5)  # Increasing k spreads nodes further apart
    
    # Calculate node sizes based on degree - bigger nodes for better visibility
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = {node: (15 + (degree / max_degree) * 25) for node, degree in degrees.items()}
    
    # Calculate edge weights based on transaction value
    if 'value' in df.columns:
        edge_weights = nx.get_edge_attributes(G, 'value')
        max_weight = max(edge_weights.values()) if edge_weights else 1
        # Reduce line width range for cleaner appearance
        edge_widths = {edge: (0.5 + (weight / max_weight) * 3) for edge, weight in edge_weights.items()}
    else:
        edge_widths = {edge: 1 for edge in G.edges()}
    
    # Create node traces - improved visibility and hover information
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        marker=dict(
            size=[node_sizes[node] for node in G.nodes()],
            color=[degrees[node] for node in G.nodes()],
            colorscale='Viridis',
            colorbar=dict(title='Node Connections'),
            line=dict(width=1.5, color='rgba(50, 50, 50, 0.9)'),
            opacity=0.9
        ),
        # Truncate long addresses for display clarity
        text=[node[:6] + '...' if len(str(node)) > 10 else node for node in G.nodes()],
        textposition="bottom center",
        textfont=dict(size=10, color='rgba(0, 0, 0, 0.7)'),
        # Show full details on hover
        hovertext=[f"Address: {node}<br>Connections: {degrees[node]}" for node in G.nodes()],
        hoverinfo='text',
        name='Addresses'
    )
    
    # Create edge traces with better styling and hover information
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        width = edge_widths.get((edge[0], edge[1]), 1)
        
        # Get the transaction value if available
        if 'value' in df.columns:
            # Find transactions between these addresses
            transactions = df[(df['from_address'] == edge[0]) & (df['to_address'] == edge[1])]
            if not transactions.empty:
                value = transactions['value'].sum()
                hover_text = f"From: {edge[0][:6]}...<br>To: {edge[1][:6]}...<br>Total Value: {value:.2f}"
            else:
                hover_text = f"From: {edge[0][:6]}...<br>To: {edge[1][:6]}..."
        else:
            hover_text = f"From: {edge[0][:6]}...<br>To: {edge[1][:6]}..."
        
        # Create a more visually appealing edge trace
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=width, 
                color='rgba(70, 130, 180, 0.6)',  # Steel blue color
                dash='solid'
            ),
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create figure with improved styling and annotations
    fig = go.Figure(data=edge_traces + [node_trace],
                   layout=go.Layout(
                       title={
                           'text': 'Blockchain Transaction Network',
                           'font': {'size': 24},
                           'x': 0.5,
                           'y': 0.95
                       },
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=40, l=10, r=10, t=60),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       template='plotly_dark',
                       legend=dict(
                           x=0.01, 
                           y=0.99,
                           bgcolor='rgba(50, 50, 50, 0.8)',
                           borderwidth=1
                       ),
                       annotations=[
                           dict(
                               text="<b>Node size</b>: Number of connections | <b>Edge thickness</b>: Transaction value",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.5, y=0.02,
                               align="center",
                               font=dict(size=12, color='rgba(200, 200, 200, 0.9)')
                           )
                       ]
                   ))
    
    return fig

def plot_risk_heatmap(risk_df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap visualization of transaction risk factors.
    
    Args:
        risk_df: DataFrame containing risk assessment data
    
    Returns:
        Plotly Figure object containing the risk heatmap
    """
    # Ensure risk_df has necessary columns
    if 'risk_score' not in risk_df.columns:
        return go.Figure().update_layout(title="No risk data available")
    
    # Extract risk factors from the risk_factors column
    risk_factors = []
    if 'risk_factors' in risk_df.columns:
        for factors in risk_df['risk_factors'].str.split(';'):
            factors = [f.strip() for f in factors if f.strip()]
            risk_factors.extend(factors)
    
    risk_factor_counts = pd.Series(risk_factors).value_counts()
    
    # If no risk factors found, create a simpler visualization
    if len(risk_factor_counts) == 0:
        fig = px.histogram(
            risk_df, x='risk_score', 
            color_discrete_sequence=['#FF4B4B'],
            title='Transaction Risk Distribution',
            template='plotly_dark'
        )
        fig.update_layout(xaxis_title='Risk Score', yaxis_title='Number of Transactions')
        return fig
    
    # Create a pivot table of risk factors by risk score
    risk_pivot = pd.DataFrame({'count': risk_factor_counts})
    risk_pivot.reset_index(inplace=True)
    risk_pivot.columns = ['Risk Factor', 'Count']
    
    # If we have too many risk factors, limit to top N
    if len(risk_pivot) > 10:
        risk_pivot = risk_pivot.sort_values('Count', ascending=False).head(10)
    
    # Create horizontal bar chart with improved styling
    fig = px.bar(
        risk_pivot, y='Risk Factor', x='Count',
        color='Count', color_continuous_scale='Reds',
        orientation='h',
        title='Risk Factors Distribution',
        template='plotly_dark',
        labels={'Count': 'Number of Transactions'},
        height=500  # Taller graph for better readability
    )
    
    # Improve formatting and readability
    fig.update_layout(
        title={
            'text': 'Risk Factors Distribution',
            'font': {'size': 22},
            'x': 0.5,
            'y': 0.95
        },
        margin=dict(l=20, r=20, t=60, b=40)
    )
    
    # Sort by count for better visualization
    fig.update_yaxes(categoryorder='total ascending')
    
    # Add a secondary plot showing risk score distribution
    fig2 = px.histogram(
        risk_df, x='risk_score', 
        color_discrete_sequence=['#FF4B4B'],
        title='Risk Score Distribution'
    )
    
    # Combine the plots
    fig3 = go.Figure()
    
    # Add all traces from the first figure
    for trace in fig.data:
        fig3.add_trace(trace)
    
    # Create a separate y-axis for the second plot
    if len(fig2.data) > 0:
        histogram_trace = fig2.data[0]
        histogram_trace.xaxis = 'x2'
        histogram_trace.yaxis = 'y2'
        fig3.add_trace(histogram_trace)
    
    # Update layout to accommodate both plots
    fig3.update_layout(
        title='Transaction Risk Analysis',
        xaxis=dict(domain=[0, 0.7], title='Count'),
        yaxis=dict(title='Risk Factor'),
        xaxis2=dict(domain=[0.75, 1], title='Risk Score'),
        yaxis2=dict(anchor='x2', title='Frequency'),
        template='plotly_dark',
        height=600
    )
    
    return fig3

def plot_anomaly_detection(df: pd.DataFrame, anomaly_indices: List[int]) -> go.Figure:
    """
    Create a visualization of the anomaly detection results.
    
    Args:
        df: DataFrame containing transaction data
        anomaly_indices: List of indices corresponding to anomalous transactions
    
    Returns:
        Plotly Figure object containing the anomaly visualization
    """
    # Create a copy of the dataframe with an anomaly flag
    vis_df = df.copy()
    vis_df['is_anomaly'] = False
    vis_df.loc[anomaly_indices, 'is_anomaly'] = True
    
    # Determine which features to plot
    if 'value' in vis_df.columns:
        x_feature = 'value'
        
        # For the y-axis, look for a timestamp or transaction count
        if 'timestamp' in vis_df.columns and pd.api.types.is_datetime64_dtype(vis_df['timestamp']):
            y_feature = 'timestamp'
        else:
            # Use transaction index as a proxy for time
            vis_df['transaction_index'] = np.arange(len(vis_df))
            y_feature = 'transaction_index'
        
        # Create a more informative scatter plot
        fig = px.scatter(
            vis_df, x=x_feature, y=y_feature, 
            color='is_anomaly',
            color_discrete_map={False: 'rgba(99, 110, 250, 0.7)', True: 'rgba(239, 85, 59, 1.0)'},
            size=[15 if a else 7 for a in vis_df['is_anomaly']],  # Make anomalies more prominent
            opacity=[1.0 if a else 0.7 for a in vis_df['is_anomaly']],  # Make normal points semi-transparent
            hover_data=vis_df.columns.tolist(),
            labels={
                'is_anomaly': 'Anomaly Status',
                x_feature: x_feature.title(),
                y_feature: 'Time/Sequence' if y_feature == 'transaction_index' else y_feature.title()
            },
            title='Anomaly Detection Results',
            template='plotly_dark',
            height=600  # Larger plot for better visibility
        )
        
        # Add custom hover information
        hover_template = (
            "<b>Transaction:</b> %{customdata[0]}<br>" +
            "<b>Value:</b> %{x}<br>" +
            "<b>From:</b> %{customdata[1]}<br>" +
            "<b>To:</b> %{customdata[2]}<br>" +
            "<b>Status:</b> %{customdata[3]}<br>" +
            "<b>Anomaly:</b> %{customdata[4]}"
        )
        
        # Highlight anomalies with markers and improved styling
        for i, trace in enumerate(fig.data):
            is_anomaly_trace = i == 1  # Typically the second trace contains anomalies
            
            if is_anomaly_trace:
                # Highlight anomalies with distinct styling
                fig.data[i].marker.line = dict(width=2, color='white')
                fig.data[i].name = "Anomalous Transactions"
            else:
                fig.data[i].marker.line = dict(width=0.5, color='rgba(50, 50, 50, 0.5)')
                fig.data[i].name = "Normal Transactions"
        
        return fig
    
    # If we don't have transaction values, fall back to a simpler visualization
    # Create a histogram of anomalies by index
    anomaly_df = pd.DataFrame({
        'index': np.arange(len(vis_df)),
        'is_anomaly': vis_df['is_anomaly']
    })
    
    fig = px.histogram(
        anomaly_df, x='index', color='is_anomaly',
        color_discrete_map={False: 'rgba(99, 110, 250, 0.7)', True: 'rgba(239, 85, 59, 0.9)'},
        title='Anomaly Distribution',
        template='plotly_dark'
    )
    
    return fig

def plot_transaction_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Create a timeline visualization of blockchain transactions.
    
    Args:
        df: DataFrame containing transaction data
    
    Returns:
        Plotly Figure object containing the timeline visualization
    """
    # Check if we have timestamp data
    if 'timestamp' not in df.columns:
        # Create a placeholder timeline using transaction indices
        df = df.copy()
        df['transaction_index'] = np.arange(len(df))
        
        if 'value' in df.columns:
            fig = px.line(
                df, x='transaction_index', y='value',
                title='Transaction Value Timeline',
                template='plotly_dark'
            )
        else:
            # If we don't have values either, just show transaction counts
            transaction_counts = pd.DataFrame({
                'transaction_index': np.arange(len(df)), 
                'count': 1
            }).rolling(window=10).sum()
            
            fig = px.line(
                transaction_counts, x='transaction_index', y='count',
                title='Transaction Count Timeline (Rolling Window)',
                template='plotly_dark'
            )
        
        fig.update_layout(xaxis_title='Transaction Index', yaxis_title='Value/Count')
        return fig
    
    # If we have timestamp data, create a proper timeline
    timeline_df = df.copy()
    
    # Ensure timestamp is in datetime format
    if not pd.api.types.is_datetime64_dtype(timeline_df['timestamp']):
        timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])
    
    # Sort by timestamp
    timeline_df = timeline_df.sort_values('timestamp')
    
    # Create a timeline visualization
    if 'value' in timeline_df.columns:
        # If we have value data, create a value timeline (using 'h' instead of deprecated 'H')
        timeline_df['hour'] = timeline_df['timestamp'].dt.floor('1h')
        hourly_values = timeline_df.groupby('hour')['value'].sum().reset_index()
        
        fig = px.line(
            hourly_values, x='hour', y='value',
            title='Hourly Transaction Value Timeline',
            template='plotly_dark'
        )
        
        # Add a second line for transaction counts
        hourly_counts = timeline_df.groupby('hour').size().reset_index(name='count')
        fig.add_trace(
            go.Scatter(
                x=hourly_counts['hour'], 
                y=hourly_counts['count'],
                mode='lines',
                name='Transaction Count',
                line=dict(color='rgba(255, 165, 0, 0.8)'),
                yaxis='y2'
            )
        )
        
        # Update layout to include secondary y-axis
        fig.update_layout(
            yaxis2=dict(
                title='Count',
                overlaying='y',
                side='right'
            ),
            xaxis_title='Time',
            yaxis_title='Value',
            legend_title='Metric'
        )
    else:
        # If we don't have value data, just show transaction counts
        timeline_df['hour'] = timeline_df['timestamp'].dt.floor('H')
        hourly_counts = timeline_df.groupby('hour').size().reset_index(name='count')
        
        fig = px.line(
            hourly_counts, x='hour', y='count',
            title='Hourly Transaction Count Timeline',
            template='plotly_dark'
        )
        
        fig.update_layout(xaxis_title='Time', yaxis_title='Transaction Count')
    
    return fig
