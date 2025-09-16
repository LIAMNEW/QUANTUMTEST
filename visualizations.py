import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import List, Dict

def plot_transaction_network(df: pd.DataFrame) -> go.Figure:
    """
    Create an enhanced, modern transaction network visualization with quantum-themed styling.
    
    Args:
        df: DataFrame containing transaction data
    
    Returns:
        Plotly Figure object containing the enhanced network visualization
    """
    # Create a graph from transactions
    G = nx.from_pandas_edgelist(
        df, 'from_address', 'to_address', 
        edge_attr=['value'] if 'value' in df.columns else None,
        create_using=nx.DiGraph()
    )
    
    if len(G.nodes()) == 0:
        # Return empty figure if no data
        return go.Figure().update_layout(
            title="No transaction data available",
            template='plotly_dark'
        )
    
    # Use hierarchical layout for better visual structure
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
    except:
        # Fallback to spring layout with optimized parameters
        pos = nx.spring_layout(G, seed=42, k=2.0, iterations=50)
    
    # Calculate enhanced node metrics
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    # Calculate centrality for importance weighting
    try:
        centrality = nx.betweenness_centrality(G)
    except:
        centrality = {node: degrees[node] / max_degree for node in G.nodes()}
    
    # Enhanced node sizing with quantum glow effect
    base_size = 20
    node_sizes = {}
    node_colors = {}
    for node in G.nodes():
        importance = centrality.get(node, 0)
        degree = degrees[node]
        # Size based on both degree and centrality
        node_sizes[node] = base_size + (degree / max_degree * 30) + (importance * 20)
        # Color intensity based on importance
        node_colors[node] = importance
    
    # Calculate transaction flows for enhanced edge styling
    edge_flows = {}
    edge_values = {}
    if 'value' in df.columns:
        for edge in G.edges():
            transactions = df[(df['from_address'] == edge[0]) & (df['to_address'] == edge[1])]
            if not transactions.empty:
                total_value = transactions['value'].sum()
                edge_values[edge] = total_value
                edge_flows[edge] = len(transactions)
    
    max_value = max(edge_values.values()) if edge_values else 1
    max_flow = max(edge_flows.values()) if edge_flows else 1
    
    # Create quantum-themed background grid effect
    grid_traces = []
    x_range = [min(pos[n][0] for n in pos) - 1, max(pos[n][0] for n in pos) + 1]
    y_range = [min(pos[n][1] for n in pos) - 1, max(pos[n][1] for n in pos) + 1]
    
    # Add subtle grid lines for quantum effect
    for i in np.linspace(x_range[0], x_range[1], 8):
        grid_traces.append(go.Scatter(
            x=[i, i], y=y_range,
            mode='lines',
            line=dict(color='rgba(0, 255, 127, 0.05)', width=0.5),
            showlegend=False, hoverinfo='skip'
        ))
    
    for i in np.linspace(y_range[0], y_range[1], 8):
        grid_traces.append(go.Scatter(
            x=x_range, y=[i, i],
            mode='lines',
            line=dict(color='rgba(0, 255, 127, 0.05)', width=0.5),
            showlegend=False, hoverinfo='skip'
        ))
    
    # Create enhanced edge traces with quantum flow effects
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Calculate edge properties
        value = edge_values.get(edge, 0)
        flow = edge_flows.get(edge, 1)
        
        # Dynamic width and opacity based on transaction value and frequency
        base_width = 1.0
        width = base_width + (value / max_value * 6) if max_value > 0 else base_width
        opacity = 0.3 + (flow / max_flow * 0.5) if max_flow > 0 else 0.3
        
        # Quantum glow color based on value intensity
        if value > max_value * 0.7:
            color = f'rgba(255, 215, 0, {opacity})'  # Gold for high value
            glow_color = 'rgba(255, 215, 0, 0.8)'
        elif value > max_value * 0.3:
            color = f'rgba(0, 255, 127, {opacity})'  # Bright green for medium
            glow_color = 'rgba(0, 255, 127, 0.6)'
        else:
            color = f'rgba(64, 224, 255, {opacity})'  # Cyan for normal
            glow_color = 'rgba(64, 224, 255, 0.4)'
        
        # Create main transaction flow line
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color),
            hovertext=f"<b>Transaction Flow</b><br>From: {edge[0][:8]}...<br>To: {edge[1][:8]}...<br>Value: {value:.2f}<br>Frequency: {flow} transactions",
            hoverinfo='text',
            showlegend=False
        )
        edge_traces.append(edge_trace)
        
        # Add quantum glow effect for important connections
        if value > max_value * 0.5:
            glow_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=width + 3, color=glow_color),
                showlegend=False,
                hoverinfo='skip',
                opacity=0.3
            )
            edge_traces.append(glow_trace)
    
    # Create enhanced node traces with quantum effects
    # Main nodes with gradient effect
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        marker=dict(
            size=[node_sizes[node] for node in G.nodes()],
            color=[node_colors[node] for node in G.nodes()],
            colorscale=[
                [0, 'rgba(25, 25, 112, 0.8)'],      # Midnight blue (low)
                [0.3, 'rgba(0, 191, 255, 0.9)'],    # Deep sky blue
                [0.6, 'rgba(0, 255, 127, 0.95)'],   # Spring green  
                [0.8, 'rgba(255, 215, 0, 0.98)'],   # Gold
                [1, 'rgba(255, 20, 147, 1.0)']      # Deep pink (high)
            ],
            colorbar=dict(
                title=dict(text='<b>Network Importance</b>', font=dict(color='white')),
                tickfont=dict(color='white'),
                x=1.02
            ),
            line=dict(width=2, color='rgba(255, 255, 255, 0.6)'),
            opacity=0.9
        ),
        text=[f"{str(node)[:4]}..." if len(str(node)) > 6 else str(node) for node in G.nodes()],
        textposition="middle center",
        textfont=dict(size=9, color='white', family='Arial Black'),
        hovertext=[
            f"<b>Address:</b> {node}<br><b>Connections:</b> {degrees[node]}<br><b>Importance:</b> {centrality.get(node, 0):.3f}<br><b>Type:</b> {'Hub' if degrees[node] > max_degree * 0.7 else 'Regular'}"
            for node in G.nodes()
        ],
        hoverinfo='text',
        name='🔐 Wallet Addresses'
    )
    
    # Add glow effect for important nodes
    important_nodes = [node for node in G.nodes() if centrality.get(node, 0) > 0.5]
    if important_nodes:
        glow_trace = go.Scatter(
            x=[pos[node][0] for node in important_nodes],
            y=[pos[node][1] for node in important_nodes],
            mode='markers',
            marker=dict(
                size=[node_sizes[node] + 15 for node in important_nodes],
                color='rgba(255, 255, 255, 0.3)',
                line=dict(width=0)
            ),
            showlegend=False,
            hoverinfo='skip'
        )
        edge_traces.append(glow_trace)
    
    # Create the enhanced figure
    fig = go.Figure(
        data=grid_traces + edge_traces + [node_trace],
        layout=go.Layout(
            title={
                'text': '<b>🔗 QuantumGuard Transaction Network Analysis</b>',
                'font': {'size': 26, 'color': 'rgba(0, 255, 127, 1)', 'family': 'Arial Black'},
                'x': 0.5,
                'y': 0.95
            },
            showlegend=True,
            hovermode='closest',
            margin=dict(b=50, l=20, r=20, t=80),
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                showspikes=False
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                showspikes=False
            ),
            plot_bgcolor='rgba(5, 15, 25, 1.0)',
            paper_bgcolor='rgba(5, 15, 25, 1.0)',
            font=dict(color='white'),
            legend=dict(
                x=0.01, 
                y=0.99,
                bgcolor='rgba(15, 25, 35, 0.9)',
                bordercolor='rgba(0, 255, 127, 0.5)',
                borderwidth=1,
                font=dict(color='white')
            ),
            annotations=[
                dict(
                    text="<b>🔬 Quantum Analytics:</b> Node size = Network importance | Edge thickness = Transaction value | Color intensity = Centrality",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.02,
                    align="center",
                    font=dict(size=11, color='rgba(0, 255, 127, 0.9)')
                ),
                dict(
                    text="<b>⚡ QuantumGuard AI</b>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.99, y=0.01,
                    align="right",
                    font=dict(size=10, color='rgba(255, 215, 0, 0.7)')
                )
            ],
            # Add subtle animation on hover
            transition=dict(duration=300, easing="cubic-in-out")
        )
    )
    
    # Add custom hover effects
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><extra></extra>',
        selector=dict(mode='markers')
    )
    
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
            template='plotly_dark',
            labels={'hour': 'Time', 'value': 'Transaction Value'},
            height=500
        )
        
        # Improve formatting
        fig.update_layout(
            title={
                'text': 'Hourly Transaction Value Timeline',
                'font': {'size': 22},
                'x': 0.5,
                'y': 0.95
            },
            xaxis_title='Time',
            yaxis_title='Total Value',
            hovermode='x unified',
            margin=dict(l=20, r=20, t=60, b=40)
        )
        
        # Add range slider for better time navigation
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            )
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
