import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import List, Dict, Optional
from datetime import datetime, date

# Import database function for watchlist checking
try:
    from database import check_addresses_against_watchlist
except ImportError:
    # Fallback function if database import fails
    def check_addresses_against_watchlist(addresses):
        return []

def filter_data_by_date(df: pd.DataFrame, start_date: Optional[date] = None, end_date: Optional[date] = None, date_column: str = 'timestamp') -> pd.DataFrame:
    """
    Filter DataFrame by date range if dates are provided.
    
    Args:
        df: DataFrame to filter
        start_date: Start date for filtering (inclusive)
        end_date: End date for filtering (inclusive)
        date_column: Name of the date column to filter on
    
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
        
    # Check if date column exists
    if date_column not in df.columns:
        return df
    
    # If no date filters provided, return original data
    if start_date is None and end_date is None:
        return df
    
    filtered_df = df.copy()
    
    # Ensure the date column is in datetime format
    if not pd.api.types.is_datetime64_dtype(filtered_df[date_column]):
        try:
            filtered_df[date_column] = pd.to_datetime(filtered_df[date_column])
        except:
            return df  # Return original if conversion fails
    
    # Apply date filters
    if start_date is not None:
        start_datetime = pd.to_datetime(start_date)
        filtered_df = filtered_df[filtered_df[date_column] >= start_datetime]
    
    if end_date is not None:
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Include full end day
        filtered_df = filtered_df[filtered_df[date_column] <= end_datetime]
    
    return filtered_df

def enhance_addresses_with_watchlist(addresses: List[str]) -> Dict[str, Dict]:
    """
    Check addresses against watchlist and return enhanced display information.
    
    Args:
        addresses: List of full addresses to check
        
    Returns:
        Dictionary mapping addresses to their display information including watchlist status
    """
    try:
        # Check addresses against watchlist
        watchlist_matches = check_addresses_against_watchlist(addresses)
        watchlist_dict = {match['address']: match for match in watchlist_matches}
        
        enhanced = {}
        for addr in addresses:
            if addr in watchlist_dict:
                # Address is in watchlist
                watchlist_info = watchlist_dict[addr]
                risk_colors = {
                    'Low': 'rgba(34, 197, 94, 0.9)',       # Green
                    'Medium': 'rgba(251, 191, 36, 0.9)',   # Yellow
                    'High': 'rgba(251, 146, 60, 0.9)',     # Orange  
                    'Critical': 'rgba(239, 68, 68, 0.9)'   # Red
                }
                enhanced[addr] = {
                    'display_name': f"🏷️ {addr[:8]}...",
                    'is_watchlisted': True,
                    'label': watchlist_info['label'],
                    'risk_level': watchlist_info['risk_level'],
                    'color': risk_colors.get(watchlist_info['risk_level'], 'rgba(156, 163, 175, 0.8)'),
                    'hover_extra': f"<br><b>Watchlist:</b> {watchlist_info['label']}<br><b>Risk Level:</b> {watchlist_info['risk_level']}"
                }
            else:
                # Regular address
                enhanced[addr] = {
                    'display_name': f"{addr[:8]}...",
                    'is_watchlisted': False,
                    'label': None,
                    'risk_level': None,
                    'color': 'rgba(156, 163, 175, 0.8)',  # Default gray
                    'hover_extra': ""
                }
        
        return enhanced
    except Exception as e:
        # Fallback if watchlist checking fails
        return {addr: {
            'display_name': f"{addr[:8]}...",
            'is_watchlisted': False,
            'label': None,
            'risk_level': None,
            'color': 'rgba(156, 163, 175, 0.8)',
            'hover_extra': ""
        } for addr in addresses}

def plot_transaction_network(df: pd.DataFrame, start_date: Optional[date] = None, end_date: Optional[date] = None) -> go.Figure:
    """
    Create a simple, easy-to-understand transaction overview chart.
    
    Args:
        df: DataFrame containing transaction data
        start_date: Start date for filtering (optional)
        end_date: End date for filtering (optional)
    
    Returns:
        Plotly Figure object containing the simple transaction overview
    """
    # Apply date filtering if provided
    df = filter_data_by_date(df, start_date, end_date)
    
    if df.empty:
        return go.Figure().update_layout(
            title="No transaction data available for selected date range",
            plot_bgcolor='rgba(22, 25, 37, 1)',
            paper_bgcolor='rgba(22, 25, 37, 1)',
            font=dict(color='white')
        )
    
    # Create simple transaction overview with top wallets and transaction flow
    if 'value' in df.columns:
        # Top sending wallets
        top_senders = df.groupby('from_address')['value'].sum().sort_values(ascending=False).head(10)
        # Top receiving wallets  
        top_receivers = df.groupby('to_address')['value'].sum().sort_values(ascending=False).head(10)
        
        # Get enhanced address information for watchlist checking
        all_addresses = list(set(list(top_senders.index) + list(top_receivers.index)))
        address_info = enhance_addresses_with_watchlist(all_addresses)
        
        # Create a simple bar chart showing transaction volume by top addresses
        fig = go.Figure()
        
        # Add top senders with enhanced colors and labels
        sender_colors = [address_info[addr]['color'] if address_info[addr]['is_watchlisted'] 
                        else 'rgba(34, 197, 94, 0.8)' for addr in top_senders.index]
        sender_display_names = [address_info[addr]['display_name'] for addr in top_senders.index]
        sender_hover_extras = [address_info[addr]['hover_extra'] for addr in top_senders.index]
        
        fig.add_trace(go.Bar(
            name='Top Senders',
            x=sender_display_names,
            y=top_senders.values,
            marker=dict(
                color=sender_colors,
                line=dict(color='rgba(34, 197, 94, 1)', width=1)
            ),
            hovertemplate='<b>Sender:</b> %{x}<br><b>Total Sent:</b> $%{y:,.2f}%{customdata}<extra></extra>',
            customdata=sender_hover_extras
        ))
        
        # Add top receivers with enhanced colors and labels
        receiver_colors = [address_info[addr]['color'] if address_info[addr]['is_watchlisted'] 
                          else 'rgba(59, 130, 246, 0.8)' for addr in top_receivers.index]
        receiver_display_names = [address_info[addr]['display_name'] for addr in top_receivers.index]
        receiver_hover_extras = [address_info[addr]['hover_extra'] for addr in top_receivers.index]
        
        fig.add_trace(go.Bar(
            name='Top Receivers',
            x=receiver_display_names,
            y=top_receivers.values,
            marker=dict(
                color=receiver_colors,
                line=dict(color='rgba(59, 130, 246, 1)', width=1)
            ),
            hovertemplate='<b>Receiver:</b> %{x}<br><b>Total Received:</b> $%{y:,.2f}%{customdata}<extra></extra>',
            customdata=receiver_hover_extras
        ))
        
        title_text = 'Top Transaction Participants'
        
    else:
        # If no value column, show transaction counts
        top_senders = df['from_address'].value_counts().head(10)
        top_receivers = df['to_address'].value_counts().head(10)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Most Active Senders',
            x=[addr[:8] + '...' for addr in top_senders.index],
            y=top_senders.values,
            marker=dict(
                color='rgba(34, 197, 94, 0.8)',
                line=dict(color='rgba(34, 197, 94, 1)', width=1)
            ),
            hovertemplate='<b>Sender:</b> %{x}<br><b>Transactions:</b> %{y}<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='Most Active Receivers',
            x=[addr[:8] + '...' for addr in top_receivers.index],
            y=top_receivers.values,
            marker=dict(
                color='rgba(59, 130, 246, 0.8)',
                line=dict(color='rgba(59, 130, 246, 1)', width=1)
            ),
            hovertemplate='<b>Receiver:</b> %{x}<br><b>Transactions:</b> %{y}<extra></extra>'
        ))
        
        title_text = 'Most Active Transaction Participants'
    
    # Update layout with clean dashboard styling
    fig.update_layout(
        title={
            'text': f'<b>{title_text}</b>',
            'font': {'size': 20, 'color': 'white'},
            'x': 0.5,
            'y': 0.95
        },
        plot_bgcolor='rgba(22, 25, 37, 1)',
        paper_bgcolor='rgba(22, 25, 37, 1)',
        font=dict(color='white', family='Inter, sans-serif'),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(55, 65, 81, 0.3)',
            title='Wallet Addresses',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(55, 65, 81, 0.3)',
            title='Value ($)' if 'value' in df.columns else 'Transaction Count'
        ),
        barmode='group',
        bargap=0.2,
        bargroupgap=0.1,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(31, 41, 55, 0.8)',
            bordercolor='rgba(75, 85, 99, 0.5)',
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=80, b=80),
        height=500
    )
    
    return fig

def plot_risk_heatmap(risk_df: pd.DataFrame, start_date: Optional[date] = None, end_date: Optional[date] = None) -> go.Figure:
    """
    Create a simple risk overview visualization.
    
    Args:
        risk_df: DataFrame containing risk assessment data
        start_date: Start date for filtering (optional)
        end_date: End date for filtering (optional)
    
    Returns:
        Plotly Figure object containing the clean risk overview
    """
    # Apply date filtering if provided
    risk_df = filter_data_by_date(risk_df, start_date, end_date)
    
    if risk_df.empty or 'risk_score' not in risk_df.columns:
        return go.Figure().update_layout(
            title="No risk data available for selected date range",
            plot_bgcolor='rgba(22, 25, 37, 1)',
            paper_bgcolor='rgba(22, 25, 37, 1)',
            font=dict(color='white')
        )
    
    # Create risk level categories
    risk_df = risk_df.copy()
    risk_df['risk_level'] = pd.cut(
        risk_df['risk_score'], 
        bins=[0, 0.3, 0.6, 1.0], 
        labels=['Low Risk', 'Medium Risk', 'High Risk'],
        include_lowest=True
    )
    
    # Count transactions by risk level
    risk_counts = risk_df['risk_level'].value_counts()
    
    # Define colors for each risk level
    colors = {
        'Low Risk': 'rgba(34, 197, 94, 0.8)',     # Green
        'Medium Risk': 'rgba(251, 191, 36, 0.8)', # Yellow  
        'High Risk': 'rgba(239, 68, 68, 0.8)'     # Red
    }
    
    # Create a clean gauge-style visualization
    fig = go.Figure()
    
    # Add bar chart for risk distribution
    fig.add_trace(go.Bar(
        x=risk_counts.index,
        y=risk_counts.values,
        marker=dict(
            color=[colors[level] for level in risk_counts.index],
            line=dict(width=1, color='rgba(255, 255, 255, 0.3)')
        ),
        text=risk_counts.values,
        textposition='auto',
        textfont=dict(size=14, color='white', family='Inter, sans-serif'),
        hovertemplate='<b>%{x}</b><br>Transactions: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>',
        customdata=[(count/len(risk_df)*100) for count in risk_counts.values]
    ))
    
    # Calculate overall risk metrics
    avg_risk = risk_df['risk_score'].mean()
    high_risk_pct = (risk_df['risk_score'] > 0.6).sum() / len(risk_df) * 100
    
    # Update layout with clean styling
    fig.update_layout(
        title={
            'text': f'<b>Risk Assessment Overview</b>',
            'font': {'size': 20, 'color': 'white'},
            'x': 0.5,
            'y': 0.95
        },
        plot_bgcolor='rgba(22, 25, 37, 1)',
        paper_bgcolor='rgba(22, 25, 37, 1)',
        font=dict(color='white', family='Inter, sans-serif'),
        xaxis=dict(
            showgrid=False,
            title='Risk Categories',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(55, 65, 81, 0.3)',
            title='Number of Transactions'
        ),
        showlegend=False,
        margin=dict(l=60, r=40, t=100, b=80),
        height=400,
        annotations=[
            dict(
                text=f"<b>Average Risk Score:</b> {avg_risk:.2f}<br><b>High Risk Transactions:</b> {high_risk_pct:.1f}%",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                align="left",
                font=dict(size=11, color='rgba(255, 255, 255, 0.8)'),
                bgcolor='rgba(31, 41, 55, 0.8)',
                bordercolor='rgba(75, 85, 99, 0.5)',
                borderwidth=1,
                borderpad=10
            )
        ]
    )
    
    return fig

def plot_anomaly_detection(df: pd.DataFrame, anomaly_indices: List[int], start_date: Optional[date] = None, end_date: Optional[date] = None) -> go.Figure:
    """
    Create a simple anomaly overview visualization.
    
    Args:
        df: DataFrame containing transaction data
        anomaly_indices: List of indices corresponding to anomalous transactions
        start_date: Start date for filtering (optional)
        end_date: End date for filtering (optional)
    
    Returns:
        Plotly Figure object containing the simple anomaly overview
    """
    # Apply date filtering if provided
    original_df = df.copy()
    df = filter_data_by_date(df, start_date, end_date)
    
    # Filter anomaly indices to match filtered data
    if not df.empty and anomaly_indices:
        # Get the indices that are still present after filtering
        filtered_indices = df.index.tolist()
        anomaly_indices = [idx for idx in anomaly_indices if idx in filtered_indices]
    
    if df.empty:
        return go.Figure().update_layout(
            title="No transaction data available for selected date range",
            plot_bgcolor='rgba(22, 25, 37, 1)',
            paper_bgcolor='rgba(22, 25, 37, 1)',
            font=dict(color='white')
        )
    
    # Create anomaly overview
    total_transactions = len(df)
    anomalous_transactions = len(anomaly_indices)
    normal_transactions = total_transactions - anomalous_transactions
    anomaly_rate = (anomalous_transactions / total_transactions) * 100 if total_transactions > 0 else 0
    
    # Create a simple donut chart showing normal vs anomalous
    labels = ['Normal Transactions', 'Anomalous Transactions']
    values = [normal_transactions, anomalous_transactions]
    colors = ['rgba(34, 197, 94, 0.8)', 'rgba(239, 68, 68, 0.8)']  # Green, Red
    
    fig = go.Figure()
    
    # Add donut chart
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(
            colors=colors,
            line=dict(color='rgba(255, 255, 255, 0.3)', width=2)
        ),
        textfont=dict(size=14, color='white', family='Inter, sans-serif'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    ))
    
    # Add center text showing anomaly rate
    center_text = f"<b>{anomaly_rate:.1f}%</b><br><span style='font-size:12px'>Anomaly Rate</span>"
    
    fig.update_layout(
        title={
            'text': '<b>Anomaly Detection Overview</b>',
            'font': {'size': 20, 'color': 'white'},
            'x': 0.5,
            'y': 0.95
        },
        plot_bgcolor='rgba(22, 25, 37, 1)',
        paper_bgcolor='rgba(22, 25, 37, 1)',
        font=dict(color='white', family='Inter, sans-serif'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(color='white', size=12)
        ),
        margin=dict(l=40, r=40, t=80, b=80),
        height=400,
        annotations=[
            dict(
                text=center_text,
                x=0.5, y=0.5,
                font=dict(size=16, color='white'),
                showarrow=False,
                align='center'
            ),
            dict(
                text=f"<b>Total Transactions:</b> {total_transactions:,}<br><b>Anomalous:</b> {anomalous_transactions:,}<br><b>Normal:</b> {normal_transactions:,}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                align="left",
                font=dict(size=11, color='rgba(255, 255, 255, 0.8)'),
                bgcolor='rgba(31, 41, 55, 0.8)',
                bordercolor='rgba(75, 85, 99, 0.5)',
                borderwidth=1,
                borderpad=10
            )
        ]
    )
    
    return fig

def plot_transaction_timeline(df: pd.DataFrame, start_date: Optional[date] = None, end_date: Optional[date] = None) -> go.Figure:
    """
    Create a simple transaction timeline visualization.
    
    Args:
        df: DataFrame containing transaction data
        start_date: Start date for filtering (optional)
        end_date: End date for filtering (optional)
    
    Returns:
        Plotly Figure object containing the clean timeline visualization
    """
    # Apply date filtering if provided
    df = filter_data_by_date(df, start_date, end_date)
    
    if df.empty:
        return go.Figure().update_layout(
            title="No transaction data available for selected date range",
            plot_bgcolor='rgba(22, 25, 37, 1)',
            paper_bgcolor='rgba(22, 25, 37, 1)',
            font=dict(color='white')
        )
    
    # Create simple transaction activity timeline
    timeline_df = df.copy()
    
    # Check if we have timestamp data
    if 'timestamp' not in timeline_df.columns:
        # Create simple index-based timeline
        timeline_df['hour'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')[:len(df)]
        if 'value' in df.columns:
            # Group by synthetic hours
            timeline_df['transaction_group'] = (timeline_df.index // 10) * 10  # Group every 10 transactions
            hourly_data = timeline_df.groupby('transaction_group').agg({
                'value': 'sum'
            }).reset_index()
            y_data = hourly_data['value']
            y_title = 'Transaction Value ($)'
        else:
            # Just count transactions
            timeline_df['transaction_group'] = (timeline_df.index // 10) * 10
            hourly_data = timeline_df.groupby('transaction_group').size().reset_index(name='count')
            y_data = hourly_data['count']
            y_title = 'Transaction Count'
        
        x_data = hourly_data['transaction_group']
        x_title = 'Transaction Group'
    else:
        # Use real timestamp data
        if not pd.api.types.is_datetime64_dtype(timeline_df['timestamp']):
            timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])
        
        timeline_df = timeline_df.sort_values('timestamp')
        
        # Group by hour
        timeline_df['hour'] = timeline_df['timestamp'].dt.floor('1h')
        
        if 'value' in timeline_df.columns:
            hourly_data = timeline_df.groupby('hour')['value'].sum().reset_index()
            y_data = hourly_data['value']
            y_title = 'Transaction Value ($)'
        else:
            hourly_data = timeline_df.groupby('hour').size().reset_index(name='count')
            y_data = hourly_data['count']
            y_title = 'Transaction Count'
            
        x_data = hourly_data['hour']
        x_title = 'Time'
    
    # Create a clean line chart
    fig = go.Figure()
    
    # Add main timeline
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines+markers',
        line=dict(
            color='rgba(34, 197, 94, 0.8)',  # Bright green
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=6,
            color='rgba(34, 197, 94, 1)',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1)
        ),
        fill='tonexty',
        fillcolor='rgba(34, 197, 94, 0.1)',
        hovertemplate=f'<b>{x_title}:</b> %{{x}}<br><b>{y_title}:</b> %{{y:,.2f}}<extra></extra>',
        name='Transaction Activity'
    ))
    
    # Calculate trend metrics
    avg_value = np.mean(y_data)
    peak_value = np.max(y_data)
    total_value = np.sum(y_data)
    
    # Update layout with clean styling
    fig.update_layout(
        title={
            'text': '<b>Transaction Activity Timeline</b>',
            'font': {'size': 20, 'color': 'white'},
            'x': 0.5,
            'y': 0.95
        },
        plot_bgcolor='rgba(22, 25, 37, 1)',
        paper_bgcolor='rgba(22, 25, 37, 1)',
        font=dict(color='white', family='Inter, sans-serif'),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(55, 65, 81, 0.3)',
            title=x_title,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(55, 65, 81, 0.3)',
            title=y_title
        ),
        showlegend=False,
        margin=dict(l=60, r=40, t=100, b=80),
        height=400,
        annotations=[
            dict(
                text=f"<b>Peak:</b> {peak_value:,.2f}<br><b>Average:</b> {avg_value:,.2f}<br><b>Total:</b> {total_value:,.2f}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                align="left",
                font=dict(size=11, color='rgba(255, 255, 255, 0.8)'),
                bgcolor='rgba(31, 41, 55, 0.8)',
                bordercolor='rgba(75, 85, 99, 0.5)',
                borderwidth=1,
                borderpad=10
            )
        ]
    )
    
    return fig
