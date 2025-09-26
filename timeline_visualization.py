import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np

class TimelineVisualization:
    """Interactive transaction timeline visualization with zoom capabilities"""
    
    def __init__(self):
        self.default_colors = {
            'normal': '#1f77b4',
            'suspicious': '#ff7f0e', 
            'high_risk': '#d62728',
            'anomaly': '#ff69b4',
            'background': '#f8f9fa'
        }
    
    def create_interactive_timeline(self, df: pd.DataFrame, 
                                  time_column: str = 'timestamp',
                                  value_column: str = 'value',
                                  risk_column: str = 'risk_score',
                                  title: str = "Transaction Timeline") -> go.Figure:
        """Create interactive timeline with zoom and pan capabilities"""
        
        # Ensure timestamp column is datetime
        if df[time_column].dtype == 'object':
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        
        # Create risk categories
        df_copy = df.copy()
        df_copy['risk_category'] = df_copy[risk_column].apply(self._categorize_risk)
        df_copy['color'] = df_copy['risk_category'].map(self._get_risk_colors())
        df_copy['size'] = df_copy[value_column] / df_copy[value_column].max() * 20 + 5
        
        # Create base timeline figure
        fig = go.Figure()
        
        # Add scatter plot for transactions
        for category in df_copy['risk_category'].unique():
            category_data = df_copy[df_copy['risk_category'] == category]
            
            fig.add_trace(go.Scatter(
                x=category_data[time_column],
                y=category_data[value_column],
                mode='markers',
                marker=dict(
                    color=self._get_risk_colors()[category],
                    size=category_data['size'],
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                name=category.title(),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>" +
                    "Time: %{x}<br>" +
                    "Value: %{y:,.2f}<br>" +
                    "Risk Score: %{customdata[0]:.3f}<br>" +
                    "<extra></extra>"
                ),
                customdata=category_data[risk_column].values.reshape(-1, 1)
            ))
        
        # Add volume indicator (aggregated by hour)
        hourly_volume = self._aggregate_hourly_volume(df_copy, time_column, value_column)
        
        fig.add_trace(go.Scatter(
            x=hourly_volume['hour'],
            y=hourly_volume['total_volume'],
            mode='lines+markers',
            line=dict(color='rgba(128,128,128,0.3)', width=1),
            marker=dict(size=3, color='rgba(128,128,128,0.5)'),
            name='Hourly Volume',
            yaxis='y2',
            hovertemplate="Hour: %{x}<br>Total Volume: %{y:,.2f}<extra></extra>"
        ))
        
        # Update layout for interactive features
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            xaxis=dict(
                title="Time",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                rangeslider=dict(visible=True),
                type='date'
            ),
            yaxis=dict(
                title="Transaction Value",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                type='log'
            ),
            yaxis2=dict(
                title="Hourly Volume",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='closest',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(128,128,128,0.3)',
                borderwidth=1
            ),
            plot_bgcolor='white',
            height=600,
            margin=dict(t=80, r=80, b=120, l=80)
        )
        
        # Add range selector buttons
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1H", step="hour", stepmode="backward"),
                        dict(count=6, label="6H", step="hour", stepmode="backward"),
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=7, label="7D", step="day", stepmode="backward"),
                        dict(count=30, label="1M", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                type='date'
            )
        )
        
        return fig
    
    def create_risk_timeline(self, df: pd.DataFrame, 
                           time_column: str = 'timestamp',
                           risk_column: str = 'risk_score') -> go.Figure:
        """Create timeline focused on risk score evolution"""
        
        # Aggregate risk by time periods
        df_copy = df.copy()
        df_copy[time_column] = pd.to_datetime(df_copy[time_column])
        
        # Create time bins
        df_copy['time_bin'] = df_copy[time_column].dt.floor('H')  # Hourly bins
        risk_timeline = df_copy.groupby('time_bin').agg({
            risk_column: ['mean', 'max', 'std', 'count']
        }).reset_index()
        
        risk_timeline.columns = ['time', 'avg_risk', 'max_risk', 'risk_std', 'count']
        risk_timeline['risk_std'] = risk_timeline['risk_std'].fillna(0)
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Risk Score Timeline', 'Transaction Volume'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Risk score with confidence bands
        fig.add_trace(go.Scatter(
            x=risk_timeline['time'],
            y=risk_timeline['avg_risk'] + risk_timeline['risk_std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=risk_timeline['time'],
            y=risk_timeline['avg_risk'] - risk_timeline['risk_std'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255,127,14,0.2)',
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)
        
        # Average risk line
        fig.add_trace(go.Scatter(
            x=risk_timeline['time'],
            y=risk_timeline['avg_risk'],
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=4),
            name='Average Risk',
            hovertemplate="Time: %{x}<br>Avg Risk: %{y:.3f}<extra></extra>"
        ), row=1, col=1)
        
        # Maximum risk line
        fig.add_trace(go.Scatter(
            x=risk_timeline['time'],
            y=risk_timeline['max_risk'],
            mode='lines',
            line=dict(color='#d62728', width=1, dash='dash'),
            name='Maximum Risk',
            hovertemplate="Time: %{x}<br>Max Risk: %{y:.3f}<extra></extra>"
        ), row=1, col=1)
        
        # Transaction count
        fig.add_trace(go.Bar(
            x=risk_timeline['time'],
            y=risk_timeline['count'],
            name='Transaction Count',
            marker_color='rgba(31,119,180,0.6)',
            hovertemplate="Time: %{x}<br>Count: %{y}<extra></extra>"
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title="Risk Analysis Timeline",
            height=600,
            hovermode='x unified',
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor='center')
        )
        
        # Add risk threshold line
        fig.add_hline(
            y=0.7, line_dash="dot", line_color="red",
            annotation_text="High Risk Threshold",
            row=1, col=1
        )
        
        return fig
    
    def create_network_activity_timeline(self, df: pd.DataFrame,
                                       time_column: str = 'timestamp',
                                       from_column: str = 'from_address',
                                       to_column: str = 'to_address') -> go.Figure:
        """Create timeline showing network activity patterns"""
        
        df_copy = df.copy()
        df_copy[time_column] = pd.to_datetime(df_copy[time_column])
        
        # Create hourly bins
        df_copy['hour'] = df_copy[time_column].dt.floor('H')
        
        # Calculate network metrics per hour
        count_column = 'transaction_hash' if 'transaction_hash' in df_copy.columns else time_column
        hourly_stats = df_copy.groupby('hour').agg({
            from_column: 'nunique',
            to_column: 'nunique',
            count_column: 'count'
        }).reset_index()
        
        hourly_stats.columns = ['hour', 'unique_senders', 'unique_receivers', 'transaction_count']
        hourly_stats['total_unique_addresses'] = hourly_stats['unique_senders'] + hourly_stats['unique_receivers']
        
        # Create figure
        fig = go.Figure()
        
        # Add transaction count
        fig.add_trace(go.Scatter(
            x=hourly_stats['hour'],
            y=hourly_stats['transaction_count'],
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
            name='Transactions',
            yaxis='y',
            hovertemplate="Hour: %{x}<br>Transactions: %{y}<extra></extra>"
        ))
        
        # Add unique addresses
        fig.add_trace(go.Scatter(
            x=hourly_stats['hour'],
            y=hourly_stats['total_unique_addresses'],
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=4),
            name='Unique Addresses',
            yaxis='y2',
            hovertemplate="Hour: %{x}<br>Unique Addresses: %{y}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title="Network Activity Timeline",
            xaxis=dict(
                title="Time",
                rangeslider=dict(visible=True)
            ),
            yaxis=dict(
                title="Transaction Count",
                side='left'
            ),
            yaxis2=dict(
                title="Unique Addresses",
                side='right',
                overlaying='y'
            ),
            height=400,
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def render_timeline_controls(self) -> Dict[str, Any]:
        """Render timeline control panel"""
        
        st.subheader("📊 Timeline Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            chart_type = st.selectbox(
                "Chart Type:",
                ["Transaction Timeline", "Risk Timeline", "Network Activity"],
                key="timeline_chart_type"
            )
        
        with col2:
            time_resolution = st.selectbox(
                "Time Resolution:",
                ["15min", "1hour", "6hour", "1day"],
                index=1,
                key="timeline_resolution"
            )
        
        with col3:
            color_by = st.selectbox(
                "Color By:",
                ["Risk Level", "Transaction Size", "Address Type"],
                key="timeline_color"
            )
        
        with col4:
            show_volume = st.checkbox(
                "Show Volume",
                value=True,
                key="timeline_show_volume"
            )
        
        # Advanced controls
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                log_scale = st.checkbox("Logarithmic Y-axis", key="timeline_log_scale")
                show_trends = st.checkbox("Show Trend Lines", key="timeline_trends")
            
            with col2:
                highlight_anomalies = st.checkbox("Highlight Anomalies", key="timeline_anomalies")
                animation_speed = st.slider("Animation Speed", 100, 2000, 500, key="timeline_animation")
        
        return {
            "chart_type": chart_type,
            "time_resolution": time_resolution,
            "color_by": color_by,
            "show_volume": show_volume,
            "log_scale": log_scale,
            "show_trends": show_trends,
            "highlight_anomalies": highlight_anomalies,
            "animation_speed": animation_speed
        }
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into levels"""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high_risk'
        elif risk_score >= 0.4:
            return 'medium_risk'
        elif risk_score >= 0.2:
            return 'low_risk'
        else:
            return 'normal'
    
    def _get_risk_colors(self) -> Dict[str, str]:
        """Get color mapping for risk categories"""
        return {
            'normal': '#2ca02c',
            'low_risk': '#17becf', 
            'medium_risk': '#ff7f0e',
            'high_risk': '#d62728',
            'critical': '#8b0000'
        }
    
    def _aggregate_hourly_volume(self, df: pd.DataFrame, 
                                time_column: str, value_column: str) -> pd.DataFrame:
        """Aggregate transaction volume by hour"""
        
        df_hourly = df.copy()
        df_hourly['hour'] = pd.to_datetime(df_hourly[time_column]).dt.floor('H')
        
        hourly_volume = df_hourly.groupby('hour').agg({
            value_column: ['sum', 'count', 'mean']
        }).reset_index()
        
        hourly_volume.columns = ['hour', 'total_volume', 'transaction_count', 'avg_volume']
        
        return hourly_volume


# Initialize timeline visualization
timeline_viz = TimelineVisualization()