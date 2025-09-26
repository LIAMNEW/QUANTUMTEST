import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass
from enum import Enum

class WidgetType(Enum):
    METRIC = "metric"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    HEATMAP = "heatmap"
    TABLE = "table"
    ALERT_FEED = "alert_feed"
    NETWORK_GRAPH = "network_graph"
    RISK_GAUGE = "risk_gauge"
    TRANSACTION_COUNT = "transaction_count"

@dataclass
class DashboardWidget:
    id: str
    title: str
    widget_type: WidgetType
    position: tuple  # (row, col)
    size: tuple  # (width, height)
    config: Dict[str, Any]
    refresh_interval: int = 30  # seconds

class DashboardManager:
    """Simple dashboard manager for real-time monitoring"""
    
    def __init__(self):
        self.default_colors = {
            'normal': '#1f77b4',
            'suspicious': '#ff7f0e', 
            'high_risk': '#d62728',
            'anomaly': '#ff69b4'
        }
    
    def create_overview_dashboard(self, df: Optional[pd.DataFrame] = None) -> List[DashboardWidget]:
        """Create overview dashboard widgets"""
        
        widgets = [
            DashboardWidget(
                id="system_health",
                title="System Health",
                widget_type=WidgetType.METRIC,
                position=(0, 0),
                size=(1, 1),
                config={"metric": "system_status", "format": "status"}
            ),
            DashboardWidget(
                id="total_transactions",
                title="Total Transactions",
                widget_type=WidgetType.TRANSACTION_COUNT,
                position=(0, 1),
                size=(1, 1),
                config={"time_range": "24h"}
            ),
            DashboardWidget(
                id="risk_overview",
                title="Risk Overview",
                widget_type=WidgetType.RISK_GAUGE,
                position=(0, 2),
                size=(1, 1),
                config={"aggregation": "average"}
            )
        ]
        
        if df is not None and not df.empty:
            widgets.extend([
                DashboardWidget(
                    id="transaction_timeline",
                    title="Transaction Timeline",
                    widget_type=WidgetType.LINE_CHART,
                    position=(1, 0),
                    size=(3, 2),
                    config={"x_axis": "timestamp", "y_axis": "count", "time_range": "7d"}
                ),
                DashboardWidget(
                    id="alerts_feed",
                    title="Recent Alerts",
                    widget_type=WidgetType.ALERT_FEED,
                    position=(2, 0),
                    size=(3, 2),
                    config={"limit": 10, "severity": "all"}
                )
            ])
        
        return widgets
    
    def render_widget(self, widget: DashboardWidget, data: Dict[str, Any] = None):
        """Render a dashboard widget"""
        
        # Generate mock data if none provided
        if data is None:
            data = self._generate_widget_data(widget)
        
        with st.container():
            st.subheader(widget.title)
            
            if widget.widget_type == WidgetType.METRIC:
                self._render_metric_widget(widget, data)
            elif widget.widget_type == WidgetType.LINE_CHART:
                self._render_line_chart_widget(widget, data)
            elif widget.widget_type == WidgetType.RISK_GAUGE:
                self._render_risk_gauge_widget(widget, data)
            elif widget.widget_type == WidgetType.TRANSACTION_COUNT:
                self._render_transaction_count_widget(widget, data)
            elif widget.widget_type == WidgetType.ALERT_FEED:
                self._render_alert_feed_widget(widget, data)
            elif widget.widget_type == WidgetType.TABLE:
                self._render_table_widget(widget, data)
            else:
                st.info(f"Widget type {widget.widget_type.value} not implemented yet")
    
    def _generate_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate sample data for widgets"""
        import random
        
        base_data = {}
        
        if widget.widget_type == WidgetType.METRIC:
            base_data = {
                "value": random.randint(100, 10000),
                "change": random.uniform(-5.0, 15.0),
                "format": widget.config.get("format", "number")
            }
        
        elif widget.widget_type == WidgetType.LINE_CHART:
            dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
            values = [random.randint(50, 200) for _ in dates]
            base_data = {
                "x": dates,
                "y": values,
                "title": widget.title
            }
        
        elif widget.widget_type == WidgetType.TRANSACTION_COUNT:
            base_data = {
                "count": random.randint(1000, 50000),
                "change": random.uniform(-10.0, 25.0),
                "time_range": widget.config.get("time_range", "24h")
            }
        
        elif widget.widget_type == WidgetType.RISK_GAUGE:
            base_data = {
                "risk_score": random.uniform(0.0, 1.0),
                "level": random.choice(["Low", "Medium", "High", "Critical"])
            }
        
        elif widget.widget_type == WidgetType.ALERT_FEED:
            alerts = []
            for i in range(widget.config.get("limit", 5)):
                alerts.append({
                    "timestamp": datetime.now() - timedelta(hours=random.randint(0, 72)),
                    "severity": random.choice(["Low", "Medium", "High", "Critical"]),
                    "message": f"Suspicious activity detected in transaction #{random.randint(1000, 9999)}",
                    "address": f"0x{random.randint(100000, 999999):06x}...{random.randint(1000, 9999):04x}"
                })
            base_data = {"alerts": alerts}
        
        return base_data
    
    def _render_metric_widget(self, widget: DashboardWidget, data: Dict[str, Any]):
        """Render a metric widget"""
        value = data.get("value", 0)
        change = data.get("change", 0)
        format_type = data.get("format", "number")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if format_type == "currency":
                st.metric("", f"${value:,.2f}", f"{change:+.1f}%")
            elif format_type == "percentage":
                st.metric("", f"{value:.1f}%", f"{change:+.1f}%")
            else:
                st.metric("", f"{value:,}", f"{change:+.1f}%")
        
        with col2:
            if change > 0:
                st.success("📈")
            elif change < 0:
                st.error("📉")
            else:
                st.info("➡️")
    
    def _render_line_chart_widget(self, widget: DashboardWidget, data: Dict[str, Any]):
        """Render a line chart widget"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.get("x", []),
            y=data.get("y", []),
            mode='lines+markers',
            name=data.get("title", "Data"),
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
            xaxis_title="",
            yaxis_title="",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_risk_gauge_widget(self, widget: DashboardWidget, data: Dict[str, Any]):
        """Render a risk gauge widget"""
        risk_score = data.get("risk_score", 0.0)
        level = data.get("level", "Low")
        
        # Color mapping
        color_map = {
            "Low": "green",
            "Medium": "yellow", 
            "High": "orange",
            "Critical": "red"
        }
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score * 100,
            title = {'text': f"Risk Level: {level}"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color_map.get(level, "gray")},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ]
            }
        ))
        
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_transaction_count_widget(self, widget: DashboardWidget, data: Dict[str, Any]):
        """Render transaction count widget"""
        count = data.get("count", 0)
        change = data.get("change", 0)
        time_range = data.get("time_range", "24h")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.metric(f"Transactions ({time_range})", f"{count:,}", f"{change:+.1f}%")
        with col2:
            if change > 0:
                st.success("📈")
            else:
                st.error("📉")
    
    def _render_alert_feed_widget(self, widget: DashboardWidget, data: Dict[str, Any]):
        """Render alert feed widget"""
        alerts = data.get("alerts", [])
        
        for alert in alerts:
            severity_colors = {
                "Low": "🟢",
                "Medium": "🟡", 
                "High": "🟠",
                "Critical": "🔴"
            }
            
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(severity_colors.get(alert["severity"], "⚪"))
                with col2:
                    st.write(f"**{alert['message']}**")
                    st.caption(f"Address: {alert['address']} | {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                st.divider()
    
    def _render_table_widget(self, widget: DashboardWidget, data: Dict[str, Any]):
        """Render table widget"""
        rows = data.get("rows", [])
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=300)
    
    def render_dashboard(self, df: Optional[pd.DataFrame] = None):
        """Render complete dashboard"""
        
        # Dashboard header
        st.title("📊 Live Dashboard")
        
        # Auto-refresh toggle
        col1, col2, col3 = st.columns([2, 1, 1])
        with col3:
            if st.button("🔄 Refresh", key="refresh_dashboard"):
                st.rerun()
        
        # Create and render widgets
        widgets = self.create_overview_dashboard(df)
        
        # Render widgets in grid layout
        widget_grid = {}
        for widget in widgets:
            row, col = widget.position
            if row not in widget_grid:
                widget_grid[row] = {}
            widget_grid[row][col] = widget
        
        # Render rows
        for row_idx in sorted(widget_grid.keys()):
            row_widgets = widget_grid[row_idx]
            
            # Calculate columns needed
            max_cols = max(row_widgets.keys()) + 1 if row_widgets else 1
            cols = st.columns(max_cols)
            
            for col_idx, widget in row_widgets.items():
                with cols[col_idx]:
                    self.render_widget(widget)


# Initialize simple dashboard manager
dashboard_manager = DashboardManager()