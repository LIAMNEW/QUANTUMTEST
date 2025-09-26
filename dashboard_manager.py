import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum

# Import database functionality
try:
    from database import get_db_connection
    HAS_DB_CONNECTION = True
except ImportError:
    HAS_DB_CONNECTION = False

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

class UserRole(Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"

@dataclass
class DashboardWidget:
    id: str
    title: str
    widget_type: WidgetType
    position: tuple  # (row, col)
    size: tuple  # (width, height)
    config: Dict[str, Any]
    refresh_interval: int = 30  # seconds
    role_permissions: List[UserRole] = None
    
    def __post_init__(self):
        if self.role_permissions is None:
            self.role_permissions = [UserRole.ADMIN, UserRole.ANALYST, UserRole.VIEWER]

@dataclass
class Dashboard:
    id: str
    name: str
    description: str
    widgets: List[DashboardWidget]
    layout: str = "grid"  # grid, tabs, sidebar
    theme: str = "default"
    owner_role: UserRole = UserRole.ADMIN
    is_public: bool = False
    auto_refresh: bool = True
    refresh_rate: int = 30

class DashboardManager:
    """Manages customizable real-time monitoring dashboards"""
    
    def __init__(self):
        self.current_user_role = self._get_current_user_role()
        if HAS_DB_CONNECTION:
            self.init_dashboard_tables()
    
    def init_dashboard_tables(self):
        """Initialize dashboard storage tables"""
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                
                # Create dashboards table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dashboards (
                        id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        layout VARCHAR(50) DEFAULT 'grid',
                        theme VARCHAR(50) DEFAULT 'default',
                        owner_role VARCHAR(50),
                        is_public BOOLEAN DEFAULT false,
                        auto_refresh BOOLEAN DEFAULT true,
                        refresh_rate INTEGER DEFAULT 30,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create dashboard widgets table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dashboard_widgets (
                        id VARCHAR(255) PRIMARY KEY,
                        dashboard_id VARCHAR(255) REFERENCES dashboards(id),
                        title VARCHAR(255) NOT NULL,
                        widget_type VARCHAR(50) NOT NULL,
                        position_row INTEGER,
                        position_col INTEGER,
                        size_width INTEGER,
                        size_height INTEGER,
                        config JSONB,
                        refresh_interval INTEGER DEFAULT 30,
                        role_permissions JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            st.error(f"Error initializing dashboard tables: {e}")
    
    def _get_current_user_role(self) -> UserRole:
        """Get current user's role from session state or default"""
        role_str = st.session_state.get('user_role', 'analyst')
        try:
            return UserRole(role_str.lower())
        except ValueError:
            return UserRole.ANALYST
    
    def create_default_dashboards(self):
        """Create default dashboards for different roles"""
        
        # Admin Dashboard
        admin_widgets = [
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
            ),
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
                position=(3, 0),
                size=(3, 2),
                config={"limit": 10, "severity": "all"}
            )
        ]
        
        admin_dashboard = Dashboard(
            id="admin_overview",
            name="Admin Overview",
            description="Comprehensive system monitoring and analytics",
            widgets=admin_widgets,
            owner_role=UserRole.ADMIN
        )
        
        # Analyst Dashboard
        analyst_widgets = [
            DashboardWidget(
                id="risk_analysis",
                title="Risk Analysis",
                widget_type=WidgetType.RISK_GAUGE,
                position=(0, 0),
                size=(1, 1),
                config={"aggregation": "current"}
            ),
            DashboardWidget(
                id="anomaly_detection",
                title="Anomalies Detected",
                widget_type=WidgetType.METRIC,
                position=(0, 1),
                size=(1, 1),
                config={"metric": "anomaly_count", "time_range": "24h"}
            ),
            DashboardWidget(
                id="transaction_heatmap",
                title="Transaction Heatmap",
                widget_type=WidgetType.HEATMAP,
                position=(1, 0),
                size=(2, 2),
                config={"x_axis": "hour", "y_axis": "day", "value": "count"}
            ),
            DashboardWidget(
                id="top_risks",
                title="High Risk Transactions",
                widget_type=WidgetType.TABLE,
                position=(3, 0),
                size=(3, 1),
                config={"columns": ["hash", "risk_score", "timestamp"], "limit": 20}
            )
        ]
        
        analyst_dashboard = Dashboard(
            id="analyst_workspace",
            name="Analyst Workspace",
            description="Focused analysis and investigation tools",
            widgets=analyst_widgets,
            owner_role=UserRole.ANALYST
        )
        
        # Save default dashboards
        self.save_dashboard(admin_dashboard)
        self.save_dashboard(analyst_dashboard)
    
    def save_dashboard(self, dashboard: Dashboard):
        """Save dashboard configuration to database"""
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                
                # Insert or update dashboard
                cursor.execute("""
                    INSERT INTO dashboards (id, name, description, layout, theme, owner_role, is_public, auto_refresh, refresh_rate)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        layout = EXCLUDED.layout,
                        theme = EXCLUDED.theme,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    dashboard.id, dashboard.name, dashboard.description,
                    dashboard.layout, dashboard.theme, dashboard.owner_role.value,
                    dashboard.is_public, dashboard.auto_refresh, dashboard.refresh_rate
                ))
                
                # Delete existing widgets for this dashboard
                cursor.execute("DELETE FROM dashboard_widgets WHERE dashboard_id = %s", (dashboard.id,))
                
                # Insert widgets
                for widget in dashboard.widgets:
                    cursor.execute("""
                        INSERT INTO dashboard_widgets 
                        (id, dashboard_id, title, widget_type, position_row, position_col, 
                         size_width, size_height, config, refresh_interval, role_permissions)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        widget.id, dashboard.id, widget.title, widget.widget_type.value,
                        widget.position[0], widget.position[1], widget.size[0], widget.size[1],
                        json.dumps(widget.config), widget.refresh_interval,
                        json.dumps([role.value for role in widget.role_permissions])
                    ))
                
                conn.commit()
                
        except Exception as e:
            st.error(f"Error saving dashboard: {e}")
    
    def load_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Load dashboard configuration from database"""
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                
                # Load dashboard
                cursor.execute("SELECT * FROM dashboards WHERE id = %s", (dashboard_id,))
                dashboard_row = cursor.fetchone()
                
                if not dashboard_row:
                    return None
                
                # Load widgets
                cursor.execute("SELECT * FROM dashboard_widgets WHERE dashboard_id = %s", (dashboard_id,))
                widget_rows = cursor.fetchall()
                
                widgets = []
                for row in widget_rows:
                    role_permissions = [UserRole(role) for role in json.loads(row[10] or '[]')]
                    widget = DashboardWidget(
                        id=row[0],
                        title=row[2],
                        widget_type=WidgetType(row[3]),
                        position=(row[4], row[5]),
                        size=(row[6], row[7]),
                        config=json.loads(row[8] or '{}'),
                        refresh_interval=row[9],
                        role_permissions=role_permissions
                    )
                    widgets.append(widget)
                
                return Dashboard(
                    id=dashboard_row[0],
                    name=dashboard_row[1],
                    description=dashboard_row[2],
                    widgets=widgets,
                    layout=dashboard_row[3],
                    theme=dashboard_row[4],
                    owner_role=UserRole(dashboard_row[5]),
                    is_public=dashboard_row[6],
                    auto_refresh=dashboard_row[7],
                    refresh_rate=dashboard_row[8]
                )
                
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")
            return None
    
    def get_available_dashboards(self) -> List[Dict[str, str]]:
        """Get list of dashboards accessible to current user"""
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                
                # Get dashboards based on role
                if self.current_user_role == UserRole.ADMIN:
                    cursor.execute("SELECT id, name, description FROM dashboards ORDER BY name")
                else:
                    cursor.execute("""
                        SELECT id, name, description FROM dashboards 
                        WHERE is_public = true OR owner_role = %s 
                        ORDER BY name
                    """, (self.current_user_role.value,))
                
                return [{"id": row[0], "name": row[1], "description": row[2]} for row in cursor.fetchall()]
                
        except Exception as e:
            st.error(f"Error loading dashboards: {e}")
            return []
    
    def render_widget(self, widget: DashboardWidget, data: Dict[str, Any] = None):
        """Render a dashboard widget"""
        
        # Check role permissions
        if self.current_user_role not in widget.role_permissions:
            st.warning(f"Access denied: {widget.title}")
            return
        
        # Generate mock data if none provided
        if data is None:
            data = self._generate_widget_data(widget)
        
        with st.container():
            st.subheader(widget.title)
            
            if widget.widget_type == WidgetType.METRIC:
                self._render_metric_widget(widget, data)
            elif widget.widget_type == WidgetType.LINE_CHART:
                self._render_line_chart_widget(widget, data)
            elif widget.widget_type == WidgetType.BAR_CHART:
                self._render_bar_chart_widget(widget, data)
            elif widget.widget_type == WidgetType.PIE_CHART:
                self._render_pie_chart_widget(widget, data)
            elif widget.widget_type == WidgetType.HEATMAP:
                self._render_heatmap_widget(widget, data)
            elif widget.widget_type == WidgetType.TABLE:
                self._render_table_widget(widget, data)
            elif widget.widget_type == WidgetType.ALERT_FEED:
                self._render_alert_feed_widget(widget, data)
            elif widget.widget_type == WidgetType.RISK_GAUGE:
                self._render_risk_gauge_widget(widget, data)
            elif widget.widget_type == WidgetType.TRANSACTION_COUNT:
                self._render_transaction_count_widget(widget, data)
            else:
                st.info(f"Widget type {widget.widget_type.value} not implemented yet")
    
    def _generate_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate sample data for widgets"""
        import random
        from datetime import datetime, timedelta
        
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
            for i in range(widget.config.get("limit", 10)):
                alerts.append({
                    "timestamp": datetime.now() - timedelta(hours=random.randint(0, 72)),
                    "severity": random.choice(["Low", "Medium", "High", "Critical"]),
                    "message": f"Suspicious activity detected in transaction #{random.randint(1000, 9999)}",
                    "address": f"0x{random.randint(100000, 999999):06x}...{random.randint(1000, 9999):04x}"
                })
            base_data = {"alerts": alerts}
        
        elif widget.widget_type == WidgetType.TABLE:
            rows = []
            for i in range(widget.config.get("limit", 20)):
                rows.append({
                    "hash": f"0x{random.randint(100000000, 999999999):09x}",
                    "risk_score": random.uniform(0.0, 1.0),
                    "timestamp": datetime.now() - timedelta(hours=random.randint(0, 48)),
                    "amount": random.uniform(0.01, 100.0)
                })
            base_data = {"rows": rows}
        
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
            mode = "gauge+number+delta",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color_map.get(level, "gray")},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
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
    
    def _render_heatmap_widget(self, widget: DashboardWidget, data: Dict[str, Any]):
        """Render heatmap widget - placeholder"""
        st.info("Heatmap visualization coming soon...")
    
    def _render_bar_chart_widget(self, widget: DashboardWidget, data: Dict[str, Any]):
        """Render bar chart widget - placeholder"""
        st.info("Bar chart visualization coming soon...")
    
    def _render_pie_chart_widget(self, widget: DashboardWidget, data: Dict[str, Any]):
        """Render pie chart widget - placeholder"""
        st.info("Pie chart visualization coming soon...")
    
    def render_dashboard(self, dashboard_id: str):
        """Render complete dashboard"""
        dashboard = self.load_dashboard(dashboard_id)
        
        if not dashboard:
            st.error(f"Dashboard '{dashboard_id}' not found")
            return
        
        # Dashboard header
        st.title(f"🏛️ {dashboard.name}")
        st.caption(dashboard.description)
        
        # Auto-refresh toggle
        if dashboard.auto_refresh:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Refresh", key=f"refresh_{dashboard_id}"):
                    st.rerun()
        
        # Render widgets in grid layout
        widget_grid = {}
        for widget in dashboard.widgets:
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
    
    def render_dashboard_selector(self) -> Optional[str]:
        """Render dashboard selector interface"""
        dashboards = self.get_available_dashboards()
        
        if not dashboards:
            st.warning("No dashboards available. Creating default dashboards...")
            self.create_default_dashboards()
            dashboards = self.get_available_dashboards()
        
        # Dashboard selection
        dashboard_options = {d["id"]: f"{d['name']} - {d['description']}" for d in dashboards}
        
        selected_id = st.selectbox(
            "Select Dashboard:",
            options=list(dashboard_options.keys()),
            format_func=lambda x: dashboard_options[x],
            key="dashboard_selector"
        )
        
        return selected_id


# Initialize dashboard manager
dashboard_manager = DashboardManager()