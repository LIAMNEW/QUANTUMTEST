import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta
from enum import Enum

class FilterOperator(Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    BETWEEN = "between"
    IN = "in"

class QueryBuilder:
    """Simple visual query builder for advanced search and filtering"""
    
    def __init__(self):
        self.field_types = {
            'from_address': 'text',
            'to_address': 'text',
            'value': 'number',
            'timestamp': 'date',
            'transaction_hash': 'text',
            'block_number': 'number',
            'gas_price': 'number',
            'gas_used': 'number',
            'risk_score': 'number',
            'is_anomaly': 'boolean',
            'blockchain': 'text',
            'source': 'text'
        }
        
        self.operator_labels = {
            FilterOperator.EQUALS: "Equals",
            FilterOperator.NOT_EQUALS: "Not Equals",
            FilterOperator.GREATER_THAN: "Greater Than",
            FilterOperator.LESS_THAN: "Less Than",
            FilterOperator.CONTAINS: "Contains",
            FilterOperator.NOT_CONTAINS: "Does Not Contain",
            FilterOperator.BETWEEN: "Between",
            FilterOperator.IN: "In List"
        }
    
    def render_query_builder(self) -> Optional[Dict[str, Any]]:
        """Render simple query builder interface"""
        
        st.subheader("🔍 Advanced Search")
        
        # Quick filters
        with st.expander("⚡ Quick Filters", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔴 High Risk", key="quick_high_risk"):
                    return {"type": "quick", "filter": "high_risk"}
            
            with col2:
                if st.button("⚠️ Anomalies", key="quick_anomalies"):
                    return {"type": "quick", "filter": "anomalies"}
            
            with col3:
                if st.button("📅 Last 24h", key="quick_24h"):
                    return {"type": "quick", "filter": "last_24h"}
            
            with col4:
                if st.button("💰 High Value", key="quick_high_value"):
                    return {"type": "quick", "filter": "high_value"}
        
        # Manual filters
        st.markdown("### 🎯 Custom Filters")
        
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            field = st.selectbox(
                "Field",
                options=list(self.field_types.keys()),
                key="filter_field",
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        field_type = self.field_types.get(field, 'text')
        available_operators = self._get_operators_for_type(field_type)
        
        with col2:
            operator = st.selectbox(
                "Operator",
                options=available_operators,
                key="filter_operator",
                format_func=lambda x: self.operator_labels[x]
            )
        
        with col3:
            value = self._render_value_input(field, field_type, operator)
        
        # Apply filter button
        if st.button("🔍 Apply Filter", key="apply_filter", type="primary"):
            if field and operator is not None and value is not None:
                return {
                    "type": "custom",
                    "field": field,
                    "operator": operator,
                    "value": value,
                    "field_type": field_type
                }
        
        return None
    
    def _get_operators_for_type(self, field_type: str) -> List[FilterOperator]:
        """Get available operators for a field type"""
        if field_type == 'text':
            return [FilterOperator.EQUALS, FilterOperator.NOT_EQUALS, FilterOperator.CONTAINS, FilterOperator.NOT_CONTAINS]
        elif field_type == 'number':
            return [FilterOperator.EQUALS, FilterOperator.NOT_EQUALS, FilterOperator.GREATER_THAN, FilterOperator.LESS_THAN, FilterOperator.BETWEEN]
        elif field_type == 'date':
            return [FilterOperator.EQUALS, FilterOperator.GREATER_THAN, FilterOperator.LESS_THAN, FilterOperator.BETWEEN]
        elif field_type == 'boolean':
            return [FilterOperator.EQUALS, FilterOperator.NOT_EQUALS]
        else:
            return list(FilterOperator)
    
    def _render_value_input(self, field: str, field_type: str, operator: FilterOperator):
        """Render appropriate input widget for field type and operator"""
        
        if operator == FilterOperator.BETWEEN:
            col1, col2 = st.columns(2)
            with col1:
                if field_type == 'number':
                    min_val = st.number_input("From", key="filter_min")
                elif field_type == 'date':
                    min_val = st.date_input("From", key="filter_min")
                else:
                    min_val = st.text_input("From", key="filter_min")
            
            with col2:
                if field_type == 'number':
                    max_val = st.number_input("To", key="filter_max")
                elif field_type == 'date':
                    max_val = st.date_input("To", key="filter_max")
                else:
                    max_val = st.text_input("To", key="filter_max")
            
            return [min_val, max_val]
        
        elif operator == FilterOperator.IN:
            values_text = st.text_input(
                "Values (comma-separated)",
                key="filter_list",
                help="Enter values separated by commas"
            )
            if values_text:
                return [v.strip() for v in values_text.split(',') if v.strip()]
            return []
        
        # Standard value inputs
        elif field_type == 'number':
            return st.number_input("Value", key="filter_value")
        elif field_type == 'date':
            return st.date_input("Value", key="filter_value")
        elif field_type == 'boolean':
            return st.selectbox("Value", options=[True, False], key="filter_value")
        else:  # text
            return st.text_input("Value", key="filter_value")
    
    def apply_filter(self, df: pd.DataFrame, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply filter to DataFrame"""
        
        if filter_config["type"] == "quick":
            return self._apply_quick_filter(df, filter_config["filter"])
        elif filter_config["type"] == "custom":
            return self._apply_custom_filter(df, filter_config)
        
        return df
    
    def _apply_quick_filter(self, df: pd.DataFrame, filter_type: str) -> pd.DataFrame:
        """Apply predefined quick filters"""
        
        if filter_type == "high_risk" and "risk_score" in df.columns:
            return df[df["risk_score"] > 0.7]
        elif filter_type == "anomalies" and "is_anomaly" in df.columns:
            return df[df["is_anomaly"] == True]
        elif filter_type == "last_24h" and "timestamp" in df.columns:
            yesterday = datetime.now() - timedelta(days=1)
            return df[pd.to_datetime(df["timestamp"]) >= yesterday]
        elif filter_type == "high_value" and "value" in df.columns:
            return df[df["value"] > 1.0]
        
        return df
    
    def _apply_custom_filter(self, df: pd.DataFrame, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply custom filter to DataFrame"""
        
        field = filter_config["field"]
        operator = filter_config["operator"]
        value = filter_config["value"]
        
        if field not in df.columns:
            st.warning(f"Field '{field}' not found in data")
            return df
        
        series = df[field]
        
        if operator == FilterOperator.EQUALS:
            return df[series == value]
        elif operator == FilterOperator.NOT_EQUALS:
            return df[series != value]
        elif operator == FilterOperator.GREATER_THAN:
            return df[series > value]
        elif operator == FilterOperator.LESS_THAN:
            return df[series < value]
        elif operator == FilterOperator.CONTAINS:
            return df[series.astype(str).str.contains(str(value), case=False, na=False)]
        elif operator == FilterOperator.NOT_CONTAINS:
            return df[~series.astype(str).str.contains(str(value), case=False, na=False)]
        elif operator == FilterOperator.BETWEEN:
            if len(value) == 2:
                return df[(series >= value[0]) & (series <= value[1])]
        elif operator == FilterOperator.IN:
            return df[series.isin(value) if value else pd.Series([False] * len(df))]
        
        return df
    
    def render_saved_searches_manager(self):
        """Render simple saved searches interface"""
        
        st.subheader("💾 Saved Searches")
        
        # Initialize saved searches in session state
        if 'saved_searches' not in st.session_state:
            st.session_state.saved_searches = []
        
        # Save current search
        with st.expander("💾 Save Current Search"):
            search_name = st.text_input("Search Name", key="save_search_name")
            search_description = st.text_area("Description", key="save_search_desc")
            
            if st.button("Save Search", key="save_search_btn"):
                if search_name:
                    saved_search = {
                        'id': len(st.session_state.saved_searches),
                        'name': search_name,
                        'description': search_description,
                        'created_at': datetime.now(),
                        'use_count': 0
                    }
                    st.session_state.saved_searches.append(saved_search)
                    st.success(f"Saved search: {search_name}")
                else:
                    st.warning("Please enter a search name")
        
        # Display saved searches
        if st.session_state.saved_searches:
            st.markdown("### Saved Searches")
            
            for search in st.session_state.saved_searches:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**{search['name']}**")
                        st.caption(search['description'])
                    
                    with col2:
                        st.caption(f"Used {search['use_count']} times")
                    
                    with col3:
                        if st.button("🗑️", key=f"delete_{search['id']}"):
                            st.session_state.saved_searches.remove(search)
                            st.rerun()
                    
                    st.divider()
        else:
            st.info("No saved searches yet")


# Initialize simple query builder
query_builder = QueryBuilder()