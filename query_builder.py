import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import re

class FilterOperator(Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    BETWEEN = "between"
    REGEX = "regex"

class LogicalOperator(Enum):
    AND = "and"
    OR = "or"

@dataclass
class FilterCondition:
    field: str
    operator: FilterOperator
    value: Any
    field_type: str = "text"  # text, number, date, boolean

@dataclass
class QueryFilter:
    conditions: List[FilterCondition]
    logical_operator: LogicalOperator = LogicalOperator.AND

@dataclass
class SavedSearch:
    id: str
    name: str
    description: str
    query_filter: QueryFilter
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    is_automated: bool = False
    schedule: Optional[str] = None  # cron-like schedule

class QueryBuilder:
    """Visual query builder for advanced search and filtering"""
    
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
            FilterOperator.GREATER_EQUAL: "Greater or Equal",
            FilterOperator.LESS_EQUAL: "Less or Equal",
            FilterOperator.CONTAINS: "Contains",
            FilterOperator.NOT_CONTAINS: "Does Not Contain",
            FilterOperator.STARTS_WITH: "Starts With",
            FilterOperator.ENDS_WITH: "Ends With",
            FilterOperator.IN: "In List",
            FilterOperator.NOT_IN: "Not In List",
            FilterOperator.IS_NULL: "Is Empty",
            FilterOperator.IS_NOT_NULL: "Is Not Empty",
            FilterOperator.BETWEEN: "Between",
            FilterOperator.REGEX: "Matches Pattern"
        }
    
    def get_operators_for_type(self, field_type: str) -> List[FilterOperator]:
        """Get available operators for a field type"""
        if field_type == 'text':
            return [
                FilterOperator.EQUALS, FilterOperator.NOT_EQUALS,
                FilterOperator.CONTAINS, FilterOperator.NOT_CONTAINS,
                FilterOperator.STARTS_WITH, FilterOperator.ENDS_WITH,
                FilterOperator.IN, FilterOperator.NOT_IN,
                FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL,
                FilterOperator.REGEX
            ]
        elif field_type == 'number':
            return [
                FilterOperator.EQUALS, FilterOperator.NOT_EQUALS,
                FilterOperator.GREATER_THAN, FilterOperator.LESS_THAN,
                FilterOperator.GREATER_EQUAL, FilterOperator.LESS_EQUAL,
                FilterOperator.BETWEEN, FilterOperator.IN, FilterOperator.NOT_IN,
                FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL
            ]
        elif field_type == 'date':
            return [
                FilterOperator.EQUALS, FilterOperator.NOT_EQUALS,
                FilterOperator.GREATER_THAN, FilterOperator.LESS_THAN,
                FilterOperator.GREATER_EQUAL, FilterOperator.LESS_EQUAL,
                FilterOperator.BETWEEN, FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL
            ]
        elif field_type == 'boolean':
            return [
                FilterOperator.EQUALS, FilterOperator.NOT_EQUALS,
                FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL
            ]
        else:
            return list(FilterOperator)
    
    def render_condition_builder(self, condition_key: str = "condition") -> Optional[FilterCondition]:
        """Render interface for building a single filter condition"""
        
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            field = st.selectbox(
                "Field",
                options=list(self.field_types.keys()),
                key=f"{condition_key}_field",
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        field_type = self.field_types.get(field, 'text')
        available_operators = self.get_operators_for_type(field_type)
        
        with col2:
            operator = st.selectbox(
                "Operator",
                options=available_operators,
                key=f"{condition_key}_operator",
                format_func=lambda x: self.operator_labels[x]
            )
        
        with col3:
            value = self._render_value_input(field, field_type, operator, condition_key)
        
        if field and operator is not None:
            return FilterCondition(
                field=field,
                operator=operator,
                value=value,
                field_type=field_type
            )
        
        return None
    
    def _render_value_input(self, field: str, field_type: str, operator: FilterOperator, key: str):
        """Render appropriate input widget for field type and operator"""
        
        # No value needed for null checks
        if operator in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL]:
            return None
        
        # Special handling for specific operators
        if operator == FilterOperator.BETWEEN:
            col1, col2 = st.columns(2)
            with col1:
                if field_type == 'number':
                    min_val = st.number_input("From", key=f"{key}_min")
                elif field_type == 'date':
                    min_val = st.date_input("From", key=f"{key}_min")
                else:
                    min_val = st.text_input("From", key=f"{key}_min")
            
            with col2:
                if field_type == 'number':
                    max_val = st.number_input("To", key=f"{key}_max")
                elif field_type == 'date':
                    max_val = st.date_input("To", key=f"{key}_max")
                else:
                    max_val = st.text_input("To", key=f"{key}_max")
            
            return [min_val, max_val]
        
        elif operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            values_text = st.text_input(
                "Values (comma-separated)",
                key=f"{key}_list",
                help="Enter values separated by commas"
            )
            if values_text:
                return [v.strip() for v in values_text.split(',') if v.strip()]
            return []
        
        # Standard value inputs
        elif field_type == 'number':
            return st.number_input(
                "Value",
                key=f"{key}_value"
            )
        
        elif field_type == 'date':
            return st.date_input(
                "Value",
                key=f"{key}_value"
            )
        
        elif field_type == 'boolean':
            return st.selectbox(
                "Value",
                options=[True, False],
                key=f"{key}_value"
            )
        
        else:  # text
            if operator == FilterOperator.REGEX:
                return st.text_input(
                    "Pattern",
                    key=f"{key}_value",
                    help="Enter regular expression pattern"
                )
            else:
                return st.text_input(
                    "Value",
                    key=f"{key}_value"
                )
    
    def render_query_builder(self) -> Optional[QueryFilter]:
        """Render complete visual query builder interface"""
        
        st.subheader("🔍 Advanced Query Builder")
        
        # Initialize conditions in session state
        if 'query_conditions' not in st.session_state:
            st.session_state.query_conditions = []
        
        # Logical operator selection
        logical_op = st.radio(
            "Combine conditions with:",
            options=[LogicalOperator.AND, LogicalOperator.OR],
            format_func=lambda x: x.value.upper(),
            horizontal=True,
            key="logical_operator"
        )
        
        # Render existing conditions
        conditions = []
        conditions_to_remove = []
        
        for i, _ in enumerate(st.session_state.query_conditions):
            with st.container():
                col1, col2 = st.columns([10, 1])
                
                with col1:
                    condition = self.render_condition_builder(f"condition_{i}")
                    if condition:
                        conditions.append(condition)
                
                with col2:
                    if st.button("❌", key=f"remove_{i}", help="Remove condition"):
                        conditions_to_remove.append(i)
        
        # Remove conditions marked for deletion
        for idx in reversed(conditions_to_remove):
            st.session_state.query_conditions.pop(idx)
            st.rerun()
        
        # Add condition button
        col1, col2, col3 = st.columns([2, 2, 6])
        
        with col1:
            if st.button("➕ Add Condition", key="add_condition"):
                st.session_state.query_conditions.append({})
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All", key="clear_conditions"):
                st.session_state.query_conditions = []
                st.rerun()
        
        # Quick filters section
        self._render_quick_filters()
        
        if conditions:
            return QueryFilter(
                conditions=conditions,
                logical_operator=logical_op
            )
        
        return None
    
    def _render_quick_filters(self):
        """Render quick filter presets"""
        
        with st.expander("⚡ Quick Filters", expanded=False):
            st.markdown("**Common Filter Presets:**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔴 High Risk", key="quick_high_risk"):
                    self._apply_quick_filter("high_risk")
            
            with col2:
                if st.button("⚠️ Anomalies", key="quick_anomalies"):
                    self._apply_quick_filter("anomalies")
            
            with col3:
                if st.button("📅 Last 24h", key="quick_24h"):
                    self._apply_quick_filter("last_24h")
            
            with col4:
                if st.button("💰 High Value", key="quick_high_value"):
                    self._apply_quick_filter("high_value")
    
    def _apply_quick_filter(self, filter_type: str):
        """Apply predefined quick filters"""
        
        if filter_type == "high_risk":
            condition = FilterCondition(
                field="risk_score",
                operator=FilterOperator.GREATER_THAN,
                value=0.7,
                field_type="number"
            )
        elif filter_type == "anomalies":
            condition = FilterCondition(
                field="is_anomaly",
                operator=FilterOperator.EQUALS,
                value=True,
                field_type="boolean"
            )
        elif filter_type == "last_24h":
            yesterday = date.today() - timedelta(days=1)
            condition = FilterCondition(
                field="timestamp",
                operator=FilterOperator.GREATER_EQUAL,
                value=yesterday,
                field_type="date"
            )
        elif filter_type == "high_value":
            condition = FilterCondition(
                field="value",
                operator=FilterOperator.GREATER_THAN,
                value=1.0,
                field_type="number"
            )
        else:
            return
        
        # Add to session state
        if 'query_conditions' not in st.session_state:
            st.session_state.query_conditions = []
        
        st.session_state.query_conditions.append({})
        st.rerun()
    
    def apply_filter(self, df: pd.DataFrame, query_filter: QueryFilter) -> pd.DataFrame:
        """Apply query filter to DataFrame"""
        
        if not query_filter.conditions:
            return df
        
        condition_results = []
        
        for condition in query_filter.conditions:
            result = self._apply_condition(df, condition)
            condition_results.append(result)
        
        # Combine results based on logical operator
        if query_filter.logical_operator == LogicalOperator.AND:
            final_mask = condition_results[0]
            for mask in condition_results[1:]:
                final_mask = final_mask & mask
        else:  # OR
            final_mask = condition_results[0]
            for mask in condition_results[1:]:
                final_mask = final_mask | mask
        
        return df[final_mask]
    
    def _apply_condition(self, df: pd.DataFrame, condition: FilterCondition) -> pd.Series:
        """Apply single condition to DataFrame and return boolean mask"""
        
        if condition.field not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        
        series = df[condition.field]
        
        if condition.operator == FilterOperator.EQUALS:
            return series == condition.value
        elif condition.operator == FilterOperator.NOT_EQUALS:
            return series != condition.value
        elif condition.operator == FilterOperator.GREATER_THAN:
            return series > condition.value
        elif condition.operator == FilterOperator.LESS_THAN:
            return series < condition.value
        elif condition.operator == FilterOperator.GREATER_EQUAL:
            return series >= condition.value
        elif condition.operator == FilterOperator.LESS_EQUAL:
            return series <= condition.value
        elif condition.operator == FilterOperator.CONTAINS:
            return series.astype(str).str.contains(str(condition.value), case=False, na=False)
        elif condition.operator == FilterOperator.NOT_CONTAINS:
            return ~series.astype(str).str.contains(str(condition.value), case=False, na=False)
        elif condition.operator == FilterOperator.STARTS_WITH:
            return series.astype(str).str.startswith(str(condition.value), na=False)
        elif condition.operator == FilterOperator.ENDS_WITH:
            return series.astype(str).str.endswith(str(condition.value), na=False)
        elif condition.operator == FilterOperator.IN:
            return series.isin(condition.value) if condition.value else pd.Series([False] * len(df))
        elif condition.operator == FilterOperator.NOT_IN:
            return ~series.isin(condition.value) if condition.value else pd.Series([True] * len(df))
        elif condition.operator == FilterOperator.IS_NULL:
            return series.isna()
        elif condition.operator == FilterOperator.IS_NOT_NULL:
            return series.notna()
        elif condition.operator == FilterOperator.BETWEEN:
            if len(condition.value) == 2:
                return (series >= condition.value[0]) & (series <= condition.value[1])
            return pd.Series([False] * len(df), index=df.index)
        elif condition.operator == FilterOperator.REGEX:
            try:
                return series.astype(str).str.match(condition.value, na=False)
            except re.error:
                st.error(f"Invalid regular expression: {condition.value}")
                return pd.Series([False] * len(df), index=df.index)
        
        return pd.Series([False] * len(df), index=df.index)
    
    def save_search(self, name: str, description: str, query_filter: QueryFilter, 
                   is_automated: bool = False, schedule: str = None):
        """Save search query for later use"""
        
        from database import DatabaseManager
        
        saved_search = SavedSearch(
            id=f"search_{int(datetime.now().timestamp())}",
            name=name,
            description=description,
            query_filter=query_filter,
            created_at=datetime.now(),
            last_used=datetime.now(),
            is_automated=is_automated,
            schedule=schedule
        )
        
        # Save to database
        try:
            db = DatabaseManager()
            with db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS saved_searches (
                        id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        query_filter JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        use_count INTEGER DEFAULT 0,
                        is_automated BOOLEAN DEFAULT false,
                        schedule VARCHAR(255)
                    )
                """)
                
                # Serialize query filter
                filter_json = {
                    'conditions': [
                        {
                            'field': c.field,
                            'operator': c.operator.value,
                            'value': c.value,
                            'field_type': c.field_type
                        } for c in query_filter.conditions
                    ],
                    'logical_operator': query_filter.logical_operator.value
                }
                
                cursor.execute("""
                    INSERT INTO saved_searches 
                    (id, name, description, query_filter, is_automated, schedule)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    saved_search.id, saved_search.name, saved_search.description,
                    json.dumps(filter_json), saved_search.is_automated, saved_search.schedule
                ))
                
                conn.commit()
                return saved_search.id
                
        except Exception as e:
            st.error(f"Error saving search: {e}")
            return None
    
    def load_saved_searches(self) -> List[SavedSearch]:
        """Load all saved searches"""
        
        from database import DatabaseManager
        
        try:
            db = DatabaseManager()
            with db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, name, description, query_filter, created_at, last_used, use_count, is_automated, schedule
                    FROM saved_searches ORDER BY last_used DESC
                """)
                
                saved_searches = []
                for row in cursor.fetchall():
                    # Deserialize query filter
                    filter_data = json.loads(row[3])
                    conditions = []
                    
                    for c_data in filter_data.get('conditions', []):
                        condition = FilterCondition(
                            field=c_data['field'],
                            operator=FilterOperator(c_data['operator']),
                            value=c_data['value'],
                            field_type=c_data['field_type']
                        )
                        conditions.append(condition)
                    
                    query_filter = QueryFilter(
                        conditions=conditions,
                        logical_operator=LogicalOperator(filter_data.get('logical_operator', 'and'))
                    )
                    
                    saved_search = SavedSearch(
                        id=row[0],
                        name=row[1],
                        description=row[2],
                        query_filter=query_filter,
                        created_at=row[4],
                        last_used=row[5],
                        use_count=row[6],
                        is_automated=row[7],
                        schedule=row[8]
                    )
                    
                    saved_searches.append(saved_search)
                
                return saved_searches
                
        except Exception as e:
            st.error(f"Error loading saved searches: {e}")
            return []
    
    def render_saved_searches_manager(self):
        """Render saved searches management interface"""
        
        st.subheader("💾 Saved Searches")
        
        saved_searches = self.load_saved_searches()
        
        if not saved_searches:
            st.info("No saved searches found. Create some queries and save them for quick access!")
            return
        
        # Display saved searches
        for search in saved_searches:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                
                with col1:
                    st.write(f"**{search.name}**")
                    st.caption(search.description)
                
                with col2:
                    st.caption(f"Used {search.use_count} times")
                    st.caption(f"Last used: {search.last_used.strftime('%Y-%m-%d')}")
                
                with col3:
                    if st.button("🔄 Load", key=f"load_{search.id}"):
                        # Load the search into the query builder
                        st.session_state.saved_search_loaded = search
                        st.success(f"Loaded search: {search.name}")
                
                with col4:
                    if st.button("🗑️", key=f"delete_{search.id}"):
                        self._delete_saved_search(search.id)
                        st.rerun()
                
                st.divider()
    
    def _delete_saved_search(self, search_id: str):
        """Delete a saved search"""
        from database import DatabaseManager
        
        try:
            db = DatabaseManager()
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM saved_searches WHERE id = %s", (search_id,))
                conn.commit()
                
        except Exception as e:
            st.error(f"Error deleting search: {e}")


# Initialize query builder
query_builder = QueryBuilder()