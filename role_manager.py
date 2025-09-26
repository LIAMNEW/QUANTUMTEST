import streamlit as st
from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

class UserRole(Enum):
    ADMIN = "admin"
    ANALYST = "analyst" 
    VIEWER = "viewer"

class Permission(Enum):
    # Data access permissions
    VIEW_TRANSACTIONS = "view_transactions"
    VIEW_ANALYSIS = "view_analysis"
    VIEW_SENSITIVE_DATA = "view_sensitive_data"
    
    # Analysis permissions
    CREATE_ANALYSIS = "create_analysis"
    MODIFY_ANALYSIS = "modify_analysis"
    DELETE_ANALYSIS = "delete_analysis"
    
    # System permissions
    MANAGE_USERS = "manage_users"
    MANAGE_SYSTEM = "manage_system"
    ACCESS_ADMIN_PANEL = "access_admin_panel"
    
    # Configuration permissions
    MANAGE_API_KEYS = "manage_api_keys"
    MODIFY_SETTINGS = "modify_settings"
    
    # Dashboard permissions
    CREATE_DASHBOARDS = "create_dashboards"
    MODIFY_DASHBOARDS = "modify_dashboards"
    DELETE_DASHBOARDS = "delete_dashboards"

@dataclass
class User:
    username: str
    role: UserRole
    permissions: Set[Permission]
    created_at: datetime
    last_login: datetime
    is_active: bool = True

class RoleBasedAccessControl:
    """Role-based access control system"""
    
    # Expose Permission enum as class attribute for easy access
    Permission = Permission
    
    def __init__(self):
        self.role_permissions = self._define_role_permissions()
        self.current_user = self._get_current_user()
    
    def _define_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Define permissions for each role"""
        return {
            UserRole.ADMIN: {
                Permission.VIEW_TRANSACTIONS,
                Permission.VIEW_ANALYSIS,
                Permission.VIEW_SENSITIVE_DATA,
                Permission.CREATE_ANALYSIS,
                Permission.MODIFY_ANALYSIS,
                Permission.DELETE_ANALYSIS,
                Permission.MANAGE_USERS,
                Permission.MANAGE_SYSTEM,
                Permission.ACCESS_ADMIN_PANEL,
                Permission.MANAGE_API_KEYS,
                Permission.MODIFY_SETTINGS,
                Permission.CREATE_DASHBOARDS,
                Permission.MODIFY_DASHBOARDS,
                Permission.DELETE_DASHBOARDS
            },
            
            UserRole.ANALYST: {
                Permission.VIEW_TRANSACTIONS,
                Permission.VIEW_ANALYSIS,
                Permission.CREATE_ANALYSIS,
                Permission.MODIFY_ANALYSIS,
                Permission.CREATE_DASHBOARDS,
                Permission.MODIFY_DASHBOARDS
            },
            
            UserRole.VIEWER: {
                Permission.VIEW_TRANSACTIONS,
                Permission.VIEW_ANALYSIS
            }
        }
    
    def _get_current_user(self) -> User:
        """Get current user from session state"""
        
        # Initialize default user if not set
        if 'user' not in st.session_state:
            default_role = UserRole.ANALYST
            st.session_state.user = User(
                username="analyst_user",
                role=default_role,
                permissions=self.role_permissions[default_role],
                created_at=datetime.now(),
                last_login=datetime.now()
            )
        
        return st.session_state.user
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if current user has specific permission"""
        return permission in self.current_user.permissions
    
    def require_permission(self, permission: Permission) -> bool:
        """Require specific permission, show warning if denied"""
        if not self.has_permission(permission):
            st.warning(f"⛔ Access denied: You don't have permission to {permission.value.replace('_', ' ')}")
            return False
        return True
    
    def get_available_roles(self) -> List[UserRole]:
        """Get roles available to current user"""
        if self.has_permission(Permission.MANAGE_USERS):
            return list(UserRole)
        else:
            return [self.current_user.role]
    
    def render_role_selector(self) -> Optional[UserRole]:
        """Render role selection interface for testing purposes"""
        
        if not self.has_permission(Permission.MANAGE_SYSTEM):
            return None
        
        st.subheader("👤 Role Management")
        
        # Current user display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"**Current Role:** {self.current_user.role.value.title()}")
            
            # Role switching (for demo purposes)
            new_role = st.selectbox(
                "Switch Role (Demo):",
                options=list(UserRole),
                format_func=lambda x: x.value.title(),
                index=list(UserRole).index(self.current_user.role)
            )
            
            if st.button("Switch Role"):
                self.switch_user_role(new_role)
                st.success(f"Switched to {new_role.value.title()} role")
                st.rerun()
        
        with col2:
            st.markdown("**Permissions:**")
            for permission in self.current_user.permissions:
                st.caption(f"✅ {permission.value.replace('_', ' ').title()}")
        
        return new_role
    
    def switch_user_role(self, new_role: UserRole):
        """Switch current user role"""
        st.session_state.user.role = new_role
        st.session_state.user.permissions = self.role_permissions[new_role]
        self.current_user = st.session_state.user
    
    def render_permission_guard(self, permission: Permission, content_func):
        """Render content only if user has permission"""
        if self.has_permission(permission):
            return content_func()
        else:
            st.warning(f"🔒 Access restricted: {permission.value.replace('_', ' ').title()} permission required")
            return None
    
    def filter_data_by_role(self, data: Dict) -> Dict:
        """Filter data based on user role permissions"""
        
        filtered_data = data.copy()
        
        # Sensitive data filtering for non-admin users
        if not self.has_permission(Permission.VIEW_SENSITIVE_DATA):
            sensitive_fields = ['private_key', 'api_secret', 'password', 'secret']
            
            for field in sensitive_fields:
                if field in filtered_data:
                    filtered_data[field] = "***HIDDEN***"
        
        return filtered_data
    
    def get_accessible_features(self) -> Dict[str, bool]:
        """Get feature accessibility for UI rendering"""
        
        return {
            'admin_panel': self.has_permission(Permission.ACCESS_ADMIN_PANEL),
            'user_management': self.has_permission(Permission.MANAGE_USERS),
            'system_settings': self.has_permission(Permission.MANAGE_SYSTEM),
            'api_configuration': self.has_permission(Permission.MANAGE_API_KEYS),
            'create_analysis': self.has_permission(Permission.CREATE_ANALYSIS),
            'modify_analysis': self.has_permission(Permission.MODIFY_ANALYSIS),
            'delete_analysis': self.has_permission(Permission.DELETE_ANALYSIS),
            'create_dashboards': self.has_permission(Permission.CREATE_DASHBOARDS),
            'modify_dashboards': self.has_permission(Permission.MODIFY_DASHBOARDS),
            'delete_dashboards': self.has_permission(Permission.DELETE_DASHBOARDS),
            'view_sensitive': self.has_permission(Permission.VIEW_SENSITIVE_DATA)
        }
    
    def render_access_summary(self):
        """Render access summary for current user"""
        
        st.subheader(f"🛡️ Access Level: {self.current_user.role.value.title()}")
        
        features = self.get_accessible_features()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Available Features:**")
            available = [k.replace('_', ' ').title() for k, v in features.items() if v]
            for feature in available:
                st.success(f"✅ {feature}")
        
        with col2:
            st.markdown("**Restricted Features:**")
            restricted = [k.replace('_', ' ').title() for k, v in features.items() if not v]
            for feature in restricted:
                st.error(f"❌ {feature}")
        
        if self.current_user.role == UserRole.VIEWER:
            st.info("💡 **Viewer Role**: You have read-only access. Contact your administrator to request additional permissions.")
        elif self.current_user.role == UserRole.ANALYST:
            st.info("💡 **Analyst Role**: You can create and modify analyses. Some system settings are restricted.")
        else:
            st.success("💡 **Admin Role**: You have full system access including user management and system configuration.")
    
    def create_audit_log_entry(self, action: str, resource: str, details: str = ""):
        """Create audit log entry for user actions"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': self.current_user.username,
            'role': self.current_user.role.value,
            'action': action,
            'resource': resource,
            'details': details,
            'success': True
        }
        
        # In a real implementation, this would write to a secure audit log
        st.session_state.setdefault('audit_logs', []).append(log_entry)
    
    def get_role_specific_dashboard_config(self) -> Dict[str, any]:
        """Get dashboard configuration based on user role"""
        
        if self.current_user.role == UserRole.ADMIN:
            return {
                'show_system_metrics': True,
                'show_user_activity': True,
                'show_audit_logs': True,
                'show_all_analyses': True,
                'enable_bulk_operations': True
            }
        elif self.current_user.role == UserRole.ANALYST:
            return {
                'show_system_metrics': False,
                'show_user_activity': False,
                'show_audit_logs': False,
                'show_all_analyses': False,  # Only own analyses
                'enable_bulk_operations': False
            }
        else:  # VIEWER
            return {
                'show_system_metrics': False,
                'show_user_activity': False,
                'show_audit_logs': False,
                'show_all_analyses': False,
                'enable_bulk_operations': False,
                'read_only': True
            }


# Initialize role-based access control
rbac = RoleBasedAccessControl()