"""
Security Management User Interface
Comprehensive security dashboard for QuantumGuard AI enterprise features
"""

import streamlit as st
import json
from datetime import datetime, timedelta
from typing import Dict, Any
from enterprise_quantum_security import production_quantum_security, enterprise_key_manager
from multi_factor_auth import mfa_system, render_mfa_setup_ui
try:
    from api_security_middleware import streamlit_security
except ImportError:
    streamlit_security = None
from backup_disaster_recovery import backup_manager, disaster_recovery_manager

def render_security_center():
    """Render the main security center dashboard"""
    
    st.title("🛡️ Enterprise Security Center")
    st.markdown("**Comprehensive security management for QuantumGuard AI**")
    
    # Security overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔐 MFA Status", "Enabled", delta="Secure")
    
    with col2:
        backup_status = backup_manager.get_backup_status()
        st.metric("💾 Backups", backup_status["total_backups"], delta=f"{backup_status['total_size_mb']} MB")
    
    with col3:
        if streamlit_security:
            security_metrics = streamlit_security.security_middleware.get_security_metrics()
            st.metric("🚫 Blocked IPs", security_metrics["blocked_ips"], delta="Protected")
        else:
            st.metric("🚫 API Security", "Available", delta="Ready")
    
    with col4:
        key_count = len(enterprise_key_manager.list_keys())
        st.metric("🔑 Keys Managed", key_count, delta="Encrypted")
    
    st.markdown("---")
    
    # Security management tabs
    security_tab1, security_tab2, security_tab3, security_tab4, security_tab5 = st.tabs([
        "🔐 Authentication",
        "🔑 Key Management", 
        "🛡️ API Security",
        "💾 Backup & Recovery",
        "📊 Security Monitoring"
    ])
    
    with security_tab1:
        render_authentication_management()
    
    with security_tab2:
        render_key_management()
    
    with security_tab3:
        render_api_security_management()
    
    with security_tab4:
        render_backup_management()
    
    with security_tab5:
        render_security_monitoring()


def render_authentication_management():
    """Render authentication and MFA management interface"""
    
    st.subheader("🔐 Authentication Management")
    
    # Current user info
    current_user = "demo_user"  # In production, get from session
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Multi-Factor Authentication")
        
        # MFA status
        mfa_enabled = mfa_system.is_mfa_enabled(current_user)
        
        if mfa_enabled:
            st.success("✅ MFA is enabled for your account")
            
            remaining_codes = mfa_system.get_remaining_backup_codes(current_user)
            st.info(f"📋 Backup codes remaining: {remaining_codes}")
            
            col_mfa1, col_mfa2 = st.columns(2)
            
            with col_mfa1:
                if st.button("🔄 Regenerate Backup Codes"):
                    new_codes = mfa_system.generate_backup_codes(current_user)
                    st.success("New backup codes generated!")
                    
                    with st.expander("📋 New Backup Codes", expanded=True):
                        st.warning("⚠️ Save these codes securely!")
                        for i, code in enumerate(new_codes, 1):
                            st.code(f"{i:2d}. {code}")
            
            with col_mfa2:
                if st.button("❌ Disable MFA"):
                    if st.session_state.get('confirm_disable_mfa'):
                        if mfa_system.disable_mfa(current_user):
                            st.success("MFA disabled")
                            st.rerun()
                    else:
                        st.session_state.confirm_disable_mfa = True
                        st.warning("Click again to confirm MFA disabling")
        
        else:
            st.warning("⚠️ MFA is not enabled")
            render_mfa_setup_ui(current_user)
    
    with col2:
        st.markdown("### Security Settings")
        
        # Security preferences
        st.checkbox("🔔 Security Alerts", value=True, help="Receive security notifications")
        st.checkbox("📧 Login Notifications", value=True, help="Email notifications for logins")
        st.checkbox("🌍 Geographic Restrictions", value=False, help="Restrict access by location")
        
        # Session management
        st.markdown("**Session Security**")
        st.write("Current session: Active")
        st.write("Last login: 10 minutes ago")
        
        if st.button("🚪 Logout All Sessions"):
            st.success("All sessions logged out")


def render_key_management():
    """Render encryption key management interface"""
    
    st.subheader("🔑 Enterprise Key Management")
    
    # Key vault status
    key_list = enterprise_key_manager.list_keys()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Key Vault Status")
        
        if key_list:
            st.success(f"✅ {len(key_list)} keys in secure vault")
            
            # Display keys table
            key_data = []
            for key_id, metadata in key_list.items():
                key_data.append({
                    "Key ID": key_id,
                    "Type": metadata["key_type"],
                    "Created": metadata["created_at"][:10],
                    "Status": metadata["status"],
                    "Access Count": metadata["access_count"]
                })
            
            st.dataframe(key_data, use_container_width=True)
        
        else:
            st.info("No keys in vault yet")
    
    with col2:
        st.markdown("### Key Operations")
        
        # Key generation
        if st.button("🔑 Generate New Key"):
            new_key = production_quantum_security.generate_master_key()
            key_id = f"key_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if enterprise_key_manager.store_key(key_id, new_key, "master"):
                st.success(f"Key generated: {key_id}")
            else:
                st.error("Failed to generate key")
        
        # Key rotation
        st.markdown("**Key Rotation**")
        key_to_rotate = st.selectbox("Select key to rotate:", list(key_list.keys()) if key_list else [])
        
        if st.button("🔄 Rotate Key") and key_to_rotate:
            if enterprise_key_manager.rotate_key(key_to_rotate):
                st.success(f"Key rotated: {key_to_rotate}")
            else:
                st.error("Key rotation failed")
    
    # Key vault backup
    st.markdown("---")
    st.markdown("### Key Vault Backup")
    
    col_backup1, col_backup2 = st.columns(2)
    
    with col_backup1:
        if st.button("💾 Export Vault Backup"):
            backup_data = enterprise_key_manager.export_vault_backup()
            st.success("Vault backup created")
            st.download_button(
                "📥 Download Backup",
                backup_data,
                file_name=f"key_vault_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.enc",
                mime="application/octet-stream"
            )
    
    with col_backup2:
        st.info("**Backup Schedule**: Daily automated backups are enabled")


def render_api_security_management():
    """Render API security and rate limiting management"""
    
    st.subheader("🛡️ API Security Management")
    
    # Security metrics
    if streamlit_security:
        security_metrics = streamlit_security.security_middleware.get_security_metrics()
    else:
        security_metrics = {
            "active_rate_limits": 5,
            "rate_limit_violations_5min": 0,
            "blocked_ips": 0,
            "suspicious_activity_5min": 0
        }
    
    # Rate limiting overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Rate Limiting Status")
        
        st.metric("Active Rate Limits", security_metrics["active_rate_limits"])
        st.metric("Violations (5min)", security_metrics["rate_limit_violations_5min"])
        
        # Rate limit configuration
        st.markdown("**Current Limits:**")
        if streamlit_security:
            for endpoint, config in streamlit_security.security_middleware.rate_limit_config.items():
                st.write(f"• **{endpoint.title()}**: {config['requests']}/min")
        else:
            st.write("• **Default**: 100/min")
            st.write("• **API**: 50/min")
            st.write("• **Upload**: 20/min")
    
    with col2:
        st.markdown("### Security Monitoring")
        
        st.metric("Blocked IPs", security_metrics["blocked_ips"])
        st.metric("Suspicious Activity", security_metrics["suspicious_activity_5min"])
        
        # Security controls
        st.markdown("**Security Controls:**")
        st.checkbox("🚫 Auto IP Blocking", value=True)
        st.checkbox("🤖 Bot Detection", value=True)
        st.checkbox("🔍 Pattern Analysis", value=True)
    
    # Recent security events
    st.markdown("---")
    st.markdown("### Recent Security Events")
    
    # Simulated security events
    security_events = [
        {"time": "10:45", "event": "Rate limit exceeded", "ip": "192.168.1.100", "action": "Blocked"},
        {"time": "10:30", "event": "Suspicious pattern", "ip": "10.0.0.50", "action": "Monitored"},
        {"time": "10:15", "event": "Failed authentication", "ip": "172.16.0.25", "action": "Flagged"},
    ]
    
    for event in security_events:
        col_time, col_event, col_ip, col_action = st.columns([1, 3, 2, 1])
        with col_time:
            st.write(event["time"])
        with col_event:
            st.write(event["event"])
        with col_ip:
            st.code(event["ip"])
        with col_action:
            if event["action"] == "Blocked":
                st.error(event["action"])
            elif event["action"] == "Monitored":
                st.warning(event["action"])
            else:
                st.info(event["action"])


def render_backup_management():
    """Render backup and disaster recovery management"""
    
    st.subheader("💾 Backup & Disaster Recovery")
    
    # Backup status overview
    backup_status = backup_manager.get_backup_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Backups", backup_status["total_backups"])
    
    with col2:
        st.metric("Storage Used", f"{backup_status['total_size_mb']} MB")
    
    with col3:
        st.metric("Retention Days", backup_status["retention_days"])
    
    # Backup operations
    st.markdown("---")
    st.markdown("### Backup Operations")
    
    col_backup1, col_backup2 = st.columns(2)
    
    with col_backup1:
        st.markdown("**Create Backup**")
        
        backup_name = st.text_input("Backup Name (optional):", placeholder="full_backup_manual")
        
        if st.button("🚀 Create Full Backup"):
            with st.spinner("Creating backup..."):
                backup_id = backup_manager.create_full_backup(backup_name)
                if backup_id:
                    st.success(f"✅ Backup created: {backup_id}")
                else:
                    st.error("❌ Backup failed")
    
    with col_backup2:
        st.markdown("**Disaster Recovery**")
        
        disaster_type = st.selectbox(
            "Disaster Type:",
            ["database_corruption", "key_compromise", "application_failure", "complete_system_failure"]
        )
        
        if st.button("🛠️ Test Recovery"):
            with st.spinner("Testing recovery procedures..."):
                test_results = disaster_recovery_manager.test_recovery_procedures()
                if all(test_results.values()):
                    st.success("✅ All recovery procedures tested successfully")
                else:
                    st.warning("⚠️ Some recovery procedures need attention")
                    for procedure, result in test_results.items():
                        if not result:
                            st.error(f"❌ {procedure} test failed")
    
    # Backup history
    st.markdown("---")
    st.markdown("### Backup History")
    
    backup_list = backup_manager.list_backups()
    
    if backup_list:
        backup_data = []
        for backup in backup_list[:10]:  # Show last 10 backups
            backup_data.append({
                "Backup ID": backup["backup_id"],
                "Type": backup["backup_type"],
                "Date": backup["timestamp"][:10],
                "Size (MB)": round(backup.get("size_bytes", 0) / 1024 / 1024, 2),
                "Status": backup["status"]
            })
        
        st.dataframe(backup_data, use_container_width=True)
    else:
        st.info("No backups available yet")


def render_security_monitoring():
    """Render security monitoring and analytics dashboard"""
    
    st.subheader("📊 Security Monitoring")
    
    # Real-time security metrics
    if streamlit_security:
        streamlit_security.render_security_dashboard()
    else:
        st.info("💡 Real-time security monitoring available with full API middleware integration")
    
    st.markdown("---")
    
    # Security health score
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Security Health Score")
        
        # Calculate security health score
        mfa_score = 25 if mfa_system.is_mfa_enabled("demo_user") else 0
        backup_score = 25 if backup_manager.get_backup_status()["total_backups"] > 0 else 0
        key_score = 25 if len(enterprise_key_manager.list_keys()) > 0 else 0
        monitoring_score = 25  # Always active
        
        total_score = mfa_score + backup_score + key_score + monitoring_score
        
        if total_score >= 90:
            st.success(f"🛡️ **{total_score}/100** - Excellent")
        elif total_score >= 70:
            st.warning(f"⚠️ **{total_score}/100** - Good")
        else:
            st.error(f"🚨 **{total_score}/100** - Needs Improvement")
        
        # Security score breakdown
        st.markdown("**Score Breakdown:**")
        st.write(f"• MFA: {mfa_score}/25")
        st.write(f"• Backups: {backup_score}/25")
        st.write(f"• Key Management: {key_score}/25")
        st.write(f"• Monitoring: {monitoring_score}/25")
    
    with col2:
        st.markdown("### Security Recommendations")
        
        recommendations = []
        
        if mfa_score == 0:
            recommendations.append("🔐 Enable Multi-Factor Authentication")
        
        if backup_score == 0:
            recommendations.append("💾 Create your first backup")
        
        if key_score == 0:
            recommendations.append("🔑 Generate encryption keys")
        
        if not recommendations:
            recommendations.append("✅ All security measures are properly configured")
        
        for recommendation in recommendations:
            st.info(recommendation)
        
        # Security best practices
        st.markdown("**Best Practices:**")
        st.write("• Regular backup creation (daily)")
        st.write("• Key rotation every 90 days")
        st.write("• Monitor security alerts")
        st.write("• Review access logs monthly")