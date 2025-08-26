#!/usr/bin/env python3
"""
QuantumGuard AI - Quantum-Safe Session Management

This module provides quantum-resistant session management for the application,
ensuring all user sessions and temporary data are protected with post-quantum cryptography.
"""

import streamlit as st
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from quantum_backend_security import (
    create_quantum_session, 
    validate_quantum_session,
    get_backend_security_status,
    quantum_backend
)

class QuantumSessionManager:
    """Quantum-safe session manager for Streamlit applications"""
    
    def __init__(self):
        self.session_key = "quantum_session_id"
        
    def initialize_session(self) -> str:
        """Initialize a new quantum-safe session"""
        if self.session_key not in st.session_state:
            session_id = str(uuid.uuid4())
            
            # Create quantum-safe session with user data
            user_data = {
                "created_at": datetime.now().isoformat(),
                "user_agent": "QuantumGuard_AI_User",
                "session_type": "analysis",
                "security_level": "quantum_safe"
            }
            
            try:
                encrypted_session = create_quantum_session(session_id, user_data)
                st.session_state[self.session_key] = session_id
                st.session_state["quantum_session_encrypted"] = encrypted_session
                st.session_state["session_initialized"] = True
                return session_id
            except Exception as e:
                st.error(f"Failed to create quantum-safe session: {str(e)}")
                return ""
        
        return st.session_state[self.session_key]
    
    def validate_current_session(self) -> bool:
        """Validate the current quantum-safe session"""
        if self.session_key not in st.session_state:
            return False
        
        session_id = st.session_state[self.session_key]
        session_data = validate_quantum_session(session_id)
        
        if session_data is None:
            # Session invalid or expired
            self.clear_session()
            return False
        
        return True
    
    def get_session_data(self) -> Optional[Dict[str, Any]]:
        """Get quantum-safe session data"""
        if not self.validate_current_session():
            return None
        
        session_id = st.session_state[self.session_key]
        return validate_quantum_session(session_id)
    
    def clear_session(self):
        """Clear quantum-safe session data"""
        if self.session_key in st.session_state:
            del st.session_state[self.session_key]
        if "quantum_session_encrypted" in st.session_state:
            del st.session_state["quantum_session_encrypted"]
        if "session_initialized" in st.session_state:
            del st.session_state["session_initialized"]
    
    def store_analysis_data_securely(self, key: str, data: Any):
        """Store analysis data with quantum-safe encryption in session"""
        if not self.validate_current_session():
            self.initialize_session()
        
        try:
            encrypted_data = quantum_backend.encrypt_sensitive_data(data, "session")
            st.session_state[f"quantum_encrypted_{key}"] = encrypted_data
        except Exception as e:
            st.error(f"Failed to securely store data: {str(e)}")
    
    def retrieve_analysis_data_securely(self, key: str) -> Any:
        """Retrieve analysis data with quantum-safe decryption from session"""
        encrypted_key = f"quantum_encrypted_{key}"
        
        if encrypted_key not in st.session_state:
            return None
        
        try:
            encrypted_data = st.session_state[encrypted_key]
            return quantum_backend.decrypt_sensitive_data(encrypted_data, "session")
        except Exception as e:
            st.error(f"Failed to securely retrieve data: {str(e)}")
            return None
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get quantum session security metrics"""
        session_data = self.get_session_data()
        backend_status = get_backend_security_status()
        
        if session_data:
            created_at = datetime.fromisoformat(session_data["user_data"]["created_at"])
            session_age = (datetime.now() - created_at).total_seconds() / 3600  # hours
        else:
            session_age = 0
        
        return {
            "session_active": session_data is not None,
            "session_age_hours": round(session_age, 2),
            "quantum_encrypted": True,
            "algorithm": backend_status["algorithm"],
            "security_level": backend_status["security_level"],
            "quantum_safe": backend_status["quantum_safe"]
        }

# Global quantum session manager
quantum_session_manager = QuantumSessionManager()

def init_quantum_session() -> str:
    """Initialize quantum-safe session for the application"""
    return quantum_session_manager.initialize_session()

def store_secure_data(key: str, data: Any):
    """Store data securely with quantum encryption"""
    quantum_session_manager.store_analysis_data_securely(key, data)

def retrieve_secure_data(key: str) -> Any:
    """Retrieve data securely with quantum decryption"""
    return quantum_session_manager.retrieve_analysis_data_securely(key)

def get_session_security_status() -> Dict[str, Any]:
    """Get comprehensive session security status"""
    return quantum_session_manager.get_security_metrics()