"""
Multi-Factor Authentication Module
Enterprise-grade MFA implementation with TOTP, backup codes, and security features
"""

import pyotp
import qrcode
import qrcode.image.svg
import bcrypt
import secrets
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from io import BytesIO
import base64
import streamlit as st
from enterprise_quantum_security import enterprise_key_manager, security_logger

class MultiFactorAuth:
    """Enterprise multi-factor authentication system"""
    
    def __init__(self):
        self.failed_attempts = {}
        self.lockout_duration = 900  # 15 minutes
        self.max_attempts = 5
        self.totp_window = 1  # Allow 1 time step tolerance
        
    def generate_totp_secret(self, user_id: str) -> str:
        """Generate TOTP secret for user"""
        secret = pyotp.random_base32()
        
        # Store encrypted secret in key manager
        secret_key_id = f"totp_secret_{user_id}"
        enterprise_key_manager.store_key(secret_key_id, secret.encode(), "totp_secret")
        
        security_logger.info(f"TOTP secret generated for user: {user_id}")
        return secret
    
    def generate_qr_code(self, user_id: str, secret: str, issuer: str = "QuantumGuard AI") -> str:
        """Generate QR code for TOTP setup"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_id,
            issuer_name=issuer
        )
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        # Create image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64 string
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        security_logger.info(f"QR code generated for user: {user_id}")
        return img_str
    
    def verify_totp(self, user_id: str, token: str) -> bool:
        """Verify TOTP token"""
        if self._is_user_locked_out(user_id):
            security_logger.warning(f"TOTP verification blocked - user locked out: {user_id}")
            return False
        
        try:
            # Retrieve secret from key manager
            secret_key_id = f"totp_secret_{user_id}"
            secret_bytes = enterprise_key_manager.retrieve_key(secret_key_id)
            
            if not secret_bytes:
                security_logger.error(f"TOTP secret not found for user: {user_id}")
                return False
            
            secret = secret_bytes.decode()
            totp = pyotp.TOTP(secret)
            
            # Verify token with window tolerance
            is_valid = totp.verify(token, valid_window=self.totp_window)
            
            if is_valid:
                self._reset_failed_attempts(user_id)
                security_logger.info(f"TOTP verification successful for user: {user_id}")
                return True
            else:
                self._record_failed_attempt(user_id)
                security_logger.warning(f"TOTP verification failed for user: {user_id}")
                return False
                
        except Exception as e:
            self._record_failed_attempt(user_id)
            security_logger.error(f"TOTP verification error for user {user_id}: {str(e)}")
            return False
    
    def generate_backup_codes(self, user_id: str, count: int = 10) -> List[str]:
        """Generate backup codes for MFA recovery"""
        backup_codes = []
        
        for _ in range(count):
            # Generate 8-character alphanumeric code
            code = ''.join(secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(8))
            backup_codes.append(code)
        
        # Hash and store backup codes
        hashed_codes = [bcrypt.hashpw(code.encode(), bcrypt.gensalt()).decode() for code in backup_codes]
        
        # Store in key manager
        backup_key_id = f"backup_codes_{user_id}"
        enterprise_key_manager.store_key(
            backup_key_id, 
            json.dumps(hashed_codes).encode(), 
            "backup_codes"
        )
        
        security_logger.info(f"Backup codes generated for user: {user_id}")
        return backup_codes
    
    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify and consume backup code"""
        if self._is_user_locked_out(user_id):
            security_logger.warning(f"Backup code verification blocked - user locked out: {user_id}")
            return False
        
        try:
            # Retrieve backup codes
            backup_key_id = f"backup_codes_{user_id}"
            codes_bytes = enterprise_key_manager.retrieve_key(backup_key_id)
            
            if not codes_bytes:
                security_logger.error(f"Backup codes not found for user: {user_id}")
                return False
            
            hashed_codes = json.loads(codes_bytes.decode())
            
            # Check if code matches any stored hash
            for i, hashed_code in enumerate(hashed_codes):
                if bcrypt.checkpw(code.encode(), hashed_code.encode()):
                    # Remove used code
                    hashed_codes.pop(i)
                    
                    # Update stored codes
                    enterprise_key_manager.store_key(
                        backup_key_id,
                        json.dumps(hashed_codes).encode(),
                        "backup_codes"
                    )
                    
                    self._reset_failed_attempts(user_id)
                    security_logger.info(f"Backup code verification successful for user: {user_id}")
                    return True
            
            # No matching code found
            self._record_failed_attempt(user_id)
            security_logger.warning(f"Backup code verification failed for user: {user_id}")
            return False
            
        except Exception as e:
            self._record_failed_attempt(user_id)
            security_logger.error(f"Backup code verification error for user {user_id}: {str(e)}")
            return False
    
    def _is_user_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out due to failed attempts"""
        if user_id not in self.failed_attempts:
            return False
        
        attempts_data = self.failed_attempts[user_id]
        
        if attempts_data['count'] >= self.max_attempts:
            lockout_end = attempts_data['first_attempt'] + timedelta(seconds=self.lockout_duration)
            if datetime.now() < lockout_end:
                return True
            else:
                # Lockout period expired, reset
                self._reset_failed_attempts(user_id)
                return False
        
        return False
    
    def _record_failed_attempt(self, user_id: str):
        """Record failed authentication attempt"""
        now = datetime.now()
        
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = {
                'count': 1,
                'first_attempt': now,
                'last_attempt': now
            }
        else:
            self.failed_attempts[user_id]['count'] += 1
            self.failed_attempts[user_id]['last_attempt'] = now
    
    def _reset_failed_attempts(self, user_id: str):
        """Reset failed attempts for user"""
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
    
    def get_remaining_backup_codes(self, user_id: str) -> int:
        """Get count of remaining backup codes"""
        try:
            backup_key_id = f"backup_codes_{user_id}"
            codes_bytes = enterprise_key_manager.retrieve_key(backup_key_id)
            
            if not codes_bytes:
                return 0
            
            hashed_codes = json.loads(codes_bytes.decode())
            return len(hashed_codes)
            
        except Exception:
            return 0
    
    def is_mfa_enabled(self, user_id: str) -> bool:
        """Check if MFA is enabled for user"""
        secret_key_id = f"totp_secret_{user_id}"
        return enterprise_key_manager.retrieve_key(secret_key_id) is not None
    
    def disable_mfa(self, user_id: str) -> bool:
        """Disable MFA for user (admin function)"""
        try:
            # Remove TOTP secret
            secret_key_id = f"totp_secret_{user_id}"
            if secret_key_id in enterprise_key_manager.key_vault:
                del enterprise_key_manager.key_vault[secret_key_id]
                del enterprise_key_manager.key_metadata[secret_key_id]
            
            # Remove backup codes
            backup_key_id = f"backup_codes_{user_id}"
            if backup_key_id in enterprise_key_manager.key_vault:
                del enterprise_key_manager.key_vault[backup_key_id]
                del enterprise_key_manager.key_metadata[backup_key_id]
            
            # Reset failed attempts
            self._reset_failed_attempts(user_id)
            
            security_logger.info(f"MFA disabled for user: {user_id}")
            return True
            
        except Exception as e:
            security_logger.error(f"Failed to disable MFA for user {user_id}: {str(e)}")
            return False


def render_mfa_setup_ui(user_id: str):
    """Render MFA setup interface in Streamlit"""
    mfa = MultiFactorAuth()
    
    st.subheader("🔐 Multi-Factor Authentication Setup")
    
    if mfa.is_mfa_enabled(user_id):
        st.success("✅ MFA is already enabled for your account")
        
        # Show backup codes status
        remaining_codes = mfa.get_remaining_backup_codes(user_id)
        if remaining_codes > 0:
            st.info(f"📋 You have {remaining_codes} backup codes remaining")
        else:
            st.warning("⚠️ No backup codes remaining. Generate new ones!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Regenerate Backup Codes"):
                new_codes = mfa.generate_backup_codes(user_id)
                st.success("New backup codes generated!")
                with st.expander("📋 Your New Backup Codes", expanded=True):
                    st.warning("⚠️ Save these codes in a secure location. They will not be shown again!")
                    for i, code in enumerate(new_codes, 1):
                        st.code(f"{i:2d}. {code}")
        
        with col2:
            if st.button("❌ Disable MFA"):
                if mfa.disable_mfa(user_id):
                    st.success("MFA has been disabled")
                    st.rerun()
                else:
                    st.error("Failed to disable MFA")
    
    else:
        st.info("🛡️ Enhance your account security with multi-factor authentication")
        
        if st.button("🚀 Enable MFA"):
            # Generate TOTP secret
            secret = mfa.generate_totp_secret(user_id)
            
            # Generate QR code
            qr_image = mfa.generate_qr_code(user_id, secret)
            
            st.success("MFA setup initiated!")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Step 1: Scan QR Code**")
                st.markdown("Use an authenticator app (Google Authenticator, Authy, etc.) to scan this QR code:")
                st.image(f"data:image/png;base64,{qr_image}", width=250)
            
            with col2:
                st.markdown("**Step 2: Enter Verification Code**")
                verification_code = st.text_input("Enter 6-digit code from your authenticator app:")
                
                if st.button("Verify & Complete Setup"):
                    if verification_code and mfa.verify_totp(user_id, verification_code):
                        # Generate backup codes
                        backup_codes = mfa.generate_backup_codes(user_id)
                        
                        st.success("🎉 MFA setup completed successfully!")
                        
                        with st.expander("📋 Your Backup Codes", expanded=True):
                            st.warning("⚠️ Save these codes in a secure location. They will not be shown again!")
                            for i, code in enumerate(backup_codes, 1):
                                st.code(f"{i:2d}. {code}")
                        
                        st.rerun()
                    else:
                        st.error("Invalid verification code. Please try again.")


def render_mfa_login_ui(user_id: str) -> bool:
    """Render MFA login interface and return authentication status"""
    mfa = MultiFactorAuth()
    
    if not mfa.is_mfa_enabled(user_id):
        return True  # MFA not enabled, skip
    
    st.subheader("🔐 Multi-Factor Authentication")
    
    # Check if user is locked out
    if mfa._is_user_locked_out(user_id):
        st.error("🚫 Account temporarily locked due to multiple failed attempts. Please try again later.")
        return False
    
    auth_method = st.radio(
        "Choose authentication method:",
        ["📱 Authenticator App", "🔑 Backup Code"]
    )
    
    if auth_method == "📱 Authenticator App":
        totp_code = st.text_input("Enter 6-digit code from your authenticator app:")
        
        if st.button("Verify Code"):
            if totp_code and mfa.verify_totp(user_id, totp_code):
                st.success("✅ Authentication successful!")
                return True
            else:
                st.error("❌ Invalid code. Please try again.")
                return False
    
    else:  # Backup Code
        backup_code = st.text_input("Enter backup code:")
        
        if st.button("Verify Backup Code"):
            if backup_code and mfa.verify_backup_code(user_id, backup_code):
                st.success("✅ Authentication successful!")
                
                remaining = mfa.get_remaining_backup_codes(user_id)
                if remaining <= 2:
                    st.warning(f"⚠️ Only {remaining} backup codes remaining. Consider regenerating them.")
                
                return True
            else:
                st.error("❌ Invalid backup code. Please try again.")
                return False
    
    return False


# Global MFA instance
mfa_system = MultiFactorAuth()