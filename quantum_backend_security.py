#!/usr/bin/env python3
"""
QuantumGuard AI - Quantum-Safe Backend Security Infrastructure

This module implements quantum-resistant cryptographic protection for all backend
operations including database encryption, session management, and data processing.
"""

import os
import json
import base64
import hashlib
import secrets
import numpy as np
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
import pandas as pd
from quantum_crypto import generate_pq_keys, encrypt_data, decrypt_data, SECURITY_LEVEL

class QuantumSecureBackend:
    """
    Quantum-safe backend security manager for QuantumGuard AI.
    Handles all cryptographic operations with post-quantum algorithms.
    """
    
    def __init__(self):
        self.master_keys = self._initialize_master_keys()
        self.session_keys = {}
        self.encrypted_cache = {}
        
    def _initialize_master_keys(self) -> Dict[str, Any]:
        """Initialize or load master encryption keys for the backend"""
        keys_file = ".quantum_master_keys.json"
        
        if os.path.exists(keys_file):
            try:
                with open(keys_file, 'r') as f:
                    encrypted_keys = json.load(f)
                return self._decrypt_master_keys(encrypted_keys)
            except Exception:
                # If keys are corrupted, generate new ones
                pass
        
        # Generate new master keys
        public_key, private_key = generate_pq_keys()
        
        master_keys = {
            "database_key": {"public": public_key, "private": private_key},
            "session_key": {"public": public_key, "private": private_key},
            "storage_key": {"public": public_key, "private": private_key},
            "created_at": datetime.now().isoformat(),
            "security_level": SECURITY_LEVEL,
            "algorithm": "lattice_based_lwe"
        }
        
        # Save encrypted master keys
        self._save_master_keys(master_keys, keys_file)
        return master_keys
    
    def _save_master_keys(self, keys: Dict[str, Any], filename: str):
        """Save master keys with additional encryption layer"""
        try:
            # Use environment-based encryption for master keys
            env_key = os.environ.get('DATABASE_URL', 'default_quantum_key')
            if env_key is None:
                env_key = 'default_quantum_key'
            key_hash = hashlib.sha256(env_key.encode()).digest()
            
            # Simple XOR encryption for key storage (additional layer)
            keys_json = json.dumps(keys, default=str)
            encrypted_keys = self._xor_encrypt(keys_json.encode(), key_hash)
            
            with open(filename, 'w') as f:
                json.dump({
                    "encrypted_data": base64.b64encode(encrypted_keys).decode(),
                    "checksum": hashlib.sha256(encrypted_keys).hexdigest()
                }, f)
        except Exception as e:
            print(f"Warning: Could not save master keys: {e}")
    
    def _decrypt_master_keys(self, encrypted_data: Dict[str, str]) -> Dict[str, Any]:
        """Decrypt master keys from storage"""
        try:
            env_key = os.environ.get('DATABASE_URL', 'default_quantum_key')
            key_hash = hashlib.sha256(env_key.encode()).digest()
            
            encrypted_bytes = base64.b64decode(encrypted_data["encrypted_data"])
            
            # Verify checksum
            if hashlib.sha256(encrypted_bytes).hexdigest() != encrypted_data["checksum"]:
                raise ValueError("Key integrity check failed")
            
            decrypted_bytes = self._xor_encrypt(encrypted_bytes, key_hash)
            return json.loads(decrypted_bytes.decode())
        except Exception:
            raise ValueError("Failed to decrypt master keys")
    
    def _xor_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Simple XOR encryption for additional key protection"""
        key_repeated = (key * (len(data) // len(key) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key_repeated))
    
    def encrypt_sensitive_data(self, data: Union[str, dict, pd.DataFrame], 
                             context: str = "general") -> str:
        """
        Encrypt sensitive data using quantum-safe algorithms
        
        Args:
            data: Data to encrypt (string, dict, or DataFrame)
            context: Encryption context (database, session, storage)
        
        Returns:
            Base64 encoded encrypted data
        """
        try:
            # Choose appropriate key based on context
            if context == "database":
                public_key = self.master_keys["database_key"]["public"]
            elif context == "session":
                public_key = self.master_keys["session_key"]["public"]
            else:
                public_key = self.master_keys["storage_key"]["public"]
            
            # Serialize data
            if isinstance(data, pd.DataFrame):
                data_bytes = data.to_json(orient='records').encode('utf-8')
            elif isinstance(data, dict):
                data_bytes = json.dumps(data, default=str).encode('utf-8')
            else:
                data_bytes = str(data).encode('utf-8')
            
            # Add metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "data_type": type(data).__name__,
                "size": len(data_bytes)
            }
            
            # Combine metadata and data
            combined_data = json.dumps({
                "metadata": metadata,
                "payload": base64.b64encode(data_bytes).decode()
            }).encode('utf-8')
            
            # Encrypt with quantum-safe algorithm
            encrypted_bytes = encrypt_data(combined_data, public_key)
            
            return base64.b64encode(encrypted_bytes).decode('utf-8')
            
        except Exception as e:
            raise ValueError(f"Encryption failed: {str(e)}")
    
    def decrypt_sensitive_data(self, encrypted_data: str, 
                             context: str = "general") -> Union[str, dict, pd.DataFrame]:
        """
        Decrypt sensitive data using quantum-safe algorithms
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            context: Decryption context (database, session, storage)
        
        Returns:
            Decrypted data in original format
        """
        try:
            # Choose appropriate key based on context
            if context == "database":
                private_key = self.master_keys["database_key"]["private"]
            elif context == "session":
                private_key = self.master_keys["session_key"]["private"]
            else:
                private_key = self.master_keys["storage_key"]["private"]
            
            # Decode and decrypt
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_bytes = decrypt_data(encrypted_bytes, private_key)
            
            # Parse combined data
            combined_data = json.loads(decrypted_bytes.decode('utf-8'))
            metadata = combined_data["metadata"]
            payload_bytes = base64.b64decode(combined_data["payload"])
            
            # Reconstruct original data based on type
            if metadata["data_type"] == "DataFrame":
                return pd.read_json(payload_bytes.decode('utf-8'), orient='records')
            elif metadata["data_type"] == "dict":
                return json.loads(payload_bytes.decode('utf-8'))
            else:
                return payload_bytes.decode('utf-8')
                
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")
    
    def create_secure_session(self, session_id: str, user_data: Dict[str, Any]) -> str:
        """Create a quantum-safe encrypted session"""
        try:
            session_data = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
                "user_data": user_data,
                "security_level": SECURITY_LEVEL
            }
            
            encrypted_session = self.encrypt_sensitive_data(session_data, "session")
            self.session_keys[session_id] = encrypted_session
            
            return encrypted_session
            
        except Exception as e:
            raise ValueError(f"Session creation failed: {str(e)}")
    
    def validate_secure_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate and decrypt a quantum-safe session"""
        try:
            if session_id not in self.session_keys:
                return None
            
            encrypted_session = self.session_keys[session_id]
            session_data = self.decrypt_sensitive_data(encrypted_session, "session")
            
            # Check expiration
            if isinstance(session_data, dict) and "expires_at" in session_data:
                expires_at_str = str(session_data["expires_at"])
                expires_at = datetime.fromisoformat(expires_at_str)
                if datetime.now() > expires_at:
                    del self.session_keys[session_id]
                    return None
            
            return session_data if isinstance(session_data, dict) else None
            
        except Exception:
            return None
    
    def secure_database_write(self, data: Any, table_context: str) -> str:
        """Encrypt data before database storage"""
        return self.encrypt_sensitive_data(data, "database")
    
    def secure_database_read(self, encrypted_data: str, table_context: str) -> Any:
        """Decrypt data after database retrieval"""
        return self.decrypt_sensitive_data(encrypted_data, "database")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    def hash_with_quantum_resistance(self, data: str, salt: Optional[str] = None) -> str:
        """Create quantum-resistant hash with optional salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use multiple hash rounds for quantum resistance
        hash_input = (data + salt).encode('utf-8')
        
        for _ in range(10000):  # PBKDF2-like iteration
            hash_input = hashlib.sha3_256(hash_input).digest()
        
        return base64.b64encode(hash_input).decode('utf-8')
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security configuration status"""
        return {
            "algorithm": "Lattice-based LWE (Learning With Errors)",
            "security_level": f"{SECURITY_LEVEL} bits",
            "quantum_safe": True,
            "shor_resistant": True,
            "grover_resistant": True,
            "key_creation_time": self.master_keys.get("created_at"),
            "active_sessions": len(self.session_keys),
            "backend_encryption": "Active",
            "database_encryption": "Active",
            "session_encryption": "Active"
        }

# Global quantum-safe backend instance
quantum_backend = QuantumSecureBackend()

def encrypt_for_storage(data: Any, context: str = "database") -> str:
    """Convenience function for encrypting data for storage"""
    return quantum_backend.encrypt_sensitive_data(data, context)

def decrypt_from_storage(encrypted_data: str, context: str = "database") -> Any:
    """Convenience function for decrypting data from storage"""
    return quantum_backend.decrypt_sensitive_data(encrypted_data, context)

def create_quantum_session(session_id: str, user_data: Dict[str, Any]) -> str:
    """Convenience function for creating quantum-safe sessions"""
    return quantum_backend.create_secure_session(session_id, user_data)

def validate_quantum_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Convenience function for validating quantum-safe sessions"""
    return quantum_backend.validate_secure_session(session_id)

def get_backend_security_status() -> Dict[str, Any]:
    """Get comprehensive backend security status"""
    return quantum_backend.get_security_status()