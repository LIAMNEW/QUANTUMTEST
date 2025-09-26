"""
API Security Middleware
Enterprise-grade API rate limiting, DDoS protection, and security middleware
"""

import time
import hashlib
import json
import redis
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import streamlit as st
from enterprise_quantum_security import security_logger

class SecurityMiddleware:
    """Enterprise security middleware with rate limiting and DDoS protection"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = None
        self.rate_limits = {}
        self.blocked_ips = set()
        self.suspicious_patterns = {}
        
        # Initialize Redis connection if available
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            security_logger.info("Redis connection established for rate limiting")
        except Exception:
            security_logger.warning("Redis not available, using in-memory rate limiting")
            self.redis_client = None
        
        # Rate limiting configuration
        self.rate_limit_config = {
            "default": {"requests": 100, "window": 60},  # 100 requests per minute
            "auth": {"requests": 5, "window": 60},       # 5 auth attempts per minute
            "api": {"requests": 1000, "window": 60},     # 1000 API calls per minute
            "upload": {"requests": 10, "window": 300},   # 10 uploads per 5 minutes
        }
        
        # DDoS protection thresholds
        self.ddos_thresholds = {
            "requests_per_second": 10,
            "concurrent_connections": 50,
            "suspicious_patterns": 5
        }
    
    def check_rate_limit(self, identifier: str, endpoint_type: str = "default") -> Tuple[bool, Dict]:
        """Check if request is within rate limits"""
        config = self.rate_limit_config.get(endpoint_type, self.rate_limit_config["default"])
        current_time = int(time.time())
        window_start = current_time - config["window"]
        
        if self.redis_client:
            return self._check_rate_limit_redis(identifier, endpoint_type, config, current_time, window_start)
        else:
            return self._check_rate_limit_memory(identifier, endpoint_type, config, current_time, window_start)
    
    def _check_rate_limit_redis(self, identifier: str, endpoint_type: str, config: Dict, current_time: int, window_start: int) -> Tuple[bool, Dict]:
        """Redis-based rate limiting"""
        key = f"rate_limit:{endpoint_type}:{identifier}"
        
        try:
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, config["window"])
            
            results = pipe.execute()
            request_count = results[1]
            
            rate_limit_info = {
                "limit": config["requests"],
                "remaining": max(0, config["requests"] - request_count),
                "reset_time": current_time + config["window"],
                "window": config["window"]
            }
            
            if request_count >= config["requests"]:
                security_logger.warning(f"Rate limit exceeded for {identifier} on {endpoint_type}")
                return False, rate_limit_info
            
            return True, rate_limit_info
            
        except Exception as e:
            security_logger.error(f"Redis rate limiting error: {str(e)}")
            return True, {}  # Fail open
    
    def _check_rate_limit_memory(self, identifier: str, endpoint_type: str, config: Dict, current_time: int, window_start: int) -> Tuple[bool, Dict]:
        """Memory-based rate limiting"""
        key = f"{endpoint_type}:{identifier}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Remove old entries
        self.rate_limits[key] = [t for t in self.rate_limits[key] if t > window_start]
        
        # Add current request
        self.rate_limits[key].append(current_time)
        
        request_count = len(self.rate_limits[key])
        
        rate_limit_info = {
            "limit": config["requests"],
            "remaining": max(0, config["requests"] - request_count),
            "reset_time": current_time + config["window"],
            "window": config["window"]
        }
        
        if request_count > config["requests"]:
            security_logger.warning(f"Rate limit exceeded for {identifier} on {endpoint_type}")
            return False, rate_limit_info
        
        return True, rate_limit_info
    
    def detect_ddos_patterns(self, ip_address: str, user_agent: str, request_data: Dict) -> bool:
        """Detect potential DDoS attack patterns"""
        current_time = time.time()
        
        # Check for rapid successive requests
        if self._check_rapid_requests(ip_address, current_time):
            return True
        
        # Check for suspicious user agent patterns
        if self._check_suspicious_user_agent(user_agent):
            return True
        
        # Check for payload anomalies
        if self._check_payload_anomalies(request_data):
            return True
        
        return False
    
    def _check_rapid_requests(self, ip_address: str, current_time: float) -> bool:
        """Check for rapid successive requests from same IP"""
        if ip_address not in self.suspicious_patterns:
            self.suspicious_patterns[ip_address] = []
        
        # Clean old entries (last 10 seconds)
        self.suspicious_patterns[ip_address] = [
            t for t in self.suspicious_patterns[ip_address] 
            if current_time - t < 10
        ]
        
        self.suspicious_patterns[ip_address].append(current_time)
        
        # Check if too many requests in short time
        if len(self.suspicious_patterns[ip_address]) > self.ddos_thresholds["requests_per_second"]:
            security_logger.warning(f"Rapid requests detected from IP: {ip_address}")
            return True
        
        return False
    
    def _check_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check for suspicious user agent patterns"""
        suspicious_patterns = [
            "bot", "crawler", "spider", "scraper", "scanner",
            "curl", "wget", "python-requests", "go-http-client"
        ]
        
        if not user_agent or user_agent.lower() in ["", "unknown"]:
            return True
        
        user_agent_lower = user_agent.lower()
        for pattern in suspicious_patterns:
            if pattern in user_agent_lower:
                security_logger.info(f"Suspicious user agent detected: {user_agent}")
                return True
        
        return False
    
    def _check_payload_anomalies(self, request_data: Dict) -> bool:
        """Check for payload anomalies that might indicate attacks"""
        if not request_data:
            return False
        
        # Check for excessively large payloads
        payload_str = json.dumps(request_data)
        if len(payload_str) > 100000:  # 100KB limit
            security_logger.warning("Excessively large payload detected")
            return True
        
        # Check for SQL injection patterns
        sql_patterns = ["'", "\"", "drop", "delete", "insert", "update", "select", "union"]
        for value in str(request_data).lower().split():
            if any(pattern in value for pattern in sql_patterns):
                security_logger.warning("Potential SQL injection attempt detected")
                return True
        
        return False
    
    def block_ip(self, ip_address: str, duration: int = 3600) -> bool:
        """Block IP address for specified duration (seconds)"""
        try:
            self.blocked_ips.add(ip_address)
            
            if self.redis_client:
                # Store in Redis with expiration
                self.redis_client.setex(f"blocked_ip:{ip_address}", duration, "1")
            
            security_logger.warning(f"IP blocked: {ip_address} for {duration} seconds")
            return True
            
        except Exception as e:
            security_logger.error(f"Failed to block IP {ip_address}: {str(e)}")
            return False
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        if ip_address in self.blocked_ips:
            return True
        
        if self.redis_client:
            try:
                return self.redis_client.exists(f"blocked_ip:{ip_address}")
            except Exception:
                pass
        
        return False
    
    def get_security_metrics(self) -> Dict:
        """Get security metrics for monitoring"""
        current_time = time.time()
        
        # Count recent rate limit violations
        recent_violations = 0
        for key, timestamps in self.rate_limits.items():
            recent_violations += len([t for t in timestamps if current_time - t < 300])  # Last 5 minutes
        
        # Count blocked IPs
        blocked_count = len(self.blocked_ips)
        
        # Count suspicious activity
        suspicious_count = len([
            ip for ip, patterns in self.suspicious_patterns.items()
            if len([t for t in patterns if current_time - t < 300]) > 0
        ])
        
        return {
            "rate_limit_violations_5min": recent_violations,
            "blocked_ips": blocked_count,
            "suspicious_activity_5min": suspicious_count,
            "active_rate_limits": len(self.rate_limits),
            "timestamp": datetime.now().isoformat()
        }


class StreamlitSecurityWrapper:
    """Security wrapper for Streamlit applications"""
    
    def __init__(self):
        self.security_middleware = SecurityMiddleware()
        self.session_security = {}
    
    def get_client_ip(self) -> str:
        """Get client IP address from Streamlit session"""
        # In Streamlit, we simulate client IP since it's not directly available
        session_id = id(st.session_state) if hasattr(st, 'session_state') else "unknown"
        return f"client_{hash(session_id) % 1000000}"
    
    def enforce_rate_limit(self, endpoint_type: str = "default") -> bool:
        """Enforce rate limiting for current session"""
        client_id = self.get_client_ip()
        allowed, rate_info = self.security_middleware.check_rate_limit(client_id, endpoint_type)
        
        if not allowed:
            st.error(f"🚫 Rate limit exceeded. Please wait {rate_info.get('window', 60)} seconds before trying again.")
            st.stop()
        
        return True
    
    def check_ddos_protection(self, request_data: Dict = None) -> bool:
        """Check DDoS protection for current request"""
        client_id = self.get_client_ip()
        user_agent = "streamlit-app"  # Streamlit doesn't provide real user agent
        
        if self.security_middleware.is_ip_blocked(client_id):
            st.error("🚫 Your access has been temporarily blocked due to suspicious activity.")
            st.stop()
        
        if self.security_middleware.detect_ddos_patterns(client_id, user_agent, request_data or {}):
            self.security_middleware.block_ip(client_id, 300)  # Block for 5 minutes
            st.error("🚫 Suspicious activity detected. Access temporarily blocked.")
            st.stop()
        
        return True
    
    def render_security_dashboard(self):
        """Render security monitoring dashboard"""
        st.subheader("🛡️ Security Monitoring Dashboard")
        
        metrics = self.security_middleware.get_security_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rate Limit Violations (5min)", metrics["rate_limit_violations_5min"])
        
        with col2:
            st.metric("Blocked IPs", metrics["blocked_ips"])
        
        with col3:
            st.metric("Suspicious Activity (5min)", metrics["suspicious_activity_5min"])
        
        with col4:
            st.metric("Active Rate Limits", metrics["active_rate_limits"])
        
        # Rate limiting configuration
        with st.expander("⚙️ Rate Limiting Configuration"):
            for endpoint, config in self.security_middleware.rate_limit_config.items():
                st.write(f"**{endpoint.title()}**: {config['requests']} requests per {config['window']} seconds")
        
        # Recent security events
        with st.expander("📋 Recent Security Events"):
            st.info("Security events would be displayed here in a production environment")


# Global security instances
security_middleware = SecurityMiddleware()
streamlit_security = StreamlitSecurityWrapper()