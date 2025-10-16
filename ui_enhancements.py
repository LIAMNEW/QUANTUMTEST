"""
QuantumGuard AI - Modern UI Enhancement Module
Provides glassmorphism effects, smooth animations, loading states, and responsive components
"""

import streamlit as st
import time
from typing import Optional, List, Dict, Any

# ============================================================================
# MODERN GLASSMORPHISM CSS WITH SMOOTH ANIMATIONS
# ============================================================================

MODERN_CSS = """
<style>
    /* ========== GLASSMORPHISM EFFECTS ========== */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(102, 126, 234, 0.5);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .glass-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* ========== MODERN METRIC CARDS ========== */
    .modern-metric {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .modern-metric::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .modern-metric:hover::before {
        left: 100%;
    }
    
    .modern-metric:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 40px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* ========== LOADING ANIMATIONS ========== */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    @keyframes slideInUp {
        from {
            transform: translateY(30px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    .skeleton {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.05) 25%, rgba(255, 255, 255, 0.1) 50%, rgba(255, 255, 255, 0.05) 75%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
        border-radius: 12px;
        height: 100px;
        margin: 10px 0;
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    .slide-in-up {
        animation: slideInUp 0.6s ease-out;
    }
    
    /* ========== MODERN BUTTONS ========== */
    .modern-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .modern-button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .modern-button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .modern-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .modern-button:active {
        transform: translateY(0);
    }
    
    /* ========== PROGRESS INDICATORS ========== */
    .modern-progress {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        border-radius: 50px;
        height: 8px;
        overflow: hidden;
        position: relative;
    }
    
    .modern-progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% 100%;
        height: 100%;
        border-radius: 50px;
        animation: progressShine 2s linear infinite;
        transition: width 0.3s ease;
    }
    
    @keyframes progressShine {
        0% {
            background-position: 200% 0;
        }
        100% {
            background-position: -200% 0;
        }
    }
    
    /* ========== ALERT MESSAGES ========== */
    .alert-success {
        background: linear-gradient(135deg, rgba(67, 233, 123, 0.2) 0%, rgba(56, 249, 215, 0.2) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(67, 233, 123, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        animation: slideInUp 0.5s ease-out;
    }
    
    .alert-error {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2) 0%, rgba(238, 90, 82, 0.2) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        animation: slideInUp 0.5s ease-out;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(250, 112, 154, 0.2) 0%, rgba(254, 225, 64, 0.2) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(250, 112, 154, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        animation: slideInUp 0.5s ease-out;
    }
    
    .alert-info {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.2) 0%, rgba(0, 242, 254, 0.2) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        animation: slideInUp 0.5s ease-out;
    }
    
    /* ========== TOOLTIP ========== */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background: rgba(0, 0, 0, 0.9);
        backdrop-filter: blur(10px);
        color: #fff;
        text-align: center;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s, visibility 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* ========== MOBILE RESPONSIVENESS ========== */
    @media (max-width: 768px) {
        .glass-card, .glass-header {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .modern-metric {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .modern-button {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
    }
    
    /* ========== ENHANCED TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* ========== SCROLLBAR STYLING ========== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
"""

# ============================================================================
# COMPONENT FUNCTIONS
# ============================================================================

def apply_modern_css():
    """Apply modern glassmorphism CSS to the Streamlit app"""
    st.markdown(MODERN_CSS, unsafe_allow_html=True)

def glass_card(title: str, content: str, icon: str = ""):
    """
    Create a modern glassmorphism card
    
    Args:
        title: Card title
        content: Card content (HTML supported)
        icon: Optional emoji icon
    """
    st.markdown(f"""
    <div class="glass-card slide-in-up">
        <h3>{icon} {title}</h3>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def glass_header(title: str, subtitle: str = "", icon: str = ""):
    """
    Create a modern glassmorphism header
    
    Args:
        title: Header title
        subtitle: Optional subtitle
        icon: Optional emoji icon
    """
    subtitle_html = f'<p style="font-size: 1rem; margin-top: 0.5rem; opacity: 0.9;">{subtitle}</p>' if subtitle else ''
    
    st.markdown(f"""
    <div class="glass-header fade-in">
        <h2 style="margin: 0;">{icon} {title}</h2>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)

def modern_metric(label: str, value: str, icon: str = "", delta: Optional[str] = None):
    """
    Create a modern animated metric card
    
    Args:
        label: Metric label
        value: Metric value
        icon: Optional emoji icon
        delta: Optional delta/change indicator
    """
    delta_html = f'<p style="font-size: 0.9rem; color: #43e97b; margin-top: 0.5rem;">{delta}</p>' if delta else ''
    
    st.markdown(f"""
    <div class="modern-metric slide-in-up">
        <p style="font-size: 0.9rem; opacity: 0.8; margin: 0;">{icon} {label}</p>
        <h2 style="margin: 0.5rem 0; font-size: 2rem;">{value}</h2>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def loading_skeleton(count: int = 3):
    """
    Display loading skeleton screens
    
    Args:
        count: Number of skeleton placeholders to show
    """
    for i in range(count):
        st.markdown(f'<div class="skeleton"></div>', unsafe_allow_html=True)

def modern_alert(message: str, alert_type: str = "info"):
    """
    Display a modern alert message
    
    Args:
        message: Alert message
        alert_type: Type of alert (success, error, warning, info)
    """
    icon_map = {
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️"
    }
    
    icon = icon_map.get(alert_type, "ℹ️")
    
    st.markdown(f"""
    <div class="alert-{alert_type}">
        <strong>{icon} {message}</strong>
    </div>
    """, unsafe_allow_html=True)

def modern_progress_bar(value: float, max_value: float = 100, label: str = ""):
    """
    Display a modern animated progress bar
    
    Args:
        value: Current progress value
        max_value: Maximum value (default 100)
        label: Optional progress label
    """
    percentage = (value / max_value) * 100
    
    label_html = f'<p style="margin-bottom: 0.5rem; font-size: 0.9rem;">{label}</p>' if label else ''
    
    st.markdown(f"""
    {label_html}
    <div class="modern-progress">
        <div class="modern-progress-bar" style="width: {percentage}%;"></div>
    </div>
    <p style="text-align: right; font-size: 0.8rem; margin-top: 0.25rem; opacity: 0.8;">{value}/{max_value}</p>
    """, unsafe_allow_html=True)

def tooltip_text(text: str, tooltip: str):
    """
    Create text with tooltip on hover
    
    Args:
        text: Display text
        tooltip: Tooltip content
    """
    st.markdown(f"""
    <div class="tooltip">{text}
        <span class="tooltiptext">{tooltip}</span>
    </div>
    """, unsafe_allow_html=True)

def animated_spinner(message: str = "Processing..."):
    """
    Show an animated loading spinner with message
    
    Args:
        message: Loading message
    """
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem;">
        <div class="modern-progress" style="width: 200px; margin: 0 auto 1rem;">
            <div class="modern-progress-bar" style="width: 100%;"></div>
        </div>
        <p style="opacity: 0.8;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

def create_export_button(data: Any, filename: str, file_type: str = "csv", label: str = "Export"):
    """
    Create a modern export button for data
    
    Args:
        data: Data to export (DataFrame, dict, etc.)
        filename: Name of the file to export
        file_type: Type of file (csv, json, excel)
        label: Button label
    """
    import pandas as pd
    import json
    import io
    
    if file_type == "csv" and isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        st.download_button(
            label=f"📥 {label} as CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            key=f"export_{filename}_{file_type}"
        )
    elif file_type == "json":
        if isinstance(data, pd.DataFrame):
            json_str = data.to_json(orient='records', indent=2)
        else:
            json_str = json.dumps(data, indent=2)
        st.download_button(
            label=f"📥 {label} as JSON",
            data=json_str,
            file_name=filename,
            mime="application/json",
            key=f"export_{filename}_{file_type}"
        )
    elif file_type == "excel" and isinstance(data, pd.DataFrame):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            data.to_excel(writer, index=False, sheet_name='Data')
        st.download_button(
            label=f"📥 {label} as Excel",
            data=buffer.getvalue(),
            file_name=filename,
            mime="application/vnd.ms-excel",
            key=f"export_{filename}_{file_type}"
        )
