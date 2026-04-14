"""
Custom CSS Styles
=================
"""

import streamlit as st


def apply_custom_css():
    """Apply custom CSS styling to the app."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        .upload-box {
            border: 2px dashed #1E88E5;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            background-color: #f0f8ff;
        }
        .info-box {
            background-color: #e3f2fd;
            border-left: 4px solid #1E88E5;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #e8f5e9;
            border-left: 4px solid #4CAF50;
            padding: 1rem;
            border-radius: 5px;
        }
        .warning-box {
            background-color: #fff3e0;
            border-left: 4px solid #FF9800;
            padding: 1rem;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
