"""
🛰️ SLCCI Standalone Dashboard
==============================
Streamlit entry point for SLCCI-only analysis.
No CMEMS, no DTU, no comparison mode.
"""

import streamlit as st

st.set_page_config(
    page_title="🛰️ SLCCI Satellite Altimetry",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app_slcci.main import run_slcci_app

if __name__ == "__main__":
    run_slcci_app()
