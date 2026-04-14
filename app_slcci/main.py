"""
SLCCI Standalone Dashboard — Main Entry Point
==============================================
Streamlit app for SLCCI-only analysis.
No CMEMS, no DTU, no comparison mode.
"""

import sys
from pathlib import Path

# Ensure project root on path
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
from app_slcci.state import init_slcci_state
from app_slcci.sidebar import render_slcci_sidebar
from app_slcci.tabs import render_slcci_tabs
from app.styles import apply_custom_css

# Register Plotly template from main app
try:
    from app.components.chart_style import register_plotly_template
    register_plotly_template()
except ImportError:
    pass


def run_slcci_app():
    """Main entry point for SLCCI standalone dashboard."""
    apply_custom_css()
    init_slcci_state()

    st.markdown(
        '<div class="main-header">🛰️ SLCCI Satellite Altimetry</div>',
        unsafe_allow_html=True,
    )

    config = render_slcci_sidebar()
    st.session_state.app_config = config

    render_slcci_tabs(config)


if __name__ == "__main__":
    run_slcci_app()
