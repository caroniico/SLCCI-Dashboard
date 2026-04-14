"""
SLCCI Standalone — Tabs
========================
All analysis tabs for SLCCI-only dashboard.
Reuses rendering functions from the main app where possible.
"""

import sys
from pathlib import Path

# Ensure project root is on path (for src.* imports)
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any

from app_slcci.state import SLCCIAppConfig, get_slcci_data

# Re-use chart rendering from the main app
from app.components.charts import (
    render_slope_timeline,
    render_dot_profile,
    render_spatial_map,
    render_geostrophic_velocity,
    render_volume_transport_tab,
    get_pass_data_attributes,
)
from app.components.salt_flux_clean import render_salt_flux_clean
from app.components.freshwater_transport_clean import render_freshwater_transport_clean
from app.components.salinity_profile_tab import render_salinity_profile_tab

# SLCCI color
COLOR_SLCCI = "#FF7F0E"


def _ds_info() -> dict:
    """Standard SLCCI dataset info dict."""
    data = get_slcci_data()
    pass_num = getattr(data, "pass_number", "") if data else ""
    name = f"SLCCI Pass {pass_num}" if pass_num else "SLCCI"
    return {
        "emoji": "🟠",
        "name": name,
        "color": COLOR_SLCCI,
        "type": "slcci",
    }


def render_slcci_tabs(config: SLCCIAppConfig):
    """
    Render the 10 analysis tabs for SLCCI data.

    Matches the full tab set of CMEMS L4 / DTUSpace:
    1. Slope Timeline
    2. DOT Profile
    3. Spatial Map
    4. Monthly Analysis
    5. Geostrophic Velocity
    6. Volume Transport
    7. Freshwater Transport
    8. Salinity Profile
    9. Salt Flux
    10. Export
    """
    data = get_slcci_data()
    if data is None:
        _render_welcome()
        return

    info = _ds_info()

    (tab_slope, tab_dot, tab_spatial, tab_monthly, tab_vgeo,
     tab_vt, tab_fw, tab_sal, tab_sf, tab_export) = st.tabs([
        "📈 Slope Timeline",
        "🌊 DOT Profile",
        "🗺️ Spatial Map",
        "📅 Monthly Analysis",
        "🌀 Geostrophic Velocity",
        "🚢 Volume Transport",
        "💧 Freshwater Transport",
        "🧪 Salinity Profile",
        "🧂 Salt Flux",
        "📥 Export",
    ])

    # We delegate to the unified renderers from the main app.
    # They accept any dataset that has the PassData interface.
    from app.components.tabs import (
        _render_unified_slope_timeline,
        _render_unified_dot_profile,
        _render_unified_spatial_map,
        _render_unified_monthly_analysis,
        _render_unified_geostrophic_velocity,
        _render_unified_export_tab,
        _render_volume_transport_tab_cmems_l4,
    )

    # Create a lightweight AppConfig wrapper compatible with main app
    from app.state import AppConfig
    app_cfg = AppConfig(
        selected_gate=config.selected_gate,
        lon_bin_size=config.lon_bin_size,
        depth_method=config.depth_method,
        fixed_depth_m=config.fixed_depth_m,
        gebco_nc_path=config.gebco_nc_path,
    )

    with tab_slope:
        _render_unified_slope_timeline(data, app_cfg, info)
    with tab_dot:
        _render_unified_dot_profile(data, app_cfg, info)
    with tab_spatial:
        _render_unified_spatial_map(data, app_cfg, info)
    with tab_monthly:
        _render_unified_monthly_analysis(data, app_cfg, info)
    with tab_vgeo:
        _render_unified_geostrophic_velocity(data, app_cfg, info)
    with tab_vt:
        _render_volume_transport_tab_cmems_l4(data, app_cfg)
    with tab_fw:
        render_freshwater_transport_clean(data, app_cfg, info)
    with tab_sal:
        render_salinity_profile_tab(data, app_cfg, info)
    with tab_sf:
        render_salt_flux_clean(data, app_cfg, info)
    with tab_export:
        _render_unified_export_tab(data, app_cfg, info)


def _render_welcome():
    """Welcome screen when no data is loaded."""
    st.markdown("## 🛰️ SLCCI Satellite Altimetry Dashboard")
    st.markdown("*ESA Sea Level CCI — Jason-2 Along-Track Analysis*")

    st.info("""
    **Getting Started:**

    1. Select a **gate** from the sidebar
    2. Set **pass number** and **cycle range**
    3. Click **🚀 Load SLCCI Data**

    The dashboard will show:
    - 📈 DOT slope time series
    - 🌊 Along-gate DOT profiles
    - 🌀 Geostrophic velocity (from DOT slope)
    - 🚢 Volume / Freshwater / Salt transport
    - 🧪 Salinity profiles (CCI SSS v5.5)
    """)

    st.markdown("### 📡 About SLCCI")
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | **Dataset** | SLCCI Altimeter Database V2 |
    | **Satellite** | Jason-2 |
    | **Variable** | Corrected SSH (corssh) |
    | **Geoid** | TUM_ogmoc.nc |
    | **DOT** | corssh − geoid |
    | **Slope** | Linear fit of DOT vs distance (m/100km) |
    | **Velocity** | v = −(g/f) × ∂(DOT)/∂x |
    """)
