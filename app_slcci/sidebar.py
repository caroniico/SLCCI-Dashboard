"""
SLCCI Standalone — Sidebar
===========================
Gate selection + SLCCI data loading.
No CMEMS, no DTU references.
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
import yaml

from app_slcci.state import SLCCIAppConfig, store_slcci_data, clear_slcci_data

# Gate shapefile directory
GATES_DIR = Path(__file__).parent.parent / "gates"

# Config files
CONFIG_DIR = Path(__file__).parent.parent / "config"


def _list_gates() -> dict:
    """
    List available gates from shapefiles in gates/ directory.
    Returns dict: {display_name: shapefile_path}
    """
    gates = {}
    if not GATES_DIR.exists():
        return gates

    for shp in sorted(GATES_DIR.glob("*.shp")):
        name = shp.stem.replace("_", " ").title()
        gates[name] = str(shp)

    return gates


def _extract_pass_from_gate(gate_name: str) -> Optional[int]:
    """Extract pass number from gate name (e.g., _TPJ_pass_248)."""
    if not gate_name:
        return None
    match = re.search(r"pass[_\s]?(\d+)", gate_name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def render_slcci_sidebar() -> SLCCIAppConfig:
    """Render the SLCCI-only sidebar. Returns config."""
    config = SLCCIAppConfig()

    st.sidebar.markdown("## 🛰️ SLCCI Configuration")

    # ── Gate Selection ──────────────────────────────────────────────
    st.sidebar.markdown("### 🚪 Gate Selection")
    gates = _list_gates()

    if not gates:
        st.sidebar.warning("No gate shapefiles found in `gates/`")
        return config

    gate_names = list(gates.keys())
    selected_name = st.sidebar.selectbox("Gate", gate_names, key="slcci_gate")
    config.selected_gate = selected_name
    config.gate_geometry = gates[selected_name]

    # Auto-detect pass number
    auto_pass = _extract_pass_from_gate(selected_name)
    default_pass = auto_pass or 248

    # ── SLCCI Settings ──────────────────────────────────────────────
    st.sidebar.markdown("### 📡 Data Settings")

    config.slcci_base_dir = st.sidebar.text_input(
        "J2 Data Directory",
        value=config.slcci_base_dir,
        key="slcci_basedir",
    )
    config.slcci_geoid_path = st.sidebar.text_input(
        "Geoid File (TUM_ogmoc.nc)",
        value=config.slcci_geoid_path,
        key="slcci_geoid",
    )

    config.pass_number = st.sidebar.number_input(
        "Pass Number", min_value=1, max_value=999,
        value=default_pass, key="slcci_pass",
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        config.cycle_start = st.number_input(
            "Cycle Start", min_value=1, max_value=500,
            value=1, key="slcci_cyc_start",
        )
    with col2:
        config.cycle_end = st.number_input(
            "Cycle End", min_value=1, max_value=500,
            value=281, key="slcci_cyc_end",
        )

    # ── Processing ──────────────────────────────────────────────────
    with st.sidebar.expander("⚙️ Processing", expanded=False):
        config.use_flag = st.checkbox("Quality Flag Filter", value=True, key="slcci_flag")
        config.lon_bin_size = st.slider(
            "Lon Bin Size (°)", 0.01, 0.50, 0.10, 0.01, key="slcci_bin",
        )
        config.lat_buffer_deg = st.slider(
            "Lat Buffer (°)", 0.5, 5.0, 2.0, 0.5, key="slcci_latbuf",
        )
        config.lon_buffer_deg = st.slider(
            "Lon Buffer (°)", 1.0, 10.0, 5.0, 0.5, key="slcci_lonbuf",
        )
        config.force_reload = st.checkbox("Force Reload", value=False, key="slcci_force")

    # ── Bathymetry ──────────────────────────────────────────────────
    with st.sidebar.expander("🏔️ Bathymetry", expanded=False):
        config.depth_method = st.selectbox(
            "Depth Method", ["fixed", "gebco"], key="slcci_depth",
        )
        config.fixed_depth_m = st.number_input(
            "Fixed Depth (m)", 50, 1000, 250, 50, key="slcci_fixdepth",
        )
        config.gebco_nc_path = st.text_input(
            "GEBCO NetCDF Path", value=config.gebco_nc_path, key="slcci_gebco",
        )

    # ── Load Button ─────────────────────────────────────────────────
    st.sidebar.markdown("---")

    if st.sidebar.button("🚀 Load SLCCI Data", use_container_width=True, type="primary"):
        _load_slcci_data(config)

    if st.sidebar.button("🗑️ Clear Data", use_container_width=True):
        clear_slcci_data()
        st.rerun()

    return config


def _load_slcci_data(config: SLCCIAppConfig):
    """Load SLCCI data using the loader."""
    from app.components.loaders.slcci_loader import load_slcci_data

    with st.spinner(f"Loading pass {config.pass_number}..."):
        result = load_slcci_data(
            base_dir=config.slcci_base_dir,
            geoid_path=config.slcci_geoid_path,
            gate_id=_gate_name_to_id(config.selected_gate),
            pass_number=config.pass_number,
            cycle_start=config.cycle_start,
            cycle_end=config.cycle_end,
            lon_filter_min=config.lon_filter_min,
            lon_filter_max=config.lon_filter_max,
            use_flag=config.use_flag,
            lat_buffer_deg=config.lat_buffer_deg,
            lon_buffer_deg=config.lon_buffer_deg,
            lon_bin_size=config.lon_bin_size,
            use_cache=not config.force_reload,
        )

        if result.success:
            store_slcci_data(result.data)
            st.sidebar.success(f"✅ {result.info_message}")
            st.rerun()
        else:
            st.sidebar.error(f"❌ {result.error_message}")


def _gate_name_to_id(display_name: str) -> str:
    """Convert display name back to gate ID."""
    if not display_name:
        return ""
    # Reverse the title + space → underscore + lowercase
    return display_name.lower().replace(" ", "_")
