"""
SLCCI Standalone — State Management
====================================
Minimal session state for SLCCI-only dashboard.
No CMEMS, no DTU, no comparison mode.
"""

import streamlit as st
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SLCCIAppConfig:
    """SLCCI-only configuration from sidebar."""
    # Gate selection
    selected_gate: Optional[str] = None
    gate_geometry: object = None
    gate_buffer_km: float = 50.0

    # Longitude filter (Fram West/East, Davis West/East)
    lon_filter_min: Optional[float] = None
    lon_filter_max: Optional[float] = None

    # SLCCI settings
    slcci_base_dir: str = "/Users/nicolocaron/Desktop/ARCFRESH/J2"
    slcci_geoid_path: str = "/Users/nicolocaron/Desktop/ARCFRESH/TUM_ogmoc.nc"

    # Pass selection
    pass_number: int = 248
    cycle_start: int = 1
    cycle_end: int = 281

    # Processing
    use_flag: bool = True
    lon_bin_size: float = 0.10
    lat_buffer_deg: float = 2.0
    lon_buffer_deg: float = 5.0
    force_reload: bool = False

    # Bathymetry
    depth_method: str = "fixed"
    fixed_depth_m: float = 250.0
    gebco_nc_path: str = "/Users/nicolocaron/Desktop/ARCFRESH/GEBCO_06_Feb_2026_c91df93f54b8/gebco_2025_n90.0_s55.0_w0.0_e360.0.nc"


def init_slcci_state():
    """Initialize session state for SLCCI-only app."""
    defaults = {
        "dataset_slcci": None,
        "slcci_pass_data": None,
        "app_config": SLCCIAppConfig(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def store_slcci_data(pass_data):
    """Store SLCCI pass data."""
    st.session_state["dataset_slcci"] = pass_data
    st.session_state["slcci_pass_data"] = pass_data


def get_slcci_data():
    """Get SLCCI data from session state."""
    return st.session_state.get("dataset_slcci")


def clear_slcci_data():
    """Clear SLCCI data."""
    st.session_state["dataset_slcci"] = None
    st.session_state["slcci_pass_data"] = None
