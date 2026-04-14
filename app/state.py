"""
Session State Management
========================
Manages session state for NICO dashboard with support for:
- SLCCI and CMEMS datasets (separate keys for comparison mode)
- Comparison mode toggle
- Export settings
"""

import streamlit as st
from dataclasses import dataclass, field
from typing import Any, Optional, List


@dataclass
class AppConfig:
    """Application configuration from sidebar."""
    # UI mode
    ui_mode: str = "New UI"

    # Gate selection
    selected_gate: Optional[str] = None
    gate_geometry: Any = None
    gate_buffer_km: float = 50.0
    
    # Longitude filter for subdivided gates (e.g., Fram West/East, Davis West/East)
    lon_filter_min: Optional[float] = None
    lon_filter_max: Optional[float] = None
    
    # Data source
    selected_dataset_type: str = "SLCCI"
    data_source_mode: str = "local"  # "local" or "api"
    
    # === COMPARISON MODE ===
    comparison_mode: bool = False  # Enable overlay of SLCCI and CMEMS
    compare_datasets: List[str] = field(default_factory=list)  # ["SLCCI", "CMEMS"]
    
    # === SLCCI SETTINGS ===
    slcci_base_dir: str = "/Users/nicolocaron/Desktop/ARCFRESH/J2"
    slcci_geoid_path: str = "/Users/nicolocaron/Desktop/ARCFRESH/TUM_ogmoc.nc"
    
    # Pass selection (SLCCI only)
    pass_mode: str = "manual"
    pass_number: int = 248
    
    # Cycle range (SLCCI only)
    cycle_start: int = 1
    cycle_end: int = 10  # Default: primi 10 cicli per test veloci
    
    # Processing parameters (from SLCCI PLOTTER notebook)
    use_flag: bool = True           # Quality flag filter
    lon_bin_size: float = 0.10      # Longitude binning size (degrees) - UNIFIED for all outputs
    lat_buffer_deg: float = 2.0     # Latitude buffer for spatial filter
    lon_buffer_deg: float = 5.0     # Longitude buffer for spatial filter
    force_reload: bool = False      # Bypass cache and reload from source
    
    # === CMEMS SETTINGS ===
    cmems_base_dir: str = "/Users/nicolocaron/Desktop/ARCFRESH/COPERNICUS DATA"
    cmems_source_mode: str = "local"  # "local" or "api"
    cmems_start_date: Any = None    # datetime.date
    cmems_end_date: Any = None      # datetime.date
    cmems_lon_bin_size: float = 0.10  # Coarser than SLCCI (0.05-0.50)
    cmems_buffer_deg: float = 5.0   # Buffer around gate (from Copernicus notebook)
    cmems_track_number: Optional[int] = None  # Filter by track (like SLCCI pass)
    
    # CMEMS Performance options
    cmems_use_parallel: bool = True  # Use parallel file loading
    cmems_use_cache: bool = True     # Cache processed DataFrames
    
    # === DTUSpace SETTINGS (ISOLATED - gridded product) ===
    dtu_nc_path: str = ""  # Path to DTUSpace NetCDF file
    dtu_start_year: int = 2006
    dtu_end_year: int = 2017
    dtu_n_gate_pts: int = 400  # Points to interpolate along gate
    
    # === CMEMS L4 SETTINGS (GRIDDED via API) ===
    cmems_l4_variables: List[str] = field(default_factory=lambda: ["adt", "sla"])
    cmems_l4_buffer: float = 2.0  # Buffer around gate (degrees)
    cmems_l4_start: Any = None  # datetime.date
    cmems_l4_end: Any = None  # datetime.date
    cmems_l4_filter_ice: bool = False  # Filter out ice-covered observations using flag_ice
    
    # === BATHYMETRY / VOLUME TRANSPORT SETTINGS ===
    depth_method: str = "fixed"  # "fixed" or "gebco"
    fixed_depth_m: float = 250.0  # Fixed depth cap in meters
    gebco_nc_path: str = "/Users/nicolocaron/Desktop/ARCFRESH/GEBCO_06_Feb_2026_c91df93f54b8/gebco_2025_n90.0_s55.0_w0.0_e360.0.nc"


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        # UI mode
        "ui_mode": "New UI",

        # Generic datasets (for ERA5 etc)
        "datasets": {},
        "cycle_info": {},
        "analysis_results": None,
        
        # SLCCI-specific (separate key for comparison)
        "dataset_slcci": None,
        
        # CMEMS-specific (separate key for comparison)
        "dataset_cmems": None,
        
        # CMEMS L4 Gridded (API) - similar to DTUSpace
        "dataset_cmems_l4": None,
        
        # DTUSpace-specific (ISOLATED - gridded product)
        "dataset_dtu": None,
        
        # Comparison mode
        "comparison_mode": False,
        "selected_dataset_type": "SLCCI",
        
        # Legacy key (for backward compatibility)
        "slcci_pass_data": None,
        
        # Config
        "config": {
            "mss_var": "mean_sea_surface",
            "bin_size": 0.01,
            "lat_range": None,
            "lon_range": None,
        },
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def update_datasets(datasets: dict, cycle_info: dict):
    """Update loaded datasets in session state."""
    st.session_state.datasets = datasets
    st.session_state.cycle_info = cycle_info


def get_datasets():
    """Get currently loaded datasets."""
    return st.session_state.get("datasets", {})


def get_cycle_info():
    """Get cycle information."""
    return st.session_state.get("cycle_info", {})


def store_slcci_data(pass_data):
    """Store SLCCI data with dedicated key for comparison mode."""
    st.session_state["dataset_slcci"] = pass_data
    # Also store in legacy key for backward compatibility
    st.session_state["slcci_pass_data"] = pass_data


def store_cmems_data(pass_data):
    """Store CMEMS data with dedicated key for comparison mode."""
    st.session_state["dataset_cmems"] = pass_data
    # Also store in legacy key when not in comparison mode
    if not st.session_state.get("comparison_mode", False):
        st.session_state["slcci_pass_data"] = pass_data


def get_slcci_data():
    """Get SLCCI data from session state."""
    return st.session_state.get("dataset_slcci")


def get_cmems_data():
    """Get CMEMS data from session state."""
    return st.session_state.get("dataset_cmems")


def set_comparison_mode(enabled: bool):
    """Enable or disable comparison mode."""
    st.session_state["comparison_mode"] = enabled


def is_comparison_mode():
    """Check if comparison mode is enabled."""
    return st.session_state.get("comparison_mode", False)


def clear_data():
    """Clear all loaded data."""
    st.session_state.datasets = {}
    st.session_state.cycle_info = {}
    st.session_state.analysis_results = None
    st.session_state.dataset_slcci = None
    st.session_state.dataset_cmems = None
    st.session_state.slcci_pass_data = None


def clear_slcci_data():
    """Clear only SLCCI data."""
    st.session_state.dataset_slcci = None
    if not st.session_state.get("dataset_cmems"):
        st.session_state.slcci_pass_data = None


def clear_cmems_data():
    """Clear only CMEMS data."""
    st.session_state.dataset_cmems = None
    if not st.session_state.get("dataset_slcci"):
        st.session_state.slcci_pass_data = None


# ==============================================================================
# DTUSpace Functions (ISOLATED - does not affect SLCCI/CMEMS)
# ==============================================================================

def store_dtu_data(pass_data):
    """Store DTUSpace data with dedicated key."""
    st.session_state["dataset_dtu"] = pass_data


def get_dtu_data():
    """Get DTUSpace data from session state."""
    return st.session_state.get("dataset_dtu")


def clear_dtu_data():
    """Clear only DTUSpace data."""
    st.session_state["dataset_dtu"] = None


# ==============================================================================
# CMEMS L4 Functions (GRIDDED via API - similar to DTU)
# ==============================================================================

def store_cmems_l4_data(pass_data):
    """Store CMEMS L4 gridded data with dedicated key."""
    st.session_state["dataset_cmems_l4"] = pass_data


def get_cmems_l4_data():
    """Get CMEMS L4 data from session state."""
    return st.session_state.get("dataset_cmems_l4")


def clear_cmems_l4_data():
    """Clear only CMEMS L4 data."""
    st.session_state["dataset_cmems_l4"] = None


# ==============================================================================
# MULTI-DATASET UTILITIES
# ==============================================================================

def get_all_loaded_datasets() -> dict:
    """
    Get all currently loaded datasets from session state.
    
    Returns:
        Dict with keys: 'slcci', 'cmems', 'cmems_l4', 'dtu'
        Only includes datasets that are actually loaded.
    """
    loaded = {}
    
    if st.session_state.get("dataset_slcci") is not None:
        loaded["slcci"] = st.session_state.get("dataset_slcci")
    if st.session_state.get("dataset_cmems") is not None:
        loaded["cmems"] = st.session_state.get("dataset_cmems")
    if st.session_state.get("dataset_cmems_l4") is not None:
        loaded["cmems_l4"] = st.session_state.get("dataset_cmems_l4")
    if st.session_state.get("dataset_dtu") is not None:
        loaded["dtu"] = st.session_state.get("dataset_dtu")
    
    return loaded


def count_loaded_datasets() -> int:
    """Count how many datasets are currently loaded."""
    return len(get_all_loaded_datasets())


def clear_all_datasets():
    """Clear ALL loaded datasets."""
    st.session_state["dataset_slcci"] = None
    st.session_state["dataset_cmems"] = None
    st.session_state["dataset_cmems_l4"] = None
    st.session_state["dataset_dtu"] = None
    st.session_state["slcci_pass_data"] = None
    st.session_state["comparison_mode"] = False
