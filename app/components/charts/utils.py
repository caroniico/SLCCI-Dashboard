"""
Utility functions and constants for chart rendering.
"""
import numpy as np
from typing import Any, Dict, Optional

# Dataset colors
DATASET_COLORS = {
    "slcci": "darkorange",
    "cmems": "steelblue", 
    "cmems_l4": "mediumpurple",
    "dtu": "seagreen"
}

# Dataset display names
DATASET_NAMES = {
    "slcci": "SLCCI",
    "cmems": "CMEMS L3",
    "cmems_l4": "CMEMS L4",
    "dtu": "DTUSpace"
}


def get_pass_data_attributes(pass_data: Any) -> Dict[str, Any]:
    """
    Extract common attributes from any PassData object.
    Works with SLCCI, CMEMS, CMEMS L4, DTU data structures.
    """
    return {
        # Core data
        "slope_series": getattr(pass_data, 'slope_series', None),
        "time_array": getattr(pass_data, 'time_array', None),
        "time_periods": getattr(pass_data, 'time_periods', None),
        
        # DOT profile
        "profile_mean": getattr(pass_data, 'profile_mean', None),
        "x_km": getattr(pass_data, 'x_km', None),
        "dot_matrix": getattr(pass_data, 'dot_matrix', None),
        
        # Spatial
        "df": getattr(pass_data, 'df', None),
        "gate_lon_pts": getattr(pass_data, 'gate_lon_pts', None),
        "gate_lat_pts": getattr(pass_data, 'gate_lat_pts', None),
        
        # Geostrophic velocity (CMEMS L4)
        "ugos_matrix": getattr(pass_data, 'ugos_matrix', None),
        "vgos_matrix": getattr(pass_data, 'vgos_matrix', None),
        
        # Metadata
        "strait_name": getattr(pass_data, 'strait_name', 'Unknown'),
        "pass_number": getattr(pass_data, 'pass_number', 0),
        "data_source": getattr(pass_data, 'data_source', 'Unknown'),
        "n_observations": getattr(pass_data, 'n_observations', 0),
        
        # Raw dataset
        "ds": getattr(pass_data, 'ds', None),
    }


def color_to_rgba(color: str, alpha: float = 0.2) -> str:
    """Convert a named color to RGBA string for fill areas."""
    color_map = {
        "darkorange": f"rgba(255, 140, 0, {alpha})",
        "steelblue": f"rgba(70, 130, 180, {alpha})",
        "mediumpurple": f"rgba(147, 112, 219, {alpha})",
        "seagreen": f"rgba(46, 139, 87, {alpha})",
        "red": f"rgba(255, 0, 0, {alpha})",
        "blue": f"rgba(0, 0, 255, {alpha})",
        "green": f"rgba(0, 128, 0, {alpha})",
    }
    return color_map.get(color, f"rgba(128, 128, 128, {alpha})")
