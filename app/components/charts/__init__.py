"""
Charts Module - Unified chart rendering functions.
Eliminates code duplication across SLCCI, CMEMS, DTU renderers.
"""

from .slope_chart import render_slope_timeline, render_multi_slope
from .dot_profile_chart import render_dot_profile, render_multi_dot_profile
from .spatial_chart import render_spatial_map, render_multi_spatial_overview
from .geostrophic_chart import render_geostrophic_velocity, render_multi_geostrophic
from .volume_transport_chart import render_volume_transport_tab, render_bathymetry_profile
from .utils import DATASET_COLORS, DATASET_NAMES, get_pass_data_attributes, color_to_rgba

__all__ = [
    "render_slope_timeline",
    "render_multi_slope",
    "render_dot_profile", 
    "render_multi_dot_profile",
    "render_spatial_map",
    "render_multi_spatial_overview",
    "render_geostrophic_velocity",
    "render_multi_geostrophic",
    "render_volume_transport_tab",
    "render_bathymetry_profile",
    "DATASET_COLORS",
    "DATASET_NAMES",
    "get_pass_data_attributes",
    "color_to_rgba",
]
