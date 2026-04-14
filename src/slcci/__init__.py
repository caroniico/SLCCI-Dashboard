"""
ARCFRESH SLCCI Module
=====================
Pure-numpy functions for ESA Sea Level CCI (SLCCI) along-track altimetry.

Responsibilities:
    models   — PassData, SLCCIConfig dataclasses
    loader   — Read SLCCI_ALTDB_J2_CycleXXX_V2.nc files
    geoid    — Interpolate TUM_ogmoc.nc geoidal height
    dot      — DOT = corssh − geoid, slope computation
    binning  — Longitude binning, mean profiles, climatology
    spatial  — Spatial filter, pass/gate matching

No Streamlit, no service-layer coupling.
All functions take and return numpy arrays / pandas DataFrames.

Usage:
    from src.slcci.models import PassData, SLCCIConfig
    from src.slcci.loader import load_cycle, load_cycles_serial
    from src.slcci.geoid import load_geoid, interpolate_geoid
    from src.slcci.dot import compute_dot, build_dot_matrix, compute_slope_series
    from src.slcci.binning import longitude_bin, mean_profile_pooled, monthly_climatology
    from src.slcci.spatial import filter_near_gate, find_passes_for_gate
"""

from .models import PassData, SLCCIConfig
from .loader import load_cycle, load_cycles_serial
from .geoid import load_geoid, interpolate_geoid
from .dot import compute_dot, build_dot_matrix, compute_slope_series
from .binning import longitude_bin, mean_profile_pooled, monthly_climatology_profiles
from .spatial import filter_near_gate

__all__ = [
    # models
    "PassData", "SLCCIConfig",
    # loader
    "load_cycle", "load_cycles_serial",
    # geoid
    "load_geoid", "interpolate_geoid",
    # dot
    "compute_dot", "build_dot_matrix", "compute_slope_series",
    # binning
    "longitude_bin", "mean_profile_pooled", "monthly_climatology_profiles",
    # spatial
    "filter_near_gate",
]
