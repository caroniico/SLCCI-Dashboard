"""
SLCCI Data Models
=================
Dataclasses for SLCCI configuration and pass data containers.
Pure Python — no framework dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Literal
import numpy as np
import pandas as pd


# Data source type
DataSource = Literal["local", "api"]


@dataclass
class SLCCIConfig:
    """Configuration for SLCCI data loading and processing."""
    # Paths
    base_dir: str = ""
    geoid_path: str = ""

    # Cycles to load
    cycles: List[int] = field(default_factory=lambda: list(range(1, 282)))

    # Processing
    use_flag: bool = True
    lat_buffer_deg: float = 2.0
    lon_buffer_deg: float = 5.0
    lon_bin_size: float = 0.1  # Longitude binning resolution (degrees)

    # Source
    source: DataSource = "local"
    satellite: str = "J2"  # J1 or J2


@dataclass
class PassData:
    """
    Container for a fully-processed SLCCI satellite pass.

    Holds raw observations (df), spatial grids (gate_lon/lat, x_km),
    the DOT matrix, slope time series, and climatological profiles.
    """
    pass_number: int
    strait_name: str
    satellite: str

    # Raw observations
    df: pd.DataFrame

    # Gate geometry (from longitude binning)
    gate_lon_pts: np.ndarray
    gate_lat_pts: np.ndarray
    x_km: np.ndarray              # distance along gate [km]

    # Time
    time_periods: List
    time_array: np.ndarray

    # DOT analysis
    slope_series: np.ndarray      # m / 100 km per time period
    profile_mean: np.ndarray      # mean DOT from pooled observations
    dot_matrix: np.ndarray        # (n_lon_bins, n_time)

    # Monthly climatology
    monthly_profiles: Optional[Dict[int, np.ndarray]] = None
    monthly_lon_centers: Optional[np.ndarray] = None
    monthly_x_km: Optional[np.ndarray] = None

    # Derived (computed lazily by UI / analysis layer)
    v_geostrophic_series: Optional[np.ndarray] = None
    mean_latitude: Optional[float] = None
    coriolis_f: Optional[float] = None
