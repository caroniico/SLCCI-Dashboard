"""
Spatial Filtering
=================
Filter observations near a gate geometry (shapefile).
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _load_gate_gdf(gate_path: str) -> gpd.GeoDataFrame:
    """Load a gate shapefile."""
    return gpd.read_file(gate_path)


def gate_bounds(
    gate_path: str,
    lat_buffer: float = 2.0,
    lon_buffer: float = 5.0,
) -> Tuple[float, float, float, float]:
    """
    Get bounding box around a gate, with buffer.

    Returns
    -------
    (lat_min, lat_max, lon_min_wrapped, lon_max_wrapped)
    in [0,360) longitude convention.
    """
    gdf = _load_gate_gdf(gate_path)
    lon_min_g, lat_min_g, lon_max_g, lat_max_g = gdf.total_bounds
    lat_min = lat_min_g - lat_buffer
    lat_max = lat_max_g + lat_buffer
    lon_min = (lon_min_g - lon_buffer) % 360
    lon_max = (lon_max_g + lon_buffer) % 360
    return lat_min, lat_max, lon_min, lon_max


def gate_profile_points(
    gate_path: str,
    n_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate equally-spaced points along a gate line.

    Parameters
    ----------
    gate_path : str
        Path to gate shapefile
    n_points : int
        Number of points to interpolate

    Returns
    -------
    lon : (n_points,)
    lat : (n_points,)
    x_km : (n_points,) distance from start [km]
    """
    from shapely.geometry import LineString
    gdf = _load_gate_gdf(gate_path)
    geom = gdf.geometry.iloc[0]

    if not isinstance(geom, LineString):
        geom = LineString(geom.coords)

    total_len = geom.length
    distances = np.linspace(0, total_len, n_points)
    points = [geom.interpolate(d) for d in distances]

    lon = np.array([p.x for p in points])
    lat = np.array([p.y for p in points])

    # Convert to km (approximate haversine)
    R = 6371.0
    mean_lat = np.deg2rad(lat.mean())
    lon_rad = np.deg2rad(lon)
    dlon = lon_rad - lon_rad[0]
    x_km = R * dlon * np.cos(mean_lat)

    return lon, lat, x_km


def filter_near_gate(
    df: pd.DataFrame,
    gate_path: str,
    lat_buffer: float = 2.0,
    lon_buffer: float = 5.0,
) -> pd.DataFrame:
    """
    Keep only observations within lat/lon buffer of a gate.

    Parameters
    ----------
    df : DataFrame with 'lat', 'lon' columns
    gate_path : str
    lat_buffer, lon_buffer : degrees

    Returns
    -------
    Filtered DataFrame (copy)
    """
    lat_min, lat_max, lon_min, lon_max = gate_bounds(gate_path, lat_buffer, lon_buffer)

    mask_lat = (df["lat"] >= lat_min) & (df["lat"] <= lat_max)

    lon = df["lon"].values % 360
    if lon_min <= lon_max:
        mask_lon = (lon >= lon_min) & (lon <= lon_max)
    else:
        mask_lon = (lon >= lon_min) | (lon <= lon_max)

    return df[mask_lat & mask_lon].copy()
