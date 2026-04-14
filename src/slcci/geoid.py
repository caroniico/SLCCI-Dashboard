"""
Geoid Interpolation
===================
Load TUM_ogmoc.nc geoidal height and interpolate to arbitrary (lat, lon) points.

Uses scipy RegularGridInterpolator with fill_value=np.nan (NO extrapolation).
"""

import logging
import numpy as np
import xarray as xr
from typing import Tuple
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

# Module-level cache (so we build the interpolator only once per session)
_GEOID_CACHE: dict = {}


def _wrap_longitude(lon: np.ndarray) -> np.ndarray:
    """Wrap longitudes to [0, 360)."""
    return lon % 360


def load_geoid(geoid_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load geoid file and return (lat, lon, values).

    Parameters
    ----------
    geoid_path : str
        Path to TUM_ogmoc.nc (or compatible geoid NetCDF)

    Returns
    -------
    lat : 1-D array
    lon : 1-D array (wrapped to [0,360), sorted, deduplicated)
    values : 2-D array (lat × lon)
    """
    ds = xr.open_dataset(geoid_path)
    lat = ds["lat"].values
    lon_raw = ds["lon"].values
    vals = ds["value"].values  # (lat, lon)

    # Wrap and sort longitudes, remove duplicates
    lon_wrapped = _wrap_longitude(lon_raw)
    sort_idx = np.argsort(lon_wrapped)
    lon_sorted = lon_wrapped[sort_idx]
    unique = np.concatenate(([True], np.diff(lon_sorted) != 0))
    lon_sorted = lon_sorted[unique]
    vals_sorted = vals[:, sort_idx][:, unique]

    return lat, lon_sorted, vals_sorted


def _get_interpolator(geoid_path: str) -> RegularGridInterpolator:
    """Build (or retrieve cached) RegularGridInterpolator for the geoid."""
    if geoid_path in _GEOID_CACHE:
        return _GEOID_CACHE[geoid_path]

    lat, lon, vals = load_geoid(geoid_path)
    interp = RegularGridInterpolator(
        (lat, lon),
        vals,
        method="nearest",
        bounds_error=False,
        fill_value=np.nan,          # ← NO extrapolation, banned project-wide
    )
    _GEOID_CACHE[geoid_path] = interp
    logger.info(f"Built geoid interpolator from {geoid_path} "
                f"(lat {lat.min():.1f}–{lat.max():.1f}, "
                f"lon {lon.min():.1f}–{lon.max():.1f})")
    return interp


def interpolate_geoid(
    geoid_path: str,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> np.ndarray:
    """
    Interpolate geoid height at target positions.

    Parameters
    ----------
    geoid_path : str
        Path to TUM_ogmoc.nc
    target_lats : (N,) array of latitudes
    target_lons : (N,) array of longitudes (any convention)

    Returns
    -------
    geoid_h : (N,) array of geoidal heights [m].
        NaN where outside geoid grid.
    """
    interp = _get_interpolator(geoid_path)
    lons_wrapped = _wrap_longitude(target_lons)
    pts = np.column_stack([target_lats, lons_wrapped])
    return interp(pts)
