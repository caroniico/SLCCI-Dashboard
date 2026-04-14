"""
DOT Computation & Slope
=======================
DOT = corssh − geoid
Slope = linear fit of DOT vs along-gate distance (m / 100 km).

All functions are pure numpy — no Streamlit, no service-layer coupling.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# Earth radius for distance calculation
_R_EARTH_KM = 6371.0


def compute_dot(
    corssh: np.ndarray,
    geoid: np.ndarray,
) -> np.ndarray:
    """
    Compute Dynamic Ocean Topography.

    DOT = corssh − geoid  [m]

    Parameters
    ----------
    corssh : (N,) corrected sea surface height [m]
    geoid  : (N,) geoidal height at same positions [m]

    Returns
    -------
    dot : (N,) dynamic ocean topography [m]
    """
    return corssh - geoid


def build_dataframe(
    ds,
    geoid_values: np.ndarray,
    pass_number: int,
) -> Optional[pd.DataFrame]:
    """
    Build a DOT DataFrame from an xarray Dataset + interpolated geoid.

    Parameters
    ----------
    ds : xr.Dataset with latitude, longitude, corssh, time, cycle coords
    geoid_values : (N,) geoidal height at observation positions
    pass_number : int

    Returns
    -------
    DataFrame with columns: cycle, pass, lat, lon, corssh, geoid, dot,
    time, month, year, year_month.  None if no corssh variable.
    """
    if "corssh" not in ds.data_vars:
        logger.warning("No 'corssh' variable in dataset")
        return None

    dot = compute_dot(ds["corssh"].values, geoid_values)

    df = pd.DataFrame({
        "cycle": ds["cycle"].values,
        "pass": pass_number,
        "lat": ds["latitude"].values,
        "lon": ds["longitude"].values,
        "corssh": ds["corssh"].values,
        "geoid": geoid_values,
        "dot": dot,
        "time": pd.to_datetime(ds["time"].values),
    })
    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    df["year_month"] = df["time"].dt.to_period("M")
    return df


def _lon_to_x_km(lon_centers: np.ndarray, mean_lat: float) -> np.ndarray:
    """Convert longitude centres to distance in km from first bin."""
    lat_rad = np.deg2rad(mean_lat)
    lon_rad = np.deg2rad(lon_centers)
    dlon = lon_rad - lon_rad[0]
    return _R_EARTH_KM * dlon * np.cos(lat_rad)


def build_dot_matrix(
    df: pd.DataFrame,
    lon_bin_size: float = 0.1,
) -> Tuple[np.ndarray, List, np.ndarray, np.ndarray]:
    """
    Build a DOT matrix (n_lon_bins × n_time) via longitude binning.

    Each cell is the mean DOT of all observations falling in that
    (longitude bin, time period) pair.  Missing cells are NaN
    — **no filling / interpolation**.

    Parameters
    ----------
    df : DataFrame with 'lon', 'lat', 'dot', 'year_month' columns
    lon_bin_size : float
        Bin width in degrees

    Returns
    -------
    dot_matrix  : (n_lon_bins, n_time) — NaN where no data
    time_periods : list of pd.Period
    lon_centers  : (n_lon_bins,) — centre of each bin
    x_km         : (n_lon_bins,) — distance from first bin [km]
    """
    # Determine longitude range from DATA
    lon_min = df["lon"].min()
    lon_max = df["lon"].max()

    # Handle dateline crossing
    if lon_max - lon_min > 180:
        logger.warning(f"Dateline crossing detected: [{lon_min:.2f}, {lon_max:.2f}]")
        df = df.copy()
        df.loc[df["lon"] < 0, "lon"] += 360
        lon_min = df["lon"].min()
        lon_max = df["lon"].max()

    lon_bins = np.arange(lon_min, lon_max + lon_bin_size, lon_bin_size)
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    n_lon_bins = len(lon_centers)

    time_periods = sorted(df["year_month"].unique())
    n_time = len(time_periods)

    dot_matrix = np.full((n_lon_bins, n_time), np.nan, dtype=float)

    for it, period in enumerate(time_periods):
        month_data = df[df["year_month"] == period]
        if month_data.empty:
            continue
        tmp = month_data.copy()
        tmp["lon_bin"] = pd.cut(tmp["lon"], bins=lon_bins, labels=False, include_lowest=True)
        binned = tmp.groupby("lon_bin")["dot"].mean()
        for bin_idx in binned.index:
            if pd.notna(bin_idx) and int(bin_idx) < n_lon_bins:
                dot_matrix[int(bin_idx), it] = binned[bin_idx]

    x_km = _lon_to_x_km(lon_centers, df["lat"].mean())

    valid = np.sum(np.isfinite(dot_matrix))
    total = dot_matrix.size
    logger.info(f"DOT matrix: {valid}/{total} valid ({100 * valid / total:.1f}%)")

    return dot_matrix, time_periods, lon_centers, x_km


def compute_slope_series(
    dot_matrix: np.ndarray,
    x_km: np.ndarray,
) -> np.ndarray:
    """
    Compute DOT slope (m / 100 km) for each time column via linear fit.

    Uses ``np.polyfit`` on valid (finite) points only.
    Returns NaN for timesteps with < 2 valid points.
    **No filling / interpolation of missing bins.**

    Parameters
    ----------
    dot_matrix : (n_bins, n_time)
    x_km       : (n_bins,)

    Returns
    -------
    slope_series : (n_time,)  in m / 100 km
    """
    _, n_time = dot_matrix.shape
    slope_series = np.full(n_time, np.nan, dtype=float)

    for it in range(n_time):
        y = dot_matrix[:, it]
        mask = np.isfinite(x_km) & np.isfinite(y)
        if np.sum(mask) < 2:
            continue
        a, _ = np.polyfit(x_km[mask], y[mask], 1)
        slope_series[it] = a * 100.0  # m / 100 km

    return slope_series
