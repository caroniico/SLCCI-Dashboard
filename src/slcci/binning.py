"""
Longitude Binning & Profiles
=============================
Spatial aggregation of along-track observations into fixed longitude bins.

All functions are pure numpy/pandas — no Streamlit.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

_R_EARTH_KM = 6371.0


def _lon_to_x_km(lon_centers: np.ndarray, mean_lat: float) -> np.ndarray:
    """Convert longitude centres to distance [km] from first bin."""
    lat_rad = np.deg2rad(mean_lat)
    lon_rad = np.deg2rad(lon_centers)
    dlon = lon_rad - lon_rad[0]
    return _R_EARTH_KM * dlon * np.cos(lat_rad)


def _prepare_lon(df: pd.DataFrame, lon_bin_size: float):
    """Handle dateline crossing and create bins + centres."""
    lon_min = df["lon"].min()
    lon_max = df["lon"].max()

    if lon_max - lon_min > 180:
        logger.warning(f"Dateline crossing detected: [{lon_min:.2f}, {lon_max:.2f}]")
        df = df.copy()
        df.loc[df["lon"] < 0, "lon"] += 360
        lon_min = df["lon"].min()
        lon_max = df["lon"].max()

    lon_bins = np.arange(lon_min, lon_max + lon_bin_size, lon_bin_size)
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    return df, lon_bins, lon_centers


def longitude_bin(
    df: pd.DataFrame,
    lon_bin_size: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign each observation to a longitude bin and compute bin statistics.

    Parameters
    ----------
    df : DataFrame with 'lon', 'lat', 'dot' columns
    lon_bin_size : float
        Bin width in degrees

    Returns
    -------
    bin_mean  : (N_bins,) mean DOT per bin (NaN where no data)
    bin_count : (N_bins,) number of observations per bin
    lon_centers : (N_bins,)
    x_km : (N_bins,) distance from first bin [km]
    """
    df, lon_bins, lon_centers = _prepare_lon(df.copy(), lon_bin_size)
    n = len(lon_centers)

    df["lon_bin"] = pd.cut(df["lon"], bins=lon_bins, labels=False, include_lowest=True)
    stats = df.groupby("lon_bin")["dot"].agg(["mean", "count"])

    bin_mean = np.full(n, np.nan, dtype=float)
    bin_count = np.zeros(n, dtype=int)
    for idx in stats.index:
        if pd.notna(idx) and int(idx) < n:
            bin_mean[int(idx)] = stats.loc[idx, "mean"]
            bin_count[int(idx)] = int(stats.loc[idx, "count"])

    x_km = _lon_to_x_km(lon_centers, df["lat"].mean())
    return bin_mean, bin_count, lon_centers, x_km


def mean_profile_pooled(
    df: pd.DataFrame,
    lon_bin_size: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build mean DOT profile by pooling ALL observations across ALL times.

    Each observation gets equal weight (not each time period).

    Parameters
    ----------
    df : DataFrame with 'lon', 'lat', 'dot'
    lon_bin_size : float

    Returns
    -------
    profile_mean : (N_bins,)
    lon_centers  : (N_bins,)
    x_km         : (N_bins,)
    """
    df, lon_bins, lon_centers = _prepare_lon(df.copy(), lon_bin_size)
    n = len(lon_centers)

    df["lon_bin"] = pd.cut(df["lon"], bins=lon_bins, labels=False, include_lowest=True)
    stats = df.groupby("lon_bin")["dot"].agg(["mean", "count"])

    profile_mean = np.full(n, np.nan, dtype=float)
    for idx in stats.index:
        if pd.notna(idx) and int(idx) < n:
            profile_mean[int(idx)] = stats.loc[idx, "mean"]

    x_km = _lon_to_x_km(lon_centers, df["lat"].mean())

    valid = np.sum(np.isfinite(profile_mean))
    logger.info(f"Pooled profile: {valid}/{n} bins have data")
    return profile_mean, lon_centers, x_km


def monthly_climatology_profiles(
    df: pd.DataFrame,
    lon_bin_size: float = 0.1,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """
    Build monthly climatological DOT profiles (month 1-12, all years pooled).

    Parameters
    ----------
    df : DataFrame with 'lon', 'lat', 'dot', 'month' columns
    lon_bin_size : float

    Returns
    -------
    monthly_profiles : {1: Jan_profile, 2: Feb_profile, ...}
    lon_centers : (N_bins,)
    x_km : (N_bins,)
    """
    df, lon_bins, lon_centers = _prepare_lon(df.copy(), lon_bin_size)
    n = len(lon_centers)

    df["lon_bin"] = pd.cut(df["lon"], bins=lon_bins, labels=False, include_lowest=True)

    monthly_profiles: Dict[int, np.ndarray] = {}
    for month in range(1, 13):
        mdata = df[df["month"] == month]
        profile = np.full(n, np.nan, dtype=float)

        if not mdata.empty:
            binned = mdata.groupby("lon_bin")["dot"].mean()
            for idx in binned.index:
                if pd.notna(idx) and int(idx) < n:
                    profile[int(idx)] = binned[idx]

        monthly_profiles[month] = profile

    x_km = _lon_to_x_km(lon_centers, df["lat"].mean())

    months_ok = sum(1 for p in monthly_profiles.values() if np.any(np.isfinite(p)))
    logger.info(f"Monthly climatology: {months_ok}/12 months, {n} bins × {lon_bin_size}°")
    return monthly_profiles, lon_centers, x_km
