"""
Temporal Aggregation Utilities
==============================
Monthly profiles, climatology, annual means, rolling mean.

All functions take **numpy arrays** and pandas DatetimeIndex — no xarray.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                   ALONG-GATE PROFILES                                 ║
# ╚════════════════════════════════════════════════════════════════════════╝

def monthly_along_gate_profile(
    field: np.ndarray,
    time: pd.DatetimeIndex,
    x_km: np.ndarray,
    sigma: Optional[np.ndarray] = None,
) -> Dict[int, dict]:
    """
    Compute mean along-gate profile for each calendar month.

    Parameters
    ----------
    field : (N_pts, N_time) — e.g. v_perp or sss.
    time  : DatetimeIndex of length N_time.
    x_km  : (N_pts,) — distance array.
    sigma : (N_pts, N_time) optional formal uncertainty array.
            If provided, the monthly RMS formal error is included as
            'sigma_mean' in the output dict.

    Returns
    -------
    dict : {month_int: {'mean': (N_pts,), 'std': (N_pts,), 'count': int,
                         'sigma_mean': (N_pts,)  # only if sigma given}}
    """
    months = time.month
    result: Dict[int, dict] = {}

    for m in range(1, 13):
        mask = months == m
        n = int(mask.sum())

        if n == 0:
            entry: dict = {
                "mean": np.full(len(x_km), np.nan),
                "std": np.full(len(x_km), np.nan),
                "count": 0,
            }
            if sigma is not None:
                entry["sigma_mean"] = np.full(len(x_km), np.nan)
            result[m] = entry
        else:
            subset = field[:, mask]
            entry = {
                "mean": np.nanmean(subset, axis=1),
                "std": np.nanstd(subset, axis=1),
                "count": n,
            }
            if sigma is not None:
                sig_sub = sigma[:, mask]
                entry["sigma_mean"] = np.sqrt(np.nanmean(sig_sub ** 2, axis=1))
            result[m] = entry

    return result


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                   TIME SERIES AGGREGATION                             ║
# ╚════════════════════════════════════════════════════════════════════════╝

def monthly_mean(
    values: np.ndarray,
    time: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute monthly means from a daily time series.

    Parameters
    ----------
    values : 1-D array (N_time,).
    time   : DatetimeIndex of same length.

    Returns
    -------
    DataFrame with columns ['date', 'value'] indexed monthly.
    """
    df = pd.DataFrame({"value": values}, index=time)
    return df.resample("MS").mean().dropna()


def monthly_climatology(
    values: np.ndarray,
    time: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Climatological monthly mean ± std (all years collapsed).

    Returns
    -------
    DataFrame with columns ['month', 'mean', 'std', 'count'].
    """
    df = pd.DataFrame({"value": values, "month": time.month}, index=time)
    clim = df.groupby("month")["value"].agg(["mean", "std", "count"]).reset_index()
    return clim


def annual_mean(
    values: np.ndarray,
    time: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Annual means from daily data."""
    df = pd.DataFrame({"value": values}, index=time)
    return df.resample("YS").mean().dropna()


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                        ROLLING / FILTER                               ║
# ╚════════════════════════════════════════════════════════════════════════╝

def rolling_mean(
    values: np.ndarray,
    window: int = 365,
) -> np.ndarray:
    """Simple centred rolling mean (NaN-aware). Returns same length as input."""
    s = pd.Series(values)
    return s.rolling(window, center=True, min_periods=window // 2).mean().values
