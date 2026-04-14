"""
ARCFRESH Gate Analysis — Utility Functions
===========================================
Works exclusively with self-contained NetCDF files produced by
scripts/build_gate_netcdf.py (one file per gate).

**Delegates core physics to ``src.physics``** — this module is an
xarray-aware wrapper that extracts numpy arrays from ``xr.Dataset``,
calls the pure-numpy functions in ``src.physics``, and returns results.

NetCDF variables:
    ugos, vgos          – CMEMS L4 geostrophic velocity  (point, time)
    err_ugosa, err_vgosa – formal mapping errors          (point, time)
    sss                  – CCI SSS v5.5 (interpolated)    (point, time)
    sss_random_error     – CCI SSS random error           (point, time)
    psal_isas_surface    – ISAS PSAL first layer          (point, time)
    depth                – GEBCO 2025 bathymetry           (point,)
    dx                   – segment width in metres         (point,)

Coordinates:
    time, longitude, latitude, x_km

Sign convention:
    v_perp > 0 → flow INTO the Arctic side of the gate
    (local per-point normal oriented toward Arctic centre: 0E, 90N).
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

# ── Physical constants — re-exported from src.physics ─────────────────────
from src.physics.constants import SVERDRUP, DEPTH_CAP, S_REF, RHO, MONTH_NAMES, ARCTIC_CENTER

# ── Pure-numpy helpers imported for delegation ────────────────────────────
from src.physics.coordinates import (
    local_into_arctic_unit_vectors as _physics_normals,
    unwrap_longitudes as _unwrap_longitudes,
)
import src.physics.transport as _phys_transport
import src.physics.aggregation as _phys_agg

# ╔════════════════════════════════════════════════════════════════════════╗
# ║                           I/O                                         ║
# ╚════════════════════════════════════════════════════════════════════════╝

def load_gate(nc_path: str | Path) -> xr.Dataset:
    """Load a gate NetCDF and ensure points go W→E (lon increasing).

    If longitude is decreasing, reverses the 'point' dimension and
    recalculates x_km from 0.  Pure spatial reorder — no physics change.
    """
    ds = xr.open_dataset(nc_path)
    lon = ds['longitude'].values
    if lon.size > 1 and lon[-1] < lon[0]:             # E→W  → flip to W→E
        idx = np.arange(ds.sizes['point'] - 1, -1, -1)
        ds = ds.isel(point=idx)
        xkm = ds['x_km'].values
        ds['x_km'] = ('point', xkm.max() - xkm)     # restart from 0
    return ds


def list_available_gates(nc_dir: str | Path) -> list[str]:
    """Return sorted list of gate NetCDF files in a directory."""
    d = Path(nc_dir)
    return sorted(f.name for f in d.glob("arcfresh_*.nc"))


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                     PERPENDICULAR VELOCITY                            ║
# ╚════════════════════════════════════════════════════════════════════════╝


def _unwrap_longitudes(lon: np.ndarray) -> np.ndarray:
    """Return longitudes unwrapped to a continuous sequence (degrees)."""
    lon = np.asarray(lon, dtype=float)
    return np.rad2deg(np.unwrap(np.deg2rad(lon)))


def _safe_unit(x: float, y: float) -> tuple[float, float]:
    mag = float(np.hypot(x, y))
    if mag < 1e-12:
        return (0.0, 1.0)
    return (x / mag, y / mag)


def _local_tangent_unit_vectors(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute local unit tangent vectors along the gate at each point.

    Uses central differences for interior points and forward/backward
    differences at endpoints. Longitudinal metric is scaled by cos(lat).
    """
    lon_u = _unwrap_longitudes(lon)
    lat = np.asarray(lat, dtype=float)
    n = len(lon_u)
    tx = np.zeros(n, dtype=float)
    ty = np.zeros(n, dtype=float)

    for i in range(n):
        if i == 0:
            dlon = lon_u[1] - lon_u[0]
            dlat = lat[1] - lat[0]
            lat_mid = lat[0]
        elif i == n - 1:
            dlon = lon_u[-1] - lon_u[-2]
            dlat = lat[-1] - lat[-2]
            lat_mid = lat[-1]
        else:
            dlon = lon_u[i + 1] - lon_u[i - 1]
            dlat = lat[i + 1] - lat[i - 1]
            lat_mid = lat[i]

        dx = dlon * np.cos(np.deg2rad(lat_mid))
        dy = dlat
        tx[i], ty[i] = _safe_unit(dx, dy)

    return tx, ty


def _to_arctic_unit_vectors(
    lon: np.ndarray, lat: np.ndarray, arctic_center: tuple[float, float] = ARCTIC_CENTER
) -> tuple[np.ndarray, np.ndarray]:
    """Vector from each gate point toward Arctic centre (unit, metric-corrected)."""
    lon_u = _unwrap_longitudes(lon)
    lat = np.asarray(lat, dtype=float)
    n = len(lon_u)
    ax = np.zeros(n, dtype=float)
    ay = np.zeros(n, dtype=float)

    ac_lon, ac_lat = arctic_center
    ac_lon_u = float(ac_lon + 360.0 * np.round((np.nanmean(lon_u) - ac_lon) / 360.0))

    for i in range(n):
        dx = (ac_lon_u - lon_u[i]) * np.cos(np.deg2rad(lat[i]))
        dy = ac_lat - lat[i]
        ax[i], ay[i] = _safe_unit(dx, dy)

    return ax, ay


def local_into_arctic_unit_vectors(
    ds: xr.Dataset, arctic_center: tuple[float, float] = ARCTIC_CENTER
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute local per-point normal unit vectors oriented toward Arctic side.

    **Delegates to ``src.physics.coordinates.local_into_arctic_unit_vectors``.**
    This wrapper just extracts lon/lat from the Dataset.
    """
    lon = ds["longitude"].values.astype(float)
    lat = ds["latitude"].values.astype(float)
    return _physics_normals(lon, lat, arctic_center=arctic_center)


def _projection_components(
    ds: xr.Dataset,
    gate_name: Optional[str] = None,
    u_into: Optional[float] = None,
    v_into: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return projection coefficients for ugos and vgos.
    New contract: always local per-point normals into Arctic side.
    Legacy inputs are ignored intentionally.
    """
    _ = (gate_name, u_into, v_into)  # Explicitly unused in the new workflow
    u_loc, v_loc = local_into_arctic_unit_vectors(ds)
    return u_loc[:, np.newaxis], v_loc[:, np.newaxis]


def perpendicular_velocity(
    ds: xr.Dataset,
    gate_name: Optional[str] = None,
    u_into: Optional[float] = None,
    v_into: Optional[float] = None,
) -> np.ndarray:
    """
    v_perp = ugos * coeff_u + vgos * coeff_v   [m/s]
    Positive → into Arctic.  Shape: (point, time).
    """
    ugos = ds["ugos"].values
    vgos = ds["vgos"].values
    coeff_u, coeff_v = _projection_components(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    return ugos * coeff_u + vgos * coeff_v


def perpendicular_velocity_uncertainty(
    ds: xr.Dataset,
    gate_name: Optional[str] = None,
    u_into: Optional[float] = None,
    v_into: Optional[float] = None,
) -> np.ndarray:
    """
    σ_v_perp = √( (σ_u · coeff_u)² + (σ_v · coeff_v)² )   [m/s]
    Shape: (point, time).
    """
    eu = ds["err_ugosa"].values
    ev = ds["err_vgosa"].values
    coeff_u, coeff_v = _projection_components(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    return np.sqrt((eu * coeff_u) ** 2 + (ev * coeff_v) ** 2)


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                     VOLUME TRANSPORT                                  ║
# ╚════════════════════════════════════════════════════════════════════════╝

def volume_transport(
    ds: xr.Dataset,
    depth_cap: float = DEPTH_CAP,
    gate_name: Optional[str] = None,
    u_into: Optional[float] = None,
    v_into: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    VT(t) = Σᵢ v_perp(i,t) · min(depth(i), cap) · dx(i)   →  Sverdrups

    Returns
    -------
    vt_sv : (time,)  volume transport in Sv
    time  : pandas DatetimeIndex
    """
    v_perp = perpendicular_velocity(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    depth = np.minimum(ds['depth'].values, depth_cap)  # (pt,)
    dx = ds['dx'].values                               # (pt,)
    # Integrate: sum over points
    q = v_perp * depth[:, None] * dx[:, None]          # (pt, time)
    vt = np.nansum(q, axis=0)                          # (time,)
    # Mark all-NaN columns
    vt[np.all(np.isnan(v_perp), axis=0)] = np.nan
    vt_sv = vt / SVERDRUP
    time = pd.to_datetime(ds['time'].values)
    return vt_sv, time


def volume_transport_uncertainty(
    ds: xr.Dataset,
    depth_cap: float = DEPTH_CAP,
    gate_name: Optional[str] = None,
    u_into: Optional[float] = None,
    v_into: Optional[float] = None,
) -> np.ndarray:
    """
    σ_VT(t) = √( Σᵢ (σ_v_perp(i,t) · H(i) · dx(i))² ) / 1e6   [Sv]
    """
    sigma_vp = perpendicular_velocity_uncertainty(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    H = np.minimum(ds['depth'].values, depth_cap)
    dx = ds['dx'].values
    terms = sigma_vp * H[:, None] * dx[:, None]        # (pt, time)
    sigma_q = np.sqrt(np.nansum(terms**2, axis=0))     # (time,)
    return sigma_q / SVERDRUP


def volume_transport_per_point(
    ds: xr.Dataset,
    depth_cap: float = DEPTH_CAP,
    gate_name: Optional[str] = None,
    u_into: Optional[float] = None,
    v_into: Optional[float] = None,
) -> np.ndarray:
    """
    Transport contribution per gate point per timestep (Sv).
    Shape: (point, time).  Useful for along-gate 4×3 profiles.
    """
    v_perp = perpendicular_velocity(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    H = np.minimum(ds['depth'].values, depth_cap)
    dx = ds['dx'].values
    return v_perp * H[:, None] * dx[:, None] / SVERDRUP


def volume_transport_per_point_uncertainty(ds: xr.Dataset,
                                           depth_cap: float = DEPTH_CAP,
                                           gate_name: Optional[str] = None,
                                           u_into: Optional[float] = None,
                                           v_into: Optional[float] = None
                                           ) -> np.ndarray:
    """σ of per-point VT (Sv). Shape: (point, time)."""
    sigma_vp = perpendicular_velocity_uncertainty(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    H = np.minimum(ds['depth'].values, depth_cap)
    dx = ds['dx'].values
    return sigma_vp * H[:, None] * dx[:, None] / SVERDRUP


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                     FRESHWATER TRANSPORT                              ║
# ╚════════════════════════════════════════════════════════════════════════╝

def freshwater_transport(ds: xr.Dataset,
                         depth_cap: float = DEPTH_CAP,
                         s_ref: float = S_REF,
                         gate_name: Optional[str] = None,
                         u_into: Optional[float] = None,
                         v_into: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    FW(t) = Σᵢ v_perp(i,t) · (1 − SSS(i,t)/S_ref) · H(i) · dx(i)   [m³/s]

    Returns (fw_m3s, time).  NaN where SSS is unavailable.
    """
    if 'sss' not in ds:
        raise ValueError("No SSS variable in this NetCDF – FW transport unavailable.")
    v_perp = perpendicular_velocity(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    sss = ds['sss'].values                             # (pt, time)
    H = np.minimum(ds['depth'].values, depth_cap)
    dx = ds['dx'].values
    integrand = v_perp * (1.0 - sss / s_ref) * H[:, None] * dx[:, None]
    fw = np.where(np.all(np.isnan(integrand), axis=0), np.nan,
                  np.nansum(integrand, axis=0))
    time = pd.to_datetime(ds['time'].values)
    return fw, time


def freshwater_transport_uncertainty(ds: xr.Dataset,
                                     depth_cap: float = DEPTH_CAP,
                                     s_ref: float = S_REF,
                                     gate_name: Optional[str] = None,
                                     u_into: Optional[float] = None,
                                     v_into: Optional[float] = None) -> np.ndarray:
    """
    σ_FW(t) = √( Σᵢ (σ_v_perp · |1 − S/S_ref| · H · dx)² )   [m³/s]
    Only velocity uncertainty propagated (SSS error not included here).
    """
    sigma_vp = perpendicular_velocity_uncertainty(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    sss = ds['sss'].values
    H = np.minimum(ds['depth'].values, depth_cap)
    dx = ds['dx'].values
    fw_factor = np.abs(1.0 - sss / s_ref)
    terms = sigma_vp * fw_factor * H[:, None] * dx[:, None]
    return np.sqrt(np.nansum(terms**2, axis=0))


def freshwater_transport_per_point(ds: xr.Dataset,
                                    depth_cap: float = DEPTH_CAP,
                                    s_ref: float = S_REF,
                                    gate_name: Optional[str] = None,
                                    u_into: Optional[float] = None,
                                    v_into: Optional[float] = None) -> np.ndarray:
    """Per-point FW contribution (m³/s). Shape: (point, time)."""
    v_perp = perpendicular_velocity(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    sss = ds['sss'].values
    H = np.minimum(ds['depth'].values, depth_cap)
    dx = ds['dx'].values
    return v_perp * (1.0 - sss / s_ref) * H[:, None] * dx[:, None]


def freshwater_transport_per_point_uncertainty(ds: xr.Dataset,
                                               depth_cap: float = DEPTH_CAP,
                                               s_ref: float = S_REF,
                                               gate_name: Optional[str] = None,
                                               u_into: Optional[float] = None,
                                               v_into: Optional[float] = None) -> np.ndarray:
    """σ of per-point FW (m³/s). Shape: (point, time)."""
    sigma_vp = perpendicular_velocity_uncertainty(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    sss = ds['sss'].values
    H = np.minimum(ds['depth'].values, depth_cap)
    dx = ds['dx'].values
    return sigma_vp * np.abs(1.0 - sss / s_ref) * H[:, None] * dx[:, None]


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                          SALT FLUX                                    ║
# ╚════════════════════════════════════════════════════════════════════════╝

def salt_flux(ds: xr.Dataset,
              depth_cap: float = DEPTH_CAP,
              rho: float = RHO,
              gate_name: Optional[str] = None,
              u_into: Optional[float] = None,
              v_into: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sm(t) = Σᵢ ρ · (SSS(i,t)/1000) · v_perp(i,t) · H(i) · dx(i)   [kg/s]

    Returns (salt_kgs, time).
    """
    if 'sss' not in ds:
        raise ValueError("No SSS variable – salt flux unavailable.")
    v_perp = perpendicular_velocity(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    sss = ds['sss'].values
    H = np.minimum(ds['depth'].values, depth_cap)
    dx = ds['dx'].values
    integrand = rho * (sss / 1000.0) * v_perp * H[:, None] * dx[:, None]
    sm = np.where(np.all(np.isnan(integrand), axis=0), np.nan,
                  np.nansum(integrand, axis=0))
    time = pd.to_datetime(ds['time'].values)
    return sm, time


def salt_flux_uncertainty(ds: xr.Dataset,
                          depth_cap: float = DEPTH_CAP,
                          rho: float = RHO,
                          gate_name: Optional[str] = None,
                          u_into: Optional[float] = None,
                          v_into: Optional[float] = None) -> np.ndarray:
    """
    σ_Sm(t) = √( Σᵢ (ρ · S/1000 · σ_v_perp · H · dx)² )   [kg/s]
    """
    sigma_vp = perpendicular_velocity_uncertainty(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    sss = ds['sss'].values
    H = np.minimum(ds['depth'].values, depth_cap)
    dx = ds['dx'].values
    terms = rho * (sss / 1000.0) * sigma_vp * H[:, None] * dx[:, None]
    return np.sqrt(np.nansum(terms**2, axis=0))


def salt_flux_per_point(ds: xr.Dataset,
                        depth_cap: float = DEPTH_CAP,
                        rho: float = RHO,
                        gate_name: Optional[str] = None,
                        u_into: Optional[float] = None,
                        v_into: Optional[float] = None) -> np.ndarray:
    """Per-point salt flux (kg/s). Shape: (point, time)."""
    v_perp = perpendicular_velocity(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    sss = ds['sss'].values
    H = np.minimum(ds['depth'].values, depth_cap)
    dx = ds['dx'].values
    return rho * (sss / 1000.0) * v_perp * H[:, None] * dx[:, None]


def salt_flux_per_point_uncertainty(ds: xr.Dataset,
                                    depth_cap: float = DEPTH_CAP,
                                    rho: float = RHO,
                                    gate_name: Optional[str] = None,
                                    u_into: Optional[float] = None,
                                    v_into: Optional[float] = None) -> np.ndarray:
    """σ of per-point salt flux (kg/s). Shape: (point, time)."""
    sigma_vp = perpendicular_velocity_uncertainty(
        ds, gate_name=gate_name, u_into=u_into, v_into=v_into
    )
    sss = ds['sss'].values
    H = np.minimum(ds['depth'].values, depth_cap)
    dx = ds['dx'].values
    return rho * np.abs(sss / 1000.0) * sigma_vp * H[:, None] * dx[:, None]


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                     MONTHLY AGGREGATION                               ║
# ╚════════════════════════════════════════════════════════════════════════╝

def monthly_mean(values: np.ndarray, time: pd.DatetimeIndex
                 ) -> pd.DataFrame:
    """
    Compute monthly means from a daily time series.

    Parameters
    ----------
    values : 1-D array (n_time,)
    time   : DatetimeIndex of same length

    Returns
    -------
    DataFrame with columns ['date', 'value'] indexed monthly.
    """
    df = pd.DataFrame({'value': values}, index=time)
    return df.resample('MS').mean().dropna()


def monthly_climatology(values: np.ndarray, time: pd.DatetimeIndex
                        ) -> pd.DataFrame:
    """
    Climatological monthly mean ± std (all years collapsed).

    Returns DataFrame with columns ['month', 'mean', 'std', 'count'].
    """
    df = pd.DataFrame({'value': values, 'month': time.month}, index=time)
    clim = df.groupby('month')['value'].agg(['mean', 'std', 'count']).reset_index()
    return clim


def annual_mean(values: np.ndarray, time: pd.DatetimeIndex
                ) -> pd.DataFrame:
    """Annual means from daily data."""
    df = pd.DataFrame({'value': values}, index=time)
    return df.resample('YS').mean().dropna()


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                   ALONG-GATE PROFILES                                 ║
# ╚════════════════════════════════════════════════════════════════════════╝

def monthly_along_gate_profile(field: np.ndarray,
                               time: pd.DatetimeIndex,
                               x_km: np.ndarray,
                               sigma: Optional[np.ndarray] = None,
                               ) -> dict:
    """
    Compute mean along-gate profile for each calendar month.

    Parameters
    ----------
    field : (point, time) array — e.g. v_perp or sss
    time  : DatetimeIndex of length n_time
    x_km  : (point,) distance array
    sigma : (point, time) optional formal uncertainty array.
            If provided, the monthly mean formal error is included
            as 'sigma_mean' in the output dict.

    Returns
    -------
    dict  {month_int: {'mean': (point,), 'std': (point,), 'count': int,
                        'sigma_mean': (point,)  # only if sigma given}}
    """
    months = time.month
    result = {}
    for m in range(1, 13):
        mask = months == m
        n = int(mask.sum())
        if n == 0:
            entry = {'mean': np.full(len(x_km), np.nan),
                     'std': np.full(len(x_km), np.nan), 'count': 0}
            if sigma is not None:
                entry['sigma_mean'] = np.full(len(x_km), np.nan)
            result[m] = entry
        else:
            subset = field[:, mask]
            entry = {
                'mean': np.nanmean(subset, axis=1),
                'std': np.nanstd(subset, axis=1),
                'count': n,
            }
            if sigma is not None:
                # Mean formal error per point across the month's days
                # RMS of daily σ: √(mean(σ²))
                sig_sub = sigma[:, mask]
                entry['sigma_mean'] = np.sqrt(np.nanmean(sig_sub**2, axis=1))
            result[m] = entry
    return result


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                        ROLLING / FILTER                               ║
# ╚════════════════════════════════════════════════════════════════════════╝

def rolling_mean(values: np.ndarray, window: int = 365) -> np.ndarray:
    """Simple centred rolling mean (NaN-aware). Returns same length as input."""
    s = pd.Series(values)
    return s.rolling(window, center=True, min_periods=window // 2).mean().values
