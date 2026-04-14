"""
Geostrophic Velocity Projection
================================
Compute the velocity component perpendicular to a gate (v_perp) using
local per-point normal vectors oriented toward the Arctic side.

All functions take **numpy arrays** — no xarray dependency.

Sign convention:
    v_perp > 0 → flow INTO the Arctic side of the gate.

Two methods are provided:

1. **Local per-point projection** (primary):
   v_perp(i,t) = ugos(i,t) · nx(i) + vgos(i,t) · ny(i)
   where (nx, ny) are the into-Arctic unit normals from
   `coordinates.local_into_arctic_unit_vectors`.

2. **DOT slope method** (for SLCCI along-track altimetry):
   v_geo = -(g/f) · ∂(DOT)/∂x
   where x is the along-gate distance.
"""

import numpy as np
from typing import Tuple

from .constants import GRAVITY
from .coordinates import local_into_arctic_unit_vectors


# ╔════════════════════════════════════════════════════════════════════════╗
# ║           METHOD 1 — LOCAL PER-POINT PROJECTION (CMEMS L4)           ║
# ╚════════════════════════════════════════════════════════════════════════╝

def perpendicular_velocity(
    ugos: np.ndarray,
    vgos: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """
    Compute perpendicular (cross-gate) velocity using local normals.

    v_perp(i,t) = ugos(i,t) · nx(i) + vgos(i,t) · ny(i)

    Parameters
    ----------
    ugos : (N_points, N_time) — eastward geostrophic velocity [m/s].
    vgos : (N_points, N_time) — northward geostrophic velocity [m/s].
    lon  : (N_points,) — gate point longitudes [degrees].
    lat  : (N_points,) — gate point latitudes [degrees].

    Returns
    -------
    v_perp : (N_points, N_time) — positive = into Arctic side [m/s].
    """
    nx, ny = local_into_arctic_unit_vectors(lon, lat)
    return ugos * nx[:, np.newaxis] + vgos * ny[:, np.newaxis]


def perpendicular_velocity_uncertainty(
    err_u: np.ndarray,
    err_v: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """
    Formal uncertainty on v_perp from CMEMS mapping errors.

    σ_v_perp = √( (σ_u · nx)² + (σ_v · ny)² )

    Parameters
    ----------
    err_u : (N_points, N_time) — formal error on ugos [m/s].
    err_v : (N_points, N_time) — formal error on vgos [m/s].
    lon   : (N_points,) — gate point longitudes [degrees].
    lat   : (N_points,) — gate point latitudes [degrees].

    Returns
    -------
    sigma_vp : (N_points, N_time) [m/s].
    """
    nx, ny = local_into_arctic_unit_vectors(lon, lat)
    return np.sqrt(
        (err_u * nx[:, np.newaxis]) ** 2
        + (err_v * ny[:, np.newaxis]) ** 2
    )


# ╔════════════════════════════════════════════════════════════════════════╗
# ║            METHOD 2 — DOT SLOPE (SLCCI ALONG-TRACK)                  ║
# ╚════════════════════════════════════════════════════════════════════════╝

def geostrophic_velocity_from_slope(
    slope_m_per_100km: np.ndarray,
    lat_mean: float,
) -> np.ndarray:
    """
    Convert a DOT slope to cross-gate geostrophic velocity.

    v_geo = -(g / f) · ∂η/∂x

    where slope is already in m / 100 km, so we convert to m/m first.

    Parameters
    ----------
    slope_m_per_100km : (N_time,) — DOT slope [m / 100 km].
    lat_mean : float — mean latitude of the gate [degrees].

    Returns
    -------
    v_geo : (N_time,) — geostrophic velocity [m/s].
             Positive = to the right of the slope direction (Northern Hemisphere).
    """
    from .constants import coriolis_parameter
    f = coriolis_parameter(lat_mean)
    if abs(f) < 1e-12:
        return np.full_like(slope_m_per_100km, np.nan)
    slope_m_per_m = np.asarray(slope_m_per_100km, dtype=float) / 1e5
    return -(GRAVITY / f) * slope_m_per_m


def dot_slope_along_gate(
    dot_profile: np.ndarray,
    x_km: np.ndarray,
) -> float:
    """
    Compute the linear DOT slope along a gate profile.

    Parameters
    ----------
    dot_profile : (N_points,) — DOT values along gate [m].
    x_km : (N_points,) — distance along gate [km].

    Returns
    -------
    slope : float — DOT slope in m / 100 km.
            Returns NaN if fewer than 2 valid points.
    """
    mask = np.isfinite(x_km) & np.isfinite(dot_profile)
    if np.sum(mask) < 2:
        return np.nan
    a, _ = np.polyfit(x_km[mask], dot_profile[mask], 1)
    return a * 100.0  # m/km → m/100km


def dot_slope_timeseries(
    dot_matrix: np.ndarray,
    x_km: np.ndarray,
) -> np.ndarray:
    """
    Compute DOT slope for each time step from a (space, time) matrix.

    Parameters
    ----------
    dot_matrix : (N_points, N_time) — DOT along gate for each time step [m].
    x_km : (N_points,) — distance along gate [km].

    Returns
    -------
    slopes : (N_time,) — slope in m / 100 km per time step.
    """
    n_time = dot_matrix.shape[1]
    slopes = np.full(n_time, np.nan, dtype=float)
    for t in range(n_time):
        slopes[t] = dot_slope_along_gate(dot_matrix[:, t], x_km)
    return slopes
