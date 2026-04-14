"""
Gate Geometry & Coordinate Utilities
=====================================
Functions for computing gate tangent vectors, local normals oriented
toward the Arctic, and distance metrics.

All functions take **numpy arrays** (lon, lat in degrees) — no xarray.

Sign convention:
    The local normal at each gate point is oriented so that a positive
    dot product with the velocity vector means flow INTO the Arctic side.
"""

import numpy as np
from typing import Tuple

from .constants import ARCTIC_CENTER, EARTH_RADIUS


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                     LONGITUDE UTILITIES                               ║
# ╚════════════════════════════════════════════════════════════════════════╝

def unwrap_longitudes(lon: np.ndarray) -> np.ndarray:
    """Return longitudes unwrapped to a continuous sequence (degrees).

    Handles dateline crossings by unwrapping via radians.
    """
    lon = np.asarray(lon, dtype=float)
    return np.rad2deg(np.unwrap(np.deg2rad(lon)))


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                     TANGENT & NORMAL VECTORS                          ║
# ╚════════════════════════════════════════════════════════════════════════╝

def _safe_unit(x: float, y: float) -> Tuple[float, float]:
    """Normalise (x, y) to unit length; returns (0, 1) for zero vectors."""
    mag = float(np.hypot(x, y))
    if mag < 1e-12:
        return (0.0, 1.0)
    return (x / mag, y / mag)


def local_tangent_unit_vectors(
    lon: np.ndarray,
    lat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local unit tangent vectors along the gate at each point.

    Uses central differences for interior points and forward/backward
    differences at endpoints.  The longitudinal metric is scaled by
    cos(lat) to convert degree increments to approximate metres on the
    sphere.

    Parameters
    ----------
    lon, lat : (N,) arrays in degrees.

    Returns
    -------
    tx, ty : (N,) arrays — eastward and northward components of the unit
             tangent at each gate point.
    """
    lon_u = unwrap_longitudes(lon)
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
    lon: np.ndarray,
    lat: np.ndarray,
    arctic_center: Tuple[float, float] = ARCTIC_CENTER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unit vector from each gate point toward the Arctic centre (metric-corrected)."""
    lon_u = unwrap_longitudes(lon)
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
    lon: np.ndarray,
    lat: np.ndarray,
    arctic_center: Tuple[float, float] = ARCTIC_CENTER,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local per-point normal unit vectors oriented toward the Arctic side.

    Algorithm:
        1. Compute local tangent at each point.
        2. Build right-hand normal: n = (-ty, tx).
        3. Flip per-point when the normal points away from Arctic centre.
        4. Conservative continuity pass to remove isolated sign flips.

    Parameters
    ----------
    lon, lat : (N,) arrays in degrees.
    arctic_center : (lon°, lat°) of the Arctic pole.

    Returns
    -------
    nx, ny : (N,) arrays — eastward and northward components of the
             into-Arctic unit normal at each gate point.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    if len(lon) < 2:
        return np.array([0.0], dtype=float), np.array([1.0], dtype=float)

    tx, ty = local_tangent_unit_vectors(lon, lat)
    nx = -ty
    ny = tx

    ax, ay = _to_arctic_unit_vectors(lon, lat, arctic_center=arctic_center)

    # Per-point orientation toward Arctic side
    dot_arc = nx * ax + ny * ay
    flip = dot_arc < 0.0
    nx[flip] *= -1.0
    ny[flip] *= -1.0

    # Conservative continuity pass:
    # If local direction flips against the previous point, flip only when
    # the flipped vector still points toward Arctic.
    for i in range(1, len(nx)):
        if not np.isfinite(nx[i - 1]) or not np.isfinite(nx[i]):
            continue
        if nx[i] * nx[i - 1] + ny[i] * ny[i - 1] < 0.0:
            flipped_dot_arc = (-nx[i]) * ax[i] + (-ny[i]) * ay[i]
            if flipped_dot_arc >= 0.0:
                nx[i] *= -1.0
                ny[i] *= -1.0

    # Safety normalisation
    mag = np.hypot(nx, ny)
    mag = np.where(mag < 1e-12, 1.0, mag)
    nx = nx / mag
    ny = ny / mag

    return nx, ny


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                     DISTANCE UTILITIES                                ║
# ╚════════════════════════════════════════════════════════════════════════╝

def haversine_distances(
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """
    Compute segment lengths (metres) between consecutive gate points.

    Parameters
    ----------
    lon, lat : (N,) arrays in degrees.

    Returns
    -------
    dx : (N,) array — dx[0] = distance to point 1; dx[-1] copies dx[-2].
         This matches the convention used in the consolidated NetCDF files.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n = len(lon)
    if n < 2:
        return np.array([0.0])

    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)

    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = EARTH_RADIUS * c  # metres

    dx = np.empty(n, dtype=float)
    dx[0] = d[0]
    dx[1:] = d
    return dx


def cumulative_distance_km(
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """
    Cumulative along-gate distance in km, starting from 0.

    Parameters
    ----------
    lon, lat : (N,) arrays in degrees.

    Returns
    -------
    x_km : (N,) array starting at 0.
    """
    dx = haversine_distances(lon, lat)
    x_km = np.concatenate([[0.0], np.cumsum(dx[1:])]) / 1e3
    return x_km
