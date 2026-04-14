"""
Transport Integrals
===================
Volume transport, freshwater transport, and salt flux through Arctic gates.

All functions take **numpy arrays** — no xarray dependency.
They integrate over gate points (axis=0) for each time step (axis=1).

Sign convention:
    Positive = INTO the Arctic side of the gate.

Formulas
--------
Volume Transport (Sv):
    VT(t) = Σᵢ v_perp(i,t) · H(i) · dx(i)  /  10⁶

Freshwater Transport (m³/s):
    FW(t) = Σᵢ v_perp(i,t) · (1 − S(i,t)/S_ref) · H(i) · dx(i)

Salt Flux (kg/s):
    SF(t) = Σᵢ ρ · (S(i,t)/1000) · v_perp(i,t) · H(i) · dx(i)
"""

import numpy as np
from typing import Tuple

from .constants import SVERDRUP, DEPTH_CAP, S_REF, RHO


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _effective_depth(depth: np.ndarray, depth_cap: float = DEPTH_CAP) -> np.ndarray:
    """Clip bathymetry to a maximum integration depth. Shape preserved."""
    return np.minimum(np.asarray(depth, dtype=float), depth_cap)


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                     VOLUME TRANSPORT                                  ║
# ╚════════════════════════════════════════════════════════════════════════╝

def volume_transport(
    v_perp: np.ndarray,
    depth: np.ndarray,
    dx: np.ndarray,
    depth_cap: float = DEPTH_CAP,
) -> np.ndarray:
    """
    Integrated volume transport per time step.

    Parameters
    ----------
    v_perp : (N_pts, N_time) — perpendicular velocity [m/s].
    depth  : (N_pts,) — bathymetry (positive down) [m].
    dx     : (N_pts,) — segment width [m].
    depth_cap : float — max integration depth [m].

    Returns
    -------
    vt_sv : (N_time,) — volume transport [Sv].
    """
    H = _effective_depth(depth, depth_cap)
    q = v_perp * H[:, None] * dx[:, None]          # (pts, time)
    vt = np.nansum(q, axis=0)                       # (time,)
    # Mark all-NaN columns
    vt[np.all(np.isnan(v_perp), axis=0)] = np.nan
    return vt / SVERDRUP


def volume_transport_uncertainty(
    sigma_vp: np.ndarray,
    depth: np.ndarray,
    dx: np.ndarray,
    depth_cap: float = DEPTH_CAP,
) -> np.ndarray:
    """
    Formal uncertainty on integrated volume transport.

    σ_VT(t) = √( Σᵢ (σ_v_perp(i,t) · H(i) · dx(i))² ) / 10⁶

    Returns
    -------
    sigma_vt : (N_time,) [Sv].
    """
    H = _effective_depth(depth, depth_cap)
    terms = sigma_vp * H[:, None] * dx[:, None]
    return np.sqrt(np.nansum(terms ** 2, axis=0)) / SVERDRUP


def volume_transport_per_point(
    v_perp: np.ndarray,
    depth: np.ndarray,
    dx: np.ndarray,
    depth_cap: float = DEPTH_CAP,
) -> np.ndarray:
    """
    Transport contribution per gate point per time step.

    Returns
    -------
    vt_pp : (N_pts, N_time) [Sv].
    """
    H = _effective_depth(depth, depth_cap)
    return v_perp * H[:, None] * dx[:, None] / SVERDRUP


def volume_transport_per_point_uncertainty(
    sigma_vp: np.ndarray,
    depth: np.ndarray,
    dx: np.ndarray,
    depth_cap: float = DEPTH_CAP,
) -> np.ndarray:
    """
    σ of per-point volume transport.

    Returns
    -------
    sigma_pp : (N_pts, N_time) [Sv].
    """
    H = _effective_depth(depth, depth_cap)
    return sigma_vp * H[:, None] * dx[:, None] / SVERDRUP


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                     FRESHWATER TRANSPORT                              ║
# ╚════════════════════════════════════════════════════════════════════════╝

def freshwater_transport(
    v_perp: np.ndarray,
    sss: np.ndarray,
    depth: np.ndarray,
    dx: np.ndarray,
    depth_cap: float = DEPTH_CAP,
    s_ref: float = S_REF,
) -> np.ndarray:
    """
    Integrated freshwater transport per time step.

    FW(t) = Σᵢ v_perp(i,t) · (1 − SSS(i,t)/S_ref) · H(i) · dx(i)

    Parameters
    ----------
    v_perp : (N_pts, N_time) [m/s].
    sss    : (N_pts, N_time) [PSU].
    depth  : (N_pts,) [m].
    dx     : (N_pts,) [m].

    Returns
    -------
    fw : (N_time,) [m³/s].  NaN where SSS is unavailable.
    """
    H = _effective_depth(depth, depth_cap)
    integrand = v_perp * (1.0 - sss / s_ref) * H[:, None] * dx[:, None]
    fw = np.where(
        np.all(np.isnan(integrand), axis=0),
        np.nan,
        np.nansum(integrand, axis=0),
    )
    return fw


def freshwater_transport_uncertainty(
    sigma_vp: np.ndarray,
    sss: np.ndarray,
    depth: np.ndarray,
    dx: np.ndarray,
    depth_cap: float = DEPTH_CAP,
    s_ref: float = S_REF,
) -> np.ndarray:
    """
    Formal uncertainty on integrated freshwater transport.

    Only velocity uncertainty is propagated (SSS error not included).

    Returns
    -------
    sigma_fw : (N_time,) [m³/s].
    """
    H = _effective_depth(depth, depth_cap)
    fw_factor = np.abs(1.0 - sss / s_ref)
    terms = sigma_vp * fw_factor * H[:, None] * dx[:, None]
    return np.sqrt(np.nansum(terms ** 2, axis=0))


def freshwater_transport_per_point(
    v_perp: np.ndarray,
    sss: np.ndarray,
    depth: np.ndarray,
    dx: np.ndarray,
    depth_cap: float = DEPTH_CAP,
    s_ref: float = S_REF,
) -> np.ndarray:
    """Per-point freshwater contribution. Returns (N_pts, N_time) [m³/s]."""
    H = _effective_depth(depth, depth_cap)
    return v_perp * (1.0 - sss / s_ref) * H[:, None] * dx[:, None]


def freshwater_transport_per_point_uncertainty(
    sigma_vp: np.ndarray,
    sss: np.ndarray,
    depth: np.ndarray,
    dx: np.ndarray,
    depth_cap: float = DEPTH_CAP,
    s_ref: float = S_REF,
) -> np.ndarray:
    """σ of per-point freshwater transport. Returns (N_pts, N_time) [m³/s]."""
    H = _effective_depth(depth, depth_cap)
    return sigma_vp * np.abs(1.0 - sss / s_ref) * H[:, None] * dx[:, None]


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                          SALT FLUX                                    ║
# ╚════════════════════════════════════════════════════════════════════════╝

def salt_flux(
    v_perp: np.ndarray,
    sss: np.ndarray,
    depth: np.ndarray,
    dx: np.ndarray,
    depth_cap: float = DEPTH_CAP,
    rho: float = RHO,
) -> np.ndarray:
    """
    Integrated salt flux per time step.

    SF(t) = Σᵢ ρ · (SSS(i,t)/1000) · v_perp(i,t) · H(i) · dx(i)

    Returns
    -------
    sf : (N_time,) [kg/s].
    """
    H = _effective_depth(depth, depth_cap)
    integrand = rho * (sss / 1000.0) * v_perp * H[:, None] * dx[:, None]
    sf = np.where(
        np.all(np.isnan(integrand), axis=0),
        np.nan,
        np.nansum(integrand, axis=0),
    )
    return sf


def salt_flux_uncertainty(
    sigma_vp: np.ndarray,
    sss: np.ndarray,
    depth: np.ndarray,
    dx: np.ndarray,
    depth_cap: float = DEPTH_CAP,
    rho: float = RHO,
) -> np.ndarray:
    """
    Formal uncertainty on integrated salt flux.

    Returns
    -------
    sigma_sf : (N_time,) [kg/s].
    """
    H = _effective_depth(depth, depth_cap)
    terms = rho * (sss / 1000.0) * sigma_vp * H[:, None] * dx[:, None]
    return np.sqrt(np.nansum(terms ** 2, axis=0))


def salt_flux_per_point(
    v_perp: np.ndarray,
    sss: np.ndarray,
    depth: np.ndarray,
    dx: np.ndarray,
    depth_cap: float = DEPTH_CAP,
    rho: float = RHO,
) -> np.ndarray:
    """Per-point salt flux. Returns (N_pts, N_time) [kg/s]."""
    H = _effective_depth(depth, depth_cap)
    return rho * (sss / 1000.0) * v_perp * H[:, None] * dx[:, None]


def salt_flux_per_point_uncertainty(
    sigma_vp: np.ndarray,
    sss: np.ndarray,
    depth: np.ndarray,
    dx: np.ndarray,
    depth_cap: float = DEPTH_CAP,
    rho: float = RHO,
) -> np.ndarray:
    """σ of per-point salt flux. Returns (N_pts, N_time) [kg/s]."""
    H = _effective_depth(depth, depth_cap)
    return rho * np.abs(sss / 1000.0) * sigma_vp * H[:, None] * dx[:, None]
