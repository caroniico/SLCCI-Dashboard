"""
Physical Constants
==================
Shared constants used across all ARCFRESH physics calculations.

These are the same values stored as NetCDF attributes in the consolidated
gate files and used by the Streamlit/API services.
"""

import numpy as np

# ── Fundamental constants ─────────────────────────────────────────────────
GRAVITY = 9.81          # m/s² — standard gravity
OMEGA = 7.2921e-5       # rad/s — Earth's angular velocity
EARTH_RADIUS = 6371e3   # m — mean Earth radius

# ── Oceanographic constants ───────────────────────────────────────────────
SVERDRUP = 1e6          # 1 Sv = 10⁶ m³/s
DEPTH_CAP = 250.0       # m — maximum integration depth
S_REF = 34.8            # PSU — reference salinity for freshwater transport
RHO = 1025.0            # kg/m³ — constant seawater density

# ── Reference geometry ────────────────────────────────────────────────────
ARCTIC_CENTER = (0.0, 90.0)  # (lon°E, lat°N) — Arctic centre for normal orientation

# ── Display helpers ───────────────────────────────────────────────────────
MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def coriolis_parameter(lat_deg: np.ndarray) -> np.ndarray:
    """
    Coriolis parameter f = 2Ω sin(φ).

    Parameters
    ----------
    lat_deg : array-like
        Latitude in degrees.

    Returns
    -------
    f : ndarray
        Coriolis parameter in rad/s.
    """
    return 2.0 * OMEGA * np.sin(np.deg2rad(np.asarray(lat_deg, dtype=float)))
