"""
ARCFRESH Physics Module
=======================
Pure numpy functions for Arctic gate oceanographic calculations.

No xarray, no Streamlit, no service layer dependencies.
All functions take and return numpy arrays.

Sub-modules:
    constants    — Physical constants (ρ, S_ref, g, etc.)
    coordinates  — Gate geometry, local normals, tangent vectors
    geostrophy   — Perpendicular velocity projection + uncertainty
    transport    — Volume, Freshwater, Salt Flux transport integrals
    aggregation  — Monthly profiles, climatology, rolling mean

Usage:
    from src.physics.constants import SVERDRUP, DEPTH_CAP, S_REF, RHO
    from src.physics.coordinates import local_into_arctic_unit_vectors
    from src.physics.geostrophy import perpendicular_velocity
    from src.physics.transport import volume_transport, freshwater_transport, salt_flux
    from src.physics.aggregation import monthly_along_gate_profile
"""

from .constants import SVERDRUP, DEPTH_CAP, S_REF, RHO, MONTH_NAMES, ARCTIC_CENTER
from .coordinates import (
    local_into_arctic_unit_vectors,
    unwrap_longitudes,
)
from .geostrophy import (
    perpendicular_velocity,
    perpendicular_velocity_uncertainty,
)
from .transport import (
    volume_transport,
    volume_transport_uncertainty,
    volume_transport_per_point,
    volume_transport_per_point_uncertainty,
    freshwater_transport,
    freshwater_transport_uncertainty,
    freshwater_transport_per_point,
    freshwater_transport_per_point_uncertainty,
    salt_flux,
    salt_flux_uncertainty,
    salt_flux_per_point,
    salt_flux_per_point_uncertainty,
)
from .aggregation import (
    monthly_along_gate_profile,
    monthly_mean,
    monthly_climatology,
    annual_mean,
    rolling_mean,
)

__all__ = [
    # constants
    "SVERDRUP", "DEPTH_CAP", "S_REF", "RHO", "MONTH_NAMES", "ARCTIC_CENTER",
    # coordinates
    "local_into_arctic_unit_vectors", "unwrap_longitudes",
    # geostrophy
    "perpendicular_velocity", "perpendicular_velocity_uncertainty",
    # transport
    "volume_transport", "volume_transport_uncertainty",
    "volume_transport_per_point", "volume_transport_per_point_uncertainty",
    "freshwater_transport", "freshwater_transport_uncertainty",
    "freshwater_transport_per_point", "freshwater_transport_per_point_uncertainty",
    "salt_flux", "salt_flux_uncertainty",
    "salt_flux_per_point", "salt_flux_per_point_uncertainty",
    # aggregation
    "monthly_along_gate_profile", "monthly_mean", "monthly_climatology",
    "annual_mean", "rolling_mean",
]
