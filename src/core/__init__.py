"""
Core Module - Essential models and logging
==========================================

Note: Legacy submodules (satellite, coordinates, helpers, config, resolver)
moved to .cemetery/src_dead/core_unused/ on 2026-02-19.
"""
from .logging_config import setup_streamlit_logging, get_logger, LogContext

# Unified models
try:
    from .models import (
        BoundingBox,
        TimeRange,
        GateModel,
        DataRequest,
        TemporalResolution,
        SpatialResolution,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

__all__ = [
    # Logging
    "setup_streamlit_logging",
    "get_logger",
    "LogContext",
    # Models (if available)
    "BoundingBox",
    "TimeRange",
    "GateModel",
    "DataRequest",
    "TemporalResolution",
    "SpatialResolution",
]
