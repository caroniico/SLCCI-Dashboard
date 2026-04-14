"""
Data Loaders Module - Separated from sidebar UI.
Handles loading data from various sources.
"""

from .base import DataLoaderResult, BaseDataLoader, get_gate_shapefile, apply_longitude_filter
from .slcci_loader import load_slcci_data
from .dtu_loader import load_dtu_data
from .cmems_l4_loader import load_cmems_l4_data

__all__ = [
    "DataLoaderResult",
    "BaseDataLoader",
    "get_gate_shapefile",
    "apply_longitude_filter",
    "load_slcci_data",
    "load_dtu_data",
    "load_cmems_l4_data",
]
