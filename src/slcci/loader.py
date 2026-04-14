"""
SLCCI Cycle Loader
==================
Read SLCCI_ALTDB_J2_CycleXXX_V2.nc files and build a raw DataFrame.

Pure I/O — no binning, no DOT computation here.
DOT is computed in dot.py after geoid interpolation.
"""

import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


def _wrap_longitude(lon: np.ndarray) -> np.ndarray:
    """Wrap longitudes to [0, 360)."""
    return lon % 360


def _lon_in_bounds(lon: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Boolean mask: lon ∈ [lo, hi], handling wrap-around."""
    if lo <= hi:
        return (lon >= lo) & (lon <= hi)
    else:
        # wraps around 360
        return (lon >= lo) | (lon <= hi)


def _detect_satellite_type(base_dir: str) -> str:
    """Auto-detect satellite type from filenames in base_dir."""
    p = Path(base_dir)
    if not p.exists():
        return "J2"
    for f in p.iterdir():
        if "SLCCI_ALTDB_J1" in f.name:
            return "J1"
        if "SLCCI_ALTDB_J2" in f.name:
            return "J2"
    return "J2"


def load_cycle(
    filepath: str,
    pass_number: Optional[int] = None,
    lat_bounds: Optional[Tuple[float, float]] = None,
    lon_bounds: Optional[Tuple[float, float]] = None,
    use_flag: bool = True,
    cycle_number: Optional[int] = None,
) -> Optional[xr.Dataset]:
    """
    Load a single SLCCI cycle NetCDF file with optional spatial + pass filter.

    Parameters
    ----------
    filepath : str
        Full path to SLCCI_ALTDB_J2_CycleXXX_V2.nc
    pass_number : int, optional
        If given, keep only this pass
    lat_bounds : (lat_min, lat_max), optional
    lon_bounds : (lon_min, lon_max), optional
        Longitude bounds in [0, 360) convention
    use_flag : bool
        If True, drop observations where validation_flag != 0
    cycle_number : int, optional
        Cycle number to tag (if None, parsed from filename)

    Returns
    -------
    xr.Dataset filtered, or None if empty / error
    """
    if not os.path.exists(filepath):
        return None

    try:
        with xr.open_dataset(filepath, decode_times=False) as ds:
            lon = ds["longitude"].values
            lat = ds["latitude"].values
            lon_wrapped = _wrap_longitude(lon)

            # --- spatial filter ---
            mask = np.ones(len(lat), dtype=bool)
            if lat_bounds is not None:
                mask &= (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
            if lon_bounds is not None:
                mask &= _lon_in_bounds(lon_wrapped, lon_bounds[0], lon_bounds[1])

            if mask.sum() == 0:
                return None

            ds_f = ds.isel(time=mask)
            ds_f = ds_f.assign_coords(longitude=("time", lon_wrapped[mask]))

            # --- decode time (days since 1950-01-01) ---
            time_vals = pd.to_datetime(ds_f["time"].values, origin="1950-01-01", unit="D")
            ds_f = ds_f.assign_coords(time=time_vals)

            # --- quality flag ---
            if use_flag and "validation_flag" in ds_f:
                valid = ds_f["validation_flag"] == 0
                ds_f = ds_f.isel(time=valid)

            # --- standardise pass variable name ---
            for var in ("pass", "track", "pass_number", "track_number"):
                if var in ds_f.variables and var != "pass":
                    ds_f = ds_f.rename({var: "pass"})
                    break

            # --- filter by pass ---
            if pass_number is not None and "pass" in ds_f:
                pvals = np.round(ds_f["pass"].values).astype(int)
                pmask = pvals == int(pass_number)
                if pmask.sum() == 0:
                    return None
                ds_f = ds_f.isel(time=pmask)

            if ds_f.sizes.get("time", 0) == 0:
                return None

            # --- tag cycle ---
            if cycle_number is None:
                # parse from filename  e.g. SLCCI_ALTDB_J2_Cycle003_V2.nc
                import re
                m = re.search(r"Cycle(\d+)", Path(filepath).name)
                cycle_number = int(m.group(1)) if m else 0

            ds_f = ds_f.assign_coords(cycle=("time", np.full(ds_f.sizes["time"], cycle_number)))

            return ds_f.load()  # materialise to memory

    except Exception as e:
        logger.debug(f"Error loading {filepath}: {e}")
        return None


def load_cycles_serial(
    base_dir: str,
    cycles: List[int],
    pass_number: Optional[int] = None,
    lat_bounds: Optional[Tuple[float, float]] = None,
    lon_bounds: Optional[Tuple[float, float]] = None,
    use_flag: bool = True,
    satellite: str = "J2",
) -> Optional[xr.Dataset]:
    """
    Load multiple cycles sequentially and concatenate.

    Parameters
    ----------
    base_dir : str
        Directory containing SLCCI_ALTDB_{sat}_CycleXXX_V2.nc files
    cycles : list of int
        Cycle numbers to load
    pass_number, lat_bounds, lon_bounds, use_flag : see load_cycle
    satellite : str
        "J1" or "J2" (determines filename pattern)

    Returns
    -------
    xr.Dataset concatenated across all cycles, or None
    """
    sat = satellite or _detect_satellite_type(base_dir)
    datasets = []

    for cyc in cycles:
        fname = f"SLCCI_ALTDB_{sat}_Cycle{cyc:03d}_V2.nc"
        fpath = os.path.join(base_dir, fname)
        ds = load_cycle(
            fpath,
            pass_number=pass_number,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            use_flag=use_flag,
            cycle_number=cyc,
        )
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        return None

    combined = xr.concat(datasets, dim="time")
    combined.attrs["satellite_type"] = sat
    return combined
