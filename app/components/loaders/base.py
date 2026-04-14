"""
Base loader classes and utilities.
"""

from dataclasses import dataclass
from typing import Optional, Any
from pathlib import Path


@dataclass
class DataLoaderResult:
    """Result of a data loading operation."""
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    info_message: Optional[str] = None
    
    # Metadata
    n_observations: int = 0
    n_cycles: int = 0
    lon_range: tuple = (0, 0)
    lat_range: tuple = (0, 0)
    strait_name: str = ""
    pass_number: int = 0


class BaseDataLoader:
    """Base class for data loaders."""
    
    def __init__(self, config):
        self.config = config
    
    def validate_paths(self) -> DataLoaderResult:
        """Validate required paths exist."""
        raise NotImplementedError
    
    def load(self) -> DataLoaderResult:
        """Load data and return result."""
        raise NotImplementedError


def get_gate_shapefile(gate_id: Optional[str], use_parent: bool = False) -> Optional[str]:
    """
    Get the shapefile path for a gate.
    
    For divided gates (e.g., davis_strait_west), returns the parent gate
    shapefile (davis_strait.shp) to show the full gate line.
    
    Args:
        gate_id: Gate identifier
        use_parent: If True, return parent gate for divided gates
        
    Returns:
        Path to shapefile or None
    """
    if not gate_id:
        return None
    
    gates_dir = Path(__file__).parent.parent.parent.parent / "gates"
    
    # Check for _west or _east suffix -> return parent gate
    parent_gate_id = None
    if gate_id.endswith("_west") or gate_id.endswith("_east"):
        parent_gate_id = gate_id.rsplit("_", 1)[0]
    
    # If use_parent and we have a parent, use it
    if use_parent and parent_gate_id:
        gate_id = parent_gate_id
    
    # Try exact match
    shp_path = gates_dir / f"{gate_id}.shp"
    if shp_path.exists():
        return str(shp_path)
    
    # Try with different patterns
    patterns = [
        f"{gate_id}_TPJ_pass_*.shp",
        f"{gate_id}_S3_pass_*.shp",
        f"{gate_id}_*.shp",
    ]
    
    for pattern in patterns:
        matches = list(gates_dir.glob(pattern))
        if matches:
            return str(matches[0])
    
    return None


def apply_longitude_filter(pass_data, lon_min: float = None, lon_max: float = None, gate_name: str = ""):
    """
    Apply longitude filter to PassData, filtering gate points and recomputing derived values.
    
    This is used for divided gates (East/West) where we load the FULL gate shapefile
    but only want data from a specific longitude range.
    
    Args:
        pass_data: PassData object
        lon_min: Minimum longitude (data must be > lon_min)
        lon_max: Maximum longitude (data must be < lon_max)
        gate_name: Name of selected gate for logging
        
    Returns:
        Modified PassData or None if no data remains
    """
    import numpy as np
    import pandas as pd
    import copy
    from dataclasses import fields, is_dataclass, replace
    
    if lon_min is None and lon_max is None:
        return pass_data
    
    gate_lon = getattr(pass_data, 'gate_lon_pts', None)
    if gate_lon is None or len(gate_lon) == 0:
        return pass_data
    
    # Build longitude mask for gate points
    mask = np.ones(len(gate_lon), dtype=bool)
    if lon_min is not None:
        mask &= (gate_lon > lon_min)
    if lon_max is not None:
        mask &= (gate_lon < lon_max)
    
    n_filtered = np.sum(mask)
    if n_filtered == 0:
        return None
    
    # Filter gate points
    new_gate_lon = gate_lon[mask]
    new_gate_lat = pass_data.gate_lat_pts[mask] if hasattr(pass_data, 'gate_lat_pts') and pass_data.gate_lat_pts is not None else None
    
    # Recompute x_km
    R_earth = 6371.0
    if new_gate_lat is not None and len(new_gate_lat) > 0:
        lat_rad = np.deg2rad(np.mean(new_gate_lat))
        lon_rad = np.deg2rad(new_gate_lon)
        new_x_km = (lon_rad - lon_rad[0]) * np.cos(lat_rad) * R_earth
    else:
        old_x_km = getattr(pass_data, 'x_km', None)
        new_x_km = old_x_km[mask] if old_x_km is not None else None
    
    # Filter DataFrame
    df = getattr(pass_data, 'df', None)
    new_df = None
    if df is not None and not df.empty and 'lon' in df.columns:
        df_mask = pd.Series(True, index=df.index)
        if lon_min is not None:
            df_mask &= (df['lon'] > lon_min)
        if lon_max is not None:
            df_mask &= (df['lon'] < lon_max)
        
        new_df = df[df_mask].copy()
        
        if new_df.empty:
            return None
    
    # Filter DOT matrix and recompute derived values
    dot_matrix = getattr(pass_data, 'dot_matrix', None)
    new_dot_matrix = None
    new_profile_mean = None
    new_slope_series = None
    
    if dot_matrix is not None and len(dot_matrix) > 0:
        new_dot_matrix = dot_matrix[mask, :]
        
        # Recompute profile_mean using POOLED method from filtered DataFrame
        # This gives equal weight to each observation, not each time period
        if new_df is not None and not new_df.empty and 'dot' in new_df.columns and 'lon' in new_df.columns:
            lon_min_df = new_df['lon'].min()
            lon_max_df = new_df['lon'].max()
            lon_bin_size = 0.01  # Same as in slcci_service
            lon_bins = np.arange(lon_min_df, lon_max_df + lon_bin_size, lon_bin_size)
            
            df_temp = new_df.copy()
            df_temp['lon_bin'] = pd.cut(df_temp['lon'], bins=lon_bins, labels=False, include_lowest=True)
            binned_stats = df_temp.groupby('lon_bin')['dot'].mean()
            
            # Match profile to new_gate_lon
            new_profile_mean = np.full(len(new_gate_lon), np.nan, dtype=float)
            lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
            
            for i, lon in enumerate(new_gate_lon):
                # Find closest bin
                bin_idx = np.argmin(np.abs(lon_centers - lon))
                if bin_idx in binned_stats.index:
                    new_profile_mean[i] = binned_stats[bin_idx]
        else:
            # Fallback: mean-of-means (less accurate but works without df)
            new_profile_mean = np.nanmean(new_dot_matrix, axis=1)
        
        # Recompute slope_series
        n_time = new_dot_matrix.shape[1]
        new_slope_series = np.full(n_time, np.nan, dtype=float)
        
        for it in range(n_time):
            y = new_dot_matrix[:, it]
            valid = np.isfinite(new_x_km) & np.isfinite(y)
            if np.sum(valid) >= 2:
                try:
                    a, b = np.polyfit(new_x_km[valid], y[valid], 1)
                    new_slope_series[it] = a * 100.0
                except:
                    pass
    
    # Filter velocity matrices (ugos, vgos) for divided gates
    ugos_matrix = getattr(pass_data, 'ugos_matrix', None)
    vgos_matrix = getattr(pass_data, 'vgos_matrix', None)
    err_ugosa_matrix = getattr(pass_data, 'err_ugosa_matrix', None)
    err_vgosa_matrix = getattr(pass_data, 'err_vgosa_matrix', None)
    new_ugos_matrix = None
    new_vgos_matrix = None
    new_err_ugosa_matrix = None
    new_err_vgosa_matrix = None
    
    if ugos_matrix is not None and len(ugos_matrix) > 0:
        new_ugos_matrix = ugos_matrix[mask, :]
    if vgos_matrix is not None and len(vgos_matrix) > 0:
        new_vgos_matrix = vgos_matrix[mask, :]
    if err_ugosa_matrix is not None and len(err_ugosa_matrix) > 0:
        new_err_ugosa_matrix = err_ugosa_matrix[mask, :]
    if err_vgosa_matrix is not None and len(err_vgosa_matrix) > 0:
        new_err_vgosa_matrix = err_vgosa_matrix[mask, :]
    
    # Update strait_name suffix
    suffix = ""
    if lon_max is not None and lon_min is None:
        suffix = " (West)"
    elif lon_min is not None and lon_max is None:
        suffix = " (East)"
    
    new_strait_name = getattr(pass_data, 'strait_name', 'Unknown')
    if suffix and suffix not in new_strait_name:
        new_strait_name = new_strait_name + suffix
    
    updates = {
        "gate_lon_pts": new_gate_lon,
        "gate_lat_pts": new_gate_lat,
        "x_km": new_x_km,
        "dot_matrix": new_dot_matrix if new_dot_matrix is not None else dot_matrix,
        "profile_mean": new_profile_mean if new_profile_mean is not None else getattr(pass_data, 'profile_mean', None),
        "slope_series": new_slope_series if new_slope_series is not None else getattr(pass_data, 'slope_series', None),
        "ugos_matrix": new_ugos_matrix if new_ugos_matrix is not None else ugos_matrix,
        "vgos_matrix": new_vgos_matrix if new_vgos_matrix is not None else vgos_matrix,
        "err_ugosa_matrix": new_err_ugosa_matrix if new_err_ugosa_matrix is not None else err_ugosa_matrix,
        "err_vgosa_matrix": new_err_vgosa_matrix if new_err_vgosa_matrix is not None else err_vgosa_matrix,
        "strait_name": new_strait_name,
    }
    if hasattr(pass_data, "df"):
        updates["df"] = new_df if new_df is not None else df

    # Create a new object to avoid mutating cached/session objects in-place.
    if is_dataclass(pass_data):
        valid_fields = {f.name for f in fields(pass_data)}
        safe_updates = {k: v for k, v in updates.items() if k in valid_fields}
        return replace(pass_data, **safe_updates)

    out = copy.deepcopy(pass_data)
    for key, value in updates.items():
        if hasattr(out, key):
            setattr(out, key, value)
    return out
