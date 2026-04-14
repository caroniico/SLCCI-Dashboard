"""
SLCCI Data Loader - Handles loading ESA Sea Level CCI data.
"""

from pathlib import Path
from typing import Optional
from .base import DataLoaderResult, get_gate_shapefile, apply_longitude_filter


def load_slcci_data(
    base_dir: str,
    geoid_path: str,
    gate_id: str,
    pass_number: int,
    cycle_start: int = 1,
    cycle_end: int = 281,
    lon_filter_min: Optional[float] = None,
    lon_filter_max: Optional[float] = None,
    use_flag: bool = True,
    lat_buffer_deg: float = 1.0,
    lon_buffer_deg: float = 1.0,
    lon_bin_size: float = 0.01,
    source: str = "local",
    use_cache: bool = True,
) -> DataLoaderResult:
    """
    Load SLCCI data for a gate.
    
    Args:
        base_dir: Path to J2 data directory
        geoid_path: Path to TUM geoid file
        gate_id: Gate identifier
        pass_number: Satellite pass number
        cycle_start: Start cycle
        cycle_end: End cycle
        lon_filter_min: Minimum longitude filter (for East sections)
        lon_filter_max: Maximum longitude filter (for West sections)
        use_flag: Whether to use quality flags
        lat_buffer_deg: Latitude buffer for filtering
        lon_buffer_deg: Longitude buffer for filtering
        lon_bin_size: Longitude bin size for gridding
        source: "local" or "api"
        use_cache: Whether to use data cache
        
    Returns:
        DataLoaderResult with success status and data/error
    """
    # Validate paths
    if not Path(geoid_path).exists():
        return DataLoaderResult(
            success=False,
            error_message=f"Geoid not found: {geoid_path}"
        )
    
    if source == "local" and not Path(base_dir).exists():
        return DataLoaderResult(
            success=False,
            error_message=f"Path not found: {base_dir}"
        )
    
    # Get gate shapefile (use parent for divided gates)
    gate_path = get_gate_shapefile(gate_id, use_parent=True)
    if not gate_path:
        return DataLoaderResult(
            success=False,
            error_message="Gate shapefile not found"
        )
    
    try:
        from src.services.slcci_service import SLCCIService, SLCCIConfig
        
        cycles = list(range(cycle_start, cycle_end + 1))
        
        slcci_config = SLCCIConfig(
            base_dir=base_dir,
            geoid_path=geoid_path,
            cycles=cycles,
            use_flag=use_flag,
            lat_buffer_deg=lat_buffer_deg,
            lon_buffer_deg=lon_buffer_deg,
            lon_bin_size=lon_bin_size,
            source=source,
            satellite="J2",
        )
        
        service = SLCCIService(slcci_config)
        
        # Load pass data
        pass_data = service.load_pass_data(
            gate_path=gate_path,
            pass_number=pass_number,
            cycles=cycles,
            use_cache=use_cache,
        )
        
        if pass_data is None:
            return DataLoaderResult(
                success=False,
                error_message=f"No data for pass {pass_number} in this gate area"
            )
        
        # Apply longitude filter if needed
        if lon_filter_min is not None or lon_filter_max is not None:
            pass_data = apply_longitude_filter(
                pass_data, lon_filter_min, lon_filter_max, gate_id
            )
            if pass_data is None:
                return DataLoaderResult(
                    success=False,
                    error_message="No data after longitude filter! The pass may not cross this section."
                )
        
        # Extract metadata
        df = getattr(pass_data, 'df', None)
        n_obs = len(df) if df is not None else 0
        n_cyc = df['cycle'].nunique() if df is not None and 'cycle' in df.columns else 0
        lon_min = df['lon'].min() if df is not None and 'lon' in df.columns else 0
        lon_max = df['lon'].max() if df is not None and 'lon' in df.columns else 0
        
        return DataLoaderResult(
            success=True,
            data=pass_data,
            n_observations=n_obs,
            n_cycles=n_cyc,
            lon_range=(lon_min, lon_max),
            strait_name=getattr(pass_data, 'strait_name', 'Unknown'),
            pass_number=pass_number,
            info_message=f"Loaded {n_obs:,} observations, {n_cyc} cycles, LON [{lon_min:.2f}, {lon_max:.2f}]"
        )
        
    except ImportError as e:
        return DataLoaderResult(
            success=False,
            error_message=f"Service not available: {e}"
        )
    except Exception as e:
        return DataLoaderResult(
            success=False,
            error_message=str(e)
        )
