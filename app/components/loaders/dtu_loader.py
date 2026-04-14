"""
DTU Data Loader - Handles loading DTU Space gridded data.
"""

from pathlib import Path
from typing import Optional
from .base import DataLoaderResult, get_gate_shapefile, apply_longitude_filter


def load_dtu_data(
    file_path: str,
    gate_id: str,
    start_date: str = "2006-01",
    end_date: str = "2017-12",
    lon_filter_min: Optional[float] = None,
    lon_filter_max: Optional[float] = None,
    use_cache: bool = True,
) -> DataLoaderResult:
    """
    Load DTU Space gridded DOT data for a gate.
    
    Args:
        file_path: Path to DTU NetCDF file
        gate_id: Gate identifier
        start_date: Start date (YYYY-MM)
        end_date: End date (YYYY-MM)
        lon_filter_min: Minimum longitude filter (for East sections)
        lon_filter_max: Maximum longitude filter (for West sections)
        use_cache: Whether to use data cache
        
    Returns:
        DataLoaderResult with success status and data/error
    """
    # Validate path
    if not Path(file_path).exists():
        return DataLoaderResult(
            success=False,
            error_message=f"DTU file not found: {file_path}"
        )
    
    # Get gate shapefile (use parent for divided gates)
    gate_path = get_gate_shapefile(gate_id, use_parent=True)
    if not gate_path:
        return DataLoaderResult(
            success=False,
            error_message="Gate shapefile not found"
        )
    
    try:
        from src.services.dtu_service import DTUService, DTUConfig
        import pandas as pd
        
        dtu_config = DTUConfig(
            file_path=file_path,
            time_start=start_date,
            time_end=end_date,
        )
        
        service = DTUService(dtu_config)
        
        # Load gate data
        pass_data = service.load_gate_data(
            gate_path=gate_path,
            use_cache=use_cache,
        )
        
        if pass_data is None:
            return DataLoaderResult(
                success=False,
                error_message="No DTU data for this gate"
            )
        
        # Apply longitude filter if needed
        if lon_filter_min is not None or lon_filter_max is not None:
            pass_data = apply_longitude_filter(
                pass_data, lon_filter_min, lon_filter_max, gate_id
            )
            if pass_data is None:
                return DataLoaderResult(
                    success=False,
                    error_message="No data after longitude filter!"
                )
        
        # Extract metadata
        df = getattr(pass_data, 'df', None)
        n_obs = len(df) if df is not None else 0
        time_array = getattr(pass_data, 'time_array', None)
        n_periods = len(time_array) if time_array is not None else 0
        
        gate_lon = getattr(pass_data, 'gate_lon_pts', None)
        lon_min = gate_lon.min() if gate_lon is not None else 0
        lon_max = gate_lon.max() if gate_lon is not None else 0
        
        return DataLoaderResult(
            success=True,
            data=pass_data,
            n_observations=n_obs,
            n_cycles=n_periods,
            lon_range=(lon_min, lon_max),
            strait_name=getattr(pass_data, 'strait_name', 'Unknown'),
            pass_number=0,  # DTU doesn't have pass numbers
            info_message=f"Loaded {n_periods} time periods, {len(gate_lon) if gate_lon is not None else 0} gate points"
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
