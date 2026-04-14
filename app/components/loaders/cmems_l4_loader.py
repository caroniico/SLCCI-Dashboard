"""
CMEMS L4 Data Loader - Handles loading Copernicus Marine L4 gridded data.
"""

from pathlib import Path
from typing import Optional
from .base import DataLoaderResult, get_gate_shapefile, apply_longitude_filter


def load_cmems_l4_data(
    gate_id: str,
    product_id: str = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
    start_date: str = "2010-01-01",
    end_date: str = "2020-12-31",
    lon_filter_min: Optional[float] = None,
    lon_filter_max: Optional[float] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_cache: bool = True,
) -> DataLoaderResult:
    """
    Load CMEMS L4 gridded data for a gate.
    
    Args:
        gate_id: Gate identifier
        product_id: CMEMS product identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        lon_filter_min: Minimum longitude filter (for East sections)
        lon_filter_max: Maximum longitude filter (for West sections)
        username: CMEMS username
        password: CMEMS password
        use_cache: Whether to use data cache
        
    Returns:
        DataLoaderResult with success status and data/error
    """
    # Get gate shapefile (use parent for divided gates)
    gate_path = get_gate_shapefile(gate_id, use_parent=True)
    if not gate_path:
        return DataLoaderResult(
            success=False,
            error_message="Gate shapefile not found"
        )
    
    try:
        from src.services.cmems_l4_service import CMEML4Service, CMEML4Config
        
        cmems_config = CMEML4Config(
            product_id=product_id,
            start_date=start_date,
            end_date=end_date,
            username=username,
            password=password,
        )
        
        service = CMEML4Service(cmems_config)
        
        # Load gate data
        pass_data = service.load_gate_data(
            gate_path=gate_path,
            use_cache=use_cache,
        )
        
        if pass_data is None:
            return DataLoaderResult(
                success=False,
                error_message="No CMEMS L4 data for this gate"
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
            pass_number=0,  # L4 gridded doesn't have pass numbers
            info_message=f"Loaded {n_periods} time periods, LON [{lon_min:.2f}, {lon_max:.2f}]"
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
