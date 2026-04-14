"""
SLCCI Service
=============
Service for loading and processing ESA Sea Level CCI (SLCCI) data.

This service wraps the functions from legacy/j2_utils.py into a clean service layer
following the NICO Unified Architecture pattern.

Data Flow:
    UI → SLCCIService → load_filtered_cycles_serial_J2 → NetCDF files (local)
                      → OR fetch via CEDAClient → CEDA API (remote)
                      → interpolate_geoid → TUM_ogmoc.nc
                      → DOT calculation → DataFrame
                      
Supported Sources:
    - "local": Load from local NetCDF files (default)
    - "api": Load via CEDA OPeNDAP API
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Literal
from dataclasses import dataclass, field
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from src.core.logging_config import get_logger, log_call

logger = get_logger(__name__)

# Data source type
DataSource = Literal["local", "api"]


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SLCCIConfig:
    """Configuration for SLCCI data loading."""
    # Local source settings
    base_dir: str = "/Users/nicolocaron/Desktop/ARCFRESH/J2"
    geoid_path: str = "/Users/nicolocaron/Desktop/ARCFRESH/TUM_ogmoc.nc"
    
    # Cycles
    cycles: List[int] = field(default_factory=lambda: list(range(1, 282)))
    
    # Processing settings
    use_flag: bool = True
    lat_buffer_deg: float = 2.0
    lon_buffer_deg: float = 5.0
    lon_bin_size: float = 0.1  # Longitude binning resolution (degrees) - UNIFIED for all outputs
    
    # Data source: "local" or "api"
    source: DataSource = "local"
    
    # API settings
    satellite: str = "J2"  # J1 or J2


@dataclass
class PassData:
    """Container for pass data analysis results."""
    pass_number: int
    strait_name: str
    satellite: str
    df: pd.DataFrame  # Main data
    gate_lon_pts: np.ndarray
    gate_lat_pts: np.ndarray
    x_km: np.ndarray
    time_periods: List
    slope_series: np.ndarray
    profile_mean: np.ndarray
    dot_matrix: np.ndarray  # (n_gate_pts, ntime)
    time_array: np.ndarray
    # Monthly climatology profiles (all years combined by month)
    monthly_profiles: Optional[Dict[int, np.ndarray]] = None  # {1: Jan profile, 2: Feb profile, ...}
    monthly_lon_centers: Optional[np.ndarray] = None
    monthly_x_km: Optional[np.ndarray] = None


# ==============================================================================
# INTELLIGENT CACHE
# ==============================================================================

@dataclass
class CacheConfig:
    """Configuration for intelligent caching."""
    enabled: bool = True
    ttl_days: int = 14  # Time-to-live in days
    max_entries: int = 50  # Max cached entries


class SLCCICache:
    """
    Intelligent in-memory cache for SLCCI data.
    
    Two-level cache:
    - Level 1 (raw_data): Raw DataFrame after loading from files (before binning)
    - Level 2 (processed): Full PassData objects (after processing with specific bin_size)
    
    Cache keys are based on:
    - gate_path (hash)
    - pass_number
    - cycles range
    - bin_size (for processed cache only)
    
    Invalidation rules:
    - bin_size change → invalidate processed, keep raw
    - cycles change → invalidate all
    - gate change → invalidate all
    - TTL expired → invalidate entry
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Level 1: Raw data cache (before binning)
        # Key: (gate_hash, pass_number, cycles_hash) → (DataFrame, timestamp)
        self._raw_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        
        # Level 2: Processed data cache (after binning)
        # Key: (gate_hash, pass_number, cycles_hash, bin_size) → (PassData, timestamp)
        self._processed_cache: Dict[str, Tuple[Any, float]] = {}
        
        # Metadata for logging
        self._stats = {"hits": 0, "misses": 0, "invalidations": 0}
    
    @staticmethod
    def _hash_gate(gate_path: str) -> str:
        """Create short hash from gate path."""
        import hashlib
        return hashlib.md5(gate_path.encode()).hexdigest()[:12]
    
    @staticmethod
    def _hash_cycles(cycles: List[int]) -> str:
        """Create hash from cycles list."""
        import hashlib
        cycles_str = f"{min(cycles)}-{max(cycles)}-{len(cycles)}"
        return hashlib.md5(cycles_str.encode()).hexdigest()[:8]
    
    def _make_raw_key(self, gate_path: str, pass_number: int, cycles: List[int]) -> str:
        """Generate cache key for raw data."""
        return f"raw_{self._hash_gate(gate_path)}_{pass_number}_{self._hash_cycles(cycles)}"
    
    def _make_processed_key(self, gate_path: str, pass_number: int, 
                            cycles: List[int], bin_size: float) -> str:
        """Generate cache key for processed data."""
        return f"proc_{self._hash_gate(gate_path)}_{pass_number}_{self._hash_cycles(cycles)}_{bin_size:.3f}"
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        import time
        age_days = (time.time() - timestamp) / (24 * 3600)
        return age_days > self.config.ttl_days
    
    def _enforce_max_entries(self, cache: dict):
        """Remove oldest entries if cache exceeds max size."""
        if len(cache) > self.config.max_entries:
            # Sort by timestamp and remove oldest
            sorted_keys = sorted(cache.keys(), key=lambda k: cache[k][1])
            for key in sorted_keys[:len(cache) - self.config.max_entries]:
                del cache[key]
                logger.debug(f"Cache evicted: {key}")
    
    # -------------------------------------------------------------------------
    # Raw Data Cache (Level 1)
    # -------------------------------------------------------------------------
    
    def get_raw(self, gate_path: str, pass_number: int, 
                cycles: List[int]) -> Optional[pd.DataFrame]:
        """Get raw DataFrame from cache."""
        if not self.config.enabled:
            return None
        
        key = self._make_raw_key(gate_path, pass_number, cycles)
        
        if key in self._raw_cache:
            df, timestamp = self._raw_cache[key]
            if not self._is_expired(timestamp):
                self._stats["hits"] += 1
                logger.debug(f"Cache HIT (raw): {key}")
                return df.copy()  # Return copy to prevent mutation
            else:
                # Expired - remove
                del self._raw_cache[key]
                logger.debug(f"Cache EXPIRED (raw): {key}")
        
        self._stats["misses"] += 1
        return None
    
    def set_raw(self, gate_path: str, pass_number: int, 
                cycles: List[int], df: pd.DataFrame):
        """Store raw DataFrame in cache."""
        if not self.config.enabled:
            return
        
        import time
        key = self._make_raw_key(gate_path, pass_number, cycles)
        self._raw_cache[key] = (df.copy(), time.time())
        self._enforce_max_entries(self._raw_cache)
        logger.debug(f"Cache SET (raw): {key}, {len(df)} rows")
    
    # -------------------------------------------------------------------------
    # Processed Data Cache (Level 2)
    # -------------------------------------------------------------------------
    
    def get_processed(self, gate_path: str, pass_number: int, 
                      cycles: List[int], bin_size: float) -> Optional[Any]:
        """Get processed PassData from cache."""
        if not self.config.enabled:
            return None
        
        key = self._make_processed_key(gate_path, pass_number, cycles, bin_size)
        
        if key in self._processed_cache:
            data, timestamp = self._processed_cache[key]
            if not self._is_expired(timestamp):
                self._stats["hits"] += 1
                logger.debug(f"Cache HIT (processed): {key}")
                return data
            else:
                del self._processed_cache[key]
                logger.debug(f"Cache EXPIRED (processed): {key}")
        
        self._stats["misses"] += 1
        return None
    
    def set_processed(self, gate_path: str, pass_number: int, 
                      cycles: List[int], bin_size: float, data: Any):
        """Store processed PassData in cache."""
        if not self.config.enabled:
            return
        
        import time
        key = self._make_processed_key(gate_path, pass_number, cycles, bin_size)
        self._processed_cache[key] = (data, time.time())
        self._enforce_max_entries(self._processed_cache)
        logger.debug(f"Cache SET (processed): {key}")
    
    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------
    
    def invalidate_for_bin_size(self, gate_path: str, pass_number: int, cycles: List[int]):
        """Invalidate processed cache when bin_size changes (keep raw)."""
        prefix = f"proc_{self._hash_gate(gate_path)}_{pass_number}_{self._hash_cycles(cycles)}"
        keys_to_remove = [k for k in self._processed_cache.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._processed_cache[key]
            self._stats["invalidations"] += 1
        if keys_to_remove:
            logger.info(f"Cache invalidated {len(keys_to_remove)} processed entries for bin_size change")
    
    def invalidate_all(self, gate_path: Optional[str] = None):
        """Invalidate all cache entries (or just for specific gate)."""
        if gate_path:
            gate_hash = self._hash_gate(gate_path)
            raw_keys = [k for k in self._raw_cache.keys() if gate_hash in k]
            proc_keys = [k for k in self._processed_cache.keys() if gate_hash in k]
            for k in raw_keys:
                del self._raw_cache[k]
            for k in proc_keys:
                del self._processed_cache[k]
            count = len(raw_keys) + len(proc_keys)
        else:
            count = len(self._raw_cache) + len(self._processed_cache)
            self._raw_cache.clear()
            self._processed_cache.clear()
        
        self._stats["invalidations"] += count
        logger.info(f"Cache cleared: {count} entries removed")
    
    def clear(self):
        """Clear all caches."""
        self.invalidate_all()
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            **self._stats,
            "raw_entries": len(self._raw_cache),
            "processed_entries": len(self._processed_cache),
            "total_entries": len(self._raw_cache) + len(self._processed_cache),
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"SLCCICache(raw={stats['raw_entries']}, processed={stats['processed_entries']}, "
                f"hits={stats['hits']}, misses={stats['misses']})")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _load_gate_gdf(gate_path: str) -> gpd.GeoDataFrame:
    """
    Load gate shapefile and ensure it's in EPSG:4326.
    
    Handles missing CRS by assuming EPSG:3413 (Polar Stereographic).
    """
    import os
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'  # Fix missing .shx files
    
    gate_gdf = gpd.read_file(gate_path)
    
    # Handle missing CRS - assume EPSG:3413 (polar stereographic) if None
    if gate_gdf.crs is None:
        logger.info("Gate shapefile missing CRS, assuming EPSG:3413 (Polar Stereographic)")
        gate_gdf = gate_gdf.set_crs("EPSG:3413")
    
    return gate_gdf.to_crs("EPSG:4326")


# ==============================================================================
# SLCCI SERVICE CLASS
# ==============================================================================

class SLCCIService:
    """
    Service for loading and processing ESA Sea Level CCI data.
    
    Supports two data sources:
    - "local": Load from local NetCDF files
    - "api": Load via CEDA OPeNDAP API
    
    Features intelligent caching:
    - Raw data cache: stores DataFrame after loading (before binning)
    - Processed cache: stores PassData (after processing with specific bin_size)
    - Automatic invalidation when parameters change
    
    Example usage:
        # Local files
        service = SLCCIService()
        pass_data = service.load_pass_data(gate_path="/path/to/gate.shp", pass_number=248)
        
        # API (CEDA)
        config = SLCCIConfig(source="api", satellite="J2")
        service = SLCCIService(config)
        pass_data = service.load_pass_data(gate_path="/path/to/gate.shp", pass_number=248)
        
        # Check cache stats
        print(service.cache)
        
        # Clear cache
        service.clear_cache()
    """
    
    def __init__(self, config: Optional[SLCCIConfig] = None, 
                 cache_config: Optional[CacheConfig] = None):
        """Initialize SLCCI service with configuration."""
        self.config = config or SLCCIConfig()
        self._geoid_interp: Optional[RegularGridInterpolator] = None
        self._gate_points_cache: Dict = {}
        self._ceda_client = None  # Lazy-loaded
        
        # Initialize intelligent cache
        self.cache = SLCCICache(cache_config)
        
        # Validate paths exist
        if self.config.source == "local":
            if not os.path.exists(self.config.base_dir):
                logger.warning(f"SLCCI base_dir not found: {self.config.base_dir}")
        if not os.path.exists(self.config.geoid_path):
            logger.warning(f"Geoid file not found: {self.config.geoid_path}")
    
    def clear_cache(self, gate_path: Optional[str] = None):
        """Clear cache (all or for specific gate)."""
        self.cache.clear() if gate_path is None else self.cache.invalidate_all(gate_path)
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    @property
    def ceda_client(self):
        """Lazy-load CEDA client."""
        if self._ceda_client is None:
            from src.services.ceda_client import CEDAClient
            self._ceda_client = CEDAClient()
        return self._ceda_client
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    @log_call(logger)
    def load_pass_data(
        self,
        gate_path: str,
        pass_number: int,
        cycles: Optional[List[int]] = None,
        force_reload: bool = False,
    ) -> Optional[PassData]:
        """
        Load satellite data for a specific pass and compute DOT analysis.
        
        Uses two-level caching:
        - Level 1 (raw): Caches raw DataFrame before binning
        - Level 2 (processed): Caches full PassData with current bin_size
        
        Parameters
        ----------
        gate_path : str
            Path to gate shapefile
        pass_number : int
            Pass number to load
        cycles : List[int], optional
            Cycles to load (defaults to config.cycles)
        force_reload : bool
            If True, bypass cache and reload from source
            
        Returns
        -------
        PassData
            Container with all analysis results, or None if no data
        """
        cycles = cycles or self.config.cycles
        bin_size = self.config.lon_bin_size
        
        logger.info(f"Loading pass {pass_number} for gate: {Path(gate_path).name} (bin_size={bin_size}°)")
        
        # --- CHECK PROCESSED CACHE (Level 2) ---
        if not force_reload:
            cached_pass_data = self.cache.get_processed(gate_path, pass_number, cycles, bin_size)
            if cached_pass_data is not None:
                logger.info(f"✅ Cache HIT (processed) for pass {pass_number}")
                return cached_pass_data
        
        # --- CHECK RAW CACHE (Level 1) ---
        cached_df = None if force_reload else self.cache.get_raw(gate_path, pass_number, cycles)
        
        if cached_df is not None:
            logger.info(f"✅ Cache HIT (raw) for pass {pass_number}, processing with bin_size={bin_size}°")
            df = cached_df
            ds = None  # No need to reload
        else:
            # --- LOAD FROM SOURCE ---
            logger.info(f"📥 Cache MISS, loading from source...")
            
            # 1. Load gate geometry
            gate_gdf = _load_gate_gdf(gate_path)
            
            # 2. Load satellite data
            ds = self._load_filtered_cycles(
                cycles=cycles,
                gate_path=gate_path,
                pass_number=pass_number,
            )
            
            if ds is None or ds.sizes.get("time", 0) == 0:
                logger.warning(f"No data found for pass {pass_number}")
                return None
            
            # 3. Interpolate and add geoid
            geoid_values = self._interpolate_geoid(
                ds["latitude"].values,
                ds["longitude"].values,
            )
            
            # 4. Build DataFrame with DOT
            df = self._build_pass_dataframe(ds, geoid_values, pass_number)
            
            if df is None or len(df) == 0:
                logger.warning(f"Empty DataFrame for pass {pass_number}")
                return None
            
            # --- STORE IN RAW CACHE ---
            self.cache.set_raw(gate_path, pass_number, cycles, df)
            logger.info(f"💾 Cached raw DataFrame ({len(df)} rows)")
        
        # --- PROCESS WITH CURRENT BIN SIZE ---
        # (df is already loaded - either from cache or from source above)
        
        # Load gate geometry
        gate_gdf = _load_gate_gdf(gate_path)
        strait_name = self._extract_strait_name(gate_path)
        
        # 5. Build gate profile points (for reference)
        gate_lon_pts, gate_lat_pts, _ = self._get_gate_profile_points(gate_gdf)
        
        # 6. Build DOT matrix using LONGITUDE BINNING (for slope time series)
        dot_matrix, time_periods, lon_centers, x_km = self._build_dot_matrix(
            df, gate_lon_pts, gate_lat_pts, 
            lon_bin_size=bin_size
        )
        
        # Use lon_centers for profiles instead of gate points
        gate_lon_pts = lon_centers
        gate_lat_pts = np.full_like(lon_centers, df["lat"].mean())  # Approx lat
        
        # 7. Compute slope series using x_km from longitude bins
        slope_series = self._compute_slope_series(dot_matrix, x_km)
        
        # 8. Compute profile mean using POOLED method (all observations, not mean-of-means)
        # This gives equal weight to each observation, not each time period
        profile_mean, _, _ = self._build_mean_profile_pooled(
            df, lon_bin_size=bin_size
        )
        
        # 9. Build monthly climatology profiles (same bin size for consistency)
        monthly_profiles, monthly_lon_centers, monthly_x_km = self._build_monthly_climatology_profiles(
            df, lon_bin_size=bin_size
        )
        
        time_array = np.array([pd.Timestamp(str(p)) for p in time_periods])
        
        # Get satellite from ds if available, otherwise default
        satellite = ds.attrs.get("satellite_type", "J2") if ds is not None else self.config.satellite
        
        logger.info(f"Loaded {len(df)} observations for pass {pass_number}")
        
        pass_data = PassData(
            pass_number=pass_number,
            strait_name=strait_name,
            satellite=satellite,
            df=df,
            gate_lon_pts=gate_lon_pts,
            gate_lat_pts=gate_lat_pts,
            x_km=x_km,
            time_periods=time_periods,
            slope_series=slope_series,
            profile_mean=profile_mean,
            dot_matrix=dot_matrix,
            time_array=time_array,
            monthly_profiles=monthly_profiles,
            monthly_lon_centers=monthly_lon_centers,
            monthly_x_km=monthly_x_km,
        )
        
        # --- STORE IN PROCESSED CACHE ---
        self.cache.set_processed(gate_path, pass_number, cycles, bin_size, pass_data)
        logger.info(f"💾 Cached processed PassData (bin_size={bin_size}°)")
        
        return pass_data
    
    @log_call(logger)
    def find_closest_pass(
        self,
        gate_path: str,
        cycles: Optional[List[int]] = None,
        n_passes: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Find the N closest satellite passes to a gate.
        
        Uses distance from pass points to the GATE LINE (not centroid).
        This gives better results when passes cross or approach the gate.
        
        Parameters
        ----------
        gate_path : str
            Path to gate shapefile
        cycles : List[int], optional
            Cycles to search (defaults to subset for speed)
        n_passes : int
            Number of closest passes to return
            
        Returns
        -------
        List[Tuple[int, float]]
            List of (pass_number, distance_km) sorted by distance
        """
        from shapely.geometry import Point
        
        cycles = cycles or list(range(1, 100))  # Use subset for speed
        
        gate_gdf = _load_gate_gdf(gate_path)
        gate_line = gate_gdf.geometry.unary_union  # The gate as a line/multiline
        
        # Get gate centroid for bounds calculation
        gate_centroid = gate_line.centroid
        gate_lon, gate_lat = gate_centroid.x, gate_centroid.y
        
        logger.info(f"Finding closest passes to gate line (centroid: {gate_lat:.4f}, {gate_lon:.4f})")
        
        # Load data with expanded bounds
        ds = self._load_filtered_cycles(
            cycles=cycles,
            gate_path=gate_path,
            pass_number=None,  # Load all passes
            lat_buffer_deg=10.0,
            lon_buffer_deg=15.0,
        )
        
        if ds is None or ds.sizes.get("time", 0) == 0:
            logger.warning("No data found for closest pass search")
            return [(1, float('inf'))]
        
        # Get unique passes
        if "pass" not in ds:
            logger.warning("No 'pass' variable in dataset")
            return [(1, float('inf'))]
        
        all_passes = set(int(p) for p in np.unique(ds["pass"].values) if not np.isnan(p))
        
        # Calculate min distance for each pass to the gate LINE
        pass_distances = {}
        R_earth = 6371.0  # km
        
        for pass_num in all_passes:
            mask = ds["pass"].values == pass_num
            lons = ds["longitude"].values[mask]
            lats = ds["latitude"].values[mask]
            
            if len(lons) == 0:
                continue
            
            # Find minimum distance from any pass point to the gate line
            min_dist_deg = float('inf')
            for lon, lat in zip(lons, lats):
                pt = Point(lon, lat)
                dist_deg = gate_line.distance(pt)  # Distance in degrees
                if dist_deg < min_dist_deg:
                    min_dist_deg = dist_deg
            
            # Convert degrees to km (approximate at gate latitude)
            dist_km = min_dist_deg * 111.0 * np.cos(np.radians(gate_lat))
            pass_distances[pass_num] = dist_km
        
        # Sort by distance
        sorted_passes = sorted(pass_distances.items(), key=lambda x: x[1])
        
        logger.info(f"Found {len(sorted_passes)} passes, closest: {sorted_passes[:3] if sorted_passes else 'none'}")
        
        return sorted_passes[:n_passes]
    
    def get_available_passes_for_gate(self, gate_path: str) -> List[int]:
        """
        Get list of available pass numbers for a gate.
        
        Extracts from filename first, then searches data.
        """
        # Check filename first
        _, pass_from_filename = self._extract_strait_info(gate_path)
        
        if pass_from_filename:
            return [pass_from_filename]
        
        # Search in data
        closest = self.find_closest_pass(gate_path, n_passes=10)
        return [p[0] for p in closest]
    
    # ==========================================================================
    # PRIVATE METHODS - DATA LOADING
    # ==========================================================================
    
    def _load_filtered_cycles(
        self,
        cycles: List[int],
        gate_path: str,
        pass_number: Optional[int] = None,
        lat_buffer_deg: Optional[float] = None,
        lon_buffer_deg: Optional[float] = None,
    ) -> Optional[xr.Dataset]:
        """
        Load and filter satellite altimetry cycles for a specific region.
        
        Routes to local file loading or API based on config.source.
        """
        if self.config.source == "api":
            return self._load_filtered_cycles_api(
                cycles, gate_path, pass_number, lat_buffer_deg, lon_buffer_deg
            )
        else:
            return self._load_filtered_cycles_local(
                cycles, gate_path, pass_number, lat_buffer_deg, lon_buffer_deg
            )
    
    def _load_filtered_cycles_api(
        self,
        cycles: List[int],
        gate_path: str,
        pass_number: Optional[int] = None,
        lat_buffer_deg: Optional[float] = None,
        lon_buffer_deg: Optional[float] = None,
    ) -> Optional[xr.Dataset]:
        """
        Load cycles from CEDA API with spatial filtering.
        """
        lat_buffer = lat_buffer_deg or self.config.lat_buffer_deg
        lon_buffer = lon_buffer_deg or self.config.lon_buffer_deg
        
        # Load gate bounds
        gate = _load_gate_gdf(gate_path)
        lon_min_g, lat_min_g, lon_max_g, lat_max_g = gate.total_bounds
        
        # Build bbox for API (lon_min, lat_min, lon_max, lat_max)
        bbox = (
            lon_min_g - lon_buffer,
            lat_min_g - lat_buffer,
            lon_max_g + lon_buffer,
            lat_max_g + lat_buffer,
        )
        
        logger.info(f"Loading from CEDA API, cycles {min(cycles)}-{max(cycles)}, bbox={bbox}")
        
        # Fetch via API
        ds = self.ceda_client.fetch_cycles(
            satellite=self.config.satellite,
            cycles=cycles,
            bbox=bbox,
            use_cache=True,
        )
        
        if ds is None or ds.sizes.get("time", 0) == 0:
            logger.warning("No data returned from CEDA API")
            return None
        
        # Post-process to match local format
        ds = self._postprocess_api_data(ds, pass_number)
        
        return ds
    
    def _postprocess_api_data(
        self,
        ds: xr.Dataset,
        pass_number: Optional[int] = None,
    ) -> xr.Dataset:
        """Post-process API data to match local file format."""
        # Wrap longitude
        if "longitude" in ds:
            lon = ds["longitude"].values
            lon_wrapped = self._wrap_longitude(lon)
            ds = ds.assign_coords(longitude=(("time",), lon_wrapped))
        
        # Decode time if needed (days since 1950-01-01)
        if "time" in ds and not np.issubdtype(ds["time"].dtype, np.datetime64):
            try:
                time_vals = pd.to_datetime(ds["time"].values, origin="1950-01-01", unit="D")
                ds = ds.assign_coords(time=time_vals)
            except Exception as e:
                logger.warning(f"Could not decode time: {e}")
        
        # Quality filtering
        if self.config.use_flag and "flag" in ds:
            valid_mask = ds["flag"] == 0
            ds = ds.where(valid_mask, drop=True)
        
        # Standardize pass variable
        for var in ["pass", "track", "pass_number", "track_number"]:
            if var in ds.variables and var != "pass":
                ds = ds.rename({var: "pass"})
                break
        
        # Filter by pass number
        if pass_number is not None and "pass" in ds:
            pass_vals = np.round(ds["pass"].values).astype(int)
            mask_pass = pass_vals == int(pass_number)
            if mask_pass.sum() > 0:
                ds = ds.isel(time=mask_pass)
            else:
                logger.warning(f"No data for pass {pass_number} in API response")
        
        ds.attrs["satellite_type"] = self.config.satellite
        
        return ds
    
    def _load_filtered_cycles_local(
        self,
        cycles: List[int],
        gate_path: str,
        pass_number: Optional[int] = None,
        lat_buffer_deg: Optional[float] = None,
        lon_buffer_deg: Optional[float] = None,
    ) -> Optional[xr.Dataset]:
        """
        Load and filter satellite altimetry cycles from local files.
        
        Adapted from legacy/j2_utils.py::load_filtered_cycles_serial_J2
        """
        lat_buffer = lat_buffer_deg or self.config.lat_buffer_deg
        lon_buffer = lon_buffer_deg or self.config.lon_buffer_deg
        
        # Load gate bounds
        gate = _load_gate_gdf(gate_path)
        lon_min_g, lat_min_g, lon_max_g, lat_max_g = gate.total_bounds
        
        lat_min = lat_min_g - lat_buffer
        lat_max = lat_max_g + lat_buffer
        lon_min = self._wrap_longitude(lon_min_g - lon_buffer)
        lon_max = self._wrap_longitude(lon_max_g + lon_buffer)
        
        satellite_type = self._detect_satellite_type()
        
        cycle_datasets = []
        
        for cycle in cycles:
            cycle_str = str(cycle).zfill(3)
            filename = f"SLCCI_ALTDB_{satellite_type}_Cycle{cycle_str}_V2.nc"
            filepath = os.path.join(self.config.base_dir, filename)
            
            if not os.path.exists(filepath):
                continue
            
            try:
                with xr.open_dataset(filepath, decode_times=False) as ds:
                    # Spatial filtering
                    lon = ds["longitude"].values
                    lat = ds["latitude"].values
                    lon_wrapped = self._wrap_longitude(lon)
                    
                    mask_spatial = (
                        (lat >= lat_min) & (lat <= lat_max) &
                        self._lon_in_bounds(lon_wrapped, lon_min, lon_max)
                    )
                    
                    if mask_spatial.sum() == 0:
                        continue
                    
                    ds_filtered = ds.isel(time=mask_spatial)
                    
                    # Update longitude to wrapped values
                    ds_filtered = ds_filtered.assign_coords(
                        longitude=(("time",), lon_wrapped[mask_spatial])
                    )
                    
                    # Decode time
                    time_vals = pd.to_datetime(
                        ds_filtered["time"].values, origin="1950-01-01", unit="D"
                    )
                    ds_filtered = ds_filtered.assign_coords(time=time_vals)
                    
                    # Quality filtering
                    if self.config.use_flag and "validation_flag" in ds_filtered:
                        valid_mask = ds_filtered["validation_flag"] == 0
                        ds_filtered = ds_filtered.isel(time=valid_mask)
                    
                    # Standardize pass variable name
                    for var in ["pass", "track", "pass_number", "track_number"]:
                        if var in ds_filtered.variables:
                            if var != "pass":
                                ds_filtered = ds_filtered.rename({var: "pass"})
                            break
                    
                    # Filter by pass number if specified
                    if pass_number is not None and "pass" in ds_filtered:
                        pass_vals = np.round(ds_filtered["pass"].values).astype(int)
                        mask_pass = pass_vals == int(pass_number)
                        if mask_pass.sum() == 0:
                            continue
                        ds_filtered = ds_filtered.isel(time=mask_pass)
                    
                    if ds_filtered.sizes.get("time", 0) == 0:
                        continue
                    
                    # Add cycle coordinate
                    ds_filtered = ds_filtered.assign_coords(
                        cycle=("time", np.full(ds_filtered.sizes["time"], cycle))
                    )
                    
                    cycle_datasets.append(ds_filtered)
                    
            except Exception as e:
                logger.debug(f"Error loading cycle {cycle}: {e}")
                continue
        
        if not cycle_datasets:
            return None
        
        combined = xr.concat(cycle_datasets, dim="time")
        combined.attrs["satellite_type"] = satellite_type
        
        return combined
    
    # ==========================================================================
    # PRIVATE METHODS - GEOID INTERPOLATION
    # ==========================================================================
    
    def _get_geoid_interpolator(self) -> RegularGridInterpolator:
        """Get or create the geoid interpolator (cached)."""
        if self._geoid_interp is not None:
            return self._geoid_interp
        
        logger.info("Building geoid interpolator...")
        
        ds_geoid = xr.open_dataset(self.config.geoid_path)
        lat_geoid = ds_geoid["lat"].values
        lon_geoid = ds_geoid["lon"].values
        geoid_values = ds_geoid["value"].values
        
        # Wrap and sort longitudes
        lon_wrapped = self._wrap_longitude(lon_geoid)
        sort_idx = np.argsort(lon_wrapped)
        lon_sorted = lon_wrapped[sort_idx]
        
        # Remove duplicates
        unique_idx = np.concatenate(([True], np.diff(lon_sorted) != 0))
        lon_sorted = lon_sorted[unique_idx]
        geoid_sorted = geoid_values[:, sort_idx][:, unique_idx]
        
        self._geoid_interp = RegularGridInterpolator(
            (lat_geoid, lon_sorted),
            geoid_sorted,
            method="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )
        
        return self._geoid_interp
    
    def _interpolate_geoid(
        self,
        target_lats: np.ndarray,
        target_lons: np.ndarray,
    ) -> np.ndarray:
        """Interpolate geoid values at target positions."""
        interp = self._get_geoid_interpolator()
        
        target_lons_wrapped = self._wrap_longitude(target_lons)
        points = np.column_stack([target_lats, target_lons_wrapped])
        
        return interp(points)
    
    # ==========================================================================
    # PRIVATE METHODS - DOT COMPUTATION
    # ==========================================================================
    
    def _build_pass_dataframe(
        self,
        ds: xr.Dataset,
        geoid_values: np.ndarray,
        pass_number: int,
    ) -> Optional[pd.DataFrame]:
        """Build DataFrame with DOT computed from corssh - geoid."""
        if "corssh" not in ds.data_vars:
            logger.warning("No 'corssh' variable in dataset")
            return None
        
        dot = ds["corssh"].values - geoid_values
        
        df = pd.DataFrame({
            "cycle": ds["cycle"].values,
            "pass": pass_number,
            "lat": ds["latitude"].values,
            "lon": ds["longitude"].values,
            "corssh": ds["corssh"].values,
            "geoid": geoid_values,
            "dot": dot,
            "time": pd.to_datetime(ds["time"].values),
        })
        
        df["month"] = df["time"].dt.month
        df["year"] = df["time"].dt.year
        df["year_month"] = df["time"].dt.to_period("M")
        
        return df
    
    def _build_dot_matrix(
        self,
        df: pd.DataFrame,
        gate_lon_pts: np.ndarray,
        gate_lat_pts: np.ndarray,
        lon_bin_size: float = 0.01,
    ) -> Tuple[np.ndarray, List, np.ndarray, np.ndarray]:
        """
        Build DOT matrix using LONGITUDE BINNING (like SLCCI PLOTTER notebook).
        
        This bins the satellite data by longitude (not by distance to gate points),
        which is the correct method for computing cross-gate DOT profiles.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'lon', 'lat', 'dot', 'year_month' columns
        gate_lon_pts : np.ndarray
            Gate longitude points (used for reference, not binning)
        gate_lat_pts : np.ndarray
            Gate latitude points
        lon_bin_size : float
            Longitude bin size in degrees (default 0.01°)
            
        Returns
        -------
        dot_matrix : np.ndarray
            Shape (n_lon_bins, n_time_periods) - DOT values binned by longitude
        time_periods : List
            List of year-month periods
        lon_centers : np.ndarray
            Center longitude of each bin
        x_km : np.ndarray
            Distance in km from first bin (for slope calculation)
        """
        time_periods = sorted(df["year_month"].unique())
        n_time = len(time_periods)
        
        # Determine longitude range from DATA (not gate)
        lon_min = df["lon"].min()
        lon_max = df["lon"].max()
        
        # Handle dateline crossing: if lon range suggests crossing, unwrap
        if lon_max - lon_min > 180:
            # Data crosses dateline - unwrap by shifting negative lons
            logger.warning(f"Dateline crossing detected: lon range [{lon_min:.2f}, {lon_max:.2f}]")
            df = df.copy()
            df.loc[df["lon"] < 0, "lon"] += 360
            lon_min = df["lon"].min()
            lon_max = df["lon"].max()
        
        # Create longitude bins (GUARANTEED monotonic increasing)
        lon_bins = np.arange(lon_min, lon_max + lon_bin_size, lon_bin_size)
        lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
        n_lon_bins = len(lon_centers)
        
        # Verify monotonicity (defensive check)
        assert np.all(np.diff(lon_centers) > 0), "lon_centers must be monotonically increasing"
        
        logger.info(f"Longitude binning: {lon_min:.3f}° to {lon_max:.3f}°, "
                    f"{n_lon_bins} bins of {lon_bin_size}°")
        
        # Create DOT matrix: rows = longitude bins, columns = time periods
        dot_matrix = np.full((n_lon_bins, n_time), np.nan, dtype=float)
        
        # Fill matrix by binning data for each time period
        for it, period in enumerate(time_periods):
            month_data = df[df["year_month"] == period]
            if month_data.empty:
                continue
            
            # Bin by longitude
            month_data_copy = month_data.copy()
            month_data_copy["lon_bin"] = pd.cut(
                month_data_copy["lon"],
                bins=lon_bins,
                labels=False,
                include_lowest=True
            )
            
            # Average DOT in each longitude bin
            binned = month_data_copy.groupby("lon_bin")["dot"].mean()
            
            # Fill matrix
            for bin_idx in binned.index:
                if pd.notna(bin_idx) and int(bin_idx) < n_lon_bins:
                    dot_matrix[int(bin_idx), it] = binned[bin_idx]
        
        # Calculate distance in km from first bin (for slope calculation)
        # lon_centers is GUARANTEED monotonic increasing (see binning above)
        # so dlon is always >= 0, no abs() needed
        R_earth = 6371.0
        mean_lat = df["lat"].mean()
        lat_rad = np.deg2rad(mean_lat)
        lon_rad = np.deg2rad(lon_centers)
        dlon = lon_rad - lon_rad[0]  # Always >= 0 since lon_centers is ascending
        x_km = R_earth * dlon * np.cos(lat_rad)
        
        valid_count = np.sum(np.isfinite(dot_matrix))
        total_count = dot_matrix.size
        logger.info(f"DOT matrix: {valid_count}/{total_count} valid values "
                    f"({100*valid_count/total_count:.1f}%)")
        
        return dot_matrix, time_periods, lon_centers, x_km
    
    def _build_mean_profile_pooled(
        self,
        df: pd.DataFrame,
        lon_bin_size: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build mean DOT profile by pooling ALL observations across ALL times.
        
        This is the INTENDED methodology:
        1. Define fixed spatial bins along longitude
        2. Pool ALL observations (from all cycles/times) that fall in each bin
        3. Compute mean DOT per bin (single value, time-collapsed)
        
        This differs from the matrix approach which computes per-month means
        then averages those means (equal weight per month, not per observation).
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'lon', 'lat', 'dot' columns (all times pooled)
        lon_bin_size : float
            Longitude bin size in degrees (default 0.01°)
            
        Returns
        -------
        profile_mean : np.ndarray
            Mean DOT for each bin (pooled across all times)
        lon_centers : np.ndarray
            Center longitude of each bin
        x_km : np.ndarray
            Distance in km from first bin
        """
        # Determine longitude range from DATA
        lon_min = df["lon"].min()
        lon_max = df["lon"].max()
        
        # Handle dateline crossing: if lon range suggests crossing, unwrap
        if lon_max - lon_min > 180:
            # Data crosses dateline - unwrap by shifting negative lons
            logger.warning(f"[_build_mean_profile_pooled] Dateline crossing detected: "
                          f"lon range [{lon_min:.2f}, {lon_max:.2f}]")
            df = df.copy()
            df.loc[df["lon"] < 0, "lon"] += 360
            lon_min = df["lon"].min()
            lon_max = df["lon"].max()
        
        # Create fixed longitude bins (GUARANTEED monotonic increasing)
        lon_bins = np.arange(lon_min, lon_max + lon_bin_size, lon_bin_size)
        lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
        n_lon_bins = len(lon_centers)
        
        # Verify monotonicity (defensive check)
        assert np.all(np.diff(lon_centers) > 0), "lon_centers must be monotonically increasing"
        
        # Assign each observation to a bin
        df_copy = df.copy()
        df_copy["lon_bin"] = pd.cut(
            df_copy["lon"],
            bins=lon_bins,
            labels=False,
            include_lowest=True
        )
        
        # Pool ALL observations and compute mean per bin
        # This gives equal weight to each observation, not each time period
        binned_stats = df_copy.groupby("lon_bin")["dot"].agg(["mean", "count", "std"])
        
        # Build profile array
        profile_mean = np.full(n_lon_bins, np.nan, dtype=float)
        obs_count = np.zeros(n_lon_bins, dtype=int)
        
        for bin_idx in binned_stats.index:
            if pd.notna(bin_idx) and int(bin_idx) < n_lon_bins:
                profile_mean[int(bin_idx)] = binned_stats.loc[bin_idx, "mean"]
                obs_count[int(bin_idx)] = int(binned_stats.loc[bin_idx, "count"])
        
        # Calculate distance in km from first bin
        # lon_centers is GUARANTEED monotonic increasing (see binning above)
        # so dlon is always >= 0, no abs() needed
        R_earth = 6371.0
        mean_lat = df["lat"].mean()
        lat_rad = np.deg2rad(mean_lat)
        lon_rad = np.deg2rad(lon_centers)
        dlon = lon_rad - lon_rad[0]  # Always >= 0 since lon_centers is ascending
        x_km = R_earth * dlon * np.cos(lat_rad)
        
        total_obs = df_copy["dot"].notna().sum()
        valid_bins = np.sum(np.isfinite(profile_mean))
        logger.info(f"Pooled profile: {total_obs} observations → {valid_bins}/{n_lon_bins} bins "
                    f"(mean {obs_count[obs_count > 0].mean():.1f} obs/bin)")
        
        return profile_mean, lon_centers, x_km
    
    def _build_monthly_climatology_profiles(
        self,
        df: pd.DataFrame,
        lon_bin_size: float = 0.1,
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
        """
        Build monthly climatological DOT profiles.
        
        Aggregates ALL observations by MONTH (1-12), regardless of year.
        This creates a climatological view: "What does January look like on average?"
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'lon', 'lat', 'dot', 'month' columns
        lon_bin_size : float
            Longitude bin size in degrees (uses config.lon_bin_size by default)
            
        Returns
        -------
        monthly_profiles : Dict[int, np.ndarray]
            Dict mapping month (1-12) to mean DOT profile
        lon_centers : np.ndarray
            Center longitude of each bin
        x_km : np.ndarray
            Distance in km from first bin
        """
        # Determine longitude range from DATA
        lon_min = df["lon"].min()
        lon_max = df["lon"].max()
        
        # Handle dateline crossing
        if lon_max - lon_min > 180:
            logger.warning(f"[_build_monthly_climatology_profiles] Dateline crossing detected")
            df = df.copy()
            df.loc[df["lon"] < 0, "lon"] += 360
            lon_min = df["lon"].min()
            lon_max = df["lon"].max()
        
        # Create fixed longitude bins (larger for smoother climatology)
        lon_bins = np.arange(lon_min, lon_max + lon_bin_size, lon_bin_size)
        lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
        n_lon_bins = len(lon_centers)
        
        # Prepare df copy with bin assignment
        df_copy = df.copy()
        df_copy["lon_bin"] = pd.cut(
            df_copy["lon"],
            bins=lon_bins,
            labels=False,
            include_lowest=True
        )
        
        # Build profile for each month (1-12)
        monthly_profiles = {}
        
        for month in range(1, 13):
            month_data = df_copy[df_copy["month"] == month]
            
            if month_data.empty:
                monthly_profiles[month] = np.full(n_lon_bins, np.nan)
                continue
            
            # Pool all observations for this month (across all years)
            binned = month_data.groupby("lon_bin")["dot"].mean()
            
            profile = np.full(n_lon_bins, np.nan, dtype=float)
            for bin_idx in binned.index:
                if pd.notna(bin_idx) and int(bin_idx) < n_lon_bins:
                    profile[int(bin_idx)] = binned[bin_idx]
            
            monthly_profiles[month] = profile
            
            # Log stats
            n_obs = len(month_data)
            n_years = month_data["year"].nunique()
            valid_bins = np.sum(np.isfinite(profile))
            logger.debug(f"Month {month}: {n_obs} obs from {n_years} years → {valid_bins}/{n_lon_bins} bins")
        
        # Calculate distance in km
        R_earth = 6371.0
        mean_lat = df["lat"].mean()
        lat_rad = np.deg2rad(mean_lat)
        lon_rad = np.deg2rad(lon_centers)
        dlon = lon_rad - lon_rad[0]
        x_km = R_earth * dlon * np.cos(lat_rad)
        
        # Summary log
        months_with_data = sum(1 for m, p in monthly_profiles.items() if np.any(np.isfinite(p)))
        logger.info(f"Monthly climatology: {months_with_data}/12 months have data, "
                    f"{n_lon_bins} bins of {lon_bin_size}°")
        
        return monthly_profiles, lon_centers, x_km
    
    def _compute_slope_series(
        self,
        dot_matrix: np.ndarray,
        x_km: np.ndarray,
    ) -> np.ndarray:
        """Compute slope time series (m / 100 km) from DOT matrix."""
        n_time = dot_matrix.shape[1]
        slope_series = np.full(n_time, np.nan, dtype=float)
        
        for it in range(n_time):
            y = dot_matrix[:, it]
            mask = np.isfinite(x_km) & np.isfinite(y)
            
            if np.sum(mask) < 2:
                continue
            
            # Linear regression: slope in m/km
            a, _ = np.polyfit(x_km[mask], y[mask], 1)
            slope_series[it] = a * 100.0  # Convert to m / 100 km
        
        return slope_series
    
    # ==========================================================================
    # PRIVATE METHODS - GATE GEOMETRY
    # ==========================================================================
    
    def _get_gate_profile_points(
        self,
        gate_gdf: gpd.GeoDataFrame,
        n_pts: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample points along the gate line for profile analysis.
        
        Returns
        -------
        gate_lon_pts : np.ndarray
        gate_lat_pts : np.ndarray
        x_km : np.ndarray (distance along gate in km)
        """
        from shapely.ops import linemerge
        from shapely.geometry import MultiLineString, LineString
        
        gate_bounds = tuple(gate_gdf.total_bounds)
        
        if gate_bounds in self._gate_points_cache:
            return self._gate_points_cache[gate_bounds]
        
        geom = gate_gdf.geometry.unary_union
        if isinstance(geom, MultiLineString):
            geom = linemerge(geom)
        
        if isinstance(geom, LineString):
            total_length = geom.length
            distances = np.linspace(0, total_length, n_pts)
            points = [geom.interpolate(d) for d in distances]
            gate_lon_pts = np.array([p.x for p in points])
            gate_lat_pts = np.array([p.y for p in points])
        else:
            # Fallback for non-linestring geometries
            bounds = gate_gdf.total_bounds
            gate_lon_pts = np.linspace(bounds[0], bounds[2], n_pts)
            gate_lat_pts = np.linspace(bounds[1], bounds[3], n_pts)
        
        # Calculate distance in km
        R_earth = 6371.0
        lat0_rad = np.deg2rad(np.mean(gate_lat_pts))
        lon_rad = np.deg2rad(gate_lon_pts)
        lat_rad = np.deg2rad(gate_lat_pts)
        
        dlon = lon_rad - lon_rad[0]
        dlat = lat_rad - lat_rad[0]
        x_km = R_earth * np.sqrt((dlon * np.cos(lat0_rad))**2 + dlat**2)
        
        result = (gate_lon_pts, gate_lat_pts, x_km)
        self._gate_points_cache[gate_bounds] = result
        
        return result
    
    # ==========================================================================
    # PRIVATE METHODS - UTILITIES
    # ==========================================================================
    
    def _detect_satellite_type(self) -> str:
        """Detect satellite type from base_dir name."""
        dir_name = os.path.basename(self.config.base_dir.rstrip("/"))
        if "J2" in dir_name.upper():
            return "J2"
        return "J1"
    
    @staticmethod
    def _wrap_longitude(lon) -> np.ndarray:
        """Wrap longitude to [-180, 180]."""
        arr = np.asarray(lon, dtype=float)
        return ((arr + 180) % 360) - 180
    
    @staticmethod
    def _lon_in_bounds(lon_wrapped, lon_min, lon_max) -> np.ndarray:
        """Dateline-aware longitude check."""
        if lon_min <= lon_max:
            return (lon_wrapped >= lon_min) & (lon_wrapped <= lon_max)
        # Crosses dateline
        return (lon_wrapped >= lon_min) | (lon_wrapped <= lon_max)
    
    def _extract_strait_name(self, path: str) -> str:
        """Extract strait name from file path."""
        filename = Path(path).stem
        return filename.replace("_", " ").replace("-", " ").title()
    
    def _extract_strait_info(self, path: str) -> Tuple[str, Optional[int]]:
        """Extract strait name and pass number from gate shapefile path."""
        import re
        filename = Path(path).stem
        strait_name = filename.replace("_", " ").replace("-", " ").title()
        match = re.search(r'pass[_\s]*(\d+)', filename, re.IGNORECASE)
        pass_from_filename = int(match.group(1)) if match else None
        return strait_name, pass_from_filename
