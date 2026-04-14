"""
Core Domain Models
==================
Pydantic models shared across all branches and layers.

These models provide:
- Type safety with runtime validation
- JSON serialization for API compatibility
- Self-documenting schemas
- Consistent data structures across Streamlit and API

Usage:
    from src.core.models import BoundingBox, GateModel, TimeRange, DataRequest
    from src.core.models import SpatialResolution, TemporalResolution

Models:
    - BoundingBox: Geographic bbox with validation and center calculation
    - TimeRange: Temporal range accepting datetime or ISO strings
    - GateModel: Ocean gate definition with optional fallback bbox
    - DataRequest: Unified data request for API and services
    - SpatialResolution: Binning resolution enum (0.1° to 1.0°)
    - TemporalResolution: Temporal aggregation enum (hourly to monthly)
    - ResolutionConfig: Combined resolution settings

Example:
    >>> from src.core.models import BoundingBox, TimeRange, SpatialResolution
    >>> bbox = BoundingBox(lat_min=78, lat_max=82, lon_min=-20, lon_max=15)
    >>> print(bbox.center)  # (80.0, -2.5)
    >>> tr = TimeRange(start="2024-01-01", end="2024-12-31")
    >>> print(tr.days)  # 365
    >>> res = SpatialResolution.MEDIUM
    >>> print(res.value)  # 0.25

Changelog:
    2025-12-29: Added BoundingBox.center, TimeRange datetime support,
                DataRequest.dataset_id, SpatialResolution as float enum
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# ENUMS
# =============================================================================

class TemporalResolution(str, Enum):
    """Temporal resolution options for data requests."""
    HOURLY = "hourly"
    THREE_HOURLY = "3-hourly"
    SIX_HOURLY = "6-hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


class SpatialResolution(float, Enum):
    """
    Spatial resolution options for data requests and binning.
    
    Values are in degrees. Use `.value` to get the float for binning operations.
    
    Example:
        >>> res = SpatialResolution.MEDIUM
        >>> bin_size = res.value  # 0.25 degrees
        >>> data.coarsen(lat=int(0.25/current_res), lon=int(0.25/current_res))
    
    Available resolutions:
        - HIGH: 0.1° (~11 km at equator)
        - MEDIUM: 0.25° (~28 km) - default for most analyses
        - LOW: 0.5° (~56 km)
        - COARSE: 1.0° (~111 km)
    
    For custom binning, you can also pass a float directly to analysis functions.
    """
    HIGH = 0.1
    MEDIUM = 0.25
    LOW = 0.5
    COARSE = 1.0
    
    @classmethod
    def from_degrees(cls, degrees: float) -> 'SpatialResolution':
        """
        Get closest resolution enum from degrees value.
        
        Args:
            degrees: Resolution in degrees
            
        Returns:
            Closest SpatialResolution enum
            
        Example:
            >>> SpatialResolution.from_degrees(0.3)
            SpatialResolution.MEDIUM
        """
        options = [(abs(r.value - degrees), r) for r in cls]
        return min(options, key=lambda x: x[0])[1]
    
    @classmethod
    def list_all(cls) -> list:
        """List all resolutions with descriptions."""
        return [
            {"name": r.name, "degrees": r.value, "km_approx": r.value * 111}
            for r in cls
        ]


class DataSource(str, Enum):
    """Available data sources."""
    CMEMS = "cmems"
    ERA5 = "era5"
    CYGNSS = "cygnss"
    CLIMATE_INDICES = "climate_indices"
    LOCAL = "local"


class GateRegion(str, Enum):
    """Ocean gate regions."""
    ATLANTIC_SECTOR = "Atlantic Sector"
    PACIFIC_SECTOR = "Pacific Sector"
    CANADIAN_ARCHIPELAGO = "Canadian Archipelago"


# =============================================================================
# CORE MODELS
# =============================================================================

class BoundingBox(BaseModel):
    """
    Geographic bounding box with validation.
    
    Attributes:
        lat_min: Minimum latitude (-90 to 90)
        lat_max: Maximum latitude (-90 to 90)
        lon_min: Minimum longitude (-180 to 180)
        lon_max: Maximum longitude (-180 to 180)
    
    Example:
        >>> bbox = BoundingBox(lat_min=78.0, lat_max=80.0, lon_min=-20.0, lon_max=10.0)
        >>> print(bbox.lat_range)
        (78.0, 80.0)
    """
    lat_min: float = Field(..., ge=-90, le=90, description="Minimum latitude")
    lat_max: float = Field(..., ge=-90, le=90, description="Maximum latitude")
    lon_min: float = Field(..., ge=-180, le=180, description="Minimum longitude")
    lon_max: float = Field(..., ge=-180, le=180, description="Maximum longitude")
    
    @model_validator(mode='after')
    def validate_ranges(self) -> 'BoundingBox':
        """Ensure lat_min <= lat_max."""
        if self.lat_min > self.lat_max:
            raise ValueError(f"lat_min ({self.lat_min}) must be <= lat_max ({self.lat_max})")
        # Note: lon_min > lon_max is valid (crosses dateline)
        return self
    
    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (lat_min, lat_max, lon_min, lon_max)."""
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)
    
    @property
    def lat_range(self) -> Tuple[float, float]:
        """Return latitude range as (min, max)."""
        return (self.lat_min, self.lat_max)
    
    @property
    def lon_range(self) -> Tuple[float, float]:
        """Return longitude range as (min, max)."""
        return (self.lon_min, self.lon_max)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Return center point as (lat, lon)."""
        center_lat = (self.lat_min + self.lat_max) / 2
        if self.crosses_dateline:
            # Handle dateline crossing
            center_lon = (self.lon_min + self.lon_max + 360) / 2
            if center_lon > 180:
                center_lon -= 360
        else:
            center_lon = (self.lon_min + self.lon_max) / 2
        return (center_lat, center_lon)
    
    @property
    def as_list(self) -> List[float]:
        """Return as [lat_min, lat_max, lon_min, lon_max] for API compatibility."""
        return [self.lat_min, self.lat_max, self.lon_min, self.lon_max]
    
    @property
    def crosses_dateline(self) -> bool:
        """Check if bounding box crosses the international dateline."""
        return self.lon_min > self.lon_max
    
    @classmethod
    def from_tuple(cls, bbox: Tuple[float, float, float, float]) -> 'BoundingBox':
        """Create from tuple (lat_min, lat_max, lon_min, lon_max)."""
        return cls(lat_min=bbox[0], lat_max=bbox[1], lon_min=bbox[2], lon_max=bbox[3])
    
    @classmethod
    def from_list(cls, bbox: List[float]) -> 'BoundingBox':
        """Create from list [lat_min, lat_max, lon_min, lon_max]."""
        if len(bbox) != 4:
            raise ValueError(f"Expected 4 elements, got {len(bbox)}")
        return cls(lat_min=bbox[0], lat_max=bbox[1], lon_min=bbox[2], lon_max=bbox[3])


class TimeRange(BaseModel):
    """
    Temporal range for data queries.
    
    Attributes:
        start: Start date (string YYYY-MM-DD or datetime)
        end: End date (string YYYY-MM-DD or datetime)
    
    Example:
        >>> tr = TimeRange(start="2024-01-01", end="2024-12-31")
        >>> print(tr.start_date)
        datetime(2024, 1, 1)
        
        >>> from datetime import datetime
        >>> tr = TimeRange(start=datetime(2024, 1, 1), end=datetime(2024, 12, 31))
    """
    start: str = Field(..., description="Start date (YYYY-MM-DD or datetime)")
    end: str = Field(..., description="End date (YYYY-MM-DD or datetime)")
    
    @field_validator('start', 'end', mode='before')
    @classmethod
    def convert_datetime(cls, v):
        """Convert datetime to string if needed."""
        if isinstance(v, datetime):
            return v.strftime('%Y-%m-%d')
        return v
    
    @field_validator('start', 'end')
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Validate ISO date format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD")
        return v
    
    @model_validator(mode='after')
    def validate_range(self) -> 'TimeRange':
        """Ensure start <= end."""
        if self.start_date > self.end_date:
            raise ValueError(f"start ({self.start}) must be <= end ({self.end})")
        return self
    
    @property
    def start_date(self) -> datetime:
        """Return start as datetime object."""
        return datetime.fromisoformat(self.start)
    
    @property
    def end_date(self) -> datetime:
        """Return end as datetime object."""
        return datetime.fromisoformat(self.end)
    
    @property
    def days(self) -> int:
        """Return number of days in range."""
        return (self.end_date - self.start_date).days


class GateModel(BaseModel):
    """
    Ocean gate definition.
    
    Attributes:
        id: Unique gate identifier (e.g., "fram_strait")
        name: Display name with emoji (e.g., "🧊 Fram Strait")
        file: Shapefile filename
        description: Human-readable description
        region: Geographic region (Atlantic, Pacific, Canadian)
        closest_passes: Pre-computed closest satellite passes
        datasets: Recommended datasets for this gate
        default_buffer_km: Default buffer around gate in km
        bbox: Optional bounding box (computed from lat/lon ranges)
    
    Example:
        >>> gate = GateModel(
        ...     id="fram_strait",
        ...     name="🧊 Fram Strait",
        ...     file="fram_strait_S3_pass_481.shp",
        ...     description="Main Arctic-Atlantic exchange",
        ...     region="Atlantic Sector",
        ...     datasets=["SLCCI", "ERA5"]
        ... )
    """
    id: str = Field(..., description="Unique gate identifier")
    name: str = Field(..., description="Display name with emoji")
    file: str = Field(..., description="Shapefile filename")
    description: str = Field(default="", description="Human-readable description")
    region: str = Field(default="", description="Geographic region")
    closest_passes: Optional[List[int]] = Field(default=None, description="Pre-computed closest satellite passes")
    datasets: Optional[List[str]] = Field(default=None, description="Recommended datasets for this gate")
    default_buffer_km: Optional[float] = Field(default=50.0, description="Default buffer around gate in km")
    # Bounding box (can be specified directly or via lat/lon ranges)
    lat_min: Optional[float] = Field(default=None, ge=-90, le=90)
    lat_max: Optional[float] = Field(default=None, ge=-90, le=90)
    lon_min: Optional[float] = Field(default=None, ge=-180, le=180)
    lon_max: Optional[float] = Field(default=None, ge=-180, le=180)
    # Alternative: latitude_range and longitude_range from YAML
    latitude_range: Optional[List[float]] = Field(default=None, description="[lat_min, lat_max]")
    longitude_range: Optional[List[float]] = Field(default=None, description="[lon_min, lon_max]")
    importance: Optional[str] = Field(default=None, description="Scientific importance")
    
    # --- Standardized Gate Division System ---
    # Parent gate reference for divided gates (e.g., fram_strait_west -> fram_strait)
    parent_gate: Optional[str] = Field(default=None, description="Parent gate ID for divided gates (uses parent shapefile)")
    # Division longitude: single value where the gate is split West/East
    division_longitude: Optional[float] = Field(default=None, description="Longitude where gate is divided (West < value < East)")
    # Longitude filters (derived from division_longitude for consistency)
    lon_filter_min: Optional[float] = Field(default=None, description="Min longitude filter (for East sections)")
    lon_filter_max: Optional[float] = Field(default=None, description="Max longitude filter (for West sections)")
    
    @property
    def bbox(self) -> Optional[BoundingBox]:
        """Get bounding box from lat/lon fields or ranges."""
        # Try direct lat/lon fields first
        if all(v is not None for v in [self.lat_min, self.lat_max, self.lon_min, self.lon_max]):
            return BoundingBox(
                lat_min=self.lat_min,
                lat_max=self.lat_max,
                lon_min=self.lon_min,
                lon_max=self.lon_max
            )
        # Try ranges (from YAML)
        if self.latitude_range and self.longitude_range:
            return BoundingBox(
                lat_min=self.latitude_range[0],
                lat_max=self.latitude_range[1],
                lon_min=self.longitude_range[0],
                lon_max=self.longitude_range[1]
            )
        return None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "fram_strait",
                "name": "🧊 Fram Strait",
                "file": "fram_strait_S3_pass_481.shp",
                "description": "Main Arctic-Atlantic exchange",
                "region": "Atlantic Sector",
                "closest_passes": [481, 254, 127, 308, 55],
                "datasets": ["SLCCI", "ERA5", "CMEMS-SST"],
                "default_buffer_km": 50.0
            }
        }


class ResolutionConfig(BaseModel):
    """
    Resolution configuration for data downloads.
    
    Attributes:
        temporal: Temporal resolution (hourly, daily, etc.)
        spatial: Spatial resolution in degrees
    """
    temporal: TemporalResolution = Field(default=TemporalResolution.DAILY)
    spatial: SpatialResolution = Field(default=SpatialResolution.MEDIUM)


class DataRequest(BaseModel):
    """
    Unified data request model for API and services.
    
    Used by both FastAPI endpoints and Streamlit to request data.
    
    Attributes:
        bbox: Geographic bounding box
        time_range: Temporal range
        variables: List of variables to retrieve
        gate_id: Optional gate identifier (auto-populates bbox)
        pass_number: Optional satellite pass filter
        source: Data source (cmems, era5, etc.)
        resolution: Resolution configuration
    
    Example:
        >>> request = DataRequest(
        ...     bbox=BoundingBox(lat_min=78, lat_max=80, lon_min=-20, lon_max=10),
        ...     time_range=TimeRange(start="2024-01-01", end="2024-12-31"),
        ...     variables=["sla", "adt"],
        ...     gate_id="fram_strait"
        ... )
    """
    bbox: BoundingBox
    time_range: TimeRange
    variables: List[str] = Field(default_factory=list)
    dataset_id: str = Field(default="cmems_sla", description="Dataset identifier")
    gate_id: Optional[str] = Field(default=None, description="Gate identifier")
    pass_number: Optional[int] = Field(default=None, description="Satellite pass number")
    source: Optional[DataSource] = Field(default=None, description="Data source")
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    
    class Config:
        json_schema_extra = {
            "example": {
                "bbox": {
                    "lat_min": 78.0,
                    "lat_max": 80.0,
                    "lon_min": -20.0,
                    "lon_max": 10.0
                },
                "time_range": {
                    "start": "2024-01-01",
                    "end": "2024-12-31"
                },
                "variables": ["sla", "adt"],
                "dataset_id": "cmems_sla",
                "gate_id": "fram_strait",
                "source": "cmems"
            }
        }


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class GateResponse(BaseModel):
    """API response for gate details."""
    gate: GateModel
    bbox: Optional[BoundingBox] = None
    available: bool = True


class DataResponse(BaseModel):
    """API response for data download status."""
    request_id: str
    status: str = Field(..., description="pending, downloading, ready, error")
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    data_url: Optional[str] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None


class GateListResponse(BaseModel):
    """API response for listing gates."""
    gates: List[GateModel]
    total: int


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def bbox_to_legacy_format(bbox: BoundingBox) -> dict:
    """
    Convert BoundingBox to legacy format used in existing code.
    
    Returns:
        dict with lat_range and lon_range tuples
    """
    return {
        "lat_range": bbox.lat_range,
        "lon_range": bbox.lon_range,
    }


def legacy_to_bbox(lat_range: Tuple[float, float], lon_range: Tuple[float, float]) -> BoundingBox:
    """
    Convert legacy lat_range/lon_range to BoundingBox.
    
    Args:
        lat_range: (lat_min, lat_max)
        lon_range: (lon_min, lon_max)
    
    Returns:
        BoundingBox instance
    """
    return BoundingBox(
        lat_min=lat_range[0],
        lat_max=lat_range[1],
        lon_min=lon_range[0],
        lon_max=lon_range[1]
    )
