"""
🗺️ Data Selector Component
===========================
Unified component for:
1. Gate/Location selection (dropdown + search bar)
2. Dataset selection with variable picker
3. Warning system for large data requests

Integrates with:
- GateService (config/gates.yaml)
- GeoResolver (Nominatim/OpenStreetMap search)
- DataService (unified data loading)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import streamlit as st

# Services imports
try:
    from src.services import GateService, DataService
    from src.core.models import BoundingBox, TimeRange, GateModel
    SERVICES_AVAILABLE = True
except ImportError as e:
    SERVICES_AVAILABLE = False
    st.warning(f"⚠️ Services not available: {e}")

# GeoResolver for location search
try:
    from src.agent.tools.geo_resolver import GeoResolver, GeoLocation
    GEO_RESOLVER_AVAILABLE = True
except ImportError:
    GEO_RESOLVER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DataSelection:
    """Result of data selection component."""
    # Location source
    source: str = "none"  # "gate", "search", "manual", "none"
    
    # Gate selection
    gate_id: Optional[str] = None
    gate: Optional[Any] = None  # GateModel
    
    # Location from search
    location: Optional[Any] = None  # GeoLocation
    location_name: str = ""
    
    # Bounding box (computed from gate or search)
    bbox: Optional[BoundingBox] = None
    buffer_km: float = 50.0
    
    # Time range
    time_range: Optional[TimeRange] = None
    
    # Dataset selection
    selected_datasets: List[str] = field(default_factory=list)
    selected_variables: Dict[str, List[str]] = field(default_factory=dict)
    
    # Flags
    confirmed: bool = False
    show_warning: bool = False


def render_data_selector() -> DataSelection:
    """
    Render the complete data selection UI.
    
    Returns DataSelection with all user choices.
    """
    selection = DataSelection()
    
    # Initialize session state
    if "data_selection" not in st.session_state:
        st.session_state.data_selection = selection
    
    st.sidebar.title("🗺️ Data Selection")
    
    # === SECTION 1: LOCATION ===
    st.sidebar.subheader("📍 Location")
    
    location_tab = st.sidebar.radio(
        "Choose location method",
        ["🚪 Ocean Gate", "🔍 Search Location", "📐 Manual BBox"],
        horizontal=True,
        key="location_method"
    )
    
    if location_tab == "🚪 Ocean Gate":
        selection = _render_gate_selection(selection)
    elif location_tab == "🔍 Search Location":
        selection = _render_location_search(selection)
    else:
        selection = _render_manual_bbox(selection)
    
    st.sidebar.divider()
    
    # === SECTION 2: TIME RANGE ===
    selection = _render_time_range(selection)
    
    st.sidebar.divider()
    
    # === SECTION 3: DATASETS & VARIABLES ===
    selection = _render_dataset_selection(selection)
    
    st.sidebar.divider()
    
    # === SECTION 4: CONFIRMATION WITH WARNING ===
    selection = _render_confirmation(selection)
    
    # Store in session
    st.session_state.data_selection = selection
    
    return selection


def _render_gate_selection(selection: DataSelection) -> DataSelection:
    """Render gate dropdown with GateService."""
    
    if not SERVICES_AVAILABLE:
        st.sidebar.warning("⚠️ GateService not available")
        return selection
    
    try:
        gate_service = GateService()
    except Exception as e:
        st.sidebar.error(f"Gate service error: {e}")
        return selection
    
    # Get regions for filtering
    regions = gate_service.get_regions()
    
    # Region filter
    selected_region = st.sidebar.selectbox(
        "🌍 Region",
        ["All Regions"] + regions,
        key="gate_region_filter"
    )
    
    # Get gates
    if selected_region == "All Regions":
        gates = gate_service.list_gates()
    else:
        gates = gate_service.list_gates_by_region(selected_region)
    
    # Build gate options
    gate_options = {"⚠️ No Gate Selected (Global)": None}
    for gate in gates:
        icon = "🚪" if "strait" in gate.name.lower() else "🌊"
        gate_options[f"{icon} {gate.name}"] = gate.id
    
    # Current selection from session
    current_gate = st.session_state.get("selected_gate_id")
    
    # Find current index
    options_list = list(gate_options.keys())
    ids_list = list(gate_options.values())
    current_idx = 0
    if current_gate in ids_list:
        current_idx = ids_list.index(current_gate)
    
    # Gate selector
    selected_option = st.sidebar.selectbox(
        "🚪 Select Gate",
        options_list,
        index=current_idx,
        key="gate_selector_main"
    )
    
    selected_gate_id = gate_options[selected_option]
    selection.gate_id = selected_gate_id
    selection.source = "gate" if selected_gate_id else "none"
    
    # Show gate info
    if selected_gate_id:
        gate = gate_service.get_gate(selected_gate_id)
        selection.gate = gate
        
        if gate:
            # Info card
            st.sidebar.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); 
                        padding: 12px; border-radius: 8px; margin: 8px 0;
                        border-left: 4px solid #4fc3f7;">
                <div style="color: #4fc3f7; font-size: 0.8em;">📍 {gate.region}</div>
                <div style="color: white; font-size: 0.9em; margin: 4px 0;">{gate.description}</div>
                <div style="color: #90caf9; font-size: 0.75em;">
                    Datasets: {', '.join(gate.datasets) if gate.datasets else 'Any'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Buffer slider
            buffer = st.sidebar.slider(
                "Buffer (km)",
                min_value=10, max_value=200,
                value=int(gate.default_buffer_km or 50),
                step=10,
                key="gate_buffer"
            )
            selection.buffer_km = buffer
            
            # Compute bbox from gate
            if gate.bbox:
                selection.bbox = BoundingBox(
                    lat_min=gate.bbox.lat_min,
                    lat_max=gate.bbox.lat_max,
                    lon_min=gate.bbox.lon_min,
                    lon_max=gate.bbox.lon_max
                )
    else:
        selection.show_warning = True  # Will trigger warning
    
    st.session_state.selected_gate_id = selected_gate_id
    return selection


def _render_location_search(selection: DataSelection) -> DataSelection:
    """
    Render Google Maps-style location search.
    
    Uses GeoResolver with Nominatim backend.
    """
    
    st.sidebar.markdown("🔍 **Search any location**")
    
    # Search input
    search_query = st.sidebar.text_input(
        "Location name",
        placeholder="e.g. Fram Strait, Lago Maggiore, Venice...",
        key="location_search_input"
    )
    
    # Search button
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        search_clicked = st.button("🔍 Search", key="location_search_btn", use_container_width=True)
    with col2:
        clear_clicked = st.button("✖️", key="location_clear_btn")
    
    if clear_clicked:
        st.session_state.pop("search_result", None)
        st.session_state.pop("search_suggestions", None)
        st.rerun()
    
    # Perform search
    if search_clicked and search_query:
        if GEO_RESOLVER_AVAILABLE:
            with st.spinner("Searching..."):
                result = _search_location(search_query)
                if result:
                    st.session_state.search_result = result
                    st.sidebar.success(f"✅ Found: {result.name}")
                else:
                    st.sidebar.error("❌ Location not found")
        else:
            st.sidebar.warning("⚠️ GeoResolver not available")
    
    # Show result
    if "search_result" in st.session_state:
        loc = st.session_state.search_result
        selection.location = loc
        selection.location_name = loc.name
        selection.source = "search"
        
        # Location info card
        st.sidebar.markdown(f"""
        <div style="background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%); 
                    padding: 12px; border-radius: 8px; margin: 8px 0;
                    border-left: 4px solid #81c784;">
            <div style="color: #c8e6c9; font-size: 0.8em;">📍 {loc.region or loc.country}</div>
            <div style="color: white; font-weight: bold; margin: 4px 0;">{loc.name}</div>
            <div style="color: #a5d6a7; font-size: 0.75em;">
                {loc.lat:.4f}°N, {loc.lon:.4f}°E
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Buffer for search results
        buffer = st.sidebar.slider(
            "Search radius (km)",
            min_value=10, max_value=500,
            value=100,
            step=10,
            key="search_buffer"
        )
        selection.buffer_km = buffer
        
        # Create bbox from location
        if loc.bbox:
            selection.bbox = BoundingBox(
                lat_min=loc.bbox[0],
                lat_max=loc.bbox[1],
                lon_min=loc.bbox[2],
                lon_max=loc.bbox[3]
            )
        else:
            # Approximate bbox from point + buffer
            import math
            lat_delta = buffer / 111.0  # ~111 km per degree latitude
            # Longitude degrees vary with latitude: 111 * cos(lat)
            cos_lat = math.cos(math.radians(loc.lat)) if loc.lat else 1.0
            lon_delta = buffer / (111.0 * cos_lat) if cos_lat > 0.01 else buffer / 111.0
            selection.bbox = BoundingBox(
                lat_min=loc.lat - lat_delta,
                lat_max=loc.lat + lat_delta,
                lon_min=loc.lon - lon_delta,
                lon_max=loc.lon + lon_delta
            )
    
    return selection


def _search_location(query: str) -> Optional[Any]:
    """Search location using GeoResolver (async wrapper)."""
    try:
        resolver = GeoResolver()
        # Run async in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(resolver.resolve(query))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Location search failed: {e}")
        return None


def _render_manual_bbox(selection: DataSelection) -> DataSelection:
    """Render manual bounding box input."""
    
    st.sidebar.markdown("📐 **Manual Bounding Box**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        lat_min = st.number_input("Lat Min", value=40.0, key="bbox_lat_min")
        lon_min = st.number_input("Lon Min", value=-10.0, key="bbox_lon_min")
    with col2:
        lat_max = st.number_input("Lat Max", value=50.0, key="bbox_lat_max")
        lon_max = st.number_input("Lon Max", value=10.0, key="bbox_lon_max")
    
    selection.bbox = BoundingBox(
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max
    )
    selection.source = "manual"
    
    # Show area estimation
    area_deg = (lat_max - lat_min) * (lon_max - lon_min)
    st.sidebar.caption(f"Area: ~{area_deg:.1f} sq degrees")
    
    if area_deg > 100:
        selection.show_warning = True
    
    return selection


def _render_time_range(selection: DataSelection) -> DataSelection:
    """Render time range selector."""
    
    st.sidebar.subheader("📅 Time Range")
    
    # Quick presets
    preset = st.sidebar.selectbox(
        "Quick selection",
        ["Last 7 days", "Last 30 days", "Last 90 days", "Last year", "Custom"],
        key="time_preset"
    )
    
    now = datetime.now()
    
    if preset == "Last 7 days":
        start = now - timedelta(days=7)
        end = now
    elif preset == "Last 30 days":
        start = now - timedelta(days=30)
        end = now
    elif preset == "Last 90 days":
        start = now - timedelta(days=90)
        end = now
    elif preset == "Last year":
        start = now - timedelta(days=365)
        end = now
    else:  # Custom
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start = st.date_input("Start", value=now - timedelta(days=30), key="time_start")
        with col2:
            end = st.date_input("End", value=now, key="time_end")
    
    selection.time_range = TimeRange(
        start=datetime.combine(start, datetime.min.time()) if hasattr(start, 'year') else start,
        end=datetime.combine(end, datetime.max.time()) if hasattr(end, 'year') else end
    )
    
    return selection


def _render_dataset_selection(selection: DataSelection) -> DataSelection:
    """
    Render dataset and variable selection.
    
    Shows available datasets and lets user pick variables.
    """
    
    st.sidebar.subheader("📊 Datasets & Variables")
    
    # Get available datasets
    try:
        from src.data_manager.intake_bridge import get_catalog
        catalog = get_catalog()
        all_datasets = catalog.list_datasets()
    except ImportError:
        all_datasets = ["SLCCI", "ERA5", "CMEMS-SST", "CMEMS-SSH"]
        st.sidebar.caption("Using default dataset list")
    
    # Filter by gate recommendations if gate selected
    recommended = []
    if selection.gate and hasattr(selection.gate, 'datasets') and selection.gate.datasets:
        recommended = selection.gate.datasets
        st.sidebar.caption(f"✨ Recommended for {selection.gate.name}: {', '.join(recommended)}")
    
    # Multi-select datasets
    selected = st.sidebar.multiselect(
        "Select datasets",
        all_datasets,
        default=recommended[:2] if recommended else [],
        key="dataset_multiselect"
    )
    selection.selected_datasets = selected
    
    # Variable selection per dataset
    if selected:
        st.sidebar.markdown("**📈 Select variables:**")
        
        for ds_id in selected:
            # Get dataset variables
            variables = _get_dataset_variables(ds_id)
            
            if variables:
                with st.sidebar.expander(f"📦 {ds_id}", expanded=True):
                    selected_vars = st.multiselect(
                        "Variables",
                        variables,
                        default=variables[:2] if len(variables) > 1 else variables,
                        key=f"vars_{ds_id}"
                    )
                    selection.selected_variables[ds_id] = selected_vars
    
    return selection


def _get_dataset_variables(dataset_id: str) -> List[str]:
    """Get available variables for a dataset."""
    
    # Try catalog first
    try:
        from src.data_manager.intake_bridge import get_catalog
        catalog = get_catalog()
        meta = catalog.get_metadata(dataset_id)
        return meta.get("variables", [])
    except Exception:
        pass
    
    # Fallback to known variables
    KNOWN_VARIABLES = {
        "SLCCI": ["mean_sea_surface", "sla", "adt", "ugos", "vgos"],
        "ERA5": ["u10", "v10", "msl", "t2m", "sst"],
        "CMEMS-SST": ["analysed_sst", "analysis_error"],
        "CMEMS-SSH": ["adt", "sla", "ugos", "vgos"],
    }
    
    return KNOWN_VARIABLES.get(dataset_id, ["data"])


def _render_confirmation(selection: DataSelection) -> DataSelection:
    """
    Render confirmation with warning for large requests.
    """
    
    st.sidebar.subheader("✅ Confirm & Load")
    
    # Calculate data size estimation
    bbox_area = 0
    if selection.bbox:
        bbox_area = (selection.bbox.lat_max - selection.bbox.lat_min) * \
                    (selection.bbox.lon_max - selection.bbox.lon_min)
    
    time_days = 0
    if selection.time_range:
        try:
            start = selection.time_range.start
            end = selection.time_range.end
            # Handle both datetime and string
            if isinstance(start, str):
                start = datetime.fromisoformat(start.replace('Z', '+00:00'))
            if isinstance(end, str):
                end = datetime.fromisoformat(end.replace('Z', '+00:00'))
            time_days = (end - start).days
        except Exception:
            time_days = 30  # Default fallback
    
    num_datasets = len(selection.selected_datasets)
    num_vars = sum(len(v) for v in selection.selected_variables.values())
    
    # Show summary
    st.sidebar.markdown(f"""
    **Summary:**
    - 📍 Location: {selection.location_name or selection.gate_id or "Global"}
    - 📐 Area: ~{bbox_area:.1f} sq degrees
    - 📅 Time: {time_days} days
    - 📊 Datasets: {num_datasets}, Variables: {num_vars}
    """)
    
    # ⚠️ WARNING for large/global requests
    show_warning = (
        selection.source == "none" or  # No location selected
        bbox_area > 100 or  # Large area
        (bbox_area > 50 and time_days > 90)  # Medium area + long time
    )
    
    if show_warning:
        st.sidebar.warning("""
        ⚠️ **Large Data Warning**
        
        No location filter selected or large area requested.
        This may download **significant amounts of data** from APIs.
        
        **Recommendation:** Select a Gate or search for a location.
        """)
        
        # Force confirmation checkbox
        confirmed = st.sidebar.checkbox(
            "I understand, proceed with global/large request",
            key="large_data_confirm"
        )
        selection.confirmed = confirmed
        
        if not confirmed:
            st.sidebar.error("❌ Please confirm to proceed or select a location")
    else:
        selection.confirmed = True
        st.sidebar.success("✅ Ready to load data")
    
    # Load button
    if st.sidebar.button(
        "🚀 Load Data",
        disabled=not selection.confirmed,
        key="load_data_btn",
        use_container_width=True,
        type="primary"
    ):
        st.session_state.data_load_requested = True
        st.session_state.current_selection = selection
    
    return selection


# === CONVENIENCE FUNCTIONS ===

def get_current_selection() -> Optional[DataSelection]:
    """Get the current data selection from session state."""
    return st.session_state.get("data_selection")


def is_data_load_requested() -> bool:
    """Check if user clicked Load Data."""
    return st.session_state.get("data_load_requested", False)


def clear_load_request():
    """Clear the load request flag after processing."""
    st.session_state.data_load_requested = False
