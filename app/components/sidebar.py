"""
Sidebar Component - SLCCI & CMEMS Analysis
===========================================
Controls following SLCCI PLOTTER notebook workflow:
1. Gate Selection (region + gate)
2. Data Source Selection (SLCCI/CMEMS with comparison mode)
3. Data Paths (SLCCI files + geoid / CMEMS folders)
4. Pass Selection (auto/manual for SLCCI, from filename for CMEMS)
5. Cycle Range (SLCCI only)
6. Processing Parameters (binning, flags)

Comparison Mode:
- Load SLCCI and CMEMS separately
- Toggle comparison to overlay plots
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple, List

import streamlit as st
import yaml

from ..state import (
    AppConfig, 
    store_slcci_data, 
    store_cmems_data, 
    is_comparison_mode, 
    set_comparison_mode,
    store_dtu_data,
    get_dtu_data
)

# Import intelligent cache (replaces old DataCache)
from src.services.intelligent_cache import get_intelligent_cache

# Import longitude filter for divided gates (Fram West/East, Davis West/East)
from app.components.loaders.base import apply_longitude_filter

# Global cache instance (IntelligentCache with disk persistence)
_cache = get_intelligent_cache()


def _render_cache_viewer():
    """Render cache viewer UI in sidebar."""
    import pandas as pd
    
    # Get stats
    stats = _cache.get_stats()
    entries = _cache.get_all_entries()
    
    # Header stats
    st.markdown(f"**📊 {stats['total_items']} items** | {stats['total_size_mb']:.1f} MB")
    
    if stats['total_items'] == 0:
        st.info("Cache is empty. Load data to populate.")
        return
    
    # Group by dataset
    st.markdown("---")
    
    for ds_name in ["slcci", "cmems_l3", "cmems_l4", "dtuspace"]:
        ds_entries = [e for e in entries if e['dataset'] == ds_name]
        if not ds_entries:
            continue
        
        # Dataset emoji
        emoji = {"slcci": "🟠", "cmems_l3": "🔵", "cmems_l4": "🟣", "dtuspace": "🟢"}.get(ds_name, "📦")
        ds_display = {"slcci": "SLCCI", "cmems_l3": "CMEMS L3", "cmems_l4": "CMEMS L4", "dtuspace": "DTUSpace"}.get(ds_name, ds_name)
        
        st.markdown(f"**{emoji} {ds_display}** ({len(ds_entries)} items)")
        
        for entry in ds_entries:
            # Build label
            label_parts = [entry['gate'].replace('_', ' ').title()]
            if entry['pass']:
                label_parts.append(f"Pass {entry['pass']}")
            if entry['track']:
                label_parts.append(f"T{entry['track']}")
            label = " | ".join(label_parts)
            
            # Format date range
            date_range = entry['date_range'] if entry['date_range'] != "N/A" else ""
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"• {label}")
                if date_range:
                    st.caption(f"  {date_range} ({entry['n_obs']} obs)")
            with col2:
                # Delete button - use clear_by_key for IntelligentCache
                if st.button("🗑️", key=f"del_{entry['key']}", help=f"Delete {entry['key']}"):
                    _cache.clear_by_key(entry['key'])
                    st.rerun()
    
    # Clear all button
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True, key="cache_refresh"):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True, type="secondary", key="cache_clear_all"):
            _cache.clear_all()
            st.success("Cache cleared!")
            st.rerun()


def _get_lon_filter_for_gate(gate_id: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Get longitude filter values for a gate from GateService.
    
    Args:
        gate_id: Gate identifier (e.g., "fram_strait_west")
        
    Returns:
        Tuple of (lon_filter_min, lon_filter_max) - both None if no filter
    """
    if not gate_id:
        return None, None
    
    try:
        from src.services.gate_service import GateService
        service = GateService()
        gate = service.get_gate(gate_id)
        if gate:
            return gate.lon_filter_min, gate.lon_filter_max
    except Exception:
        pass
    
    return None, None


def _get_parent_gate_id(gate_id: str) -> str:
    """
    Get the parent gate ID for cache key purposes.
    
    For divided gates (e.g., fram_strait_west), returns the parent (fram_strait).
    This allows sharing cache between West/East sections.
    
    Args:
        gate_id: Gate identifier (e.g., "fram_strait_west" or "fram_strait")
        
    Returns:
        Parent gate ID if divided gate, otherwise the original gate_id
    """
    if not gate_id:
        return gate_id
    
    try:
        from src.services.gate_service import GateService
        service = GateService()
        gate = service.get_gate(gate_id)
        if gate and gate.parent_gate:
            return gate.parent_gate
    except Exception:
        pass
    
    return gate_id


# ============================================================
# PASS/TRACK EXTRACTION HELPERS
# ============================================================

def _extract_pass_from_gate_name(gate_name: str) -> Optional[int]:
    """
    Extract pass/track number from gate shapefile name.
    
    Patterns supported:
    - *_TPJ_pass_XXX  (J1/J2/J3 = TOPEX/Poseidon-Jason)
    - *_S3_pass_XXX   (Sentinel-3)
    - Trailing _XXX   (generic fallback)
    
    Args:
        gate_name: Gate name or shapefile path
        
    Returns:
        Pass number as int, or None if not found
    """
    if not gate_name:
        return None
    
    # Get just the filename stem
    name = Path(gate_name).stem if "/" in gate_name or "\\" in gate_name else gate_name
    
    # Pattern 1: _TPJ_pass_XXX (J1/J2/J3)
    match = re.search(r"_TPJ_pass_(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Pattern 2: _S3_pass_XXX (Sentinel-3)
    match = re.search(r"_S3_pass_(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Pattern 3: Generic trailing _XXX (3+ digits at end)
    match = re.search(r"_(\d{3,})$", name)
    if match:
        return int(match.group(1))
    
    return None


def _extract_satellite_from_gate_name(gate_name: str) -> Optional[str]:
    """
    Extract satellite type from gate name.
    
    Returns:
        'J2' for TPJ (Jason-2), 'S3' for Sentinel-3, or None
    """
    if not gate_name:
        return None
    
    name = Path(gate_name).stem if "/" in gate_name or "\\" in gate_name else gate_name
    
    if "_TPJ_" in name.upper():
        return "J2"  # TOPEX/Poseidon-Jason series
    elif "_S3_" in name.upper():
        return "S3"  # Sentinel-3
    
    return None


# ============================================================
# PRE-COMPUTED PASSES CACHE
# ============================================================

_GATE_PASSES_CACHE = None

def _load_gate_passes_config() -> dict:
    """Load pre-computed gate passes from config/gate_passes.yaml."""
    global _GATE_PASSES_CACHE
    
    if _GATE_PASSES_CACHE is not None:
        return _GATE_PASSES_CACHE
    
    config_path = Path(__file__).parent.parent.parent / "config" / "gate_passes.yaml"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            _GATE_PASSES_CACHE = yaml.safe_load(f)
        return _GATE_PASSES_CACHE
    
    return {"gates": {}}


def _get_precomputed_passes(gate_name: str, dataset_type: str = "slcci") -> List[int]:
    """
    Get pre-computed closest passes for a gate.
    
    Args:
        gate_name: Gate name (e.g., "davis_strait" or "fram_strait_S3_pass_481")
        dataset_type: "slcci" or "cmems"
        
    Returns:
        List of pass/track numbers (up to 5), or empty list if not found
    """
    if not gate_name:
        return []
        
    config = _load_gate_passes_config()
    gates = config.get("gates", {})
    key = "slcci_passes" if dataset_type == "slcci" else "cmems_tracks"
    
    # Try exact match first
    if gate_name in gates:
        return gates[gate_name].get(key, [])
    
    # Try without extension (e.g., from filepath)
    gate_stem = Path(gate_name).stem if "/" in gate_name or "\\" in gate_name else gate_name
    if gate_stem in gates:
        return gates[gate_stem].get(key, [])
    
    # Try prefix match: gate_id "denmark_strait" should match "denmark_strait_TPJ_pass_246"
    # This handles cases where YAML key has satellite/pass suffix but gate_id doesn't
    for yaml_key in gates.keys():
        if yaml_key.startswith(gate_stem) or gate_stem.startswith(yaml_key.split("_TPJ_")[0].split("_S3_")[0]):
            return gates[yaml_key].get(key, [])
    
    # Try partial match (gate_id is substring of yaml key)
    for yaml_key in gates.keys():
        if gate_stem in yaml_key:
            return gates[yaml_key].get(key, [])
    
    return []


# ============================================================
# UNIFIED PASS/TRACK SELECTION (used by both SLCCI and CMEMS)
# ============================================================

def _render_unified_pass_selection(config: AppConfig, dataset_type: str) -> AppConfig:
    """
    Unified pass/track selection for SLCCI and CMEMS.
    
    Pass (SLCCI) and Track (CMEMS) are the same concept - satellite ground track number.
    This function handles both with appropriate naming.
    
    Args:
        config: AppConfig to update
        dataset_type: "slcci" or "cmems"
        
    Returns:
        Updated config with pass_number (SLCCI) or cmems_track_number (CMEMS)
    """
    # Terminology
    term = "Pass" if dataset_type == "slcci" else "Track"
    term_lower = term.lower()
    
    # Get gate path (filename contains pass number, e.g., "denmark_strait_TPJ_pass_246.shp")
    gate_path = _get_gate_shapefile(config.selected_gate)
    
    # Try to extract suggested pass/track from gate FILENAME (not gate_id)
    # The filename has the pass number, e.g., "denmark_strait_TPJ_pass_246.shp"
    suggested_num = None
    suggested_satellite = None
    if gate_path:
        # Extract from filename which has the pass number
        suggested_num = _extract_pass_from_gate_name(gate_path)
        suggested_satellite = _extract_satellite_from_gate_name(gate_path)
    
    # Show suggested if found
    if suggested_num:
        sat_label = f" ({suggested_satellite})" if suggested_satellite else ""
        st.sidebar.success(f"🎯 **Suggested {term}: {suggested_num}**{sat_label}")
        st.sidebar.caption("Extracted from gate shapefile name")
    
    # Build mode options
    if dataset_type == "cmems":
        # CMEMS has "all tracks" option
        mode_options = ["all", "suggested", "closest", "manual"] if suggested_num else ["all", "closest", "manual"]
        mode_labels = {
            "all": "📊 All",
            "suggested": f"🎯 Suggested ({suggested_num})" if suggested_num else "Suggested",
            "closest": "🔍 5 Closest",
            "manual": "✏️ Manual"
        }
    else:
        # SLCCI doesn't have "all" option
        mode_options = ["suggested", "closest", "manual"] if suggested_num else ["closest", "manual"]
        mode_labels = {
            "suggested": f"🎯 Suggested ({suggested_num})" if suggested_num else "Suggested",
            "closest": "🔍 5 Closest",
            "manual": "✏️ Manual"
        }
    
    selected_mode = st.sidebar.radio(
        f"{term} Mode",
        mode_options,
        format_func=lambda x: mode_labels.get(x, x),
        horizontal=True,
        key=f"sidebar_{dataset_type}_{term_lower}_mode",
        help=f"Select how to choose the satellite {term_lower}"
    )
    
    selected_number = None
    
    if selected_mode == "all":
        # CMEMS only - use all tracks
        selected_number = None
        st.sidebar.info(f"Using ALL {term_lower}s (merged)")
        
    elif selected_mode == "suggested" and suggested_num:
        selected_number = suggested_num
        st.sidebar.info(f"Using {term_lower} **{suggested_num}** from gate definition")
        
    elif selected_mode == "closest" and gate_path:
        # Use pre-computed closest passes/tracks
        try:
            closest_list = _get_precomputed_passes(config.selected_gate, dataset_type)
            
            if closest_list:
                labels = [f"{term} {p}" for p in closest_list]
                
                selected_idx = st.sidebar.selectbox(
                    f"Select {term}",
                    range(len(closest_list)),
                    format_func=lambda i: labels[i],
                    key=f"sidebar_{dataset_type}_closest_{term_lower}",
                    help=f"Pre-computed closest {term_lower}s to gate"
                )
                
                selected_number = closest_list[selected_idx]
                st.sidebar.success(f"🎯 {term} {selected_number} selected")
            else:
                st.sidebar.warning(f"No pre-computed {term_lower}s found")
                selected_number = 248 if dataset_type == "slcci" else None
                
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            selected_number = 248 if dataset_type == "slcci" else None
            
    elif selected_mode == "manual":
        selected_number = st.sidebar.number_input(
            f"{term} Number",
            min_value=1,
            max_value=500,
            value=suggested_num if suggested_num else (248 if dataset_type == "slcci" else 100),
            key=f"sidebar_{dataset_type}_manual_{term_lower}",
            help=f"Enter a specific {term_lower} number"
        )
    
    # Store in config
    if dataset_type == "slcci":
        config.pass_number = selected_number or 248
        config.pass_mode = selected_mode
    else:
        config.cmems_track_number = selected_number
    
    return config


# ============================================================
# GATE SERVICE INITIALIZATION
# ============================================================

try:
    from src.services import GateService
    _gate_service = GateService()
    GATE_SERVICE_AVAILABLE = True
except ImportError:
    _gate_service = None
    GATE_SERVICE_AVAILABLE = False


def _render_loaded_datasets_compact() -> int:
    """Compact loaded-datasets panel for task-driven UI."""
    loaded_specs = [
        ("dataset_slcci", "🟠 SLCCI"),
        ("dataset_cmems", "🔵 CMEMS L3"),
        ("dataset_cmems_l4", "🟣 CMEMS L4"),
        ("dataset_dtu", "🟢 DTUSpace"),
    ]
    loaded = [(key, label) for key, label in loaded_specs if st.session_state.get(key) is not None]

    if not loaded:
        st.sidebar.caption("No dataset loaded yet.")
        return 0

    st.sidebar.markdown(f"**Loaded ({len(loaded)})**")
    for key, label in loaded:
        col1, col2 = st.sidebar.columns([5, 1])
        with col1:
            st.caption(label)
        with col2:
            if st.button("✖", key=f"task_remove_{key}", help=f"Remove {label}"):
                st.session_state[key] = None
                if key == "dataset_slcci":
                    st.session_state["slcci_pass_data"] = None
                st.rerun()

    if len(loaded) > 1 and st.sidebar.button("🗑️ Clear Loaded", use_container_width=True, key="task_clear_loaded"):
        st.session_state["dataset_slcci"] = None
        st.session_state["dataset_cmems"] = None
        st.session_state["dataset_cmems_l4"] = None
        st.session_state["dataset_dtu"] = None
        st.session_state["slcci_pass_data"] = None
        st.rerun()

    return len(loaded)


def _render_sidebar_task_mode() -> AppConfig:
    """
    Task-driven sidebar:
    1) Gate
    2) Dataset
    3) Configure
    4) Load
    """
    st.sidebar.title("Workflow")
    config = AppConfig()
    config.ui_mode = "New UI"

    # STEP 1
    st.sidebar.markdown("### 1. Select Gate")
    config = _render_gate_selection(config)
    st.sidebar.divider()

    # STEP 2
    st.sidebar.markdown("### 2. Select Dataset")
    dataset_options = ["CMEMS L4", "SLCCI", "CMEMS L3", "DTUSpace"]
    default_dataset = st.session_state.get("selected_dataset_type", "CMEMS L4")
    default_idx = dataset_options.index(default_dataset) if default_dataset in dataset_options else 0
    config.selected_dataset_type = st.sidebar.radio(
        "Dataset",
        dataset_options,
        index=default_idx,
        key="sidebar_datasource_task",
    )
    st.session_state["selected_dataset_type"] = config.selected_dataset_type
    st.session_state["sidebar_datasource"] = config.selected_dataset_type
    st.sidebar.caption(
        {
            "CMEMS L4": "Gridded via API, optimized for transport workflow.",
            "SLCCI": "Along-track local analysis.",
            "CMEMS L3": "Along-track merged product.",
            "DTUSpace": "Gridded local DOT product.",
        }.get(config.selected_dataset_type, "")
    )
    st.sidebar.divider()

    # STEP 3
    st.sidebar.markdown("### 3. Configure")
    if config.selected_dataset_type == "CMEMS L4":
        config = _render_cmems_l4_time_range(config)
        with st.sidebar.expander("Advanced CMEMS L4 Options", expanded=False):
            config = _render_cmems_l4_config(config)
    elif config.selected_dataset_type == "SLCCI":
        config = _render_data_paths(config)
        config = _render_unified_pass_selection(config, "slcci")
        config = _render_cycle_range(config)
        with st.sidebar.expander("Advanced SLCCI Options", expanded=False):
            config = _render_processing_params(config)
            _render_latitude_warning(config)
    elif config.selected_dataset_type == "CMEMS L3":
        config = _render_cmems_paths(config)
        config = _render_unified_pass_selection(config, "cmems")
        with st.sidebar.expander("Advanced CMEMS L3 Options", expanded=False):
            config = _render_cmems_processing_params(config)
            _render_latitude_warning(config)
    elif config.selected_dataset_type == "DTUSpace":
        config = _render_dtu_paths(config)
        config = _render_dtu_time_range(config)
    else:
        st.sidebar.warning("Unsupported dataset type.")

    st.sidebar.divider()

    # STEP 4
    st.sidebar.markdown("### 4. Load Data")
    load_labels = {
        "CMEMS L4": "🌐 Load CMEMS L4 Data",
        "SLCCI": "Load SLCCI Data",
        "CMEMS L3": "Load CMEMS L3 Data",
        "DTUSpace": "🟢 Load DTUSpace Data",
    }
    load_actions = {
        "CMEMS L4": _load_cmems_l4_data,
        "SLCCI": _load_slcci_data,
        "CMEMS L3": _load_cmems_data,
        "DTUSpace": _load_dtu_data,
    }
    if st.sidebar.button(
        load_labels.get(config.selected_dataset_type, "Load Data"),
        type="primary",
        use_container_width=True,
        key="task_load_button",
    ):
        action = load_actions.get(config.selected_dataset_type, _load_generic_data)
        action(config)

    st.sidebar.divider()
    loaded_count = _render_loaded_datasets_compact()
    if loaded_count >= 2:
        st.sidebar.success(f"Comparison ready ({loaded_count} datasets loaded).")

    with st.sidebar.expander("Advanced Utilities", expanded=False):
        _render_cache_viewer()
        st.caption("Legacy interface is still available via Interface Mode.")

    return config


def render_sidebar() -> AppConfig:
    """Sidebar dispatcher: new task-driven UI with dormant legacy fallback."""
    mode_options = ["New UI", "Legacy UI"]
    current_mode = st.session_state.get("ui_mode", "New UI")
    mode_idx = mode_options.index(current_mode) if current_mode in mode_options else 0

    selected_mode = st.sidebar.radio(
        "Interface Mode",
        mode_options,
        index=mode_idx,
        key="sidebar_interface_mode",
        help="New UI is task-driven. Legacy UI preserves the original full interface.",
    )
    st.session_state["ui_mode"] = selected_mode

    if selected_mode == "Legacy UI":
        st.sidebar.caption("Legacy interface active (dormant mode re-enabled).")
        cfg = render_sidebar_legacy()
        cfg.ui_mode = "Legacy UI"
        return cfg

    cfg = _render_sidebar_task_mode()
    cfg.ui_mode = "New UI"
    return cfg


def render_sidebar_legacy() -> AppConfig:
    """
    Render sidebar following SLCCI PLOTTER notebook workflow.
    
    Returns AppConfig with all settings.
    """
    st.sidebar.title("Settings")
    
    config = AppConfig()
    
    # === 1. GATE SELECTION ===
    st.sidebar.subheader("1. Gate Selection")
    config = _render_gate_selection(config)
    
    st.sidebar.divider()
    
    # === 2. DATA SOURCE ===
    st.sidebar.subheader("2. Data Source")
    config = _render_data_source(config)
    
    # Only show SLCCI options if SLCCI selected
    if config.selected_dataset_type == "SLCCI":
        st.sidebar.divider()
        
        # === 3. DATA PATHS ===
        st.sidebar.subheader("3. Data Paths")
        config = _render_data_paths(config)
        
        st.sidebar.divider()
        
        # === 4. PASS SELECTION ===
        st.sidebar.subheader("4. Pass Selection")
        config = _render_unified_pass_selection(config, "slcci")
        
        st.sidebar.divider()
        
        # === 5. CYCLE RANGE ===
        st.sidebar.subheader("5. Cycle Range")
        config = _render_cycle_range(config)
        
        st.sidebar.divider()
        
        # === 6. PROCESSING PARAMETERS ===
        with st.sidebar.expander("6. Processing Parameters", expanded=False):
            config = _render_processing_params(config)
        
        # === 66°N WARNING ===
        _render_latitude_warning(config)
        
        st.sidebar.divider()
        
        # === LOAD BUTTON ===
        if st.sidebar.button("Load SLCCI Data", type="primary", use_container_width=True):
            _load_slcci_data(config)
    
    elif config.selected_dataset_type == "CMEMS L3":
        # === CMEMS L3 (ALONG-TRACK) OPTIONS ===
        st.sidebar.divider()
        
        # === 3. DATA PATHS (includes source mode) ===
        st.sidebar.subheader("3. CMEMS L3 Data Paths")
        config = _render_cmems_paths(config)
        
        st.sidebar.divider()
        
        # === 4. TRACK SELECTION (same as Pass for SLCCI) ===
        st.sidebar.subheader("4. Track Selection")
        config = _render_unified_pass_selection(config, "cmems")
        
        st.sidebar.divider()
        
        # === 5. PROCESSING PARAMETERS ===
        with st.sidebar.expander("5. Processing Parameters", expanded=False):
            config = _render_cmems_processing_params(config)
        
        # === 66°N WARNING ===
        _render_latitude_warning(config)
        
        st.sidebar.divider()
        
        # === LOAD BUTTON ===
        if st.sidebar.button("Load CMEMS Data", type="primary", use_container_width=True):
            _load_cmems_data(config)
    
    elif config.selected_dataset_type == "CMEMS L4":
        # === CMEMS L4 (GRIDDED VIA API) OPTIONS ===
        st.sidebar.divider()
        
        # === 3. API CONFIGURATION ===
        st.sidebar.subheader("3. CMEMS L4 API Config")
        config = _render_cmems_l4_config(config)
        
        st.sidebar.divider()
        
        # === 4. TIME RANGE ===
        st.sidebar.subheader("4. Time Range")
        config = _render_cmems_l4_time_range(config)
        
        st.sidebar.divider()
        
        # === LOAD BUTTON ===
        if st.sidebar.button("🌐 Load CMEMS L4 Data (API)", type="primary", use_container_width=True):
            _load_cmems_l4_data(config)
    
    elif config.selected_dataset_type == "DTUSpace":
        # === DTUSpace-SPECIFIC OPTIONS (ISOLATED - GRIDDED PRODUCT) ===
        st.sidebar.divider()
        
        # === 3. DATA PATH ===
        st.sidebar.subheader("3. DTUSpace Data Path")
        config = _render_dtu_paths(config)
        
        st.sidebar.divider()
        
        # === 4. TIME RANGE ===
        st.sidebar.subheader("4. Time Range")
        config = _render_dtu_time_range(config)
        
        st.sidebar.divider()
        
        # === LOAD BUTTON ===
        if st.sidebar.button("🟢 Load DTUSpace Data", type="primary", use_container_width=True):
            _load_dtu_data(config)
    
    else:
        # Fallback for unknown types
        st.sidebar.divider()
        if st.sidebar.button("Load Data", type="primary", use_container_width=True):
            _load_generic_data(config)
    
    return config


def _render_gate_selection(config: AppConfig) -> AppConfig:
    """Render region filter and gate selector."""
    
    if not GATE_SERVICE_AVAILABLE:
        st.sidebar.warning("Gate service not available")
        return config
    
    # Region filter
    regions = _gate_service.get_regions()
    selected_region = st.sidebar.selectbox(
        "Region",
        ["All Regions"] + regions,
        key="main_sidebar_region_v2"
    )
    
    # Get gates for selected region
    if selected_region == "All Regions":
        gates = _gate_service.list_gates()
    else:
        gates = _gate_service.list_gates_by_region(selected_region)
    
    # Gate selector
    gate_options = ["None (Global)"] + [g.name for g in gates]
    gate_ids = [None] + [g.id for g in gates]
    
    # Sync with globe selection (selected_gate)
    globe_selected = st.session_state.get("selected_gate")
    default_idx = 0
    if globe_selected and globe_selected in gate_ids:
        default_idx = gate_ids.index(globe_selected)
    
    selected_idx = st.sidebar.selectbox(
        "Gate",
        range(len(gate_options)),
        index=default_idx,
        format_func=lambda i: gate_options[i],
        key="main_sidebar_gate_v2"
    )
    
    config.selected_gate = gate_ids[selected_idx]
    st.session_state["selected_gate"] = config.selected_gate  # Sync back to globe
    
    # Show gate info
    if config.selected_gate:
        gate = _gate_service.get_gate(config.selected_gate)
        if gate:
            st.sidebar.caption(f"{gate.region} - {gate.description}")
    
    # Buffer
    config.gate_buffer_km = st.sidebar.slider(
        "Buffer (km)", 10, 200, 50, 10,
        key="sidebar_buffer"
    )
    
    return config


def _render_data_source(config: AppConfig) -> AppConfig:
    """Render data source selector with comparison mode option."""
    
    # Main dataset selector - 4 datasets
    # Along-track: SLCCI, CMEMS L3 (both have pass/track selection)
    # Gridded: CMEMS L4 (API), DTUSpace (local)
    config.selected_dataset_type = st.sidebar.radio(
        "Dataset",
        ["SLCCI", "CMEMS L3", "CMEMS L4", "DTUSpace"],
        horizontal=True,
        key="sidebar_datasource",
        help=(
            "SLCCI = ESA Sea Level CCI (along-track, pass selection)\n"
            "CMEMS L3 = Copernicus L3 1Hz (along-track, track selection)\n"
            "CMEMS L4 = Copernicus L4 Gridded (API download)\n"
            "DTUSpace = DTUSpace v4 Gridded (local file)"
        )
    )
    
    # Store in session state for tabs
    st.session_state["selected_dataset_type"] = config.selected_dataset_type
    
    # Show dataset type info
    if config.selected_dataset_type in ["SLCCI", "CMEMS L3"]:
        st.sidebar.caption("📡 **Along-track** - Pass/Track selection available")
    else:
        st.sidebar.caption("🗺️ **Gridded** - No pass selection (synthetic gate sampling)")
    
    # === LOADED DATASETS PANEL ===
    st.sidebar.divider()
    
    # Check all 4 datasets
    slcci_loaded = st.session_state.get("dataset_slcci") is not None
    cmems_loaded = st.session_state.get("dataset_cmems") is not None
    cmems_l4_loaded = st.session_state.get("dataset_cmems_l4") is not None
    dtu_loaded = st.session_state.get("dataset_dtu") is not None
    
    loaded_count = sum([slcci_loaded, cmems_loaded, cmems_l4_loaded, dtu_loaded])
    
    if loaded_count > 0:
        st.sidebar.markdown(f"**📊 Loaded Datasets ({loaded_count}/4):**")
        
        # Dataset info with remove buttons
        # SLCCI
        if slcci_loaded:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                slcci_data = st.session_state.get("dataset_slcci")
                pass_num = getattr(slcci_data, 'pass_number', '?')
                st.markdown(f"🟠 **SLCCI** Pass {pass_num}")
            with col2:
                if st.button("✖", key="remove_slcci", help="Remove SLCCI data"):
                    st.session_state["dataset_slcci"] = None
                    st.session_state["slcci_pass_data"] = None
                    st.rerun()
        
        # CMEMS L3
        if cmems_loaded:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                cmems_data = st.session_state.get("dataset_cmems")
                pass_num = getattr(cmems_data, 'pass_number', None)
                pass_str = f"Track {pass_num}" if pass_num else "Synthetic"
                st.markdown(f"🔵 **CMEMS L3** {pass_str}")
            with col2:
                if st.button("✖", key="remove_cmems", help="Remove CMEMS L3 data"):
                    st.session_state["dataset_cmems"] = None
                    st.rerun()
        
        # CMEMS L4
        if cmems_l4_loaded:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                st.markdown("🟣 **CMEMS L4** Gridded")
            with col2:
                if st.button("✖", key="remove_cmems_l4", help="Remove CMEMS L4 data"):
                    st.session_state["dataset_cmems_l4"] = None
                    st.rerun()
        
        # DTUSpace
        if dtu_loaded:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                st.markdown("🟢 **DTUSpace** Gridded")
            with col2:
                if st.button("✖", key="remove_dtu", help="Remove DTUSpace data"):
                    st.session_state["dataset_dtu"] = None
                    st.rerun()
        
        # Clear all button
        if loaded_count > 1:
            if st.sidebar.button("🗑️ Clear All Data", use_container_width=True, type="secondary"):
                st.session_state["dataset_slcci"] = None
                st.session_state["dataset_cmems"] = None
                st.session_state["dataset_cmems_l4"] = None
                st.session_state["dataset_dtu"] = None
                st.session_state["slcci_pass_data"] = None
                st.rerun()
        
        # Comparison mode info
        if loaded_count >= 2:
            st.sidebar.success(f"✅ {loaded_count} datasets loaded - Comparison tabs available!")
    
    # === CACHE VIEWER ===
    st.sidebar.divider()
    with st.sidebar.expander("💾 **Cache Manager**", expanded=False):
        _render_cache_viewer()
    
    # For SLCCI, add LOCAL/API selector
    if config.selected_dataset_type == "SLCCI":
        config.data_source_mode = st.sidebar.radio(
            "Source Mode",
            ["local", "api"],
            format_func=lambda x: "📁 LOCAL Files" if x == "local" else "🌐 CEDA API",
            horizontal=True,
            key="sidebar_source_mode",
            help="LOCAL=NetCDF files on disk, API=CEDA OPeNDAP"
        )
        
        if config.data_source_mode == "api":
            st.sidebar.caption("⚡ Faster downloads with bbox filtering")
    
    # For CMEMS L3, show info
    elif config.selected_dataset_type == "CMEMS L3":
        st.sidebar.info(
            "📡 **CMEMS L3 Along-Track**\n"
            "[Dataset Info](https://doi.org/10.48670/moi-00149)"
        )
    
    # For CMEMS L4, show API info
    elif config.selected_dataset_type == "CMEMS L4":
        st.sidebar.info(
            "🌐 **CMEMS L4 Gridded** (via API)\n"
            "[Dataset Info](https://doi.org/10.48670/moi-00148)\n"
            "Requires `copernicusmarine` credentials"
        )
    
    return config


def _render_data_paths(config: AppConfig) -> AppConfig:
    """Render SLCCI data path inputs."""
    
    config.slcci_base_dir = st.sidebar.text_input(
        "SLCCI Data Directory",
        value="/Users/nicolocaron/Desktop/ARCFRESH/J2",
        key="sidebar_slcci_dir",
        help="Folder containing SLCCI_ALTDB_J2_CycleXXX_V2.nc files"
    )
    
    config.slcci_geoid_path = st.sidebar.text_input(
        "Geoid File",
        value="/Users/nicolocaron/Desktop/ARCFRESH/TUM_ogmoc.nc",
        key="sidebar_geoid",
        help="TUM_ogmoc.nc for DOT calculation"
    )
    
    return config


def _render_cycle_range(config: AppConfig) -> AppConfig:
    """Render cycle range inputs."""
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        config.cycle_start = st.number_input(
            "Start",
            min_value=1,
            max_value=300,
            value=1,
            key="sidebar_cycle_start"
        )
    
    with col2:
        config.cycle_end = st.number_input(
            "End",
            min_value=1,
            max_value=300,
            value=10,  # Default 10 for fast testing
            key="sidebar_cycle_end"
        )
    
    # Show info
    n_cycles = config.cycle_end - config.cycle_start + 1
    st.sidebar.caption(f"📊 {n_cycles} cycles selected")
    
    return config


def _render_processing_params(config: AppConfig) -> AppConfig:
    """Render advanced processing parameters from SLCCI PLOTTER."""
    
    # Quality flag filter
    config.use_flag = st.checkbox(
        "Use Quality Flags",
        value=True,
        key="sidebar_use_flag",
        help="Filter data using SLCCI quality flags"
    )
    
    # Longitude binning size - UNIFIED for all outputs (slope + profiles)
    # Default 0.1° (~11km bins), range 0.01° - 0.5°
    config.lon_bin_size = st.slider(
        "🎚️ Lon Bin Size (°)",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        format="%.2f",
        key="sidebar_lon_bin",
        help="Unified binning for slope & profiles. Default 0.1° (~11km). Lower=finer but noisier."
    )
    
    # Show approximate km resolution
    km_approx = config.lon_bin_size * 111.0  # rough conversion at equator
    st.caption(f"≈ {km_approx:.1f} km bins (varies with latitude)")
    
    # Cache Management Section
    st.markdown("---")
    st.markdown("**🗄️ Cache Management**")
    
    # Import SLCCI cache to show stats
    try:
        # Get current service instance from session state if exists
        if "slcci_service" in st.session_state and st.session_state.slcci_service is not None:
            cache_stats = st.session_state.slcci_service.get_cache_stats()
            st.caption(
                f"📊 Raw: {cache_stats.get('raw_entries', 0)} | "
                f"Processed: {cache_stats.get('processed_entries', 0)} | "
                f"Hits: {cache_stats.get('hits', 0)}"
            )
    except Exception:
        pass
    
    # Clear Cache button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Cache", key="clear_slcci_cache", use_container_width=True):
            if "slcci_service" in st.session_state and st.session_state.slcci_service is not None:
                st.session_state.slcci_service.clear_cache()
                st.success("✅ Cache cleared!")
            else:
                st.info("No cache to clear")
    with col2:
        # Force reload checkbox
        config.force_reload = st.checkbox(
            "Force Reload",
            value=False,
            key="sidebar_force_reload",
            help="Bypass cache and reload from source"
        )
    
    return config


# ============================================================
# CMEMS-SPECIFIC FUNCTIONS
# ============================================================

def _render_cmems_paths(config: AppConfig) -> AppConfig:
    """Render CMEMS data path inputs and source mode."""
    
    # Source mode (like SLCCI)
    config.cmems_source_mode = st.sidebar.radio(
        "Source Mode",
        ["local", "api"],
        format_func=lambda x: "📁 LOCAL Files" if x == "local" else "🌐 CMEMS API",
        horizontal=True,
        key="sidebar_cmems_source_mode",
        help="LOCAL=NetCDF files on disk (requires track selection), API=Copernicus Marine (smart geographic filter)"
    )
    
    if config.cmems_source_mode == "api":
        st.sidebar.success(
            "🌐 **API Mode** - Uses L4 Gridded!\n"
            "- 0.125° resolution (all altimeters merged)\n"
            "- Geographic bounding box filter\n"
            "- Requires: `copernicusmarine login`"
        )
    else:
        st.sidebar.warning(
            "📁 **Local Mode** - L3 Along-Track\n"
            "- Must select a specific track\n"
            "- Cannot load 'All Tracks' (7000+ files)"
        )
    
    # Only show path input for local mode
    if config.cmems_source_mode == "local":
        config.cmems_base_dir = st.sidebar.text_input(
            "CMEMS Data Directory",
            value="/Users/nicolocaron/Desktop/ARCFRESH/COPERNICUS DATA",
            key="sidebar_cmems_dir",
            help="Folder containing J1_netcdf/, J2_netcdf/, J3_netcdf/ subfolders"
        )
        
        # Show available files info using CMEMSService
        from pathlib import Path
        try:
            from src.services.cmems_service import CMEMSService, CMEMSConfig
            temp_config = CMEMSConfig(base_dir=config.cmems_base_dir)
            temp_service = CMEMSService(temp_config)
            file_counts = temp_service.count_files()
            date_info = temp_service.get_date_range()
            
            if file_counts["total"] > 0:
                st.sidebar.caption(f"📁 {file_counts['total']:,} NetCDF files found")
                # Show per-satellite breakdown
                sat_info = " | ".join([f"{k}: {v}" for k, v in file_counts.items() if k != "total"])
                st.sidebar.caption(f"   {sat_info}")
                if date_info["years"]:
                    st.sidebar.caption(f"📅 Years: {date_info['years'][0]} - {date_info['years'][-1]}")
            else:
                st.sidebar.warning("⚠️ 0 NetCDF files found")
        except Exception as e:
            cmems_path = Path(config.cmems_base_dir)
            if not cmems_path.exists():
                st.sidebar.warning("⚠️ Directory not found")
            else:
                st.sidebar.warning(f"⚠️ Error checking files: {e}")
    else:
        # API mode: use default path (won't be used anyway)
        config.cmems_base_dir = "/tmp/cmems_api_cache"
    
    # Show pass number from gate filename (if available)
    gate_path = _get_gate_shapefile(config.selected_gate)
    if gate_path:
        try:
            from src.services.cmems_service import _extract_pass_from_gate_name
            strait_name, pass_number = _extract_pass_from_gate_name(gate_path)
            
            if pass_number is not None:
                st.sidebar.success(f"🎯 **Pass {pass_number}** found in gate filename")
                st.sidebar.caption(f"Gate: {strait_name}")
            else:
                st.sidebar.info("ℹ️ No pass number in gate filename (synthetic pass)")
                st.sidebar.caption(f"Gate: {strait_name}")
        except Exception as e:
            pass  # Silently continue if extraction fails
    
    return config


def _render_cmems_date_range(config: AppConfig) -> AppConfig:
    """
    CMEMS date range - REMOVED.
    We load ALL data from the folder, no date filtering.
    """
    st.sidebar.info("📅 Loading ALL data from folder (no date filter)")
    return config


def _render_cmems_params(config: AppConfig) -> AppConfig:
    """Render CMEMS-specific processing parameters."""
    
    # Track selection (equivalent to SLCCI pass)
    st.sidebar.markdown("### 🛤️ Track Selection")
    
    # Get gate path for track discovery (filename has track number)
    gate_path = _get_gate_shapefile(config.selected_gate)
    
    # Try to extract suggested track from gate FILENAME (not gate_id)
    suggested_track = None
    if gate_path:
        suggested_track = _extract_pass_from_gate_name(gate_path)
    
    # Show suggested track if found
    if suggested_track:
        st.sidebar.success(f"🎯 **Suggested Track: {suggested_track}**")
        st.sidebar.caption("Extracted from gate shapefile name")
    
    # Get source mode
    source_mode = getattr(config, 'cmems_source_mode', 'local')
    
    # Track selection mode - depends on source
    if source_mode == "local":
        # Local mode (L3): must select a specific track (no "all")
        track_options = ["suggested", "closest", "manual"] if suggested_track else ["closest", "manual"]
        st.sidebar.caption("⚠️ Local mode requires track selection (7000+ files)")
    else:
        # API mode (L4): only "all" makes sense (L4 is gridded, no tracks)
        st.sidebar.info("ℹ️ API mode uses L4 gridded data (no track filtering)")
        config.cmems_track_number = None  # L4 doesn't have tracks
        return config  # Skip track selection UI
    
    track_mode = st.sidebar.radio(
        "Track Mode",
        track_options,
        format_func=lambda x: {
            "suggested": f"🎯 Suggested ({suggested_track})" if suggested_track else "Suggested",
            "closest": "🔍 5 Closest",
            "manual": "✏️ Manual"
        }.get(x, x),
        horizontal=True,
        key="sidebar_cmems_track_mode",
        help="suggested=from gate name, closest=5 nearest to gate, manual=enter number"
    )
    
    if track_mode == "suggested" and suggested_track:
        config.cmems_track_number = suggested_track
        st.sidebar.info(f"Using track **{suggested_track}** from gate definition")
        
    elif track_mode == "closest" and gate_path:
        # Use pre-computed closest tracks from config/gate_passes.yaml
        try:
            closest_tracks = _get_precomputed_passes(config.selected_gate, "cmems")
            
            if closest_tracks:
                track_options = closest_tracks
                track_labels = [f"Track {t}" for t in closest_tracks]
                
                selected_idx = st.sidebar.selectbox(
                    "Select Track",
                    range(len(track_options)),
                    format_func=lambda i: track_labels[i],
                    key="sidebar_cmems_closest_track",
                    help="Pre-computed closest tracks to gate"
                )
                
                config.cmems_track_number = track_options[selected_idx]
                st.sidebar.success(f"🎯 Track {config.cmems_track_number} selected")
            else:
                # Fallback: compute on-the-fly (slower)
                st.sidebar.warning("No pre-computed tracks, computing...")
                from src.services.cmems_service import CMEMSService, CMEMSConfig
                temp_config = CMEMSConfig(
                    base_dir=config.cmems_base_dir,
                    source_mode=config.cmems_source_mode
                )
                temp_service = CMEMSService(temp_config)
                
                cache_key = f"cmems_closest_{config.selected_gate}"
                if cache_key not in st.session_state:
                    with st.spinner("Finding closest tracks..."):
                        st.session_state[cache_key] = temp_service.find_closest_tracks(gate_path, n_tracks=5)
                
                computed_tracks = st.session_state[cache_key]
                if computed_tracks:
                    track_options = [t[0] for t in computed_tracks]
                    track_labels = [f"Track {t[0]} ({t[1]:.1f} km)" for t in computed_tracks]
                    
                    selected_idx = st.sidebar.selectbox(
                        "Select Track",
                        range(len(track_options)),
                        format_func=lambda i: track_labels[i],
                        key="sidebar_cmems_closest_track_computed",
                        help="Computed closest tracks to gate line"
                    )
                    config.cmems_track_number = track_options[selected_idx]
                else:
                    st.sidebar.warning("No tracks found")
                    config.cmems_track_number = None
        except Exception as e:
            st.sidebar.error(f"Error finding tracks: {e}")
            config.cmems_track_number = None
            
    elif track_mode == "manual":
        config.cmems_track_number = st.sidebar.number_input(
            "Track Number",
            min_value=1,
            max_value=500,
            value=suggested_track if suggested_track else 100,
            key="sidebar_cmems_manual_track",
            help="Enter a specific track number"
        )
        st.sidebar.info(f"Using manual track **{config.cmems_track_number}**")
    
    # Performance options (collapsible)
    with st.expander("⚡ Performance", expanded=False):
        # Parallel processing toggle
        config.cmems_use_parallel = st.checkbox(
            "Use Parallel Loading",
            value=True,
            key="sidebar_cmems_parallel",
            help="Load files in parallel (faster for large datasets)"
        )
        
        # Cache toggle
        config.cmems_use_cache = st.checkbox(
            "Use Cache",
            value=True,
            key="sidebar_cmems_cache",
            help="Cache processed data (instant reload on second run)"
        )
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", key="sidebar_clear_cmems_cache"):
            try:
                from src.services.cmems_service import CACHE_DIR
                import shutil
                if CACHE_DIR.exists():
                    shutil.rmtree(CACHE_DIR)
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    st.success("Cache cleared!")
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")
    
    # Longitude binning size - SLIDER da 0.05 a 0.50 (lower res than SLCCI)
    config.cmems_lon_bin_size = st.slider(
        "Lon Bin Size (°)",
        min_value=0.05,
        max_value=0.50,
        value=0.10,
        step=0.05,
        format="%.2f",
        key="sidebar_cmems_lon_bin",
        help="Binning resolution for CMEMS (0.05° - 0.50°, coarser than SLCCI)"
    )
    
    # Buffer around gate - default 5.0° (from Copernicus notebook)
    config.cmems_buffer_deg = st.slider(
        "Gate Buffer (°)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,  # Changed from 0.5 to 5.0 as per Copernicus notebook
        step=0.5,
        format="%.1f",
        key="sidebar_cmems_buffer",
        help="Buffer around gate for data extraction (default 5.0° from notebook)"
    )
    
    return config


def _render_cmems_processing_params(config: AppConfig) -> AppConfig:
    """
    Render CMEMS processing parameters (without Track Selection which is now separate).
    
    This is a slimmed down version of _render_cmems_params for the refactored sidebar.
    """
    # Performance options (collapsible)
    with st.expander("⚡ Performance", expanded=False):
        # Parallel processing toggle
        config.cmems_use_parallel = st.checkbox(
            "Use Parallel Loading",
            value=True,
            key="sidebar_cmems_parallel_new",
            help="Load files in parallel (faster for large datasets)"
        )
        
        # Cache toggle
        config.cmems_use_cache = st.checkbox(
            "Use Cache",
            value=True,
            key="sidebar_cmems_cache_new",
            help="Cache processed data (instant reload on second run)"
        )
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", key="sidebar_clear_cmems_cache_new"):
            try:
                from src.services.cmems_service import CACHE_DIR
                import shutil
                if CACHE_DIR.exists():
                    shutil.rmtree(CACHE_DIR)
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    st.success("Cache cleared!")
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")
    
    # Longitude binning size - SLIDER da 0.05 a 0.50 (lower res than SLCCI)
    config.cmems_lon_bin_size = st.slider(
        "Lon Bin Size (°)",
        min_value=0.05,
        max_value=0.50,
        value=0.10,
        step=0.05,
        format="%.2f",
        key="sidebar_cmems_lon_bin_new",
        help="Binning resolution for CMEMS (0.05° - 0.50°, coarser than SLCCI)"
    )
    
    # Buffer around gate - default 5.0° (from Copernicus notebook)
    config.cmems_buffer_deg = st.slider(
        "Gate Buffer (°)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        format="%.1f",
        key="sidebar_cmems_buffer_new",
        help="Buffer around gate for data extraction (default 5.0° from notebook)"
    )
    
    return config


def _render_latitude_warning(config: AppConfig) -> AppConfig:
    """
    Show warning if gate is above 66°N (Jason satellite coverage limit).
    Applies to both SLCCI and CMEMS.
    """
    if not GATE_SERVICE_AVAILABLE or not config.selected_gate:
        return config
    
    gate = _gate_service.get_gate(config.selected_gate)
    if not gate:
        return config
    
    # Try to get gate latitude from shapefile
    gate_path = _get_gate_shapefile(config.selected_gate)
    if gate_path:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(gate_path)
            max_lat = gdf.geometry.bounds['maxy'].max()
            
            if max_lat > 66.0:
                st.sidebar.warning(f"""
                ⚠️ **Latitude Warning**
                
                Gate extends to {max_lat:.2f}°N.
                
                Jason satellites (J1/J2/J3) coverage is limited to ±66°.
                Data beyond 66°N may be sparse or unavailable.
                """)
        except Exception:
            pass
    
    return config


def _load_slcci_data(config: AppConfig):
    """Load SLCCI data using SLCCIService (local or API) with intelligent cache."""
    
    # Validate geoid path (always needed)
    if not Path(config.slcci_geoid_path).exists():
        st.sidebar.error(f"❌ Geoid not found: {config.slcci_geoid_path}")
        return
    
    # Validate local paths if using local source
    source_mode = getattr(config, 'data_source_mode', 'local')
    if source_mode == "local" and not Path(config.slcci_base_dir).exists():
        st.sidebar.error(f"❌ Path not found: {config.slcci_base_dir}")
        return
    
    # Get gate shapefile
    gate_path = _get_gate_shapefile(config.selected_gate)
    if not gate_path:
        st.sidebar.error("❌ Select a gate first")
        return
    
    try:
        from src.services.slcci_service import SLCCIService, SLCCIConfig, CacheConfig
        
        cycles = list(range(config.cycle_start, config.cycle_end + 1))
        
        # Determine pass number first (needed for cache key)
        pass_number = config.pass_number
        
        # Get bin size from config (set via slider)
        bin_size = getattr(config, 'lon_bin_size', 0.10)
        force_reload = getattr(config, 'force_reload', False)
        
        slcci_config = SLCCIConfig(
            base_dir=config.slcci_base_dir,
            geoid_path=config.slcci_geoid_path,
            cycles=cycles,
            use_flag=config.use_flag,
            lat_buffer_deg=config.lat_buffer_deg,
            lon_buffer_deg=config.lon_buffer_deg,
            lon_bin_size=bin_size,  # UNIFIED bin size from slider
            source=source_mode,  # "local" or "api"
            satellite="J2",
        )
        
        # Use existing service from session state if available (preserves cache)
        # Recreate if config changed significantly
        existing_service = st.session_state.get("slcci_service")
        if existing_service is not None:
            # Check if we need to recreate service (source or paths changed)
            old_config = st.session_state.get("slcci_config")
            if (old_config is None or 
                old_config.slcci_base_dir != config.slcci_base_dir or
                old_config.data_source_mode != source_mode):
                service = SLCCIService(slcci_config)
            else:
                # Reuse service but update config (preserves cache!)
                service = existing_service
                service.config = slcci_config  # Update with new bin_size etc.
        else:
            service = SLCCIService(slcci_config)
        
        # Auto-find pass if needed
        if config.pass_mode == "auto":
            st.sidebar.info("🔍 Finding closest pass...")
            closest = service.find_closest_pass(gate_path, n_passes=1)
            if closest:
                pass_number = closest[0][0]
                st.sidebar.success(f"Found pass {pass_number}")
            else:
                st.sidebar.error("No passes found near gate")
                return
        
        # Load data using service's intelligent cache
        with st.spinner(f"Loading {len(cycles)} cycles (bin={bin_size}°)..."):
            pass_data = service.load_pass_data(
                gate_path=gate_path,
                pass_number=pass_number,
                cycles=cycles,
                force_reload=force_reload,  # Bypass cache if requested
            )
            
            if pass_data is None:
                st.sidebar.error(f"❌ No data for pass {pass_number}")
                return
        
        # Apply longitude filter for divided gates (Fram West/East, Davis West/East)
        lon_min, lon_max = _get_lon_filter_for_gate(config.selected_gate)
        if lon_min is not None or lon_max is not None:
            pass_data = apply_longitude_filter(pass_data, lon_min, lon_max, config.selected_gate)
            if pass_data is None:
                st.sidebar.error(f"❌ No data after longitude filter ({lon_min}° to {lon_max}°)")
                return
            filter_info = f" [lon: {lon_min or '-∞'}° to {lon_max or '+∞'}°]"
        else:
            filter_info = ""
        
        # Store in session state using dedicated function
        store_slcci_data(pass_data)
        st.session_state["slcci_service"] = service
        st.session_state["slcci_config"] = config
        st.session_state["datasets"] = {}  # Clear generic
        
        # Success message with cache stats
        n_obs = len(pass_data.df) if hasattr(pass_data, 'df') else 0
        n_cyc = pass_data.df['cycle'].nunique() if hasattr(pass_data, 'df') and 'cycle' in pass_data.df.columns else 0
        cache_stats = service.get_cache_stats()
        
        st.sidebar.success(f"""
        ✅ SLCCI Data Loaded!{filter_info}
        - Pass: {pass_number}
        - Observations: {n_obs:,}
        - Cycles: {n_cyc}
        - Bin size: {bin_size}°
        - Cache: {cache_stats.get('hits', 0)} hits, {cache_stats.get('misses', 0)} misses
        """)
        
        st.rerun()
            
    except ImportError as e:
        st.sidebar.error(f"❌ Service not available: {e}")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")
        import traceback
        with st.sidebar.expander("Traceback"):
            st.code(traceback.format_exc())


def _load_generic_data(config: AppConfig):
    """Load data from ERA5 or other APIs."""
    
    if not config.selected_gate:
        st.sidebar.warning("Select a gate first")
        return
    
    st.sidebar.info(f"Loading {config.selected_dataset_type}... (not implemented yet)")


def _load_cmems_data(config: AppConfig):
    """Load CMEMS L3 1Hz data using CMEMSService."""
    from pathlib import Path
    
    # For API mode, no path validation needed
    if getattr(config, 'cmems_source_mode', 'local') == 'local':
        # Validate CMEMS path
        cmems_path = Path(config.cmems_base_dir)
        if not cmems_path.exists():
            st.sidebar.error(f"❌ Path not found: {config.cmems_base_dir}")
            return
    else:
        cmems_path = Path(config.cmems_base_dir)
    
    # Get gate shapefile
    gate_path = _get_gate_shapefile(config.selected_gate)
    if not gate_path:
        st.sidebar.error("❌ Select a gate first")
        return
    
    try:
        from src.services.cmems_service import CMEMSService, CMEMSConfig
        
        cmems_config = CMEMSConfig(
            base_dir=str(cmems_path),
            source_mode=getattr(config, 'cmems_source_mode', 'local'),
            lon_bin_size=getattr(config, 'cmems_lon_bin_size', 0.1),
            buffer_deg=getattr(config, 'cmems_buffer_deg', 5.0),
            # Track filtering (like SLCCI pass)
            track_number=getattr(config, 'cmems_track_number', None),
            # Performance options
            use_parallel=getattr(config, 'cmems_use_parallel', True),
            use_cache=getattr(config, 'cmems_use_cache', True),
        )
        
        service = CMEMSService(cmems_config)
        
        # Show info based on source mode
        if cmems_config.source_mode == "api":
            st.sidebar.info("🌐 Connecting to CMEMS API...")
        else:
            # Show file count
            file_counts = service.count_files()
            total_files = file_counts['total']
            
            # Show performance info
            perf_info = []
            if cmems_config.use_cache:
                perf_info.append("📦 Cache ON")
            if cmems_config.use_parallel:
                perf_info.append("🚀 Parallel ON")
            if cmems_config.track_number:
                perf_info.append(f"🛤️ Track {cmems_config.track_number}")
            perf_str = " | ".join(perf_info) if perf_info else ""
            
            st.sidebar.info(f"📁 Found {total_files:,} files to process... {perf_str}")
        
        # Create progress bar
        progress_bar = st.sidebar.progress(0, text="Preparing...")
        status_text = st.sidebar.empty()
        
        def update_progress(processed: int, total: int):
            """Callback to update progress bar."""
            pct = processed / total if total > 0 else 0
            progress_bar.progress(pct, text=f"Processing: {processed:,}/{total:,} files ({pct*100:.0f}%)")
        
        # Check gate coverage
        coverage_info = service.check_gate_coverage(gate_path)
        if coverage_info.get("warning"):
            st.sidebar.warning(f"⚠️ {coverage_info['warning']}")
        
        status_text.text("Loading CMEMS data... (this may take several minutes)")
        
        # Load pass data with progress callback
        pass_data = service.load_pass_data(
            gate_path=gate_path, 
            progress_callback=update_progress
        )
        
        # Clear progress bar
        progress_bar.empty()
        status_text.empty()
        
        if pass_data is None:
            st.sidebar.error("❌ No data found for this gate")
            return
        
        # Store in session state using dedicated function
        store_cmems_data(pass_data)
        st.session_state["cmems_service"] = service
        st.session_state["cmems_config"] = config
        st.session_state["datasets"] = {}  # Clear generic
        
        # Success message
        n_obs = len(pass_data.df) if hasattr(pass_data, 'df') else 0
        n_months = pass_data.df['year_month'].nunique() if hasattr(pass_data, 'df') and 'year_month' in pass_data.df.columns else 0
        time_range = ""
        if hasattr(pass_data, 'df') and 'time' in pass_data.df.columns:
            time_range = f"\n- Period: {pass_data.df['time'].min().strftime('%Y-%m')} → {pass_data.df['time'].max().strftime('%Y-%m')}"
        
        # Show pass number properly
        pass_num = pass_data.pass_number
        pass_display = f"Pass {pass_num}" if pass_num else "Synthetic pass"
        
        st.sidebar.success(f"""
        ✅ CMEMS Data Loaded!
        - Gate: {pass_data.strait_name}
        - {pass_display}
        - Observations: {n_obs:,}
        - Monthly periods: {n_months}{time_range}
        - Satellites: J1+J2+J3 merged
        """)
        
        st.rerun()
            
    except ImportError as e:
        st.sidebar.error(f"❌ CMEMSService not available: {e}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading CMEMS: {e}")
        import traceback
        with st.sidebar.expander("Traceback"):
            st.code(traceback.format_exc())


def _render_cmems_l4_config(config: AppConfig) -> AppConfig:
    """Render CMEMS L4 API configuration."""
    
    st.sidebar.info(
        "🌐 **CMEMS L4 Gridded**\n"
        "Data downloaded via Copernicus Marine API"
    )
    
    # Check if copernicusmarine is available
    try:
        import copernicusmarine
        st.sidebar.success("✅ copernicusmarine installed")
        api_available = True
    except ImportError:
        st.sidebar.error(
            "❌ `copernicusmarine` not installed.\n"
            "Run: `pip install copernicusmarine`"
        )
        api_available = False
    
    # API credentials check
    if api_available:
        with st.sidebar.expander("🔐 API Credentials", expanded=False):
            st.markdown("""
            **First time setup:**
            1. Create free account at [marine.copernicus.eu](https://marine.copernicus.eu)
            2. Run in terminal: `copernicusmarine login`
            3. Enter your credentials
            
            Credentials are stored in `~/.copernicusmarine/`
            """)
    
    # Variables to download
    config.cmems_l4_variables = st.sidebar.multiselect(
        "Variables",
        ["adt", "sla", "ugos", "vgos", "err_ugosa", "err_vgosa", "ugosa", "vgosa", "flag_ice"],
        default=["adt", "sla", "ugos", "vgos", "err_ugosa", "err_vgosa"],
        key="sidebar_cmems_l4_vars",
        help="ADT=Absolute Dynamic Topography, SLA=Sea Level Anomaly, ugos/vgos=geostrophic velocities, err_ugosa/err_vgosa=velocity uncertainty, flag_ice=ice mask"
    )
    
    # Ice filter checkbox
    config.cmems_l4_filter_ice = st.sidebar.checkbox(
        "🧊 Filter Ice-Covered Data",
        value=config.cmems_l4_filter_ice,
        key="sidebar_cmems_l4_filter_ice",
        help="Use flag_ice to mask out ice-covered observations. Automatically adds flag_ice to variables if not selected."
    )
    
    # Buffer around gate
    config.cmems_l4_buffer = st.sidebar.slider(
        "Spatial Buffer (deg)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        key="sidebar_cmems_l4_buffer",
        help="Buffer around gate for data download"
    )
    
    # Dataset info
    with st.sidebar.expander("ℹ️ About CMEMS L4", expanded=False):
        st.markdown("""
        **CMEMS L4 Gridded SSH**
        - Product: SEALEVEL_GLO_PHY_L4_MY_008_047
        - Resolution: 0.125° (~14km) daily
        - [DOI: 10.48670/moi-00148](https://doi.org/10.48670/moi-00148)
        
        **Description:**
        Gridded Sea Level Anomalies (SLA) computed with Optimal Interpolation,
        merging L3 along-track measurements from multiple altimeter missions.
        Processed by DUACS multimission system.
        
        **Variables:**
        - `adt`: Absolute Dynamic Topography (m)
        - `sla`: Sea Level Anomaly (m)
        - `ugos/vgos`: Geostrophic velocities (m/s)
        """)
    
    return config


def _render_cmems_l4_time_range(config: AppConfig) -> AppConfig:
    """Render CMEMS L4 time range selector."""
    from datetime import date
    today = date.today()
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        config.cmems_l4_start = st.date_input(
            "Start Date",
            value=date(2010, 1, 1),
            min_value=date(1993, 1, 1),
            max_value=today,
            key="sidebar_cmems_l4_start"
        )
    
    with col2:
        config.cmems_l4_end = st.date_input(
            "End Date",
            value=min(date(2020, 12, 31), today),
            min_value=date(1993, 1, 1),
            max_value=today,
            key="sidebar_cmems_l4_end"
        )
    
    # Validate
    days = (config.cmems_l4_end - config.cmems_l4_start).days
    if config.cmems_l4_start >= config.cmems_l4_end:
        st.sidebar.error("❌ Start date must be before end date")
    else:
        st.sidebar.caption(f"📅 Period: {config.cmems_l4_start} to {config.cmems_l4_end} ({days} days)")
    
    # Warn about large downloads
    if days > 3650:  # ~10 years
        st.sidebar.warning("⚠️ Large time range - download may take several minutes")
    
    return config


def _load_cmems_l4_data(config: AppConfig):
    """Load CMEMS L4 data via API for the selected gate with cache support."""
    
    # Check copernicusmarine
    try:
        import copernicusmarine
    except ImportError:
        st.sidebar.error("❌ `copernicusmarine` not installed. Run: `pip install copernicusmarine`")
        return
    
    # Get gate path
    gate_path = _get_gate_shapefile(config.selected_gate)
    if not gate_path:
        st.sidebar.error("❌ No gate selected. Select a gate first.")
        return
    
    # Validate time range
    if config.cmems_l4_start >= config.cmems_l4_end:
        st.sidebar.error("❌ Invalid time range")
        return
    
    # Prepare variables - auto-add flag_ice if filter is enabled
    variables = list(config.cmems_l4_variables)
    if config.cmems_l4_filter_ice and "flag_ice" not in variables:
        variables.append("flag_ice")
    
    try:
        from src.services.cmems_l4_service import CMEMSL4Service, CMEMSL4Config
        
        service = CMEMSL4Service()

        # Create config and load data (service handles cache internally)
        l4_config = CMEMSL4Config(
            gate_path=gate_path,
            time_start=str(config.cmems_l4_start),
            time_end=str(config.cmems_l4_end),
            buffer_deg=config.cmems_l4_buffer,
            variables=variables,
        )
        
        with st.sidebar.status("🌐 Loading CMEMS L4 data...", expanded=True) as status:
            progress_text = st.empty()
            
            def progress_callback(progress: float, message: str):
                progress_text.write(f"{message} ({progress*100:.0f}%)")
            
            st.write(f"🚪 Gate: {config.selected_gate}")
            st.write(f"📅 Period: {config.cmems_l4_start} to {config.cmems_l4_end}")
            st.write(f"📊 Variables: {', '.join(variables)}")
            st.write(f"💾 Cache: {'ON' if getattr(config, 'cmems_use_cache', True) else 'OFF'}")
            if config.cmems_l4_filter_ice:
                st.write("🧊 Ice filter: ENABLED")
            
            pass_data = service.load_gate_data(
                config=l4_config,
                progress_callback=progress_callback,
                use_cache=getattr(config, "cmems_use_cache", True),
                filter_ice=config.cmems_l4_filter_ice,
            )
            
            status.update(label="✅ CMEMS L4 loaded!", state="complete", expanded=False)
        
        if pass_data is None:
            st.sidebar.error("❌ No data returned from API")
            return
        
        # Apply longitude filter for divided gates (Fram West/East, Davis West/East)
        lon_min, lon_max = _get_lon_filter_for_gate(config.selected_gate)
        if lon_min is not None or lon_max is not None:
            pass_data = apply_longitude_filter(pass_data, lon_min, lon_max, config.selected_gate)
            if pass_data is None:
                st.sidebar.error(f"❌ No data after longitude filter ({lon_min}° to {lon_max}°)")
                return
            filter_info = f" [lon: {lon_min or '-∞'}° to {lon_max or '+∞'}°]"
        else:
            filter_info = ""
        
        # Store in session state (use cmems key for compatibility)
        st.session_state["dataset_cmems_l4"] = pass_data
        st.session_state["cmems_l4_service"] = service
        st.session_state["cmems_l4_config"] = config
        
        # Success message
        n_time = len(pass_data.time_array)
        n_valid_slopes = sum(~np.isnan(pass_data.slope_series))
        gate_length = pass_data.x_km[-1] if len(pass_data.x_km) > 0 else 0
        
        # Build filter info string
        ice_info = " 🧊" if config.cmems_l4_filter_ice else ""
        
        st.sidebar.success(f"""
        ✅ CMEMS L4 Data Loaded!{filter_info}{ice_info}
        - Gate: {pass_data.strait_name}
        - Source: {pass_data.data_source}
        - Period: {pass_data.time_range[0][:10]} to {pass_data.time_range[1][:10]}
        - Time steps: {n_time} daily
        - Valid slopes: {n_valid_slopes}/{n_time}
        - Gate length: {gate_length:.1f} km
        - Observations: {pass_data.n_observations:,}
        - Ice filter: {'✅ Enabled' if config.cmems_l4_filter_ice else '❌ Disabled'}
        """)
        
        # Rerun to display tabs
        st.rerun()
        
    except ImportError as e:
        st.sidebar.error(f"❌ CMEMSL4Service not available: {e}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading CMEMS L4: {e}")
        import traceback
        with st.sidebar.expander("Traceback"):
            st.code(traceback.format_exc())


def _get_gate_shapefile(gate_id: Optional[str]) -> Optional[str]:
    """
    Get path to gate shapefile.
    
    For divided gates (e.g., fram_strait_west), uses parent gate's shapefile.
    This allows West/East sections to share the same data source.
    """
    
    if not gate_id:
        return None
    
    gates_dir = Path(__file__).parent.parent.parent / "gates"
    
    if not gates_dir.exists():
        return None
    
    # Try GateService first
    if GATE_SERVICE_AVAILABLE and _gate_service:
        gate = _gate_service.get_gate(gate_id)
        if gate:
            # For divided gates, use parent's shapefile
            actual_gate_id = gate.parent_gate if gate.parent_gate else gate_id
            actual_gate = _gate_service.get_gate(actual_gate_id) if gate.parent_gate else gate
            
            if actual_gate and hasattr(actual_gate, 'file') and actual_gate.file:
                shp_path = gates_dir / actual_gate.file
                if shp_path.exists():
                    return str(shp_path)
    
    # Fallback: search by pattern (use parent gate id if divided)
    search_id = _get_parent_gate_id(gate_id)
    patterns = [
        f"*{search_id}*.shp",
        f"*{search_id.replace('_', '-')}*.shp",
        f"*{search_id.replace('_', ' ')}*.shp",
    ]
    
    for pattern in patterns:
        matches = list(gates_dir.glob(pattern))
        if matches:
            return str(matches[0])
    
    return None


# ==============================================================================
# DTUSpace-SPECIFIC FUNCTIONS (ISOLATED - does not affect SLCCI/CMEMS)
# ==============================================================================

def _render_dtu_paths(config: AppConfig) -> AppConfig:
    """Render DTUSpace NetCDF file path input."""
    
    st.sidebar.info("🟢 **DTUSpace v4** is a gridded DOT product (not along-track)")
    
    config.dtu_nc_path = st.sidebar.text_input(
        "NetCDF File Path",
        value="/Users/nicolocaron/Desktop/ARCFRESH/arctic_ocean_prod_DTUSpace_v4.0.nc/arctic_ocean_prod_DTUSpace_v4.0.nc",
        key="sidebar_dtu_nc_path",
        help="Full path to DTUSpace NetCDF file (arctic_ocean_prod_DTUSpace_v4.0.nc)"
    )
    
    # Validate path
    if config.dtu_nc_path:
        nc_path = Path(config.dtu_nc_path)
        if nc_path.exists():
            st.sidebar.success(f"✅ File found: {nc_path.name}")
        else:
            st.sidebar.warning("⚠️ File not found at specified path")
    
    # Info about DTUSpace
    with st.sidebar.expander("ℹ️ About DTUSpace", expanded=False):
        st.markdown("""
        **DTUSpace v4** is a gridded Dynamic Ocean Topography (DOT) product.
        
        **Key differences from SLCCI/CMEMS:**
        - 📊 **Gridded** (lat × lon × time), not along-track
        - 🚫 **No satellite passes** - gate defines the "synthetic pass"
        - 📁 **Local only** - no API access
        - 🗓️ **Monthly** resolution
        
        **Variables:**
        - `dot`: Dynamic Ocean Topography (m)
        - `lat`, `lon`, `date`: coordinates
        """)
    
    return config


def _render_dtu_time_range(config: AppConfig) -> AppConfig:
    """Render DTUSpace time range selector."""
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        config.dtu_start_year = st.number_input(
            "Start Year",
            min_value=1993,
            max_value=2020,
            value=2006,
            key="sidebar_dtu_start_year"
        )
    
    with col2:
        config.dtu_end_year = st.number_input(
            "End Year",
            min_value=1993,
            max_value=2020,
            value=2017,
            key="sidebar_dtu_end_year"
        )
    
    # Show time range
    years = config.dtu_end_year - config.dtu_start_year + 1
    st.sidebar.caption(f"📅 Period: {config.dtu_start_year}–{config.dtu_end_year} ({years} years, ~{years*12} monthly steps)")
    
    # Processing options
    with st.sidebar.expander("⚙️ Processing Options", expanded=False):
        config.dtu_n_gate_pts = st.slider(
            "Gate interpolation points",
            min_value=100,
            max_value=800,
            value=400,
            step=50,
            key="sidebar_dtu_n_gate_pts",
            help="Number of points to interpolate along the gate line"
        )
    
    return config


def _load_dtu_data(config: AppConfig):
    """Load DTUSpace data for the selected gate with cache support."""
    
    # Validate
    if not config.dtu_nc_path or not Path(config.dtu_nc_path).exists():
        st.sidebar.error("❌ DTUSpace NetCDF file not found. Check the path.")
        return
    
    gate_path = _get_gate_shapefile(config.selected_gate)
    if not gate_path:
        st.sidebar.error("❌ No gate selected. Select a gate first.")
        return
    
    try:
        from src.services.dtu_service import DTUService
        
        service = DTUService()
        
        # Check cache first - use parent gate for cache key (Fram West/East share cache)
        # Include time range in cache key so different time ranges are cached separately
        cache_gate = _get_parent_gate_id(config.selected_gate)
        gate_name = cache_gate.replace(" ", "_").lower()
        time_range = (config.dtu_start_year, config.dtu_end_year)
        cached_data = _cache.load("dtuspace", gate_name, time_range=time_range)
        
        if cached_data is not None:
            st.sidebar.success(f"📦 Loaded from cache! ({config.dtu_start_year}-{config.dtu_end_year})")
            pass_data = cached_data
        else:
            with st.sidebar.status("🟢 Loading DTUSpace data...", expanded=True) as status:
                st.write(f"📁 File: {Path(config.dtu_nc_path).name}")
                st.write(f"🚪 Gate: {config.selected_gate}")
                st.write(f"📅 Period: {config.dtu_start_year}–{config.dtu_end_year}")
                
                pass_data = service.load_gate_data(
                    nc_path=config.dtu_nc_path,
                    gate_path=gate_path,
                    start_year=config.dtu_start_year,
                    end_year=config.dtu_end_year,
                    n_gate_pts=config.dtu_n_gate_pts
                )
                
                status.update(label="✅ DTUSpace loaded!", state="complete", expanded=False)
            
            if pass_data is None:
                st.sidebar.error("❌ No data found for this gate/period")
                return
            
            # Save to cache with time range
            _cache.save("dtuspace", gate_name, pass_data, time_range=time_range)
        
        # Apply longitude filter for divided gates (Fram West/East, Davis West/East)
        lon_min, lon_max = _get_lon_filter_for_gate(config.selected_gate)
        if lon_min is not None or lon_max is not None:
            pass_data = apply_longitude_filter(pass_data, lon_min, lon_max, config.selected_gate)
            if pass_data is None:
                st.sidebar.error(f"❌ No data after longitude filter ({lon_min}° to {lon_max}°)")
                return
            filter_info = f" [lon: {lon_min or '-∞'}° to {lon_max or '+∞'}°]"
        else:
            filter_info = ""
        
        # Store in session state using dedicated DTU function
        store_dtu_data(pass_data)
        st.session_state["dtu_service"] = service
        st.session_state["dtu_config"] = config
        
        # Verify storage
        stored = get_dtu_data()
        if stored is None:
            st.sidebar.error("❌ Failed to store DTU data in session state!")
            return
        
        # Success message
        n_time = pass_data.n_time
        n_valid_slopes = sum(~np.isnan(pass_data.slope_series))
        gate_length = pass_data.x_km[-1] if len(pass_data.x_km) > 0 else 0
        
        st.sidebar.success(f"""
        ✅ DTUSpace Data Loaded!{filter_info}
        - Gate: {pass_data.strait_name}
        - Dataset: {pass_data.dataset_name}
        - Period: {config.dtu_start_year}–{config.dtu_end_year}
        - Time steps: {n_time} monthly
        - Valid slopes: {n_valid_slopes}/{n_time}
        - Gate length: {gate_length:.1f} km
        """)
        
        # Rerun to display DTU tabs (data verified above)
        st.rerun()
            
    except ImportError as e:
        st.sidebar.error(f"❌ DTUService not available: {e}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading DTUSpace: {e}")
        import traceback
        with st.sidebar.expander("Traceback"):
            st.code(traceback.format_exc())


# Import numpy for DTU functions
import numpy as np
