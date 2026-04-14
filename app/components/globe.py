"""
3D Globe Component for Landing Page
====================================
Interactive 3D globe showing all available gates.
Click on a gate to select it for analysis.

Uses Plotly's scattergeo for Streamlit compatibility.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Optional, List, Dict, Any

# Try to import GateService
try:
    from src.services import GateService
    _gate_service = GateService()
    GATE_SERVICE_AVAILABLE = True
except ImportError:
    _gate_service = None
    GATE_SERVICE_AVAILABLE = False


def render_globe_landing(on_gate_select: Optional[callable] = None):
    """
    Render the 3D globe landing page with all gates.
    
    Args:
        on_gate_select: Callback when a gate is clicked (receives gate_name)
    """
    st.markdown("## 🌍 ARCFRESH Project")
    st.markdown("*Navigate the Arctic: Select a gate to begin your analysis*")
    
    # Dataset preview selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        preview_dataset = st.selectbox(
            "Preview Dataset",
            ["All Gates", "SLCCI Coverage", "CMEMS Coverage", "DTUSpace Coverage"],
            key="globe_preview_dataset",
            help="Highlight gates by dataset availability"
        )
    
    with col2:
        show_labels = st.checkbox("Show Labels", value=True, key="globe_show_labels")
    
    with col3:
        projection = st.selectbox(
            "Projection",
            ["orthographic", "natural earth", "equirectangular"],
            key="globe_projection"
        )
    
    # Get all gates
    gates_data = _get_all_gates_positions()
    
    if not gates_data:
        # Use demo/fallback gates if service not available
        st.info("🔧 Using demo gates (GateService not configured)")
        gates_data = _get_demo_gates()
    
    if not gates_data:
        st.error("No gates available and no demo data. Check configuration.")
        return
    
    # Get currently selected gate from session state
    selected_gate = st.session_state.get("selected_gate", None)
    
    # Create the globe figure
    fig = _create_globe_figure(
        gates_data=gates_data,
        selected_gate=selected_gate,
        show_labels=show_labels,
        projection=projection
    )
    
    # Render with click events
    clicked = st.plotly_chart(
        fig, 
        use_container_width=True,
        key="globe_chart",
        on_select="rerun",  # Enable selection events
        selection_mode="points"
    )
    
    # Handle click events
    if clicked and clicked.selection and clicked.selection.points:
        point = clicked.selection.points[0]
        if "customdata" in point:
            clicked_gate = point["customdata"]
            if clicked_gate != selected_gate:
                st.session_state["selected_gate"] = clicked_gate
                # Note: sidebar will pick this up via its own sync mechanism
                if on_gate_select:
                    on_gate_select(clicked_gate)
                # Don't call st.rerun() here - on_select="rerun" already handles it
    
    # Quick stats
    st.divider()
    _render_quick_stats(gates_data, selected_gate)
    
    # Selected gate info
    if selected_gate:
        st.divider()
        _render_selected_gate_info(selected_gate)


def _get_all_gates_positions() -> List[Dict[str, Any]]:
    """
    Get all gates with their centroid positions by scanning the gates/ folder.
    
    This function directly reads shapefiles from the gates/ directory,
    independent of GateService configuration.
    """
    import geopandas as gpd
    import os
    from pathlib import Path
    
    # Suppress shapefile warnings
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    
    # Find gates folder
    gates_folder = Path(__file__).parent.parent.parent / "gates"
    
    if not gates_folder.exists():
        st.warning(f"Gates folder not found: {gates_folder}")
        return []
    
    # Find all shapefiles
    shp_files = list(gates_folder.glob("*.shp"))
    
    if not shp_files:
        st.warning(f"No shapefiles found in {gates_folder}")
        return []
    
    gates = []
    
    for shp_path in shp_files:
        try:
            # Read shapefile
            gdf = gpd.read_file(shp_path)
            
            # Convert to WGS84 if needed
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:3413")  # Assume polar stereographic
            
            if not gdf.crs.is_geographic:
                gdf = gdf.to_crs("EPSG:4326")
            
            # Get centroid
            centroid = gdf.geometry.unary_union.centroid
            lon, lat = centroid.x, centroid.y
            
            # Extract name from filename
            gate_name = shp_path.stem
            
            # Determine region from name
            region = _infer_region_from_name(gate_name)
            
            gates.append({
                "name": gate_name,
                "lon": lon,
                "lat": lat,
                "region": region,
                "path": str(shp_path)
            })
            
        except Exception as e:
            # Skip problematic files
            continue
    
    return gates


def _infer_region_from_name(gate_name: str) -> str:
    """Infer the region from gate name."""
    name_lower = gate_name.lower()
    
    if "fram" in name_lower:
        return "Nordic Seas"
    elif "bering" in name_lower:
        return "Pacific-Arctic"
    elif "davis" in name_lower:
        return "Labrador Sea"
    elif "denmark" in name_lower:
        return "Nordic Seas"
    elif "barents" in name_lower:
        return "Barents Sea"
    elif "kara" in name_lower:
        return "Kara Sea"
    elif "laptev" in name_lower:
        return "Laptev Sea"
    elif "east_siberian" in name_lower or "siberian" in name_lower:
        return "East Siberian Sea"
    elif "beaufort" in name_lower:
        return "Beaufort Sea"
    elif "canadian" in name_lower or "nares" in name_lower or "lancaster" in name_lower or "jones" in name_lower:
        return "Canadian Arctic"
    elif "norwegian" in name_lower:
        return "Norwegian Sea"
    elif "central_arctic" in name_lower:
        return "Central Arctic"
    else:
        return "Arctic Ocean"


def _get_demo_gates() -> List[Dict[str, Any]]:
    """
    Return demo gates for display when shapefiles not available.
    These are the main Arctic straits used in NICO analysis.
    """
    return [
        {"name": "fram_strait", "lon": 0.0, "lat": 79.0, "region": "Nordic Seas", "path": None},
        {"name": "bering_strait", "lon": -168.5, "lat": 65.8, "region": "Pacific-Arctic", "path": None},
        {"name": "davis_strait", "lon": -57.0, "lat": 66.5, "region": "Labrador Sea", "path": None},
        {"name": "denmark_strait", "lon": -27.0, "lat": 66.0, "region": "Nordic Seas", "path": None},
        {"name": "barents_sea_opening", "lon": 20.0, "lat": 74.0, "region": "Barents Sea", "path": None},
        {"name": "nares_strait", "lon": -70.0, "lat": 80.5, "region": "Canadian Arctic", "path": None},
        {"name": "lancaster_sound", "lon": -85.0, "lat": 74.0, "region": "Canadian Arctic", "path": None},
        {"name": "hudson_strait", "lon": -70.0, "lat": 62.0, "region": "Hudson Bay", "path": None},
    ]


def _create_globe_figure(
    gates_data: List[Dict],
    selected_gate: Optional[str] = None,
    show_labels: bool = True,
    projection: str = "orthographic"
) -> go.Figure:
    """Create the 3D globe figure with gates."""
    
    # Separate selected gate from others
    other_gates = [g for g in gates_data if g["name"] != selected_gate]
    selected_gates = [g for g in gates_data if g["name"] == selected_gate]
    
    fig = go.Figure()
    
    # Add non-selected gates (blue markers)
    if other_gates:
        fig.add_trace(go.Scattergeo(
            lon=[g["lon"] for g in other_gates],
            lat=[g["lat"] for g in other_gates],
            mode="markers+text" if show_labels else "markers",
            marker=dict(
                size=12,
                color="steelblue",
                symbol="circle",
                line=dict(width=1, color="white")
            ),
            text=[g["name"].replace("_", " ").title() for g in other_gates] if show_labels else None,
            textposition="top center",
            textfont=dict(size=10, color="white"),
            customdata=[g["name"] for g in other_gates],
            hovertemplate="<b>%{text}</b><br>Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<extra></extra>",
            name="Available Gates"
        ))
    
    # Add selected gate (highlighted - orange/gold)
    if selected_gates:
        sg = selected_gates[0]
        fig.add_trace(go.Scattergeo(
            lon=[sg["lon"]],
            lat=[sg["lat"]],
            mode="markers+text",
            marker=dict(
                size=18,
                color="darkorange",
                symbol="star",
                line=dict(width=2, color="gold")
            ),
            text=[sg["name"].replace("_", " ").title()],
            textposition="top center",
            textfont=dict(size=12, color="gold", family="Arial Black"),
            customdata=[sg["name"]],
            hovertemplate="<b>%{text}</b> ⭐ SELECTED<br>Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<extra></extra>",
            name="Selected Gate"
        ))
    
    # Configure the globe
    fig.update_geos(
        projection_type=projection,
        showland=True,
        landcolor="rgb(40, 40, 40)",
        showocean=True,
        oceancolor="rgb(20, 50, 80)",
        showcoastlines=True,
        coastlinecolor="rgb(100, 100, 100)",
        showlakes=True,
        lakecolor="rgb(20, 50, 80)",
        showcountries=True,
        countrycolor="rgb(80, 80, 80)",
        
        # Focus on Arctic
        projection_rotation=dict(lon=0, lat=70, roll=0),
        
        # Lat/lon grid lines
        lataxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.2)",
            dtick=10,
            range=[50, 90]
        ),
        lonaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.2)",
            dtick=30,
            range=[-180, 180]
        ),
    )
    
    # Layout
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white")
        ),
        title=dict(
            text="🌊 Arctic Ocean Gates",
            font=dict(size=16, color="white"),
            x=0.5
        )
    )
    
    return fig


def _render_quick_stats(gates_data: List[Dict], selected_gate: Optional[str]):
    """Render quick statistics about available gates."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Gates", len(gates_data))
    
    with col2:
        # Count by region
        regions = set(g.get("region", "Unknown") for g in gates_data)
        st.metric("Regions", len(regions))
    
    with col3:
        status = "✅ Selected" if selected_gate else "❌ None"
        st.metric("Current Gate", status)
    
    with col4:
        if selected_gate:
            st.metric("Gate Name", selected_gate.replace("_", " ").title()[:20])
        else:
            st.metric("Gate Name", "Click to select")


def _render_bathymetry_profile(gate_name: str, gate_lons: np.ndarray, gate_lats: np.ndarray) -> Optional[go.Figure]:
    """
    Load and render bathymetry profile for a gate.
    
    Args:
        gate_name: Name of the gate
        gate_lons: Longitude array along gate
        gate_lats: Latitude array along gate
    
    Returns:
        Plotly Figure with bathymetry profile, or None if error
    """
    try:
        from src.services.gebco_service import get_bathymetry_cache
        
        # Default GEBCO path
        gebco_path = "/Users/nicolocaron/Desktop/ARCFRESH/GEBCO_06_Feb_2026_c91df93f54b8/gebco_2025_n90.0_s55.0_w0.0_e360.0.nc"
        
        # Check if file exists
        from pathlib import Path
        if not Path(gebco_path).exists():
            return None
        
        # Get bathymetry cache
        cache = get_bathymetry_cache()
        
        # Get depth profile (no cap - show full bathymetry)
        depth_profile = cache.get_or_compute(
            gate_name=gate_name,
            gate_lons=gate_lons,
            gate_lats=gate_lats,
            gebco_path=gebco_path,
            depth_cap=None
        )
        
        if depth_profile is None or len(depth_profile) == 0:
            return None
        
        # Compute distance along gate (x_km)
        R_earth = 6371.0  # km
        lon_rad = np.deg2rad(gate_lons)
        lat_rad = np.deg2rad(gate_lats)
        
        # Haversine distance
        dlat = np.diff(lat_rad)
        dlon = np.diff(lon_rad)
        a = np.sin(dlat / 2)**2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        d_km = R_earth * c
        
        x_km = np.zeros(len(gate_lons))
        x_km[1:] = np.cumsum(d_km)
        
        # Create figure
        fig = go.Figure()
        
        # Fill area (ocean depth as filled area)
        fig.add_trace(go.Scatter(
            x=x_km,
            y=-depth_profile,  # Negative for depth below sea level
            fill='tozeroy',
            fillcolor='rgba(70, 130, 180, 0.4)',  # Steel blue with transparency
            line=dict(color='#1E3A5F', width=2),
            name='Bathymetry',
            hovertemplate='Distance: %{x:.1f} km<br>Depth: %{customdata:.0f} m<extra></extra>',
            customdata=depth_profile
        ))
        
        # Add sea level line
        fig.add_hline(y=0, line_color='#3498DB', line_width=2, 
                      annotation_text="Sea Level", annotation_position="top right")
        
        # Add 250m reference line (common depth cap for transport)
        fig.add_hline(y=-250, line_color='#E74C3C', line_width=1.5, line_dash='dash',
                      annotation_text="250m (transport cap)", annotation_position="bottom right",
                      annotation_font=dict(size=10, color='#E74C3C'))
        
        # Stats for title
        max_depth = np.max(depth_profile)
        mean_depth = np.mean(depth_profile)
        sill_depth = np.min(depth_profile[depth_profile > 0]) if np.any(depth_profile > 0) else 0
        gate_length = x_km[-1]
        
        fig.update_layout(
            title=dict(
                text=f"🌊 {gate_name.replace('_', ' ').title()} — Bathymetry Profile<br>"
                     f"<sup>Length: {gate_length:.0f} km | Max: {max_depth:.0f} m | Mean: {mean_depth:.0f} m | Sill: {sill_depth:.0f} m</sup>",
                font=dict(size=14)
            ),
            xaxis_title="Distance along gate (km)",
            yaxis_title="Depth (m)",
            height=350,
            margin=dict(l=60, r=40, t=80, b=50),
            font=dict(family="Inter, sans-serif", size=11),
            xaxis=dict(
                gridcolor='#E8E8E8',
                zeroline=True,
                zerolinecolor='#3498DB',
                zerolinewidth=1
            ),
            yaxis=dict(
                gridcolor='#E8E8E8',
                zeroline=True,
                zerolinecolor='#3498DB',
                zerolinewidth=2,
                autorange='reversed',  # Flip so deeper is at bottom
                range=[-(max_depth * 1.1), max_depth * 0.05]  # Add margins
            ),
            showlegend=False
        )
        
        # Flip y-axis to show depth correctly (0 at top, max depth at bottom)
        fig.update_yaxes(autorange=False, range=[-max_depth * 1.1, max_depth * 0.05])
        
        return fig
        
    except ImportError as e:
        st.warning(f"GEBCO service not available: {e}")
        return None
    except FileNotFoundError as e:
        st.warning(f"GEBCO file not found: {e}")
        return None
    except Exception as e:
        st.warning(f"Error loading bathymetry: {e}")
        return None


def _render_selected_gate_info(gate_name: str):
    """Render detailed info about the selected gate, including bathymetry profile."""
    
    if not GATE_SERVICE_AVAILABLE:
        return
    
    gate_info = _gate_service.get_gate(gate_name)
    if not gate_info:
        return
    
    st.markdown(f"### 🎯 Selected: **{gate_name.replace('_', ' ').title()}**")
    
    # =========================================================================
    # BATHYMETRY PROFILE (shown first, automatically)
    # =========================================================================
    gate_lons = None
    gate_lats = None
    
    # Try to load gate coordinates
    try:
        gate_path = getattr(gate_info, "path", None) or _gate_service.get_gate_path(gate_name)
        if gate_path:
            import geopandas as gpd
            import os
            os.environ['SHAPE_RESTORE_SHX'] = 'YES'
            gdf = gpd.read_file(gate_path)
            if gdf.crs and not gdf.crs.is_geographic:
                gdf = gdf.to_crs("EPSG:4326")
            
            # Extract line coordinates
            geom = gdf.geometry.unary_union
            if hasattr(geom, 'coords'):
                coords = np.array(geom.coords)
                gate_lons = coords[:, 0]
                gate_lats = coords[:, 1]
            elif hasattr(geom, 'geoms'):
                # MultiLineString - take first
                coords = np.array(geom.geoms[0].coords)
                gate_lons = coords[:, 0]
                gate_lats = coords[:, 1]
            
            # Interpolate to more points for smooth bathymetry
            if gate_lons is not None and len(gate_lons) > 2:
                from scipy.interpolate import interp1d
                t = np.linspace(0, 1, len(gate_lons))
                t_fine = np.linspace(0, 1, 200)  # 200 points
                
                f_lon = interp1d(t, gate_lons, kind='linear')
                f_lat = interp1d(t, gate_lats, kind='linear')
                
                gate_lons = f_lon(t_fine)
                gate_lats = f_lat(t_fine)
    except Exception as e:
        pass
    
    # Show bathymetry profile if we have coordinates
    if gate_lons is not None and gate_lats is not None:
        with st.spinner("Loading bathymetry..."):
            bathy_fig = _render_bathymetry_profile(gate_name, gate_lons, gate_lats)
            if bathy_fig is not None:
                st.plotly_chart(bathy_fig, use_container_width=True, key=f"bathy_{gate_name}")
            else:
                st.info("ℹ️ Bathymetry data not available. Configure GEBCO path in settings.")
    
    # =========================================================================
    # GATE INFO + ACTIONS
    # =========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Gate Info:**")
        st.markdown(f"- Region: `{getattr(gate_info, 'region', 'Unknown')}`")
        
        # Show coordinates if available
        if gate_lons is not None and gate_lats is not None:
            st.markdown(f"- Lon: `{gate_lons.min():.2f}°` to `{gate_lons.max():.2f}°`")
            st.markdown(f"- Lat: `{gate_lats.min():.2f}°` to `{gate_lats.max():.2f}°`")
        else:
            # Try to get from bounds
            try:
                gate_path = getattr(gate_info, "path", None) or _gate_service.get_gate_path(gate_name)
                if gate_path:
                    import geopandas as gpd
                    import os
                    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
                    gdf = gpd.read_file(gate_path)
                    if gdf.crs and not gdf.crs.is_geographic:
                        gdf = gdf.to_crs("EPSG:4326")
                    bounds = gdf.total_bounds
                    st.markdown(f"- Lon: `{bounds[0]:.2f}°` to `{bounds[2]:.2f}°`")
                    st.markdown(f"- Lat: `{bounds[1]:.2f}°` to `{bounds[3]:.2f}°`")
            except:
                pass
    
    with col2:
        st.markdown("**Next Steps:**")
        st.markdown("1. Select a **Data Source** in the sidebar")
        st.markdown("2. Configure **parameters**")
        st.markdown("3. Click **Load Data**")
        
        # Quick action buttons
        if st.button("📊 Load SLCCI", key="globe_load_slcci"):
            st.session_state["selected_dataset_type"] = "SLCCI"
            st.session_state["sidebar_datasource"] = "SLCCI"
            st.rerun()
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🌊 Load CMEMS", key="globe_load_cmems"):
                st.session_state["selected_dataset_type"] = "CMEMS"
                st.session_state["sidebar_datasource"] = "CMEMS"
                st.rerun()
        with col_b:
            if st.button("🟢 Load DTU", key="globe_load_dtu"):
                st.session_state["selected_dataset_type"] = "DTUSpace"
                st.session_state["sidebar_datasource"] = "DTUSpace"
                st.rerun()
