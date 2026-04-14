"""
Spatial Map Chart Components.
Renders spatial distribution maps for the dashboard.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from .utils import DATASET_COLORS, get_pass_data_attributes


def render_spatial_map(pass_data, color_var: str = "dot", show_gate: bool = True, height: int = 600):
    """
    Render spatial map of measurements.
    
    Args:
        pass_data: PassData object with df, gate_lon_pts, gate_lat_pts
        color_var: Variable to color points by (dot, corssh, geoid, sla_filtered, mdt, satellite, track, cycle)
        show_gate: Whether to show gate line
        height: Chart height in pixels
        
    Returns:
        go.Figure or None if no data
    """
    df = getattr(pass_data, 'df', None)
    gate_lon_pts = getattr(pass_data, 'gate_lon_pts', None)
    gate_lat_pts = getattr(pass_data, 'gate_lat_pts', None)
    strait_name = getattr(pass_data, 'strait_name', 'Unknown')
    pass_number = getattr(pass_data, 'pass_number', 0)
    
    if df is None or df.empty:
        return None
    
    # Check if color variable exists
    if color_var not in df.columns:
        # Fallback to dot or first numeric column
        if 'dot' in df.columns:
            color_var = 'dot'
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                color_var = numeric_cols[0]
            else:
                return None
    
    # Sample for performance
    if len(df) > 5000:
        plot_df = df.sample(5000)
    else:
        plot_df = df
    
    # Determine if color variable is categorical or numeric
    categorical_vars = ["satellite", "track", "cycle"]
    is_categorical = color_var in categorical_vars
    
    # Create map with appropriate color handling
    if is_categorical:
        fig = px.scatter_mapbox(
            plot_df,
            lat="lat",
            lon="lon",
            color=color_var,
            zoom=5,
            height=height,
            title=f"{strait_name} - Pass {pass_number}"
        )
    else:
        fig = px.scatter_mapbox(
            plot_df,
            lat="lat",
            lon="lon",
            color=color_var,
            color_continuous_scale="viridis",
            zoom=5,
            height=height,
            title=f"{strait_name} - Pass {pass_number}"
        )
    
    # Add gate line
    if show_gate and gate_lon_pts is not None and gate_lat_pts is not None:
        fig.add_trace(go.Scattermapbox(
            lat=gate_lat_pts,
            lon=gate_lon_pts,
            mode="lines",
            name="Gate",
            line=dict(width=3, color="red")
        ))
    
    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def get_available_color_vars(pass_data) -> list:
    """
    Get list of available color variables for spatial map.
    
    Args:
        pass_data: PassData object with df
        
    Returns:
        List of available variable names
    """
    df = getattr(pass_data, 'df', None)
    if df is None or df.empty:
        return ["dot"]
    
    available_vars = []
    
    # Common variables
    if "dot" in df.columns:
        available_vars.append("dot")
    
    # SLCCI-specific
    if "corssh" in df.columns:
        available_vars.append("corssh")
    if "geoid" in df.columns:
        available_vars.append("geoid")
    
    # CMEMS-specific
    if "sla_filtered" in df.columns:
        available_vars.append("sla_filtered")
    if "mdt" in df.columns:
        available_vars.append("mdt")
    if "satellite" in df.columns:
        available_vars.append("satellite")
    if "track" in df.columns:
        available_vars.append("track")
    if "cycle" in df.columns:
        available_vars.append("cycle")
    
    # Fallback
    if not available_vars:
        available_vars = ["dot"]
    
    return available_vars


def render_multi_spatial_overview(datasets: dict, height: int = 600):
    """
    Render spatial overview for multiple datasets (comparison mode).
    Shows all datasets on the same map with different colors.
    
    Args:
        datasets: Dict of {name: PassData} objects
        height: Chart height in pixels
        
    Returns:
        go.Figure or None if no data
    """
    if not datasets:
        return None
    
    fig = go.Figure()
    has_data = False
    
    # Collect all coordinates for zoom calculation
    all_lats = []
    all_lons = []
    
    for name, data in datasets.items():
        df = getattr(data, 'df', None)
        if df is None or df.empty:
            continue
        
        has_data = True
        color = DATASET_COLORS.get(name.lower(), 'gray')
        
        # Sample for performance
        if len(df) > 2000:
            plot_df = df.sample(2000)
        else:
            plot_df = df
        
        all_lats.extend(plot_df['lat'].tolist())
        all_lons.extend(plot_df['lon'].tolist())
        
        fig.add_trace(go.Scattermapbox(
            lat=plot_df['lat'],
            lon=plot_df['lon'],
            mode='markers',
            name=name,
            marker=dict(size=5, color=color, opacity=0.7)
        ))
    
    if not has_data:
        return None
    
    # Add gate line from first dataset with gate info
    for name, data in datasets.items():
        gate_lon_pts = getattr(data, 'gate_lon_pts', None)
        gate_lat_pts = getattr(data, 'gate_lat_pts', None)
        if gate_lon_pts is not None and gate_lat_pts is not None:
            fig.add_trace(go.Scattermapbox(
                lat=gate_lat_pts,
                lon=gate_lon_pts,
                mode="lines",
                name="Gate",
                line=dict(width=3, color="red")
            ))
            break
    
    # Calculate center
    center_lat = np.mean(all_lats) if all_lats else 70.0
    center_lon = np.mean(all_lons) if all_lons else 0.0
    
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=4
        ),
        title="Spatial Data Comparison",
        height=height,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig
