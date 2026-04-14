"""
DOT Profile Chart Components.
Renders DOT profile visualizations for the dashboard.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from .utils import DATASET_COLORS, get_pass_data_attributes


def render_dot_profile(pass_data, height: int = 600, show_std: bool = True):
    """
    Render DOT profile using PassData.profile_mean and x_km.
    
    From SLCCI PLOTTER Panel 2:
    - X-axis: x_km (Distance along longitude in km)
    - Y-axis: profile_mean (Mean DOT in m)
    
    Args:
        pass_data: PassData object with profile_mean, x_km, dot_matrix, time_periods
        height: Chart height in pixels
        show_std: Whether to show standard deviation band
        
    Returns:
        go.Figure or None if no data
    """
    # Get attributes from PassData
    profile_mean = getattr(pass_data, 'profile_mean', None)
    x_km = getattr(pass_data, 'x_km', None)
    dot_matrix = getattr(pass_data, 'dot_matrix', None)
    strait_name = getattr(pass_data, 'strait_name', 'Unknown')
    pass_number = getattr(pass_data, 'pass_number', 0)
    
    if profile_mean is None or x_km is None:
        return None
    
    # Check for valid data
    valid_mask = ~np.isnan(profile_mean)
    if not np.any(valid_mask):
        return None
    
    fig = go.Figure()
    
    # Plot mean profile
    fig.add_trace(go.Scatter(
        x=x_km[valid_mask],
        y=profile_mean[valid_mask],
        mode="lines",
        name="Mean DOT",
        line=dict(color="steelblue", width=2)
    ))
    
    # Add std band
    if show_std and dot_matrix is not None:
        std = np.nanstd(dot_matrix, axis=1)
        upper = profile_mean + std
        lower = profile_mean - std
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_km[valid_mask], x_km[valid_mask][::-1]]),
            y=np.concatenate([upper[valid_mask], lower[valid_mask][::-1]]),
            fill='toself',
            fillcolor='rgba(70,130,180,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 Std Dev'
        ))
    
    # Add WEST/EAST labels
    fig.add_annotation(
        x=x_km[valid_mask].min(),
        y=np.nanmax(profile_mean[valid_mask]),
        text="WEST",
        showarrow=False,
        font=dict(size=12, color="black"),
        xanchor="left"
    )
    fig.add_annotation(
        x=x_km[valid_mask].max(),
        y=np.nanmax(profile_mean[valid_mask]),
        text="EAST",
        showarrow=False,
        font=dict(size=12, color="black"),
        xanchor="right"
    )
    
    fig.update_layout(
        title=f"{strait_name} - Pass {pass_number} - DOT Profile",
        xaxis_title="Distance along longitude (km)",
        yaxis_title="DOT (m)",
        height=height,
        
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


def render_dot_profile_periods(pass_data, selected_periods: list = None, height: int = 600):
    """
    Render individual DOT profiles for selected time periods.
    
    Args:
        pass_data: PassData object with dot_matrix, x_km, time_periods
        selected_periods: List of period indices to show
        height: Chart height in pixels
        
    Returns:
        go.Figure or None if no data
    """
    x_km = getattr(pass_data, 'x_km', None)
    dot_matrix = getattr(pass_data, 'dot_matrix', None)
    time_periods = getattr(pass_data, 'time_periods', None)
    strait_name = getattr(pass_data, 'strait_name', 'Unknown')
    pass_number = getattr(pass_data, 'pass_number', 0)
    
    if dot_matrix is None or time_periods is None or x_km is None:
        return None
    
    n_periods = dot_matrix.shape[1]
    period_labels = [str(p)[:7] for p in time_periods]
    
    if selected_periods is None:
        selected_periods = list(range(min(5, n_periods)))
    
    if not selected_periods:
        return None
    
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    
    for i, idx in enumerate(selected_periods):
        if idx >= n_periods:
            continue
        profile = dot_matrix[:, idx]
        mask = ~np.isnan(profile)
        if np.any(mask):
            fig.add_trace(go.Scatter(
                x=x_km[mask],
                y=profile[mask],
                mode="lines",
                name=period_labels[idx],
                line=dict(color=colors[i % len(colors)])
            ))
    
    fig.update_layout(
        title=f"{strait_name} - Pass {pass_number} - Individual Periods",
        xaxis_title="Distance along longitude (km)",
        yaxis_title="DOT (m)",
        height=height,
        
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


def render_multi_dot_profile(datasets: dict, height: int = 600):
    """
    Render DOT profiles for multiple datasets (comparison mode).
    
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
    
    for name, data in datasets.items():
        profile_mean = getattr(data, 'profile_mean', None)
        x_km = getattr(data, 'x_km', None)
        
        if profile_mean is None or x_km is None:
            continue
        
        valid_mask = ~np.isnan(profile_mean)
        if not np.any(valid_mask):
            continue
        
        has_data = True
        color = DATASET_COLORS.get(name.lower(), 'gray')
        
        fig.add_trace(go.Scatter(
            x=x_km[valid_mask],
            y=profile_mean[valid_mask],
            mode="lines",
            name=name,
            line=dict(color=color, width=2)
        ))
    
    if not has_data:
        return None
    
    fig.update_layout(
        title="DOT Profile Comparison",
        xaxis_title="Distance along longitude (km)",
        yaxis_title="DOT (m)",
        height=height,
        
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig
