"""
Geostrophic Velocity Chart Components.
Renders geostrophic velocity visualizations for the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .utils import DATASET_COLORS, get_pass_data_attributes

# Physical constants
G = 9.81  # m/s² (gravity)
OMEGA = 7.2921e-5  # Earth's angular velocity (rad/s)
R_EARTH = 6371.0  # km


def compute_coriolis(lat_deg: float) -> float:
    """Compute Coriolis parameter f = 2Ω sin(lat)."""
    lat_rad = np.deg2rad(lat_deg)
    return 2 * OMEGA * np.sin(lat_rad)


def render_geostrophic_velocity(pass_data, height: int = 450):
    """
    Render geostrophic velocity analysis.
    Uses the formula: v = -g/f * (dη/dx) where f = 2Ω sin(lat)
    
    Args:
        pass_data: PassData object with df or pre-computed v_geostrophic_series
        height: Chart height in pixels
        
    Returns:
        go.Figure or None if no data
    """
    strait_name = getattr(pass_data, 'strait_name', 'Unknown')
    pass_number = getattr(pass_data, 'pass_number', 0)
    
    # Check if v_geostrophic_series is already computed (from CMEMS service)
    v_geostrophic_series = getattr(pass_data, 'v_geostrophic_series', None)
    mean_latitude = getattr(pass_data, 'mean_latitude', None)
    
    if v_geostrophic_series is not None and len(v_geostrophic_series) > 0:
        return _render_precomputed_geostrophic(pass_data, height)
    
    # Otherwise compute from raw data (SLCCI style)
    df = getattr(pass_data, 'df', None)
    if df is None or (hasattr(df, 'empty') and df.empty):
        return None
    
    # Check required columns
    required = ['lon', 'dot']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None
    
    return _render_computed_geostrophic(pass_data, height)


def _render_precomputed_geostrophic(pass_data, height: int = 450):
    """Render pre-computed geostrophic velocity time series."""
    strait_name = getattr(pass_data, 'strait_name', 'Unknown')
    v_geostrophic_series = getattr(pass_data, 'v_geostrophic_series', None)
    time_array = getattr(pass_data, 'time_array', None)
    
    # Handle both numpy arrays and pandas Series
    if hasattr(v_geostrophic_series, 'index'):
        time_index = v_geostrophic_series.index
        v_values = v_geostrophic_series.values
    else:
        if time_array is not None:
            time_index = pd.to_datetime(time_array)
        else:
            time_index = pd.date_range('2000-01', periods=len(v_geostrophic_series), freq='MS')
        v_values = v_geostrophic_series
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_index,
        y=v_values * 100,  # Convert m/s to cm/s
        mode='lines+markers',
        name='v_geostrophic',
        line=dict(color='steelblue', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{strait_name} - Geostrophic Velocity Time Series",
        xaxis_title="Time",
        yaxis_title="Geostrophic Velocity (cm/s)",
        height=height,
        
    )
    
    return fig


def _render_computed_geostrophic(pass_data, height: int = 450):
    """Compute and render geostrophic velocity from raw data."""
    df = getattr(pass_data, 'df', None)
    strait_name = getattr(pass_data, 'strait_name', 'Unknown')
    pass_number = getattr(pass_data, 'pass_number', 0)
    
    # Get mean latitude for Coriolis parameter
    if 'lat' in df.columns:
        mean_lat = df['lat'].mean()
    else:
        mean_lat = 70.0  # Default for Arctic
    
    lat_rad = np.deg2rad(mean_lat)
    f = compute_coriolis(mean_lat)
    
    # Check for year/month columns for time series
    has_year = 'year' in df.columns
    has_month = 'month' in df.columns
    
    if has_year and has_month:
        df = df.copy()
        df['year_month'] = pd.to_datetime(
            df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
        )
        groups = df.groupby('year_month')
    elif has_month:
        groups = df.groupby('month')
    else:
        # Single group
        groups = [('all', df)]
    
    # Calculate slope and geostrophic velocity for each period
    results = []
    
    for period, group_df in groups:
        lon = group_df['lon'].values
        dot = group_df['dot'].values
        
        mask = np.isfinite(lon) & np.isfinite(dot)
        if np.sum(mask) < 3:
            continue
        
        lon_valid = lon[mask]
        dot_valid = dot[mask]
        
        # Convert longitude to meters
        lon_rad_arr = np.deg2rad(lon_valid)
        dlon_rad = lon_rad_arr - lon_rad_arr.min()
        x_m = R_EARTH * 1000 * dlon_rad * np.cos(lat_rad)
        
        try:
            slope_m_m, _ = np.polyfit(x_m, dot_valid, 1)
            v_geo = -G / f * slope_m_m
            
            results.append({
                'period': period,
                'slope_m_m': slope_m_m,
                'v_geostrophic_m_s': v_geo,
                'v_geostrophic_cm_s': v_geo * 100,
                'n_points': len(lon_valid)
            })
        except Exception:
            continue
    
    if not results:
        return None
    
    results_df = pd.DataFrame(results)
    
    # Plot time series or monthly values
    fig = go.Figure()
    
    if has_year and has_month:
        x_vals = results_df['period']
        title_suffix = "Time Series"
    elif has_month:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        x_vals = [month_names[int(m)-1] for m in results_df['period']]
        title_suffix = "Monthly"
    else:
        x_vals = results_df['period']
        title_suffix = ""
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=results_df['v_geostrophic_cm_s'],
        mode='lines+markers',
        name='v_geostrophic',
        line=dict(color='steelblue', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{strait_name} - Pass {pass_number} - Geostrophic Velocity ({title_suffix})",
        xaxis_title="Time" if (has_year and has_month) else "Month" if has_month else "",
        yaxis_title="Geostrophic Velocity (cm/s)",
        height=height,
        
    )
    
    return fig


def render_geostrophic_climatology(pass_data, height: int = 400):
    """
    Render monthly climatology of geostrophic velocity.
    
    Args:
        pass_data: PassData with v_geostrophic_series
        height: Chart height in pixels
        
    Returns:
        go.Figure or None if no data
    """
    strait_name = getattr(pass_data, 'strait_name', 'Unknown')
    v_geostrophic_series = getattr(pass_data, 'v_geostrophic_series', None)
    time_array = getattr(pass_data, 'time_array', None)
    
    if v_geostrophic_series is None or len(v_geostrophic_series) == 0:
        return None
    
    # Handle both numpy arrays and pandas Series
    if hasattr(v_geostrophic_series, 'index'):
        time_index = v_geostrophic_series.index
        v_values = v_geostrophic_series.values
    else:
        if time_array is not None:
            time_index = pd.to_datetime(time_array)
        else:
            return None
        v_values = v_geostrophic_series
    
    # Create Series for groupby
    v_series = pd.Series(v_values, index=time_index)
    monthly_clim = v_series.groupby(v_series.index.month).mean()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[month_names[m-1] for m in monthly_clim.index],
        y=monthly_clim.values * 100,
        marker_color=['steelblue' if v >= 0 else 'coral' for v in monthly_clim.values],
        name='Mean Velocity'
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    fig.update_layout(
        title=f"{strait_name} - Monthly Mean Geostrophic Velocity",
        xaxis_title="Month",
        yaxis_title="Velocity (cm/s)",
        height=height,
        
    )
    
    return fig


def render_multi_geostrophic(datasets: dict, height: int = 450):
    """
    Render geostrophic velocity comparison for multiple datasets.
    
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
        v_geostrophic_series = getattr(data, 'v_geostrophic_series', None)
        time_array = getattr(data, 'time_array', None)
        
        if v_geostrophic_series is None or len(v_geostrophic_series) == 0:
            continue
        
        has_data = True
        color = DATASET_COLORS.get(name.lower(), 'gray')
        
        # Handle both numpy arrays and pandas Series
        if hasattr(v_geostrophic_series, 'index'):
            time_index = v_geostrophic_series.index
            v_values = v_geostrophic_series.values
        else:
            if time_array is not None:
                time_index = pd.to_datetime(time_array)
            else:
                time_index = pd.date_range('2000-01', periods=len(v_geostrophic_series), freq='MS')
            v_values = v_geostrophic_series
        
        fig.add_trace(go.Scatter(
            x=time_index,
            y=v_values * 100,  # Convert m/s to cm/s
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=4)
        ))
    
    if not has_data:
        return None
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Geostrophic Velocity Comparison",
        xaxis_title="Time",
        yaxis_title="Geostrophic Velocity (cm/s)",
        height=height,
        
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


def get_geostrophic_stats(pass_data) -> dict:
    """
    Get statistics for geostrophic velocity.
    
    Args:
        pass_data: PassData with v_geostrophic_series
        
    Returns:
        Dict with mean, std, max, min in cm/s
    """
    v_geostrophic_series = getattr(pass_data, 'v_geostrophic_series', None)
    
    if v_geostrophic_series is None or len(v_geostrophic_series) == 0:
        return {}
    
    if hasattr(v_geostrophic_series, 'values'):
        v_values = v_geostrophic_series.values
    else:
        v_values = v_geostrophic_series
    
    return {
        'mean_cm_s': float(v_values.mean() * 100),
        'std_cm_s': float(v_values.std() * 100),
        'max_cm_s': float(v_values.max() * 100),
        'min_cm_s': float(v_values.min() * 100)
    }
