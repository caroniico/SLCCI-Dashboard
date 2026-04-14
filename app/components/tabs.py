
"""
Main content tabs for the dashboard.
Following SLCCI PLOTTER notebook workflow exactly.
Supports comparison mode with SLCCI/CMEMS overlay.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List
import io
import logging

logger = logging.getLogger(__name__)

from .sidebar import AppConfig
from ..state import get_slcci_data, get_cmems_data, is_comparison_mode, get_dtu_data
from .charts import (
    render_slope_timeline,
    render_dot_profile,
    render_spatial_map,
    render_geostrophic_velocity,
    render_volume_transport_tab,
    get_pass_data_attributes,
)
# Aggiungi questo import in alto
from app.components.salt_flux_clean import render_salt_flux_clean
from app.components.freshwater_transport_clean import render_freshwater_transport_clean
from app.components.salinity_profile_tab import render_salinity_profile_tab

# Comparison mode colors (from COMPARISON_BATCH notebook)
COLOR_SLCCI = "darkorange"
COLOR_CMEMS = "steelblue"
COLOR_CMEMS_L4 = "mediumpurple"  # CMEMS L4 Gridded color
COLOR_DTU = "seagreen"  # DTUSpace color (green)

# Dataset display names
DATASET_NAMES = {
    "slcci": "SLCCI",
    "cmems": "CMEMS L3",
    "cmems_l4": "CMEMS L4",
    "dtu": "DTUSpace"
}

DATASET_COLORS = {
    "slcci": COLOR_SLCCI,
    "cmems": COLOR_CMEMS,
    "cmems_l4": COLOR_CMEMS_L4,
    "dtu": COLOR_DTU
}

# Month names for timelapse
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _create_velocity_timelapse(monthly_v_perp: dict, x_km: np.ndarray, gate_lon: np.ndarray, 
                                title_prefix: str = "Velocity Profile") -> go.Figure:
    """
    Create animated timelapse for velocity profile across all 12 months.
    
    Args:
        monthly_v_perp: Dict with month (1-12) as key, (bin_centers, bin_means, bin_stds) as value
        x_km: Distance array along gate (km)
        gate_lon: Longitude array along gate
        title_prefix: Title prefix for the animation
    
    Returns:
        Plotly Figure with animation frames
    """
    # Collect all months data
    frames = []
    all_y_values = []
    
    for month in range(1, 13):
        bin_centers, bin_means, bin_stds = monthly_v_perp.get(month, (np.array([]), np.array([]), np.array([])))
        if len(bin_centers) > 0:
            all_y_values.extend(bin_means * 100)  # cm/s
    
    # Calculate y-axis range for consistent scaling
    if all_y_values:
        y_min = min(all_y_values) * 1.1
        y_max = max(all_y_values) * 1.1
        # Ensure zero is visible
        y_min = min(y_min, -abs(y_max) * 0.1)
        y_max = max(y_max, abs(y_min) * 0.1)
    else:
        y_min, y_max = -10, 10
    
    # Create frames for each month
    for month in range(1, 13):
        bin_centers, bin_means, bin_stds = monthly_v_perp.get(month, (np.array([]), np.array([]), np.array([])))
        
        if len(bin_centers) > 0:
            bin_lon = np.interp(bin_centers, x_km, gate_lon)
            frame_data = go.Scatter(
                x=bin_centers,
                y=bin_means * 100,
                mode='lines+markers',
                name='v_perp',
                line=dict(color='#1E3A5F', width=2.5),
                marker=dict(size=7, color='#1E3A5F'),
                error_y=dict(type='data', array=bin_stds * 100, visible=True, color='rgba(30,58,95,0.3)')
            )
        else:
            frame_data = go.Scatter(x=[], y=[], mode='lines+markers', name='v_perp')
        
        frames.append(go.Frame(
            data=[frame_data],
            name=MONTH_NAMES[month-1],
            layout=go.Layout(title=dict(text=f"{title_prefix} — {MONTH_NAMES[month-1]}"))
        ))
    
    # Initial frame (January)
    bin_centers, bin_means, bin_stds = monthly_v_perp.get(1, (np.array([]), np.array([]), np.array([])))
    if len(bin_centers) > 0:
        initial_trace = go.Scatter(
            x=bin_centers,
            y=bin_means * 100,
            mode='lines+markers',
            name='v_perp (ugos/vgos)',
            line=dict(color='#1E3A5F', width=2.5),
            marker=dict(size=7, color='#1E3A5F'),
            error_y=dict(type='data', array=bin_stds * 100, visible=True, color='rgba(30,58,95,0.3)')
        )
    else:
        initial_trace = go.Scatter(x=[], y=[], mode='lines+markers', name='v_perp')
    
    fig = go.Figure(data=[initial_trace], frames=frames)
    
    # Add zero line
    fig.add_hline(y=0, line_color="#7F8C8D", line_width=1, line_dash="dash")
    
    # Animation controls
    fig.update_layout(
        title=dict(text=f"{title_prefix} — Jan", font=dict(size=16)),
        yaxis_title="Velocity (cm/s)",
        xaxis_title="Distance along gate (km)",
        height=480,
        
        
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor='#E8E8E8', gridwidth=1),
        yaxis=dict(gridcolor='#E8E8E8', gridwidth=1, range=[y_min, y_max]),
        margin=dict(l=60, r=40, t=80, b=100),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=-0.15,
                x=0.1,
                xanchor="right",
                buttons=[
                    dict(
                        label="▶️ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 800, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300, "easing": "cubic-in-out"}
                        }]
                    ),
                    dict(
                        label="⏸️ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 14, "color": "#1E3A5F"},
                "prefix": "Month: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.8,
            "x": 0.15,
            "y": -0.05,
            "steps": [
                {
                    "args": [[MONTH_NAMES[i]], {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300}
                    }],
                    "label": MONTH_NAMES[i],
                    "method": "animate"
                }
                for i in range(12)
            ]
        }]
    )
    
    return fig


def _create_transport_timelapse(monthly_profiles: dict, x_km: np.ndarray, gate_lon: np.ndarray,
                                 title_prefix: str = "Volume Transport") -> go.Figure:
    """
    Create animated timelapse for volume transport profile across all 12 months.
    
    Args:
        monthly_profiles: Dict with month (1-12) as key, (bin_centers, bin_means, bin_stds) as value
        x_km: Distance array along gate (km)
        gate_lon: Longitude array along gate
        title_prefix: Title prefix for the animation
    
    Returns:
        Plotly Figure with animation frames
    """
    # Collect all months data for consistent y-axis
    all_y_values = []
    
    for month in range(1, 13):
        bin_centers, bin_means, bin_stds = monthly_profiles.get(month, (np.array([]), np.array([]), np.array([])))
        if len(bin_means) > 0:
            all_y_values.extend(bin_means)
    
    # Calculate y-axis range
    if all_y_values:
        y_min = min(all_y_values) * 1.2
        y_max = max(all_y_values) * 1.2
        # Ensure zero is visible
        y_min = min(y_min, -abs(y_max) * 0.1)
        y_max = max(y_max, abs(y_min) * 0.1)
    else:
        y_min, y_max = -1, 1
    
    # Create frames for each month
    frames = []
    for month in range(1, 13):
        bin_centers, bin_means, bin_stds = monthly_profiles.get(month, (np.array([]), np.array([]), np.array([])))
        
        if len(bin_centers) > 0:
            colors = ['#3498DB' if v >= 0 else '#E74C3C' for v in bin_means]
            frame_data = go.Bar(
                x=bin_centers,
                y=bin_means,
                marker_color=colors,
                name=f'{MONTH_NAMES[month-1]} Mean',
                error_y=dict(type='data', array=bin_stds, visible=True, color='rgba(0,0,0,0.3)'),
                hovertemplate='%{x:.1f} km<br>Transport: %{y:.4f} ×10⁶ m³/s<extra></extra>'
            )
        else:
            frame_data = go.Bar(x=[], y=[], name=MONTH_NAMES[month-1])
        
        frames.append(go.Frame(
            data=[frame_data],
            name=MONTH_NAMES[month-1],
            layout=go.Layout(title=dict(text=f"{title_prefix} — {MONTH_NAMES[month-1]}"))
        ))
    
    # Initial frame (January)
    bin_centers, bin_means, bin_stds = monthly_profiles.get(1, (np.array([]), np.array([]), np.array([])))
    if len(bin_centers) > 0:
        colors = ['#3498DB' if v >= 0 else '#E74C3C' for v in bin_means]
        initial_trace = go.Bar(
            x=bin_centers,
            y=bin_means,
            marker_color=colors,
            name='Jan Mean',
            error_y=dict(type='data', array=bin_stds, visible=True, color='rgba(0,0,0,0.3)'),
            hovertemplate='%{x:.1f} km<br>Transport: %{y:.4f} ×10⁶ m³/s<extra></extra>'
        )
    else:
        initial_trace = go.Bar(x=[], y=[], name='Jan')
    
    fig = go.Figure(data=[initial_trace], frames=frames)
    
    # Add zero line
    fig.add_hline(y=0, line_color="#7F8C8D", line_width=1)
    
    # Animation controls
    fig.update_layout(
        title=dict(text=f"{title_prefix} — Jan", font=dict(size=16)),
        yaxis_title="Transport (×10⁶ m³/s)",
        xaxis_title="Distance along gate (km)",
        height=480,
        
        
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor='#E8E8E8', gridwidth=1),
        yaxis=dict(gridcolor='#E8E8E8', gridwidth=1, range=[y_min, y_max]),
        bargap=0.15,
        margin=dict(l=60, r=40, t=80, b=100),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=-0.15,
                x=0.1,
                xanchor="right",
                buttons=[
                    dict(
                        label="▶️ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 800, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300, "easing": "cubic-in-out"}
                        }]
                    ),
                    dict(
                        label="⏸️ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 14, "color": "#1E3A5F"},
                "prefix": "Month: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.8,
            "x": 0.15,
            "y": -0.05,
            "steps": [
                {
                    "args": [[MONTH_NAMES[i]], {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300}
                    }],
                    "label": MONTH_NAMES[i],
                    "method": "animate"
                }
                for i in range(12)
            ]
        }]
    )
    
    return fig


def _create_dot_monthly_timelapse(
    monthly_profiles: dict,
    lon_centers: np.ndarray,
    x_km: np.ndarray,
    title_prefix: str = "Mean DOT Profile",
    y_units: str = "cm",
    color: str = "#1E90FF"
) -> go.Figure:
    """
    Create animated timelapse for monthly climatological DOT profiles.
    
    Shows how the DOT profile varies by month (January through December),
    aggregating data from all years for each month.
    
    Args:
        monthly_profiles: Dict with month (1-12) as key, DOT profile array as value
        lon_centers: Longitude centers for each bin
        x_km: Distance array along gate (km)
        title_prefix: Title prefix for the animation
        y_units: Units for Y axis ('m', 'cm', 'mm')
        color: Line color for the profile
    
    Returns:
        Plotly Figure with animation frames
    """
    # Y scaling factor
    y_scale = {"m": 1.0, "cm": 100.0, "mm": 1000.0}.get(y_units, 100.0)
    y_label = f"DOT ({y_units})"
    
    # Collect all months data for consistent y-axis
    all_y_values = []
    for month in range(1, 13):
        profile = monthly_profiles.get(month, np.array([]))
        if len(profile) > 0:
            valid = profile[np.isfinite(profile)]
            if len(valid) > 0:
                all_y_values.extend(valid * y_scale)
    
    # Calculate y-axis range
    if all_y_values:
        y_min = min(all_y_values) * 1.1
        y_max = max(all_y_values) * 1.1
        y_range_diff = y_max - y_min
        y_min -= y_range_diff * 0.05
        y_max += y_range_diff * 0.05
    else:
        y_min, y_max = -10, 10
    
    # Create frames for each month
    frames = []
    for month in range(1, 13):
        profile = monthly_profiles.get(month, np.array([]))
        
        if len(profile) > 0 and np.any(np.isfinite(profile)):
            valid_mask = np.isfinite(profile)
            frame_data = go.Scatter(
                x=x_km[valid_mask] if len(x_km) == len(profile) else np.arange(np.sum(valid_mask)),
                y=profile[valid_mask] * y_scale,
                mode='lines',
                name=f'{MONTH_NAMES[month-1]}',
                line=dict(color=color, width=2.5),
                fill='tozeroy',
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)",
                hovertemplate='%{x:.1f} km<br>DOT: %{y:.2f} ' + y_units + '<extra></extra>'
            )
        else:
            frame_data = go.Scatter(x=[], y=[], mode='lines', name=MONTH_NAMES[month-1])
        
        frames.append(go.Frame(
            data=[frame_data],
            name=MONTH_NAMES[month-1],
            layout=go.Layout(title=dict(text=f"{title_prefix} — {MONTH_NAMES[month-1]}"))
        ))
    
    # Initial frame (January)
    profile = monthly_profiles.get(1, np.array([]))
    if len(profile) > 0 and np.any(np.isfinite(profile)):
        valid_mask = np.isfinite(profile)
        initial_trace = go.Scatter(
            x=x_km[valid_mask] if len(x_km) == len(profile) else np.arange(np.sum(valid_mask)),
            y=profile[valid_mask] * y_scale,
            mode='lines',
            name='DOT Profile',
            line=dict(color=color, width=2.5),
            fill='tozeroy',
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)",
            hovertemplate='%{x:.1f} km<br>DOT: %{y:.2f} ' + y_units + '<extra></extra>'
        )
    else:
        initial_trace = go.Scatter(x=[], y=[], mode='lines', name='DOT')
    
    fig = go.Figure(data=[initial_trace], frames=frames)
    
    # Add zero line reference
    fig.add_hline(y=0, line_color="#7F8C8D", line_width=1, line_dash="dash", opacity=0.5)
    
    # Animation controls with slider
    fig.update_layout(
        title=dict(text=f"{title_prefix} — Jan", font=dict(size=16)),
        yaxis_title=y_label,
        xaxis_title="Distance along gate (km)",
        height=500,
        
        
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor='#E8E8E8', gridwidth=1),
        yaxis=dict(gridcolor='#E8E8E8', gridwidth=1, range=[y_min, y_max]),
        margin=dict(l=60, r=40, t=80, b=100),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=-0.15,
                x=0.1,
                xanchor="right",
                buttons=[
                    dict(
                        label="▶️ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 800, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300, "easing": "cubic-in-out"}
                        }]
                    ),
                    dict(
                        label="⏸️ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 14, "color": color},
                "prefix": "Month: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.8,
            "x": 0.15,
            "y": -0.05,
            "steps": [
                {
                    "args": [[MONTH_NAMES[i]], {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300}
                    }],
                    "label": MONTH_NAMES[i],
                    "method": "animate"
                }
                for i in range(12)
            ]
        }]
    )
    
    return fig


def _get_all_loaded_datasets() -> dict:
    """Get all currently loaded datasets from session state."""
    loaded = {}
    
    if st.session_state.get("dataset_slcci") is not None:
        loaded["slcci"] = st.session_state.get("dataset_slcci")
    if st.session_state.get("dataset_cmems") is not None:
        loaded["cmems"] = st.session_state.get("dataset_cmems")
    if st.session_state.get("dataset_cmems_l4") is not None:
        loaded["cmems_l4"] = st.session_state.get("dataset_cmems_l4")
    if st.session_state.get("dataset_dtu") is not None:
        loaded["dtu"] = st.session_state.get("dataset_dtu")
    
    return loaded


def _format_loaded_names(loaded_datasets: dict) -> str:
    if not loaded_datasets:
        return "None"
    return ", ".join(DATASET_NAMES.get(k, k) for k in loaded_datasets.keys())


def _render_context_bar(config: AppConfig, loaded_datasets: dict):
    """Always-visible context summary for task-driven UI."""
    selected_dataset_type = st.session_state.get("sidebar_datasource") or st.session_state.get("selected_dataset_type", "SLCCI")
    selected_gate = st.session_state.get("selected_gate") or getattr(config, "selected_gate", None) or "None"
    cmems_l4 = st.session_state.get("dataset_cmems_l4")

    period = "n/a"
    points = "n/a"
    if cmems_l4 is not None:
        tr = getattr(cmems_l4, "time_range", ("", ""))
        if tr and tr[0]:
            period = f"{str(tr[0])[:10]} to {str(tr[1])[:10]}"
        gate_lon = getattr(cmems_l4, "gate_lon_pts", None)
        time_arr = getattr(cmems_l4, "time_array", None)
        n_pts = len(gate_lon) if gate_lon is not None else 0
        n_t = len(time_arr) if time_arr is not None else 0
        points = f"{n_pts} pts · {n_t} steps"

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset", selected_dataset_type)
        c2.metric("Gate", str(selected_gate))
        c3.metric("Period", period)
        c4.metric("Loaded", _format_loaded_names(loaded_datasets))
        st.caption(
            f"Grid stats: {points} | "
            f"Ice filter: {'ON' if getattr(config, 'cmems_l4_filter_ice', False) else 'OFF'} | "
            f"Cache: {'ON' if st.session_state.get('cmems_use_cache', True) else 'OFF'}"
        )


def _render_cmems_l4_kpi_row(cmems_l4_data):
    """Quick KPI row for CMEMS L4 task workspace."""
    strait_name = str(getattr(cmems_l4_data, "strait_name", "gate"))
    key_safe = strait_name.replace(" ", "_").lower()

    slope = np.asarray(getattr(cmems_l4_data, "slope_series", np.array([])))
    mean_slope = float(np.nanmean(slope)) if slope.size else np.nan

    gv_key = f"gv_{key_safe}"
    vt_key = f"vt_{key_safe}"
    fw_key = f"fw_{key_safe}"
    sf_key = f"salt_flux_{key_safe}"

    v_perp = st.session_state.get(f"{gv_key}_v_perp")
    vt = st.session_state.get(f"{vt_key}_transport_total_sv")
    fw = st.session_state.get(f"{fw_key}_fw")
    sf = st.session_state.get(f"{sf_key}_salt_flux")

    mean_v = float(np.nanmean(v_perp) * 100.0) if isinstance(v_perp, np.ndarray) else np.nan
    mean_vt = float(np.nanmean(vt)) if isinstance(vt, np.ndarray) else np.nan
    mean_fw = float(np.nanmean(fw) / 1e3) if isinstance(fw, np.ndarray) else np.nan  # mSv
    mean_sf = float(np.nanmean(sf) / 1e9) if isinstance(sf, np.ndarray) else np.nan  # Gg/s

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean slope", f"{mean_slope:.3f} m/100km" if np.isfinite(mean_slope) else "n/a")
    c2.metric("Mean v⊥", f"{mean_v:.2f} cm/s" if np.isfinite(mean_v) else "not computed")
    c3.metric("Mean VT", f"{mean_vt:.3f} Sv" if np.isfinite(mean_vt) else "not computed")
    c4.metric("Mean FW", f"{mean_fw:.2f} mSv" if np.isfinite(mean_fw) else "not computed")
    c5.metric("Mean SF", f"{mean_sf:.3f} Gg/s" if np.isfinite(mean_sf) else "not computed")

    velocity_done = isinstance(v_perp, np.ndarray)
    vt_done = isinstance(vt, np.ndarray)
    fw_done = bool(st.session_state.get(f"{fw_key}_done", False))
    sf_done = bool(st.session_state.get(f"{sf_key}_done", False))
    status_line = (
        f"Velocity: {'✅' if velocity_done else '⏳'}  "
        f"VT: {'✅' if vt_done else '⏳'}  "
        f"FW: {'✅' if fw_done else '⏳'}  "
        f"SF: {'✅' if sf_done else '⏳'}"
    )
    st.caption(f"Compute status · {status_line}")


def _render_cmems_l4_sign_convention_card(cmems_l4_data):
    """Single sign-convention card for the whole physics workspace."""
    strait_name = str(getattr(cmems_l4_data, "strait_name", "gate"))
    key_safe = strait_name.replace(" ", "_").lower()
    gv_key = f"gv_{key_safe}"
    v_info = st.session_state.get(f"{gv_key}_v_info", {}) or {}

    if not v_info:
        st.info(
            "Sign convention will appear here after computing velocity once. "
            "Positive v⊥ is defined as INTO Arctic."
        )
        return

    normal_dir = v_info.get("normal_direction", "?")
    mean_v = v_info.get("mean_v_perp", np.nan)
    pos_meaning = v_info.get("positive_means", "Into Arctic")
    neg_meaning = v_info.get("negative_means", "Out of Arctic")

    st.info(
        f"**Sign Convention** · INTO-Arctic direction ≈ **{normal_dir}** · "
        f"Positive v⊥ = {pos_meaning}, Negative v⊥ = {neg_meaning}, "
        f"Mean v⊥ = {mean_v * 100:.2f} cm/s"
    )


def _render_cmems_l4_task_workspace(cmems_l4_data, config: AppConfig):
    """Task-driven workspace with three macro areas."""
    st.subheader("CMEMS L4 Workspace")
    _render_cmems_l4_kpi_row(cmems_l4_data)
    st.divider()

    tab_overview, tab_physics, tab_export = st.tabs(["Overview", "Physics", "Export"])

    with tab_overview:
        st.caption("Explore core fields before transport computation.")
        ov1, ov2, ov3, ov4 = st.tabs(["Slope", "DOT", "Spatial", "Monthly"])
        with ov1:
            _render_dtu_slope_timeline(cmems_l4_data, config)
        with ov2:
            _render_dtu_dot_profile(cmems_l4_data, config)
        with ov3:
            _render_cmems_l4_spatial(cmems_l4_data, config)
        with ov4:
            _render_gridded_monthly_analysis(cmems_l4_data, config)

    with tab_physics:
        st.caption("Compute first, then inspect resulting transports.")
        _render_cmems_l4_sign_convention_card(cmems_l4_data)
        ph1, ph2, ph3, ph4, ph5 = st.tabs(["Velocity", "Volume", "Freshwater", "Salinity", "Salt"])
        with ph1:
            _render_geostrophic_velocity_tab_cmems_l4(cmems_l4_data, config)
        with ph2:
            _render_volume_transport_tab_cmems_l4(cmems_l4_data, config)
        with ph3:
            ds_info = _get_unified_dataset_info(cmems_l4_data, 'cmems_l4')
            render_freshwater_transport_clean(cmems_l4_data, config, ds_info)
        with ph4:
            ds_info = _get_unified_dataset_info(cmems_l4_data, 'cmems_l4')
            render_salinity_profile_tab(cmems_l4_data, config, ds_info)
        with ph5:
            ds_info = _get_unified_dataset_info(cmems_l4_data, 'cmems_l4')
            render_salt_flux_clean(cmems_l4_data, config, ds_info)

    with tab_export:
        st.caption("Export only after required computes are ready.")
        _render_cmems_l4_export_tab(cmems_l4_data, config)


def _render_tabs_task_mode(config: AppConfig):
    """Task-driven main content renderer."""
    slcci_data = get_slcci_data()
    cmems_data = get_cmems_data()
    dtu_data = get_dtu_data()
    cmems_l4_data = st.session_state.get("dataset_cmems_l4")

    loaded_datasets = _get_all_loaded_datasets()
    n_loaded = len(loaded_datasets)

    _render_context_bar(config, loaded_datasets)

    if n_loaded == 0:
        _render_empty_tabs(config)
        return

    if n_loaded >= 2:
        st.subheader("Comparison Workspace")
        st.caption("Dedicated page for multi-dataset comparison and consistency checks.")
        _render_multi_comparison_tabs(loaded_datasets, config)
        return

    if cmems_l4_data is not None:
        _render_cmems_l4_task_workspace(cmems_l4_data, config)
    elif dtu_data is not None:
        _render_dtu_tabs(dtu_data, config)
    elif slcci_data is not None:
        _render_slcci_tabs(slcci_data, config)
    elif cmems_data is not None:
        _render_cmems_tabs(cmems_data, config)
    else:
        render_tabs_legacy(config)


def render_tabs(config: AppConfig):
    """Render tabs using selected UI mode. Legacy renderer remains dormant but available."""
    ui_mode = st.session_state.get("ui_mode", getattr(config, "ui_mode", "New UI"))
    if ui_mode == "Legacy UI":
        render_tabs_legacy(config)
        return
    _render_tabs_task_mode(config)


def render_tabs_legacy(config: AppConfig):
    """Render main content tabs based on loaded data type and comparison mode."""
    slcci_data = get_slcci_data()
    cmems_data = get_cmems_data()
    dtu_data = get_dtu_data()
    cmems_l4_data = st.session_state.get("dataset_cmems_l4")
    
    # Get all loaded datasets
    loaded_datasets = _get_all_loaded_datasets()
    n_loaded = len(loaded_datasets)
    
    # Legacy support
    legacy_slcci = st.session_state.get("slcci_pass_data")
    datasets = st.session_state.get("datasets", {})
    selected_dataset_type = st.session_state.get("sidebar_datasource") or st.session_state.get("selected_dataset_type", "SLCCI")
    
    # Status bar showing loaded datasets
    if n_loaded > 0:
        loaded_names = [f"**{DATASET_NAMES.get(k, k)}**" for k in loaded_datasets.keys()]
        st.caption(f"� Loaded: {', '.join(loaded_names)} ({n_loaded} dataset{'s' if n_loaded > 1 else ''})")
    
    # MULTI-DATASET COMPARISON MODE (2+ datasets loaded)
    if n_loaded >= 2:
        _render_multi_comparison_tabs(loaded_datasets, config)
        return
    
    # SINGLE DATASET MODE
    if dtu_data is not None:
        _render_dtu_tabs(dtu_data, config)
    elif cmems_l4_data is not None:
        _render_cmems_l4_tabs(cmems_l4_data, config)
    elif slcci_data is not None:
        _render_slcci_tabs(slcci_data, config)
    elif cmems_data is not None:
        _render_cmems_tabs(cmems_data, config)
    elif selected_dataset_type == "SLCCI" and legacy_slcci is not None:
        _render_slcci_tabs(legacy_slcci, config)
    elif datasets:
        _render_generic_tabs(datasets, config)
    else:
        _render_empty_tabs(config)


def _render_empty_tabs(config: AppConfig):
    """Render welcome tabs when no data is loaded - with 3D Globe first."""
    tab1, tab2, tab3 = st.tabs(["🌍 Globe", "🏠 Welcome", "❓ Help"])
    
    with tab1:
        # Import and render the 3D globe component
        try:
            from .globe import render_globe_landing
            render_globe_landing()
        except ImportError as e:
            st.error(f"Globe component not available: {e}")
            st.info("Install plotly: `pip install plotly`")
        except Exception as e:
            st.error(f"Error rendering globe: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    with tab2:
        _render_welcome_landing(config)
    
    with tab3:
        _render_help_tab()


def _render_welcome_landing(config: AppConfig):
    """Render the welcome/landing page content."""
    st.markdown("## 🛰️ ARCFRESH Project")
    st.markdown("*Satellite Altimetry Analysis for Arctic Ocean*")
    
    st.info("""
    **Getting Started:**
    
    1. **🌍 Globe Tab** - Click on a gate to select it
    2. **Select Region** from the sidebar dropdown
    3. **Choose a Gate** for detailed analysis
    4. **Load Data** using one of three sources:
       - 🟠 **SLCCI** - ESA Sea Level CCI (J2 satellite, local NetCDF)
       - 🔵 **CMEMS** - Copernicus Marine (J1/J2/J3 merged, via API)
       - 🟢 **DTUSpace** - DTU gridded DOT (v4, local NetCDF)
    
    Data will appear in analysis tabs once loaded.
    """)
    
    st.divider()
    
    # Quick status dashboard
    st.markdown("### 📊 Current Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        datasets_loaded = len(st.session_state.get("datasets", {}))
        slcci = 1 if st.session_state.get("dataset_slcci") else 0
        cmems = 1 if st.session_state.get("dataset_cmems") else 0
        dtu = 1 if st.session_state.get("dataset_dtu") else 0
        total = datasets_loaded + slcci + cmems + dtu
        st.metric("Datasets Loaded", total)
    
    with col2:
        gate = st.session_state.get("selected_gate", "None")
        st.metric("Selected Gate", gate if gate else "None")
    
    with col3:
        dtype = st.session_state.get("selected_dataset_type", "None")
        st.metric("Data Source", dtype if dtype else "None")
    
    with col4:
        # Try to get gate count
        try:
            from src.services import GateService
            gs = GateService()
            gate_count = len(gs.list_gates())
        except:
            gate_count = "?"
        st.metric("Available Gates", gate_count)
    
    st.divider()
    
    # Dataset comparison
    st.markdown("### 📡 Dataset Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🟠 SLCCI")
        st.markdown("""
        - **Type**: Along-track
        - **Satellite**: Jason-2
        - **Source**: Local NetCDF
        - **Passes**: Per-orbit
        - **Best for**: Single-pass analysis
        """)
    
    with col2:
        st.markdown("#### 🔵 CMEMS")
        st.markdown("""
        - **Type**: Along-track (merged)
        - **Satellites**: J1, J2, J3
        - **Source**: Copernicus API
        - **Tracks**: Multi-satellite
        - **Best for**: Time series
        """)
    
    with col3:
        st.markdown("#### 🟢 DTUSpace")
        st.markdown("""
        - **Type**: Gridded (lat×lon×time)
        - **Product**: DOT monthly means
        - **Source**: Local NetCDF
        - **Resolution**: ~0.25° grid
        - **Best for**: Spatial patterns
        """)


def _render_help_tab():
    """Render the help documentation tab."""
    st.markdown("## ❓ Help & Documentation")
    
    with st.expander("🟠 SLCCI Data", expanded=True):
        st.markdown("""
        **ESA Sea Level Climate Change Initiative**
        
        - Uses local NetCDF files from Jason-2 satellite
        - Select pass number and cycle range in sidebar
        - Shows slope, DOT profiles, and spatial maps
        
        **Required Files:**
        - J2 data directory with cycle folders
        - TUM geoid file (ogmoc.nc)
        """)
    
    with st.expander("🔵 CMEMS Data"):
        st.markdown("""
        **Copernicus Marine Environment Monitoring Service**
        
        - Downloads data via Copernicus API
        - Merged Jason-1, Jason-2, Jason-3 tracks
        - Supports SEALEVEL_EUR_PHY_L3_NRT_019_003
        
        **Setup:**
        - Configure credentials in `config/credentials.yaml`
        - Or set COPERNICUS_USERNAME/PASSWORD env vars
        """)
    
    with st.expander("🟢 DTUSpace Data"):
        st.markdown("""
        **DTU Space Gridded DOT Products**
        
        - Monthly mean Dynamic Ocean Topography
        - Gridded at ~0.25° resolution
        - 2006-2017 coverage (v4.0)
        
        **Required Files:**
        - arctic_ocean_prod_DTUSpace_v4.0.nc
        """)
    
    with st.expander("🚪 Gate System"):
        st.markdown("""
        **Oceanographic Gates**
        
        Gates define transects across straits for flux analysis.
        
        - **Fram Strait**: Arctic-Atlantic exchange
        - **Bering Strait**: Pacific inflow
        - **Davis Strait**: Labrador Sea connection
        - **Denmark Strait**: Nordic Seas overflow
        
        Gate shapefiles are in `gates/` directory.
        """)


def _render_slcci_tabs(slcci_data: Dict[str, Any], config: AppConfig):
    """Render tabs for SLCCI satellite data — full physics pipeline."""
    ds_info = _get_unified_dataset_info(slcci_data, "slcci")

    tab1, tab2, tab3, tab4, tab5, tab_vt, tab_fw, tab_sal, tab_sf, tab_ex = st.tabs([
        f"{ds_info['emoji']} Slope Timeline",
        f"{ds_info['emoji']} DOT Profile",
        f"{ds_info['emoji']} Spatial Map",
        f"{ds_info['emoji']} Monthly Analysis",
        f"{ds_info['emoji']} Geostrophic Velocity",
        "🚢 Volume Transport",
        "💧 Freshwater Transport",
        "🧪 Salinity Profile",
        "🧂 Salt Flux",
        "📥 Export",
    ])

    with tab1:
        _render_unified_slope_timeline(slcci_data, config, ds_info)
    with tab2:
        _render_unified_dot_profile(slcci_data, config, ds_info)
    with tab3:
        _render_unified_spatial_map(slcci_data, config, ds_info)
    with tab4:
        _render_unified_monthly_analysis(slcci_data, config, ds_info)
    with tab5:
        _render_unified_geostrophic_velocity(slcci_data, config, ds_info)
    with tab_vt:
        _render_volume_transport_tab_cmems_l4(slcci_data, config)
    with tab_fw:
        render_freshwater_transport_clean(slcci_data, config, ds_info)
    with tab_sal:
        render_salinity_profile_tab(slcci_data, config, ds_info)
    with tab_sf:
        render_salt_flux_clean(slcci_data, config, ds_info)
    with tab_ex:
        _render_unified_export_tab(slcci_data, config, ds_info)


def _render_unified_dataset_tabs(data, config: AppConfig, dataset_type: str = "auto"):
    """
    Unified tab rendering for ALL datasets (SLCCI, CMEMS L4, DTUSpace).
    
    Same 6 tabs with identical structure:
    1. Slope Timeline
    2. DOT Profile  
    3. Spatial Map
    4. Monthly Analysis
    5. Geostrophic Velocity
    6. Export
    
    Each function auto-detects dataset type and renders appropriately.
    """
    # Auto-detect dataset type if not specified
    if dataset_type == "auto":
        data_source = getattr(data, 'data_source', '')
        if 'cmems' in data_source.lower() or 'CMEMS' in str(type(data)):
            dataset_type = "cmems_l4"
        elif 'dtu' in data_source.lower() or 'DTU' in str(type(data)):
            dataset_type = "dtu"
        else:
            dataset_type = "slcci"
    
    # Get dataset info for display
    ds_info = _get_unified_dataset_info(data, dataset_type)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        f"{ds_info['emoji']} Slope Timeline",
        f"{ds_info['emoji']} DOT Profile",
        f"{ds_info['emoji']} Spatial Map",
        f"{ds_info['emoji']} Monthly Analysis",
        f"{ds_info['emoji']} Geostrophic Velocity",
        "📥 Export"
    ])
    
    with tab1:
        _render_unified_slope_timeline(data, config, ds_info)
    with tab2:
        _render_unified_dot_profile(data, config, ds_info)
    with tab3:
        _render_unified_spatial_map(data, config, ds_info)
    with tab4:
        _render_unified_monthly_analysis(data, config, ds_info)
    with tab5:
        _render_unified_geostrophic_velocity(data, config, ds_info)
    with tab6:
        _render_unified_export_tab(data, config, ds_info)


def _get_unified_dataset_info(data, dataset_type: str) -> dict:
    """Get unified display info for any dataset type."""
    data_source = getattr(data, 'data_source', '')
    dataset_name = getattr(data, 'dataset_name', '')
    
    # Detect from data_source if type is generic
    if 'cmems' in data_source.lower() or 'cmems' in dataset_name.lower():
        return {
            'emoji': '🟣',
            'name': dataset_name or 'CMEMS L4',
            'color': '#9B59B6',  # Purple
            'type': 'cmems_l4'
        }
    elif 'dtu' in data_source.lower() or 'dtu' in dataset_name.lower():
        return {
            'emoji': '🟢',
            'name': dataset_name or 'DTUSpace v4',
            'color': '#2ECC71',  # Green
            'type': 'dtu'
        }
    else:
        # SLCCI
        pass_num = getattr(data, 'pass_number', '')
        name = f"SLCCI Pass {pass_num}" if pass_num else 'SLCCI'
        return {
            'emoji': '🟠',
            'name': name,
            'color': '#FF7F0E',  # Orange
            'type': 'slcci'
        }


# ==============================================================================
# UNIFIED TAB 1: SLOPE TIMELINE
# ==============================================================================
def _render_unified_slope_timeline(data, config: AppConfig, ds_info: dict):
    """
    Unified slope timeline for ALL datasets.
    Works with SLCCI, CMEMS L4, and DTUSpace data structures.
    """
    st.subheader(f"{ds_info['emoji']} {ds_info['name']} - Slope Timeline")
    
    # Get attributes (works for all dataset types)
    slope_series = getattr(data, 'slope_series', None)
    time_array = getattr(data, 'time_array', None)
    strait_name = getattr(data, 'strait_name', 'Unknown')
    
    # Get time range info
    time_range = getattr(data, 'time_range', None)
    start_year = getattr(data, 'start_year', None)
    end_year = getattr(data, 'end_year', None)
    
    if time_range and not start_year:
        start_year = time_range[0][:4] if time_range[0] else '?'
        end_year = time_range[1][:4] if time_range[1] else '?'
    
    if slope_series is None:
        st.error("❌ No slope_series available in data")
        return
    
    # Check for valid data
    valid_mask = ~np.isnan(slope_series)
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        st.warning("⚠️ All slope values are NaN")
        return
    
    # Info bar
    period_str = f"{start_year}–{end_year}" if start_year else ""
    st.info(f"📊 **{ds_info['name']}** | {strait_name} | {period_str}")
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        show_trend = st.checkbox("Show trend line", value=True, key=f"{ds_info['type']}_slope_trend")
    with col2:
        unit = st.selectbox("Units", ["m/100km", "cm/km"], key=f"{ds_info['type']}_slope_unit")
    
    # Convert units
    if unit == "cm/km":
        y_vals = slope_series * 100
        y_label = "Slope (cm/km)"
    else:
        y_vals = slope_series
        y_label = "Slope (m/100km)"
    
    # Build time axis
    if time_array is not None and len(time_array) > 0:
        time_pd = pd.to_datetime(time_array)
    else:
        time_pd = pd.date_range('2000-01', periods=len(slope_series), freq='MS')
    
    # Create figure
    fig = go.Figure()
    
    # Plot valid values only
    valid_x = time_pd[valid_mask]
    valid_y = y_vals[valid_mask]
    
    fig.add_trace(go.Scatter(
        x=valid_x,
        y=valid_y,
        mode="markers+lines",
        name="DOT Slope",
        marker=dict(size=6, color=ds_info['color']),
        line=dict(width=2, color=ds_info['color'])
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.8)
    
    # Trend line
    if show_trend and len(valid_y) > 2:
        x_numeric = np.arange(len(valid_y))
        z = np.polyfit(x_numeric, valid_y, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=valid_x,
            y=p(x_numeric),
            mode="lines",
            name=f"Trend ({z[0]:.4f}/step)",
            line=dict(dash="dash", color="darkred", width=1.5)
        ))
    
    fig.update_layout(
        title=f"{ds_info['name']} - {strait_name}<br><sup>DOT Slope Time Series</sup>",
        xaxis_title="Date",
        yaxis_title=y_label,
        height=500,
        
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_unified_slope_timeline_1")
    
    # Statistics
    with st.expander("📊 Statistics"):
        valid_slopes = slope_series[valid_mask]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{np.mean(valid_slopes):.4f} m/100km")
        with col2:
            st.metric("Std Dev", f"{np.std(valid_slopes):.4f} m/100km")
        with col3:
            st.metric("Min", f"{np.min(valid_slopes):.4f} m/100km")
        with col4:
            st.metric("Max", f"{np.max(valid_slopes):.4f} m/100km")
        
        st.caption(f"Valid time steps: {n_valid}/{len(slope_series)}")


# ==============================================================================
# UNIFIED TAB 2: DOT PROFILE
# ==============================================================================
def _render_unified_dot_profile(data, config: AppConfig, ds_info: dict):
    """
    Unified DOT profile across gate for ALL datasets.
    X-axis can be distance (km) or longitude.
    Y-axis can be m, cm, or mm.
    Supports monthly climatology view for datasets with monthly_profiles.
    """
    st.subheader(f"{ds_info['emoji']} {ds_info['name']} - Mean DOT Profile")
    
    # Get data (handle both along-track and gridded)
    profile_mean = getattr(data, 'profile_mean', None)
    x_km = getattr(data, 'x_km', None)
    gate_lon_pts = getattr(data, 'gate_lon_pts', None)
    dot_matrix = getattr(data, 'dot_matrix', None)
    df = getattr(data, 'df', None)
    strait_name = getattr(data, 'strait_name', 'Unknown')
    
    # Get monthly climatology data (if available, e.g., from SLCCI)
    monthly_profiles = getattr(data, 'monthly_profiles', None)
    monthly_lon_centers = getattr(data, 'monthly_lon_centers', None)
    monthly_x_km = getattr(data, 'monthly_x_km', None)
    
    # For SLCCI (along-track), compute profile from df
    if profile_mean is None and df is not None and 'dot' in df.columns:
        if 'lon' in df.columns:
            # Group by longitude bins to create profile
            df_sorted = df.sort_values('lon')
            profile_mean = df_sorted.groupby(pd.cut(df_sorted['lon'], bins=100))['dot'].mean().values
            x_km = np.linspace(0, 100, len(profile_mean))  # Approximate
            gate_lon_pts = np.linspace(df['lon'].min(), df['lon'].max(), len(profile_mean))
    
    if profile_mean is None or len(profile_mean) == 0:
        st.warning("⚠️ No DOT profile data available")
        return
    
    if x_km is None:
        x_km = np.arange(len(profile_mean))
    
    # Check for valid data
    valid_mask = ~np.isnan(profile_mean)
    if not np.any(valid_mask):
        st.warning("⚠️ All DOT values are NaN")
        return
    
    # Options - add Monthly Climatology if available
    col1, col2, col3 = st.columns(3)
    
    # Determine available view modes
    view_modes = ["Mean Profile", "Individual Time Steps"]
    if monthly_profiles is not None and len(monthly_profiles) > 0:
        view_modes.append("Monthly Climatology")
    
    with col1:
        view_mode = st.radio(
            "View mode",
            view_modes,
            horizontal=True,
            key=f"{ds_info['type']}_dot_view_mode"
        )
    with col2:
        x_axis_mode = st.selectbox("X-axis", ["Distance (km)", "Longitude (°)"], key=f"{ds_info['type']}_dot_xaxis")
    with col3:
        y_units = st.selectbox("Y units", ["m", "cm", "mm"], index=1, key=f"{ds_info['type']}_dot_yunits")  # Default to cm
    
    show_std = st.checkbox("Show ±1 Std Dev", value=True, key=f"{ds_info['type']}_dot_std")
    
    # Y scaling
    y_scale = {"m": 1.0, "cm": 100.0, "mm": 1000.0}[y_units]
    
    # X values
    if x_axis_mode == "Distance (km)":
        x_vals = x_km
        x_label = "Distance along gate (km)"
    else:
        x_vals = gate_lon_pts if gate_lon_pts is not None else x_km
        x_label = "Longitude (°)"
    
    # MONTHLY CLIMATOLOGY VIEW - with animated timelapse
    if view_mode == "Monthly Climatology":
        if monthly_profiles is None:
            st.warning("Monthly climatology data not available for this dataset")
            return
        
        st.markdown("### 📅 Monthly Climatological DOT Profiles")
        st.markdown("*Aggregates all observations by month across all years*")
        
        # Use monthly x_km if available
        m_x_km = monthly_x_km if monthly_x_km is not None else x_km
        m_lon_centers = monthly_lon_centers if monthly_lon_centers is not None else gate_lon_pts
        
        # Create the animated timelapse
        fig = _create_dot_monthly_timelapse(
            monthly_profiles=monthly_profiles,
            lon_centers=m_lon_centers if m_lon_centers is not None else np.array([]),
            x_km=m_x_km,
            title_prefix=f"{strait_name} — Monthly DOT Profile",
            y_units=y_units,
            color=ds_info['color']
        )
        
        st.plotly_chart(fig, use_container_width=True, key="_render_unified_dot_profile_1")
        
        # Monthly statistics
        with st.expander("📊 Monthly Statistics"):
            # Compute stats per month
            month_stats = []
            for month in range(1, 13):
                profile = monthly_profiles.get(month, np.array([]))
                if len(profile) > 0 and np.any(np.isfinite(profile)):
                    valid = profile[np.isfinite(profile)]
                    month_stats.append({
                        "Month": MONTH_NAMES[month-1],
                        "Mean DOT (cm)": f"{np.mean(valid) * 100:.2f}",
                        "Min DOT (cm)": f"{np.min(valid) * 100:.2f}",
                        "Max DOT (cm)": f"{np.max(valid) * 100:.2f}",
                        "Range (cm)": f"{(np.max(valid) - np.min(valid)) * 100:.2f}",
                        "Valid Points": f"{len(valid)}"
                    })
                else:
                    month_stats.append({
                        "Month": MONTH_NAMES[month-1],
                        "Mean DOT (cm)": "N/A",
                        "Min DOT (cm)": "N/A", 
                        "Max DOT (cm)": "N/A",
                        "Range (cm)": "N/A",
                        "Valid Points": "0"
                    })
            
            st.dataframe(pd.DataFrame(month_stats), use_container_width=True)
        
        return  # Exit early for monthly view
    
    # STANDARD VIEW MODES
    fig = go.Figure()
    
    if view_mode == "Mean Profile":
        # Plot mean profile
        fig.add_trace(go.Scatter(
            x=x_vals[valid_mask],
            y=profile_mean[valid_mask] * y_scale,
            mode="lines",
            name="Mean DOT",
            line=dict(color=ds_info['color'], width=2)
        ))
        
        # Add std band if requested
        if show_std and dot_matrix is not None:
            profile_std = np.nanstd(dot_matrix, axis=1) * y_scale
            upper = (profile_mean[valid_mask] * y_scale + profile_std[valid_mask])
            lower = (profile_mean[valid_mask] * y_scale - profile_std[valid_mask])
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_vals[valid_mask], x_vals[valid_mask][::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself',
                fillcolor=f"rgba({int(ds_info['color'][1:3], 16)}, {int(ds_info['color'][3:5], 16)}, {int(ds_info['color'][5:7], 16)}, 0.2)",
                line=dict(color='rgba(0,0,0,0)'),
                name='±1 Std Dev'
            ))
    
    else:  # Individual Time Steps
        if dot_matrix is None:
            st.warning("No time step data available for individual profiles")
        else:
            n_time = dot_matrix.shape[1]
            time_array = getattr(data, 'time_array', None)
            
            # Let user select time steps
            max_select = min(10, n_time)
            selected = st.multiselect(
                "Select time steps",
                options=list(range(n_time)),
                default=list(range(min(5, n_time))),
                format_func=lambda i: str(pd.Timestamp(time_array[i]).strftime('%Y-%m')) if time_array is not None else f"Step {i}",
                key=f"{ds_info['type']}_time_steps",
                max_selections=max_select
            )
            
            if selected:
                colors = px.colors.qualitative.Set2
                for i, idx in enumerate(selected):
                    profile = dot_matrix[:, idx]
                    mask = ~np.isnan(profile)
                    if np.any(mask):
                        color = colors[i % len(colors)]
                        label = str(pd.Timestamp(time_array[idx]).strftime('%Y-%m')) if time_array is not None else f"Step {idx}"
                        fig.add_trace(go.Scatter(
                            x=x_vals[mask],
                            y=profile[mask] * y_scale,
                            mode="lines",
                            name=label,
                            line=dict(color=color, width=1.5)
                        ))
    
    # Add WEST/EAST labels
    y_max = np.nanmax(profile_mean[valid_mask]) * y_scale
    y_min = np.nanmin(profile_mean[valid_mask]) * y_scale
    y_text = y_max - 0.05 * (y_max - y_min)
    
    fig.add_annotation(x=x_vals[valid_mask].min(), y=y_text, text="WEST", showarrow=False, font=dict(size=12, weight="bold"), xanchor="left")
    fig.add_annotation(x=x_vals[valid_mask].max(), y=y_text, text="EAST", showarrow=False, font=dict(size=12, weight="bold"), xanchor="right")
    
    fig.update_layout(
        title=f"{ds_info['name']} - {strait_name}<br><sup>Mean DOT Profile Across Gate</sup>",
        xaxis_title=x_label,
        yaxis_title=f"DOT ({y_units})",
        yaxis_tickformat=".3f",
        height=500,
        
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_unified_dot_profile_2")
    
    # Stats
    with st.expander("📊 Profile Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean DOT", f"{np.nanmean(profile_mean):.4f} m")
        with col2:
            st.metric("DOT Range", f"{np.nanmax(profile_mean) - np.nanmin(profile_mean):.4f} m")
        with col3:
            st.metric("Gate Length", f"{x_km[-1]:.1f} km" if len(x_km) > 0 else "N/A")
        with col4:
            st.metric("Valid Points", f"{np.sum(valid_mask)}/{len(profile_mean)}")


# ==============================================================================
# UNIFIED TAB 3: SPATIAL MAP
# ==============================================================================
def _load_cci_sss_coords(strait_name: str):
    """Load CCI SSS v5.5 lon/lat track for a gate. Returns (lons, lats, sss_mean) or (None, None, None)."""
    from pathlib import Path
    import xarray as xr
    NC_DIR = Path("/Users/nicolocaron/Desktop/ARCFRESH/straits/netcdf")
    SSS_MAP = {
        'fram strait': 'fram_strait_S3_pass_481_SSS_CCIv5.5.nc',
        'davis strait': 'davis_strait_SSS_CCIv5.5.nc',
        'denmark strait': 'denmark_strait_TPJ_pass_246_SSS_CCIv5.5.nc',
        'bering strait': 'bering_strait_TPJ_pass_076_SSS_CCIv5.5.nc',
        'barents sea opening': 'barents_sea_opening_S3_pass_481_SSS_CCIv5.5.nc',
        'barents opening': 'barents_sea_opening_S3_pass_481_SSS_CCIv5.5.nc',
        'norwegian sea boundary': 'norwegian_sea_boundary_TPJ_pass_220_SSS_CCIv5.5.nc',
        'nares strait': 'nares_strait_SSS_CCIv5.5.nc',
    }
    key = strait_name.lower().strip().replace('_', ' ')
    fname = SSS_MAP.get(key)
    if fname is None:
        for k, v in SSS_MAP.items():
            if k in key or key in k:
                fname = v
                break
    if fname is None:
        return None, None, None
    fp = NC_DIR / fname
    if not fp.exists():
        return None, None, None
    try:
        ds = xr.open_dataset(fp)
        lons = ds['longitude'].values.astype(float)
        lats = ds['latitude'].values.astype(float)
        sss_mean = np.nanmean(ds['sss'].values, axis=0)  # climatological mean per point
        ds.close()
        return lons, lats, sss_mean
    except Exception:
        return None, None, None


def _render_unified_spatial_map(data, config: AppConfig, ds_info: dict):
    """
    Unified spatial map for ALL datasets.
    Shows gate location, data coverage, and CCI SSS track alignment.
    """
    st.subheader(f"{ds_info['emoji']} {ds_info['name']} - Spatial Map")

    strait_name = getattr(data, 'strait_name', 'Unknown')
    gate_lon_pts = getattr(data, 'gate_lon_pts', None)
    gate_lat_pts = getattr(data, 'gate_lat_pts', None)
    df = getattr(data, 'df', None)
    time_range = getattr(data, 'time_range', ('', ''))
    n_obs = getattr(data, 'n_observations', 0)
    x_km_data = getattr(data, 'x_km', None)

    # For gridded data, also get the DOT grid
    dot_mean_grid = getattr(data, 'dot_mean_grid', None)
    lat_grid = getattr(data, 'lat_grid', None)
    lon_grid = getattr(data, 'lon_grid', None)

    # Load CCI SSS track for this gate
    cci_lons, cci_lats, cci_sss_mean = _load_cci_sss_coords(strait_name)

    # Info row
    period_str = ""
    if time_range[0]:
        period_str = f"{time_range[0][:10]} to {time_range[1][:10]}"

    info_cols = st.columns(4)
    info_cols[0].markdown(f"**Gate:** {strait_name}")
    if period_str:
        info_cols[1].markdown(f"**Period:** {period_str}")
    if n_obs:
        info_cols[2].markdown(f"**Observations:** {n_obs:,}")
    if x_km_data is not None and len(x_km_data) > 0:
        info_cols[3].markdown(f"**Gate length:** {float(x_km_data[-1]):.0f} km")

    # Options
    opt_cols = st.columns([2, 2, 2, 2])
    with opt_cols[0]:
        show_gate = st.checkbox("Show gate line", value=True, key=f"{ds_info['type']}_map_gate")
    with opt_cols[1]:
        show_gate_pts = st.checkbox("Show gate points", value=True, key=f"{ds_info['type']}_map_gate_pts")
    with opt_cols[2]:
        show_cci = st.checkbox("Show CCI SSS track", value=(cci_lons is not None), key=f"{ds_info['type']}_map_cci",
                               disabled=(cci_lons is None))
    with opt_cols[3]:
        if dot_mean_grid is not None:
            colorscale = st.selectbox("Colorscale", ["viridis", "RdBu_r", "Plasma"],
                                      key=f"{ds_info['type']}_map_colorscale")
        else:
            colorscale = "viridis"

    # ------------------------------------------------------------------ #
    # MAIN MAP                                                             #
    # ------------------------------------------------------------------ #
    fig = go.Figure()

    # GRIDDED DATA: Show heatmap
    if dot_mean_grid is not None and lat_grid is not None and lon_grid is not None:
        z_data = dot_mean_grid.values if hasattr(dot_mean_grid, 'values') else dot_mean_grid
        vmin = np.nanpercentile(z_data, 5)
        vmax = np.nanpercentile(z_data, 95)
        fig.add_trace(go.Heatmap(
            x=lon_grid, y=lat_grid, z=z_data,
            colorscale=colorscale, zmin=vmin, zmax=vmax,
            colorbar=dict(title="DOT (m)", x=1.0),
            name="Mean DOT",
        ))

    # ALONG-TRACK DATA: Show scatter points
    elif df is not None and 'lon' in df.columns and 'lat' in df.columns:
        df_sample = df.sample(min(5000, len(df))) if len(df) > 5000 else df
        fig.add_trace(go.Scatter(
            x=df_sample['lon'], y=df_sample['lat'], mode='markers',
            marker=dict(
                size=3,
                color=df_sample['dot'] if 'dot' in df_sample.columns else ds_info['color'],
                colorscale='viridis',
                colorbar=dict(title="DOT (m)", x=1.0) if 'dot' in df_sample.columns else None,
                opacity=0.6,
            ),
            name='Observations',
        ))

    # Gate line
    if show_gate and gate_lon_pts is not None and gate_lat_pts is not None:
        fig.add_trace(go.Scatter(
            x=gate_lon_pts, y=gate_lat_pts,
            mode="lines",
            name=f"Gate line ({len(gate_lon_pts)} pts)",
            line=dict(color="#E74C3C", width=2.5),
        ))

    # Gate points
    if show_gate_pts and gate_lon_pts is not None and gate_lat_pts is not None:
        fig.add_trace(go.Scatter(
            x=gate_lon_pts, y=gate_lat_pts,
            mode="markers",
            name=f"Gate points (n={len(gate_lon_pts)}, ~{x_km_data[-1]/len(gate_lon_pts):.1f} km/pt)" if x_km_data is not None else f"Gate points (n={len(gate_lon_pts)})",
            marker=dict(size=5, color="#E74C3C", symbol="circle", line=dict(color="white", width=0.5)),
            showlegend=True,
        ))

    # CCI SSS track (colour-coded by mean SSS)
    if show_cci and cci_lons is not None:
        nan_mask = np.isnan(cci_sss_mean)
        valid_mask = ~nan_mask
        # Valid points: colour by SSS value — green circles
        if np.any(valid_mask):
            fig.add_trace(go.Scatter(
                x=cci_lons[valid_mask], y=cci_lats[valid_mask],
                mode="markers",
                name=f"CCI SSS v5.5 track (n={len(cci_lons)}, valid={valid_mask.sum()})",
                marker=dict(
                    size=8,
                    color=cci_sss_mean[valid_mask],
                    colorscale="Greens",
                    cmin=np.nanpercentile(cci_sss_mean[valid_mask], 5),
                    cmax=np.nanpercentile(cci_sss_mean[valid_mask], 95),
                    colorbar=dict(title="Mean SSS (PSU)", x=1.08,
                                  tickfont=dict(color='#111111'),
                                  title_font=dict(color='#111111')),
                    symbol="circle",
                    line=dict(color="#1A1A1A", width=0.5),
                ),
            ))
        # NaN points (no CCI coverage) — grey circles
        if np.any(nan_mask):
            fig.add_trace(go.Scatter(
                x=cci_lons[nan_mask], y=cci_lats[nan_mask],
                mode="markers",
                name=f"CCI SSS — no coverage (n={nan_mask.sum()})",
                marker=dict(size=6, color="lightgray", symbol="circle",
                            line=dict(color="#888", width=0.5)),
            ))

    # If no gate coords but df exists, approximate
    elif show_gate and df is not None and 'lon' in df.columns:
        lon_range = [df['lon'].min(), df['lon'].max()]
        lat_mean = df['lat'].mean() if 'lat' in df.columns else 70
        fig.add_trace(go.Scatter(
            x=lon_range, y=[lat_mean, lat_mean],
            mode="lines", name="Approx Gate",
            line=dict(color="#E74C3C", width=2, dash="dash"),
        ))

    _ax_style = dict(showgrid=True, gridcolor='#E0E0E0', gridwidth=1,
                     zeroline=False, title_font=dict(color='#111111'),
                     tickfont=dict(color='#111111'))
    fig.update_layout(
        title=dict(text=f"{ds_info['name']} — {strait_name} — Spatial Coverage & CCI SSS Track",
                   font=dict(color='#111111', size=14)),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        height=620,
        legend=dict(orientation='v', x=1.18, y=1.0, bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#CCC', borderwidth=1, font=dict(color='#111111')),
        yaxis=dict(scaleanchor="x", scaleratio=1, **_ax_style),
        xaxis=dict(**_ax_style),
        margin=dict(r=260),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#111111'),
    )
    st.plotly_chart(fig, use_container_width=True, key="_render_unified_spatial_map_1")

    # ------------------------------------------------------------------ #
    # SSS PROFILE ALONG GATE                                               #
    # ------------------------------------------------------------------ #
    if cci_lons is not None and cci_sss_mean is not None:
        st.divider()
        st.subheader("🌊 CCI SSS v5.5 — Surface Salinity Profile Along Gate")

        valid_mask_sss = ~np.isnan(cci_sss_mean)

        # Compute cumulative along-track distance (km) for the CCI track
        # This avoids the broken x-axis when longitude is non-monotonic (e.g. Fram Strait crosses 0°)
        _cci_dist_km = np.zeros(len(cci_lons))
        for _i in range(1, len(cci_lons)):
            _dlat = (cci_lats[_i] - cci_lats[_i-1]) * 111.0
            _dlon = (cci_lons[_i] - cci_lons[_i-1]) * 111.0 * np.cos(np.deg2rad(cci_lats[_i-1]))
            _cci_dist_km[_i] = _cci_dist_km[_i-1] + np.sqrt(_dlat**2 + _dlon**2)

        # Load full time series to get std (interannual variability)
        try:
            from pathlib import Path as _Path
            import xarray as _xr
            _NC_DIR = _Path("/Users/nicolocaron/Desktop/ARCFRESH/straits/netcdf")
            _SSS_MAP = {
                'fram strait': 'fram_strait_S3_pass_481_SSS_CCIv5.5.nc',
                'davis strait': 'davis_strait_SSS_CCIv5.5.nc',
                'denmark strait': 'denmark_strait_TPJ_pass_246_SSS_CCIv5.5.nc',
                'bering strait': 'bering_strait_TPJ_pass_076_SSS_CCIv5.5.nc',
                'barents sea opening': 'barents_sea_opening_S3_pass_481_SSS_CCIv5.5.nc',
                'barents opening': 'barents_sea_opening_S3_pass_481_SSS_CCIv5.5.nc',
                'norwegian sea boundary': 'norwegian_sea_boundary_TPJ_pass_220_SSS_CCIv5.5.nc',
                'nares strait': 'nares_strait_SSS_CCIv5.5.nc',
            }
            _key = strait_name.lower().strip().replace('_', ' ')
            _fname = _SSS_MAP.get(_key)
            if _fname is None:
                for k, v in _SSS_MAP.items():
                    if k in _key or _key in k:
                        _fname = v
                        break
            _sss_std = None
            _n_months = None
            if _fname:
                _fp = _NC_DIR / _fname
                if _fp.exists():
                    _ds = _xr.open_dataset(_fp)
                    _sss_all = _ds['sss'].values  # (time, nb_prof)
                    _sss_std = np.nanstd(_sss_all, axis=0)
                    _n_months = _sss_all.shape[0]
                    _ds.close()
        except Exception:
            _sss_std = None
            _n_months = None

        fig_sss = go.Figure()

        # ±1 std band
        if _sss_std is not None and np.any(valid_mask_sss):
            _upper = cci_sss_mean + _sss_std
            _lower = cci_sss_mean - _sss_std
            _x_valid = _cci_dist_km[valid_mask_sss]
            fig_sss.add_trace(go.Scatter(
                x=np.concatenate([_x_valid, _x_valid[::-1]]),
                y=np.concatenate([_upper[valid_mask_sss], _lower[valid_mask_sss][::-1]]),
                fill='toself',
                fillcolor='rgba(41,128,185,0.18)',
                line=dict(color='rgba(0,0,0,0)'),
                name='±1 Std (interannual)',
                showlegend=True,
            ))

        # Mean SSS line — green circles
        if np.any(valid_mask_sss):
            fig_sss.add_trace(go.Scatter(
                x=_cci_dist_km[valid_mask_sss],
                y=cci_sss_mean[valid_mask_sss],
                mode='lines+markers',
                name='Climatological mean SSS',
                line=dict(color='#2E8B57', width=2.5),
                marker=dict(size=6, color='#2E8B57', symbol='circle'),
            ))

        # NaN / no-coverage points — grey circles at bottom of plot
        if np.any(~valid_mask_sss):
            _y_nan_pos = np.full(np.sum(~valid_mask_sss),
                                 np.nanmin(cci_sss_mean[valid_mask_sss]) - 0.3 if np.any(valid_mask_sss) else 30.0)
            fig_sss.add_trace(go.Scatter(
                x=_cci_dist_km[~valid_mask_sss],
                y=_y_nan_pos,
                mode='markers',
                name=f'No CCI coverage (n={np.sum(~valid_mask_sss)})',
                marker=dict(size=7, color='lightgray', symbol='circle',
                            line=dict(color='#999', width=0.8)),
            ))

        nan_pct_sss = np.isnan(cci_sss_mean).sum() / len(cci_sss_mean) * 100
        fig_sss.update_layout(
            title=dict(
                text=(
                    f"Surface Salinity Along Gate — CCI SSS v5.5 (satellite) — {strait_name}<br>"
                    f"<sup>{_n_months or '?'} monthly timesteps · "
                    f"{np.sum(valid_mask_sss)}/{len(cci_lons)} valid points · "
                    f"{nan_pct_sss:.0f}% NaN (sea-ice / no coverage)</sup>"
                ),
                font=dict(size=13, color='#111111'),
            ),
            xaxis_title="Distance along CCI track (km)",
            yaxis_title="Sea Surface Salinity (PSU)",
            height=380,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                        font=dict(color='#111111')),
            hovermode='x unified',
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='Inter, sans-serif', color='#111111'),
            xaxis=dict(
                showgrid=True, gridcolor='#E0E0E0', gridwidth=1,
                zeroline=False,
                title_font=dict(color='#111111'),
                tickfont=dict(color='#111111'),
            ),
            yaxis=dict(
                showgrid=True, gridcolor='#E0E0E0', gridwidth=1,
                zeroline=False,
                title_font=dict(color='#111111'),
                tickfont=dict(color='#111111'),
            ),
        )
        st.plotly_chart(fig_sss, use_container_width=True, key="_render_unified_spatial_map_2")
        st.caption(
            f"**Source:** ESA CCI Sea Surface Salinity v5.5 — satellite-derived monthly SSS, ~2010–2023. "
            f"Mean ± 1σ (interannual variability). "
            f"X-axis = cumulative along-track distance (km) along the CCI satellite ground track. "
            f"Grey circles = no satellite coverage (sea-ice / high-latitude gaps)."
        )

    # ------------------------------------------------------------------ #
    # SSS TRACK ALIGNMENT SECTION                                          #
    # ------------------------------------------------------------------ #
    if cci_lons is not None and gate_lon_pts is not None and gate_lat_pts is not None:
        st.divider()
        st.subheader("🔍 CCI SSS Track vs Gate Alignment")

        # Compute along-gate lengths
        def _km(lons, lats):
            d = np.zeros(len(lons))
            for i in range(1, len(lons)):
                dlat = (lats[i] - lats[i-1]) * 111.0
                dlon = (lons[i] - lons[i-1]) * 111.0 * np.cos(np.deg2rad(lats[i-1]))
                d[i] = d[i-1] + np.sqrt(dlat**2 + dlon**2)
            return d

        gate_cum_km = _km(gate_lon_pts, gate_lat_pts)
        cci_cum_km = _km(cci_lons, cci_lats)
        gate_total_km = float(gate_cum_km[-1])
        cci_total_km = float(cci_cum_km[-1])
        ratio = cci_total_km / gate_total_km if gate_total_km > 0 else 1.0

        # Metric summary
        m_cols = st.columns(5)
        m_cols[0].metric("Gate pts", len(gate_lon_pts))
        m_cols[1].metric("Gate length", f"{gate_total_km:.1f} km")
        m_cols[2].metric("CCI pts", len(cci_lons))
        m_cols[3].metric("CCI track length", f"{cci_total_km:.1f} km")
        delta_str = f"+{cci_total_km - gate_total_km:.0f} km ({(ratio-1)*100:+.0f}%)"
        m_cols[4].metric("Track discrepancy", delta_str,
                         delta=f"{(ratio-1)*100:+.0f}%",
                         delta_color="inverse")

        # Two-panel alignment figure
        fig_align = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Spatial alignment (map view)",
                "Point-to-point distance offset (km)",
            ],
            column_widths=[0.58, 0.42],
            horizontal_spacing=0.10,
        )

        # LEFT: map overlay
        fig_align.add_trace(go.Scatter(
            x=gate_lon_pts, y=gate_lat_pts,
            mode="lines+markers",
            name="Gate (shapefile)",
            line=dict(color="#E74C3C", width=2),
            marker=dict(size=4, color="#E74C3C"),
        ), row=1, col=1)

        fig_align.add_trace(go.Scatter(
            x=cci_lons, y=cci_lats,
            mode="lines+markers",
            name="CCI SSS v5.5 track",
            line=dict(color="#2E8B57", width=2, dash="dot"),
            marker=dict(size=6, color="#2E8B57", symbol="circle"),
        ), row=1, col=1)

        # RIGHT: point-by-point lat/lon offset along normalised distance
        norm_gate = gate_cum_km / gate_total_km
        norm_cci = cci_cum_km / cci_total_km
        # Interpolate CCI onto gate's normalised positions to compute offset
        from scipy.interpolate import interp1d as _interp1d
        f_cci_lon = _interp1d(norm_cci, cci_lons, kind='linear', bounds_error=False, fill_value=np.nan)
        f_cci_lat = _interp1d(norm_cci, cci_lats, kind='linear', bounds_error=False, fill_value=np.nan)
        cci_lon_at_gate = f_cci_lon(norm_gate)
        cci_lat_at_gate = f_cci_lat(norm_gate)

        # Offset in km at each gate point
        dlat_offset = (cci_lat_at_gate - gate_lat_pts) * 111.0
        dlon_offset = (cci_lon_at_gate - gate_lon_pts) * 111.0 * np.cos(np.deg2rad(gate_lat_pts))
        dist_offset_km = np.sqrt(dlat_offset**2 + dlon_offset**2)

        fig_align.add_trace(go.Scatter(
            x=gate_cum_km, y=dist_offset_km,
            mode="lines+markers",
            name="Spatial offset (km)",
            line=dict(color="#E67E22", width=2),
            marker=dict(size=4, color="#E67E22"),
            fill='tozeroy', fillcolor='rgba(230,126,34,0.12)',
        ), row=1, col=2)

        fig_align.add_hline(
            y=float(np.nanmean(dist_offset_km)),
            line_dash="dash", line_color="#E74C3C",
            annotation_text=f"Mean offset: {np.nanmean(dist_offset_km):.1f} km",
            annotation_position="right",
            row=1, col=2,
        )

        fig_align.update_xaxes(title_text="Longitude (°E)", row=1, col=1,
                               showgrid=True, gridcolor='#E0E0E0',
                               title_font=dict(color='#111111'), tickfont=dict(color='#111111'))
        fig_align.update_yaxes(title_text="Latitude (°N)", row=1, col=1,
                               scaleanchor="x", scaleratio=1,
                               showgrid=True, gridcolor='#E0E0E0',
                               title_font=dict(color='#111111'), tickfont=dict(color='#111111'))
        fig_align.update_xaxes(title_text="Distance along gate (km)", row=1, col=2,
                               showgrid=True, gridcolor='#E0E0E0',
                               title_font=dict(color='#111111'), tickfont=dict(color='#111111'))
        fig_align.update_yaxes(title_text="CCI–Gate offset (km)", row=1, col=2,
                               showgrid=True, gridcolor='#E0E0E0',
                               title_font=dict(color='#111111'), tickfont=dict(color='#111111'))

        fig_align.update_layout(
            height=480,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='Inter, sans-serif', color='#111111'),
            title=dict(
                text=f"CCI SSS v5.5 track vs Gate shapefile — {strait_name}<br>"
                     f"<sup>Track length discrepancy: {delta_str} | "
                     f"Mean point offset: {np.nanmean(dist_offset_km):.1f} km | "
                     f"Interpolation method: normalised along-gate distance (0→1)</sup>",
                font=dict(size=13, color='#111111'),
            ),
            legend=dict(orientation='h', y=-0.15, font=dict(color='#111111')),
            margin=dict(b=80),
        )
        st.plotly_chart(fig_align, use_container_width=True, key="_render_unified_spatial_map_3")

        st.caption(
            "**How to read this:** Left panel shows the two tracks on the same map. "
            "Right panel shows the spatial offset (km) between the CCI ground track and the gate shapefile "
            "at each normalised position along the gate. "
            "A large offset means the CCI salinity is assigned to a point that is physically offset "
            "from the corresponding gate point — this introduces a positional uncertainty in the salinity field."
        )

    # ------------------------------------------------------------------ #
    # STATS EXPANDER                                                        #
    # ------------------------------------------------------------------ #
    with st.expander("📊 Spatial Statistics"):
        if gate_lon_pts is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Lon Range", f"{gate_lon_pts.min():.2f}° to {gate_lon_pts.max():.2f}°")
            with col2:
                st.metric("Lat Range", f"{gate_lat_pts.min():.2f}° to {gate_lat_pts.max():.2f}°")
            with col3:
                if x_km_data is not None and len(x_km_data) > 0:
                    st.metric("Gate Length", f"{x_km_data[-1]:.1f} km")
            with col4:
                st.metric("N Observations", f"{n_obs:,}" if n_obs else "N/A")


# ==============================================================================
# UNIFIED TAB 4: MONTHLY ANALYSIS
# ==============================================================================
def _render_unified_monthly_analysis(data, config: AppConfig, ds_info: dict):
    """
    Unified 12-month DOT analysis for ALL datasets.
    Shows DOT profile vs distance/longitude for each month with linear regression.
    Includes R² and slope statistics.
    """
    st.subheader(f"{ds_info['emoji']} {ds_info['name']} - Monthly Analysis")
    
    # Get required data
    dot_matrix = getattr(data, 'dot_matrix', None)
    time_array = getattr(data, 'time_array', None)
    x_km = getattr(data, 'x_km', None)
    gate_lon_pts = getattr(data, 'gate_lon_pts', None)
    df = getattr(data, 'df', None)
    strait_name = getattr(data, 'strait_name', 'Unknown')
    
    # For along-track data (SLCCI), use df directly
    if dot_matrix is None and df is not None and 'month' in df.columns:
        # Along-track mode
        _render_monthly_from_df(df, strait_name, ds_info, config)
        return
    
    # For gridded data (DTU, CMEMS L4)
    if dot_matrix is None or time_array is None or x_km is None:
        st.warning("⚠️ Missing data for monthly analysis")
        return
    
    # Convert time to pandas for month extraction
    time_pd = pd.to_datetime(time_array)
    months = time_pd.month
    
    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_regression = st.checkbox("Show linear regression", value=True, key=f"{ds_info['type']}_monthly_reg")
    with col2:
        x_axis_mode = st.selectbox("X-axis", ["Distance (km)", "Longitude (°)"], key=f"{ds_info['type']}_monthly_xaxis")
    with col3:
        y_units = st.selectbox("Y units", ["m", "cm", "mm"], key=f"{ds_info['type']}_monthly_yunits")
    
    # Y-axis scaling
    y_scale = {"m": 1.0, "cm": 100.0, "mm": 1000.0}[y_units]
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=[f"{month_names[i]} ({i+1})" for i in range(12)],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    # X-axis values
    if x_axis_mode == "Distance (km)":
        x_vals = x_km
        x_label = "Distance (km)"
    else:
        x_vals = gate_lon_pts if gate_lon_pts is not None else x_km
        x_label = "Longitude (°)"
    
    slopes_info = []
    
    for month in range(1, 13):
        row = (month - 1) // 4 + 1
        col = (month - 1) % 4 + 1
        
        # Get time indices for this month
        month_mask = months == month
        if not np.any(month_mask):
            continue
        
        # Average DOT profile for this month
        dot_month = dot_matrix[:, month_mask]
        dot_mean = np.nanmean(dot_month, axis=1)
        
        # Valid data mask
        mask = np.isfinite(x_vals) & np.isfinite(dot_mean)
        if np.sum(mask) < 2:
            continue
        
        x_valid = x_vals[mask]
        y_valid = dot_mean[mask] * y_scale
        
        # Scatter
        fig.add_trace(
            go.Scatter(
                x=x_valid, y=y_valid, mode='markers',
                marker=dict(size=4, color=ds_info['color'], opacity=0.6),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Regression
        if show_regression and len(x_valid) > 2:
            try:
                from scipy import stats as scipy_stats
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_valid, y_valid)
                r_squared = r_value ** 2
                
                # Convert slope to m/100km for standard comparison
                if x_axis_mode == "Distance (km)":
                    slope_m_100km = (slope / y_scale) * 100
                else:
                    # For longitude, approximate
                    slope_m_100km = slope / y_scale
                
                slopes_info.append({
                    'month': month,
                    'name': month_names[month-1],
                    'slope': slope,
                    'slope_m_100km': slope_m_100km,
                    'r_squared': r_squared,
                    'n_time': np.sum(month_mask),
                    'n_points': len(x_valid)
                })
                
                # Regression line
                x_line = np.linspace(x_valid.min(), x_valid.max(), 50)
                y_line = slope * x_line + intercept
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line, y=y_line, mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False,
                        hovertemplate=f"R²={r_squared:.3f}<br>slope={slope:.4f}"
                    ),
                    row=row, col=col
                )
                
                # Add annotation with slope and R² on each subplot
                fig.add_annotation(
                    text=f"R²={r_squared:.3f}<br>slope={slope_m_100km:.3f} m/100km",
                    xref=f"x{month}" if month > 1 else "x",
                    yref=f"y{month}" if month > 1 else "y",
                    x=x_valid.max(),
                    y=y_valid.max(),
                    xanchor="right",
                    yanchor="top",
                    showarrow=False,
                    font=dict(size=9, color="#2C3E50"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#E8E8E8",
                    borderwidth=1,
                    borderpad=3
                )
            except Exception:
                pass
    
    fig.update_layout(
        title=f"{ds_info['name']} - {strait_name} - Monthly Mean DOT Profile",
        height=700,
        
        showlegend=False
    )
    
    # Axis labels
    x_label_short = "km" if x_axis_mode == "Distance (km)" else "Lon (°)"
    for i in range(1, 13):
        row = (i - 1) // 4 + 1
        col = (i - 1) % 4 + 1
        if row == 3:
            fig.update_xaxes(title_text=x_label_short, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=f"DOT ({y_units})", row=row, col=col)
    
    st.plotly_chart(fig, width='stretch', key="_render_unified_monthly_analysis_1")
    
    # Summary table with R² and slope
    if slopes_info:
        with st.expander("📊 Monthly Slopes & R² Summary"):
            slopes_df = pd.DataFrame(slopes_info)
            
            display_df = pd.DataFrame({
                'Month': slopes_df['name'],
                f'Slope ({y_units}/{x_label_short})': slopes_df['slope'].apply(lambda x: f"{x:.4f}"),
                'Slope (m/100km)': slopes_df['slope_m_100km'].apply(lambda x: f"{x:.4f}"),
                'R²': slopes_df['r_squared'].apply(lambda x: f"{x:.3f}"),
                'N time steps': slopes_df['n_time'],
                'N points': slopes_df['n_points']
            })
            
            st.dataframe(display_df, width='stretch', hide_index=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Slope", f"{slopes_df['slope_m_100km'].mean():.4f} m/100km")
            with col2:
                st.metric("Std Dev", f"{slopes_df['slope_m_100km'].std():.4f} m/100km")
            with col3:
                st.metric("Mean R²", f"{slopes_df['r_squared'].mean():.3f}")
            with col4:
                st.metric("Months with Data", len(slopes_df))


def _render_monthly_from_df(df: pd.DataFrame, strait_name: str, ds_info: dict, config: AppConfig):
    """
    Render 12-month DOT analysis for along-track data (SLCCI-style).
    Takes a DataFrame with lon, lat, dot, month columns.
    """
    st.subheader(f"{ds_info['emoji']} {ds_info['name']} - 12 Months DOT Analysis")
    
    if df is None or df.empty:
        st.warning("No data available for monthly analysis.")
        return
    
    required_cols = ['lon', 'lat', 'dot', 'month']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return
    
    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_regression = st.checkbox("Show linear regression", value=True, key=f"monthly_df_reg_{ds_info['name']}")
    with col2:
        x_axis_mode = st.selectbox("X-axis", ["Distance (km)", "Longitude (°)"], key=f"monthly_df_xaxis_{ds_info['name']}")
    with col3:
        y_units = st.selectbox("Y units", ["m", "cm", "mm"], key=f"monthly_df_yunits_{ds_info['name']}")
    
    y_scale = {"m": 1.0, "cm": 100.0, "mm": 1000.0}[y_units]
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=[f"{month_names[i]} ({i+1})" for i in range(12)],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    R_earth = 6371.0
    mean_lat = df['lat'].mean()
    lat_rad = np.deg2rad(mean_lat)
    
    # Compute x_km for distance mode
    lon_min = df['lon'].min()
    df = df.copy()
    df['x_km'] = R_earth * np.deg2rad(df['lon'] - lon_min) * np.cos(lat_rad)
    
    slopes_info = []
    
    for month in range(1, 13):
        row = (month - 1) // 4 + 1
        col = (month - 1) % 4 + 1
        
        month_df = df[df['month'] == month]
        if len(month_df) < 2:
            continue
        
        if x_axis_mode == "Distance (km)":
            x_data = month_df['x_km'].values
        else:
            x_data = month_df['lon'].values
        
        dot_data = month_df['dot'].values * y_scale
        
        mask = np.isfinite(x_data) & np.isfinite(dot_data)
        if np.sum(mask) < 2:
            continue
        
        x_valid = x_data[mask]
        y_valid = dot_data[mask]
        
        fig.add_trace(
            go.Scatter(
                x=x_valid, y=y_valid, mode='markers',
                marker=dict(size=3, color=ds_info['color'], opacity=0.5),
                showlegend=False
            ),
            row=row, col=col
        )
        
        if show_regression and len(x_valid) > 2:
            try:
                from scipy import stats as scipy_stats
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_valid, y_valid)
                r_squared = r_value ** 2
                
                if x_axis_mode == "Distance (km)":
                    slope_m_100km = (slope / y_scale) * 100
                else:
                    km_per_deg = R_earth * np.cos(lat_rad) * np.pi / 180
                    slope_m_100km = (slope / y_scale) * km_per_deg * 100
                
                slopes_info.append({
                    'month': month, 'name': month_names[month-1],
                    'slope': slope, 'slope_m_100km': slope_m_100km,
                    'r_squared': r_squared, 'n_points': len(x_valid)
                })
                
                x_line = np.linspace(x_valid.min(), x_valid.max(), 50)
                y_line = slope * x_line + intercept
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line, y=y_line, mode='lines',
                        line=dict(color='red', width=2), showlegend=False
                    ),
                    row=row, col=col
                )
            except Exception:
                pass
    
    fig.update_layout(
        title=f"{ds_info['emoji']} {ds_info['name']} - {strait_name} - Monthly DOT Analysis",
        height=700,  showlegend=False
    )
    
    x_label_short = "km" if x_axis_mode == "Distance (km)" else "Lon (°)"
    for i in range(1, 13):
        row = (i - 1) // 4 + 1
        col = (i - 1) % 4 + 1
        if row == 3:
            fig.update_xaxes(title_text=x_label_short, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=f"DOT ({y_units})", row=row, col=col)
    
    st.plotly_chart(fig, width='stretch', key="_render_monthly_from_df_1")
    
    if slopes_info:
        with st.expander("📊 Monthly Slopes & R² Summary"):
            slopes_df = pd.DataFrame(slopes_info)
            display_df = pd.DataFrame({
                'Month': slopes_df['name'],
                f'Slope ({y_units}/{x_label_short})': slopes_df['slope'].apply(lambda x: f"{x:.4f}"),
                'Slope (m/100km)': slopes_df['slope_m_100km'].apply(lambda x: f"{x:.4f}"),
                'R²': slopes_df['r_squared'].apply(lambda x: f"{x:.3f}"),
                'N Points': slopes_df['n_points']
            })
            st.dataframe(display_df, width='stretch', hide_index=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Slope", f"{slopes_df['slope_m_100km'].mean():.4f} m/100km")
            with col2:
                st.metric("Std Dev", f"{slopes_df['slope_m_100km'].std():.4f} m/100km")
            with col3:
                st.metric("Mean R²", f"{slopes_df['r_squared'].mean():.3f}")
            with col4:
                st.metric("Months with Data", len(slopes_df))


# ==============================================================================
# UNIFIED GEOSTROPHIC VELOCITY
# ==============================================================================
def _render_unified_geostrophic_velocity(data, config: AppConfig, ds_info: dict):
    """
    Render geostrophic velocity analysis for any dataset.
    Uses v = -g/f * (dη/dx) where f = 2Ω sin(lat)
    """
    st.subheader(f"{ds_info['emoji']} {ds_info['name']} - Geostrophic Velocity")
    
    g = 9.81  # m/s²
    OMEGA = 7.2921e-5  # Earth's angular velocity (rad/s)
    
    strait_name = getattr(data, 'strait_name', 'Unknown')
    
    # Try to get pre-computed values (CMEMS/DTU style)
    v_geostrophic_series = getattr(data, 'v_geostrophic_series', None)
    mean_latitude = getattr(data, 'mean_latitude', None)
    coriolis_f = getattr(data, 'coriolis_f', None)
    slope_series = getattr(data, 'slope_series', None)
    time_array = getattr(data, 'time_array', None)
    
    # For SLCCI, compute from slope_series
    if v_geostrophic_series is None and slope_series is not None:
        if mean_latitude is None:
            df = getattr(data, 'df', None)
            if df is not None and 'lat' in df.columns:
                mean_latitude = df['lat'].mean()
            else:
                mean_latitude = 45.0  # fallback
        
        lat_rad = np.deg2rad(mean_latitude)
        coriolis_f = 2 * OMEGA * np.sin(lat_rad)
        
        # slope is in m/100km, convert to m/m
        slope_m_m = slope_series / 100000.0
        v_geostrophic_series = -g / coriolis_f * slope_m_m
    
    if v_geostrophic_series is None:
        st.warning("No geostrophic velocity data available. Requires slope data.")
        return
    
    # Build time index
    if hasattr(v_geostrophic_series, 'index'):
        time_index = v_geostrophic_series.index
        v_values = v_geostrophic_series.values
    else:
        if time_array is not None:
            time_index = pd.to_datetime(time_array)
        else:
            time_index = pd.date_range('2000-01', periods=len(v_geostrophic_series), freq='MS')
        v_values = np.array(v_geostrophic_series)
    
    st.info(f"Computed at lat={mean_latitude:.2f}° (f={coriolis_f:.2e} s⁻¹)")
    
    # Options
    col1, col2 = st.columns([1, 1])
    with col1:
        show_trend = st.checkbox("Show trend line", value=True, key=f"geo_trend_{ds_info['name']}")
    with col2:
        units = st.selectbox("Units", ["cm/s", "m/s"], key=f"geo_units_{ds_info['name']}")
    
    scale = 100.0 if units == "cm/s" else 1.0
    v_scaled = v_values * scale
    
    # Time series plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_index,
        y=v_scaled,
        mode='lines+markers',
        name='Geostrophic velocity',
        line=dict(color=ds_info['color'], width=2),
        marker=dict(size=5)
    ))
    
    if show_trend and len(v_scaled) > 2:
        try:
            from scipy import stats as scipy_stats
            x_numeric = np.arange(len(v_scaled))
            mask = np.isfinite(v_scaled)
            if np.sum(mask) > 2:
                slope, intercept, r_value, _, _ = scipy_stats.linregress(x_numeric[mask], v_scaled[mask])
                y_trend = slope * x_numeric + intercept
                fig.add_trace(go.Scatter(
                    x=time_index, y=y_trend, mode='lines',
                    name=f'Trend (R²={r_value**2:.3f})',
                    line=dict(color='red', dash='dash', width=2)
                ))
        except Exception:
            pass
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title=f"{ds_info['emoji']} {strait_name} - Geostrophic Velocity",
        xaxis_title="Date",
        yaxis_title=f"Velocity ({units})",
        height=500,
        
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_unified_geostrophic_velocity_1")
    
    # Statistics
    with st.expander("📈 Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{np.nanmean(v_scaled):.2f} {units}")
        with col2:
            st.metric("Std Dev", f"{np.nanstd(v_scaled):.2f} {units}")
        with col3:
            st.metric("Max", f"{np.nanmax(v_scaled):.2f} {units}")
        with col4:
            st.metric("Min", f"{np.nanmin(v_scaled):.2f} {units}")


# ==============================================================================
# UNIFIED EXPORT TAB
# ==============================================================================
def _render_unified_export_tab(data, config: AppConfig, ds_info: dict):
    """Unified export tab for all dataset types."""
    st.subheader(f"{ds_info['emoji']} {ds_info['name']} - Data Export")
    
    strait_name = getattr(data, 'strait_name', 'Unknown')
    pass_number = getattr(data, 'pass_number', 0)
    
    # Collect available data for export
    export_options = []
    
    # 1. Slope time series
    slope_series = getattr(data, 'slope_series', None)
    time_array = getattr(data, 'time_array', None)
    if slope_series is not None and len(slope_series) > 0:
        export_options.append("Slope Time Series")
    
    # 2. DOT profile (raw dataframe)
    df = getattr(data, 'df', None)
    if df is not None and not df.empty:
        export_options.append("DOT Observations (DataFrame)")
    
    # 3. DOT matrix (for gridded data)
    dot_matrix = getattr(data, 'dot_matrix', None)
    lon_array = getattr(data, 'lon_array', None)
    if dot_matrix is not None:
        export_options.append("DOT Matrix (Gridded)")
    
    # 4. Geostrophic velocity
    v_geo = getattr(data, 'v_geostrophic_series', None)
    if v_geo is not None:
        export_options.append("Geostrophic Velocity")
    
    # 5. Salt Flux along gate (4×3 monthly grid)
    sf_prefix = f"sf_{strait_name.replace(' ', '_').lower()}"
    if st.session_state.get(f'{sf_prefix}_done', False):
        export_options.append("Salt Flux Along Gate (4×3)")
    
    # 6. Freshwater Transport along gate (4×3 monthly grid)
    fw_prefix = f"fw_{strait_name.replace(' ', '_').lower()}"
    if st.session_state.get(f'{fw_prefix}_done', False):
        export_options.append("FW Transport Along Gate (4×3)")
    
    if not export_options:
        st.warning("No data available for export.")
        return
    
    selected = st.multiselect(
        "Select data to export",
        export_options,
        default=export_options[:1]
    )
    
    if not selected:
        st.info("Select at least one data type to export.")
        return
    
    st.markdown("---")
    
    # Export each selected type
    for export_type in selected:
        st.markdown(f"### {export_type}")
        
        if export_type == "Slope Time Series":
            if hasattr(slope_series, 'index'):
                df_export = pd.DataFrame({
                    'time': slope_series.index,
                    'slope_m_100km': slope_series.values
                })
            else:
                times = time_array if time_array is not None else np.arange(len(slope_series))
                df_export = pd.DataFrame({
                    'time': times,
                    'slope_m_100km': slope_series
                })
            
            st.dataframe(df_export.head(20), width='stretch')
            csv = df_export.to_csv(index=False)
            st.download_button(
                "📥 Download Slope CSV",
                csv,
                f"{strait_name}_{ds_info['type']}_slope.csv",
                "text/csv"
            )
        
        elif export_type == "DOT Observations (DataFrame)":
            st.dataframe(df.head(50), width='stretch')
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download DOT CSV",
                csv,
                f"{strait_name}_{ds_info['type']}_dot.csv",
                "text/csv"
            )
        
        elif export_type == "DOT Matrix (Gridded)":
            st.write(f"Matrix shape: {dot_matrix.shape} (lon × time)")
            # Create matrix DataFrame with lon as index
            if time_array is not None:
                cols = [str(t)[:10] for t in time_array]
            else:
                cols = [f"t_{i}" for i in range(dot_matrix.shape[1])]
            
            if lon_array is not None:
                idx = lon_array
            else:
                idx = np.arange(dot_matrix.shape[0])
            
            df_matrix = pd.DataFrame(dot_matrix, index=idx, columns=cols)
            st.dataframe(df_matrix.head(20), width='stretch')
            csv = df_matrix.to_csv()
            st.download_button(
                "📥 Download DOT Matrix CSV",
                csv,
                f"{strait_name}_{ds_info['type']}_dot_matrix.csv",
                "text/csv"
            )
        
        elif export_type == "Geostrophic Velocity":
            if hasattr(v_geo, 'index'):
                df_export = pd.DataFrame({
                    'time': v_geo.index,
                    'v_geostrophic_m_s': v_geo.values
                })
            else:
                times = time_array if time_array is not None else np.arange(len(v_geo))
                df_export = pd.DataFrame({
                    'time': times,
                    'v_geostrophic_m_s': v_geo
                })
            
            st.dataframe(df_export.head(20), width='stretch')
            csv = df_export.to_csv(index=False)
            st.download_button(
                "📥 Download Geostrophic CSV",
                csv,
                f"{strait_name}_{ds_info['type']}_geostrophic.csv",
                "text/csv"
            )
        
        elif export_type in ("Salt Flux Along Gate (4×3)", "FW Transport Along Gate (4×3)"):
            _render_transport_4x3_export(export_type, strait_name, sf_prefix, fw_prefix)


def _render_transport_4x3_export(export_type: str, strait_name: str,
                                  sf_prefix: str, fw_prefix: str):
    """Render 4×3 along-gate transport grid and provide download as PNG."""
    from .transport_plots import plot_transport_along_gate_4x3

    if "Salt Flux" in export_type:
        prefix = sf_prefix
        mode = 'salt'
        label = "Salt Flux"
    else:
        prefix = fw_prefix
        mode = 'freshwater'
        label = "FW Transport"

    v_perp = st.session_state.get(f'{prefix}_v_perp')
    x_km = st.session_state.get(f'{prefix}_x_km')
    time_array = st.session_state.get(f'{prefix}_time')
    H_profile = st.session_state.get(f'{prefix}_H_profile')
    sss_interp = st.session_state.get(f'{prefix}_sss_interp')
    dx = st.session_state.get(f'{prefix}_dx')
    s_ref = st.session_state.get(f'{prefix}_sref_val', 34.8)

    if any(x is None for x in [v_perp, time_array, H_profile, sss_interp, dx]):
        st.warning(f"⚠️ Compute {label} first in the corresponding tab.")
        return

    fig = plot_transport_along_gate_4x3(
        v_perp=v_perp, x_km=x_km, time_array=time_array,
        H_profile=H_profile, sss_interp=sss_interp, dx=dx,
        strait_name=strait_name, mode=mode, s_ref=s_ref,
    )

    if fig is None:
        st.warning("Could not generate 4×3 grid.")
        return

    st.plotly_chart(fig, use_container_width=True, key="_render_transport_4x3_export_1")

    # Offer PNG download via kaleido
    try:
        img_bytes = fig.to_image(format="png", width=1400, height=1200, scale=2)
        safe_name = strait_name.replace(' ', '_').lower()
        st.download_button(
            f"📥 Download {label} 4×3 PNG",
            img_bytes,
            f"{safe_name}_{mode}_along_gate_4x3.png",
            "image/png",
            key=f"{prefix}_4x3_png",
        )
    except Exception as e:
        st.info(f"💡 Install **kaleido** (`pip install kaleido`) for PNG export. Error: {e}")


# ==============================================================================
# ⚠️ LEGACY FUNCTIONS - DEPRECATED (2026-01-06)
# ==============================================================================
# The following functions are LEGACY and have been replaced by unified functions:
#
# REPLACED BY _render_unified_spatial_map():
#   - _render_spatial_map()
#
# REPLACED BY _render_unified_monthly_analysis():
#   - _render_slcci_monthly_analysis()
#   - _render_gridded_monthly_analysis()
#
# REPLACED BY _render_unified_geostrophic_velocity():
#   - _render_geostrophic_velocity()
#   - _render_dtu_geostrophic_velocity()
#
# REPLACED BY _render_unified_slope_timeline():
#   - _render_dtu_slope_timeline()
#
# REPLACED BY _render_unified_dot_profile():
#   - _render_dtu_dot_profile()
#
# REPLACED BY _render_unified_spatial_map():
#   - _render_dtu_spatial_map()
#
# REPLACED BY _render_unified_export_tab():
#   - _render_dtu_export_tab()
#
# DUPLICATE (already defined above):
#   - _get_gridded_dataset_info()
#
# These functions are kept for backwards compatibility and reference.
# DO NOT USE - Use the _render_unified_* functions instead.
# ==============================================================================


# ==============================================================================
# TAB 3: SPATIAL MAP (from SLCCI PLOTTER Panel 3) [LEGACY]
# ==============================================================================
def _render_spatial_map(slcci_data, config: AppConfig):
    """Render spatial map of DOT measurements."""
    st.subheader("Spatial Distribution")
    
    df = getattr(slcci_data, 'df', None)
    gate_lon_pts = getattr(slcci_data, 'gate_lon_pts', None)
    gate_lat_pts = getattr(slcci_data, 'gate_lat_pts', None)
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    pass_number = getattr(slcci_data, 'pass_number', 0)
    
    if df is None or df.empty:
        st.warning("No spatial data available.")
        return
    
    # Build available variables dynamically based on DataFrame columns
    # SLCCI has: corssh, geoid, dot
    # CMEMS has: sla_filtered, mdt, dot, satellite, cycle, track
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
    
    # Reset session state if current selection is not valid for this dataset
    if "map_color" in st.session_state and st.session_state.map_color not in available_vars:
        st.session_state.map_color = available_vars[0]
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        color_var = st.selectbox(
            "Color by",
            available_vars,
            index=0,  # Default to first available variable
            key="map_color"
        )
    with col2:
        show_gate = st.checkbox("Show gate line", value=True, key="map_gate")
    
    # Sample for performance
    if len(df) > 5000:
        plot_df = df.sample(5000)
        st.caption("Showing 5000 random points for performance")
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
            height=600,
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
            height=600,
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
    
    st.plotly_chart(fig, width='stretch', key="_render_spatial_map_1")


# ==============================================================================
# TAB 4: MONTHLY ANALYSIS (from SLCCI PLOTTER 12-subplot figure) [LEGACY]
# ==============================================================================
def _render_slcci_monthly_analysis(slcci_data, config: AppConfig):
    """
    Render 12-month DOT analysis like SLCCI PLOTTER.
    Shows DOT vs Longitude/Distance for each month (1-12) with linear regression.
    Includes R² and slope statistics.
    """
    st.subheader("🟠 SLCCI - 12 Months DOT Analysis")
    
    df = getattr(slcci_data, 'df', None)
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    pass_number = getattr(slcci_data, 'pass_number', 0)
    
    if df is None or df.empty:
        st.warning("No data available for monthly analysis.")
        return
    
    if 'month' not in df.columns or 'lon' not in df.columns or 'dot' not in df.columns:
        st.error("Missing required columns: month, lon, dot")
        return
    
    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_regression = st.checkbox("Show linear regression", value=True, key="slcci_monthly_reg")
    with col2:
        x_axis_mode = st.selectbox("X-axis", ["Longitude (°)", "Distance (km)"], key="slcci_monthly_xaxis")
    with col3:
        y_units = st.selectbox("Y units", ["m", "cm", "mm"], key="slcci_monthly_yunits")
    
    # Y-axis scaling
    y_scale = {"m": 1.0, "cm": 100.0, "mm": 1000.0}[y_units]
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=[f"{month_names[i]} ({i+1})" for i in range(12)],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    R_earth = 6371.0
    mean_lat = df['lat'].mean()
    lat_rad = np.deg2rad(mean_lat)
    
    # Compute x_km for distance mode
    if x_axis_mode == "Distance (km)" and 'lon' in df.columns:
        lon_min = df['lon'].min()
        df['x_km'] = R_earth * np.deg2rad(df['lon'] - lon_min) * np.cos(lat_rad)
    
    slopes_info = []
    
    for month in range(1, 13):
        row = (month - 1) // 4 + 1
        col = (month - 1) % 4 + 1
        
        month_df = df[df['month'] == month]
        if len(month_df) < 2:
            continue
        
        # Get x and y values based on mode
        if x_axis_mode == "Distance (km)":
            x_data = month_df['x_km'].values
            x_label = "Distance (km)"
        else:
            x_data = month_df['lon'].values
            x_label = "Longitude (°)"
        
        dot_data = month_df['dot'].values * y_scale
        
        mask = np.isfinite(x_data) & np.isfinite(dot_data)
        if np.sum(mask) < 2:
            continue
        
        x_valid = x_data[mask]
        y_valid = dot_data[mask]
        
        # Scatter
        fig.add_trace(
            go.Scatter(
                x=x_valid, y=y_valid, mode='markers',
                marker=dict(size=3, color='#FF7F0E', opacity=0.5),  # Orange for SLCCI
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Regression
        if show_regression and len(x_valid) > 2:
            try:
                from scipy import stats as scipy_stats
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_valid, y_valid)
                r_squared = r_value ** 2
                
                # Convert slope to m/100km for standard comparison
                if x_axis_mode == "Distance (km)":
                    slope_m_100km = (slope / y_scale) * 100  # m/100km
                else:
                    # For longitude, approximate using mean latitude
                    km_per_deg = R_earth * np.cos(lat_rad) * np.pi / 180
                    slope_m_100km = (slope / y_scale) * km_per_deg * 100
                
                slopes_info.append({
                    'month': month,
                    'name': month_names[month-1],
                    'slope': slope,
                    'slope_m_100km': slope_m_100km,
                    'r_squared': r_squared,
                    'n_points': len(x_valid)
                })
                
                # Regression line
                x_line = np.linspace(x_valid.min(), x_valid.max(), 50)
                y_line = slope * x_line + intercept
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line, y=y_line, mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False,
                        hovertemplate=f"R²={r_squared:.3f}<br>slope={slope:.4f}"
                    ),
                    row=row, col=col
                )
            except Exception:
                pass
    
    fig.update_layout(
        title=f"🟠 SLCCI - {strait_name} - Pass {pass_number} - Monthly DOT Analysis",
        height=700,
        
        showlegend=False
    )
    
    # Axis labels
    x_label_short = "km" if x_axis_mode == "Distance (km)" else "Lon (°)"
    for i in range(1, 13):
        row = (i - 1) // 4 + 1
        col = (i - 1) % 4 + 1
        if row == 3:
            fig.update_xaxes(title_text=x_label_short, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=f"DOT ({y_units})", row=row, col=col)
    
    st.plotly_chart(fig, width='stretch', key="_render_slcci_monthly_analysis_1")
    
    # Summary table with R²
    if slopes_info:
        with st.expander("📊 Monthly Slopes & R² Summary"):
            slopes_df = pd.DataFrame(slopes_info)
            
            # Format columns for display
            display_df = pd.DataFrame({
                'Month': slopes_df['name'],
                f'Slope ({y_units}/{x_label_short})': slopes_df['slope'].apply(lambda x: f"{x:.4f}"),
                'Slope (m/100km)': slopes_df['slope_m_100km'].apply(lambda x: f"{x:.4f}"),
                'R²': slopes_df['r_squared'].apply(lambda x: f"{x:.3f}"),
                'N Points': slopes_df['n_points']
            })
            
            st.dataframe(display_df, width='stretch', hide_index=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Slope", f"{slopes_df['slope_m_100km'].mean():.4f} m/100km")
            with col2:
                st.metric("Std Dev", f"{slopes_df['slope_m_100km'].std():.4f} m/100km")
            with col3:
                st.metric("Mean R²", f"{slopes_df['r_squared'].mean():.3f}")
            with col4:
                st.metric("Months with Data", len(slopes_df))


# ==============================================================================
# TAB 5: GEOSTROPHIC VELOCITY [LEGACY]
# ==============================================================================
def _render_geostrophic_velocity(slcci_data, config):
    """
    Render geostrophic velocity analysis.
    Uses the formula: v = -g/f * (dη/dx) where f = 2Ω sin(lat)
    """
    st.subheader("Geostrophic Velocity Analysis")
    
    # Constants
    g = 9.81  # m/s²
    OMEGA = 7.2921e-5  # Earth's angular velocity (rad/s)
    R_earth = 6371.0  # km
    
    # Get data from PassData
    df = getattr(slcci_data, 'df', None)
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    pass_number = getattr(slcci_data, 'pass_number', 0)
    
    # Check if v_geostrophic_series is already computed (from CMEMS service)
    v_geostrophic_series = getattr(slcci_data, 'v_geostrophic_series', None)
    mean_latitude = getattr(slcci_data, 'mean_latitude', None)
    coriolis_f = getattr(slcci_data, 'coriolis_f', None)
    
    if v_geostrophic_series is not None and len(v_geostrophic_series) > 0:
        # Use pre-computed values (CMEMS style)
        st.info(f"Using pre-computed geostrophic velocities at lat={mean_latitude:.2f}° (f={coriolis_f:.2e} s⁻¹)")
        
        # Handle both numpy arrays and pandas Series
        if hasattr(v_geostrophic_series, 'index'):
            time_index = v_geostrophic_series.index
            v_values = v_geostrophic_series.values
        else:
            # Convert numpy array to series using time_array from PassData
            time_array = getattr(slcci_data, 'time_array', None)
            if time_array is not None:
                time_index = pd.to_datetime(time_array)
            else:
                time_index = pd.date_range('2000-01', periods=len(v_geostrophic_series), freq='MS')
            v_values = v_geostrophic_series
        
        # Plot time series
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
            height=450,
            
        )
        
        st.plotly_chart(fig, width='stretch', key="_render_geostrophic_velocity_1")
        
        # Statistics
        with st.expander("Geostrophic Velocity Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{v_values.mean() * 100:.2f} cm/s")
            with col2:
                st.metric("Std Dev", f"{v_values.std() * 100:.2f} cm/s")
            with col3:
                st.metric("Max", f"{v_values.max() * 100:.2f} cm/s")
            with col4:
                st.metric("Min", f"{v_values.min() * 100:.2f} cm/s")
        
        return
    
    # Otherwise compute from raw data (SLCCI style)
    if df is None or (hasattr(df, 'empty') and df.empty):
        st.warning("No data available for geostrophic velocity calculation.")
        return
    
    # Check required columns
    required = ['lon', 'dot', 'month']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return
    
    # Get mean latitude for Coriolis parameter
    if 'lat' in df.columns:
        mean_lat = df['lat'].mean()
    else:
        mean_lat = st.number_input("Mean latitude (°)", value=70.0, step=0.1)
    
    lat_rad = np.deg2rad(mean_lat)
    f = 2 * OMEGA * np.sin(lat_rad)
    
    st.info(f"Computing geostrophic velocity at lat={mean_lat:.2f}° (f={f:.2e} s⁻¹)")
    
    # Check for year column for time series
    has_year = 'year' in df.columns
    
    if has_year:
        df['year_month'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01')
        groups = df.groupby('year_month')
    else:
        groups = df.groupby('month')
    
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
        x_m = R_earth * 1000 * dlon_rad * np.cos(lat_rad)
        
        try:
            slope_m_m, _ = np.polyfit(x_m, dot_valid, 1)
            v_geo = -g / f * slope_m_m
            
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
        st.warning("Could not compute geostrophic velocities. Check data quality.")
        return
    
    results_df = pd.DataFrame(results)
    
    # Plot time series or monthly values
    fig = go.Figure()
    
    if has_year:
        x_vals = results_df['period']
        title_suffix = "Time Series"
    else:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        x_vals = [month_names[int(m)-1] for m in results_df['period']]
        title_suffix = "Monthly"
    
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
        xaxis_title="Time" if has_year else "Month",
        yaxis_title="Geostrophic Velocity (cm/s)",
        height=450,
        
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_geostrophic_velocity_2")
    
    # Statistics
    with st.expander("Geostrophic Velocity Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        v_vals = results_df['v_geostrophic_cm_s'].values
        with col1:
            st.metric("Mean", f"{np.mean(v_vals):.2f} cm/s")
        with col2:
            st.metric("Std Dev", f"{np.std(v_vals):.2f} cm/s")
        with col3:
            st.metric("Max", f"{np.max(v_vals):.2f} cm/s")
        with col4:
            st.metric("Min", f"{np.min(v_vals):.2f} cm/s")
        
        st.subheader("Detailed Results")
        display_df = results_df.copy()
        display_df.columns = ['Period', 'Slope (m/m)', 'v_geo (m/s)', 'v_geo (cm/s)', 'N Points']
        st.dataframe(display_df, width='stretch')
    
    # Physical interpretation
    with st.expander("Physical Interpretation"):
        mean_v = np.mean(v_vals)
        st.markdown(f"""
        **Geostrophic Balance Formula:**
        
        v = -g/f × (dη/dx)
        
        Where:
        - g = 9.81 m/s² (gravity)
        - f = 2Ω sin(lat) = {f:.2e} s⁻¹ (Coriolis at {mean_lat:.1f}°)
        - dη/dx = DOT slope along gate
        
        **Results:**
        - Mean geostrophic velocity: **{mean_v:.2f} cm/s**
        - Positive values → flow northward
        - Negative values → flow southward
        """)


# ==============================================================================
# GENERIC TABS (for CMEMS/ERA5)
# ==============================================================================
def _render_generic_tabs(datasets, config: AppConfig):
    """Render tabs for generic datasets."""
    tab1, tab2 = st.tabs(["Data View", "Explorer"])
    
    with tab1:
        st.subheader("Dataset Overview")
        if datasets:
            for name, data in datasets.items():
                st.write(f"**{name}**: {type(data).__name__}")
        else:
            st.info("No datasets loaded")
    
    with tab2:
        st.subheader("Data Explorer")
        st.info("Load CMEMS or ERA5 data to explore")


# ==============================================================================
# COMPARISON MODE TABS
# ==============================================================================
def _render_comparison_tabs(slcci_data, cmems_data, config: AppConfig):
    """Render comparison tabs overlaying SLCCI and CMEMS data."""
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Slope Timeline Comparison",
        "DOT Profile Comparison", 
        "Spatial Map Comparison",
        "Geostrophic Velocity Comparison",
        "📈 Correlation Analysis",
        "📊 Difference Plot",
        "📥 Export Data"
    ])
    
    with tab1:
        _render_slope_comparison(slcci_data, cmems_data, config)
    with tab2:
        _render_dot_profile_comparison(slcci_data, cmems_data, config)
    with tab3:
        _render_spatial_map_comparison(slcci_data, cmems_data, config)
    with tab4:
        _render_geostrophic_comparison(slcci_data, cmems_data, config)
    with tab5:
        _render_correlation_analysis(slcci_data, cmems_data, config)
    with tab6:
        _render_difference_plot(slcci_data, cmems_data, config)
    with tab7:
        _render_export_tab(slcci_data, cmems_data, config)


def _render_slope_comparison(slcci_data, cmems_data, config: AppConfig):
    """Render slope timeline comparison overlay."""
    st.subheader("SSH Slope Timeline - SLCCI vs CMEMS")
    
    # Get SLCCI data
    slcci_slope = getattr(slcci_data, 'slope_series', None)
    slcci_time = getattr(slcci_data, 'time_array', None)
    slcci_name = getattr(slcci_data, 'strait_name', 'Unknown')
    slcci_pass = getattr(slcci_data, 'pass_number', 0)
    
    # Get CMEMS data
    cmems_slope = getattr(cmems_data, 'slope_series', None)
    cmems_time = getattr(cmems_data, 'time_array', None)
    cmems_pass = getattr(cmems_data, 'pass_number', 0)
    
    if slcci_slope is None and cmems_slope is None:
        st.warning("No slope data available for comparison.")
        return
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        show_trend = st.checkbox("Show trend lines", value=True, key="comp_slope_trend")
    with col2:
        unit = st.selectbox("Units", ["m/100km", "cm/km"], key="comp_slope_unit")
    
    fig = go.Figure()
    
    # Plot SLCCI (Orange)
    if slcci_slope is not None:
        valid_mask = ~np.isnan(slcci_slope)
        if np.sum(valid_mask) > 0:
            y_vals = slcci_slope * 100 if unit == "cm/km" else slcci_slope
            x_vals = slcci_time if slcci_time is not None else np.arange(len(slcci_slope))
            
            valid_x = [x_vals[i] for i in range(len(x_vals)) if valid_mask[i]]
            valid_y = [y_vals[i] for i in range(len(y_vals)) if valid_mask[i]]
            
            fig.add_trace(go.Scatter(
                x=valid_x, y=valid_y,
                mode="markers+lines",
                name=f"SLCCI (Pass {slcci_pass})",
                marker=dict(size=6, color=COLOR_SLCCI),
                line=dict(width=2, color=COLOR_SLCCI)
            ))
            
            if show_trend and len(valid_y) > 2:
                x_numeric = np.arange(len(valid_y))
                z = np.polyfit(x_numeric, valid_y, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=valid_x, y=p(x_numeric),
                    mode="lines",
                    name=f"SLCCI Trend ({z[0]:.4f}/period)",
                    line=dict(dash="dash", color=COLOR_SLCCI, width=1)
                ))
    
    # Plot CMEMS (Blue)
    if cmems_slope is not None:
        valid_mask = ~np.isnan(cmems_slope)
        if np.sum(valid_mask) > 0:
            y_vals = cmems_slope * 100 if unit == "cm/km" else cmems_slope
            x_vals = cmems_time if cmems_time is not None else np.arange(len(cmems_slope))
            
            valid_x = [x_vals[i] for i in range(len(x_vals)) if valid_mask[i]]
            valid_y = [y_vals[i] for i in range(len(y_vals)) if valid_mask[i]]
            
            fig.add_trace(go.Scatter(
                x=valid_x, y=valid_y,
                mode="markers+lines",
                name=f"CMEMS (Pass {cmems_pass})",
                marker=dict(size=6, color=COLOR_CMEMS),
                line=dict(width=2, color=COLOR_CMEMS)
            ))
            
            if show_trend and len(valid_y) > 2:
                x_numeric = np.arange(len(valid_y))
                z = np.polyfit(x_numeric, valid_y, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=valid_x, y=p(x_numeric),
                    mode="lines",
                    name=f"CMEMS Trend ({z[0]:.4f}/period)",
                    line=dict(dash="dash", color=COLOR_CMEMS, width=1)
                ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.8)
    
    y_label = "Slope (cm/km)" if unit == "cm/km" else "Slope (m/100km)"
    fig.update_layout(
        title=f"Slope Comparison: {slcci_name}",
        xaxis_title="Date",
        yaxis_title=y_label,
        height=500,
        
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_slope_comparison_1")
    
    # Statistics comparison
    _render_comparison_stats(slcci_slope, cmems_slope, "Slope", unit)


def _render_dot_profile_comparison(slcci_data, cmems_data, config: AppConfig):
    """Render DOT profile comparison overlay."""
    st.subheader("Mean DOT Profile - SLCCI vs CMEMS")
    
    slcci_profile = getattr(slcci_data, 'profile_mean', None)
    slcci_x_km = getattr(slcci_data, 'x_km', None)
    slcci_pass = getattr(slcci_data, 'pass_number', 0)
    
    cmems_profile = getattr(cmems_data, 'profile_mean', None)
    cmems_x_km = getattr(cmems_data, 'x_km', None)
    cmems_pass = getattr(cmems_data, 'pass_number', 0)
    
    if slcci_profile is None and cmems_profile is None:
        st.warning("No profile data available for comparison.")
        return
    
    fig = go.Figure()
    
    if slcci_profile is not None and slcci_x_km is not None:
        valid_mask = ~np.isnan(slcci_profile)
        if np.any(valid_mask):
            fig.add_trace(go.Scatter(
                x=slcci_x_km[valid_mask],
                y=slcci_profile[valid_mask],
                mode="lines",
                name=f"SLCCI (Pass {slcci_pass})",
                line=dict(color=COLOR_SLCCI, width=2)
            ))
    
    if cmems_profile is not None and cmems_x_km is not None:
        valid_mask = ~np.isnan(cmems_profile)
        if np.any(valid_mask):
            fig.add_trace(go.Scatter(
                x=cmems_x_km[valid_mask],
                y=cmems_profile[valid_mask],
                mode="lines",
                name=f"CMEMS (Pass {cmems_pass})",
                line=dict(color=COLOR_CMEMS, width=2)
            ))
    
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    fig.update_layout(
        title=f"DOT Profile Comparison: {strait_name}",
        xaxis_title="Distance along longitude (km)",
        yaxis_title="DOT (m)",
        height=500,
        
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_dot_profile_comparison_1")
    _render_comparison_stats(slcci_profile, cmems_profile, "DOT", "m")


def _render_spatial_map_comparison(slcci_data, cmems_data, config: AppConfig):
    """Render spatial map comparison."""
    st.subheader("Spatial Distribution - SLCCI vs CMEMS")
    
    slcci_df = getattr(slcci_data, 'df', None)
    cmems_df = getattr(cmems_data, 'df', None)
    gate_lon = getattr(slcci_data, 'gate_lon_pts', None) or getattr(cmems_data, 'gate_lon_pts', None)
    gate_lat = getattr(slcci_data, 'gate_lat_pts', None) or getattr(cmems_data, 'gate_lat_pts', None)
    
    if (slcci_df is None or slcci_df.empty) and (cmems_df is None or cmems_df.empty):
        st.warning("No spatial data available for comparison.")
        return
    
    col1, col2 = st.columns([2, 1])
    with col1:
        show_gate = st.checkbox("Show gate line", value=True, key="comp_map_gate")
    with col2:
        sample_size = st.slider("Sample size", 500, 5000, 2000, key="comp_map_sample")
    
    fig = go.Figure()
    
    if slcci_df is not None and not slcci_df.empty:
        plot_df = slcci_df.sample(min(sample_size, len(slcci_df)))
        fig.add_trace(go.Scattermapbox(
            lat=plot_df["lat"], lon=plot_df["lon"],
            mode="markers", name="SLCCI",
            marker=dict(size=5, color=COLOR_SLCCI, opacity=0.6)
        ))
    
    if cmems_df is not None and not cmems_df.empty:
        plot_df = cmems_df.sample(min(sample_size, len(cmems_df)))
        fig.add_trace(go.Scattermapbox(
            lat=plot_df["lat"], lon=plot_df["lon"],
            mode="markers", name="CMEMS",
            marker=dict(size=5, color=COLOR_CMEMS, opacity=0.6)
        ))
    
    if show_gate and gate_lon is not None and gate_lat is not None:
        fig.add_trace(go.Scattermapbox(
            lat=gate_lat, lon=gate_lon,
            mode="lines", name="Gate",
            line=dict(width=3, color="red")
        ))
    
    all_lats, all_lons = [], []
    if slcci_df is not None and not slcci_df.empty:
        all_lats.extend(slcci_df["lat"].tolist())
        all_lons.extend(slcci_df["lon"].tolist())
    if cmems_df is not None and not cmems_df.empty:
        all_lats.extend(cmems_df["lat"].tolist())
        all_lons.extend(cmems_df["lon"].tolist())
    
    center_lat = np.mean(all_lats) if all_lats else 70.0
    center_lon = np.mean(all_lons) if all_lons else 0.0
    
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    fig.update_layout(
        title=f"Spatial Comparison: {strait_name}",
        mapbox=dict(style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=5),
        height=600, margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_spatial_map_comparison_1")


def _render_geostrophic_comparison(slcci_data, cmems_data, config: AppConfig):
    """Render geostrophic velocity comparison."""
    st.subheader("Geostrophic Velocity - SLCCI vs CMEMS")
    
    slcci_v_geo = getattr(slcci_data, 'v_geostrophic_series', None)
    slcci_pass = getattr(slcci_data, 'pass_number', 0)
    
    cmems_v_geo = getattr(cmems_data, 'v_geostrophic_series', None)
    cmems_pass = getattr(cmems_data, 'pass_number', 0)
    
    if slcci_v_geo is None and cmems_v_geo is None:
        st.warning("No geostrophic velocity data available.")
        return
    
    fig = go.Figure()
    
    if slcci_v_geo is not None and len(slcci_v_geo) > 0:
        fig.add_trace(go.Scatter(
            x=slcci_v_geo.index, y=slcci_v_geo.values * 100,
            mode="lines+markers", name=f"SLCCI (Pass {slcci_pass})",
            line=dict(color=COLOR_SLCCI, width=2),
            marker=dict(size=6, color=COLOR_SLCCI)
        ))
    
    if cmems_v_geo is not None and len(cmems_v_geo) > 0:
        fig.add_trace(go.Scatter(
            x=cmems_v_geo.index, y=cmems_v_geo.values * 100,
            mode="lines+markers", name=f"CMEMS (Pass {cmems_pass})",
            line=dict(color=COLOR_CMEMS, width=2),
            marker=dict(size=6, color=COLOR_CMEMS)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    fig.update_layout(
        title=f"Geostrophic Velocity Comparison: {strait_name}",
        xaxis_title="Time", yaxis_title="Geostrophic Velocity (cm/s)",
        height=500,  legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_geostrophic_comparison_1")


def _render_export_tab(slcci_data, cmems_data, config: AppConfig):
    """Render comprehensive export tab with CSV, PNG, and statistics."""
    st.subheader("📤 Export Data & Results")
    
    # Check what data is available
    has_slcci = slcci_data is not None
    has_cmems = cmems_data is not None
    
    if not has_slcci and not has_cmems:
        st.warning("No data loaded. Load SLCCI or CMEMS data first.")
        return
    
    # Create tabs for different export types
    export_tabs = st.tabs(["📊 Raw Data", "📈 Time Series", "📉 Statistics", "🖼️ Plots"])
    
    # ==========================================================================
    # TAB 1: RAW DATA EXPORT
    # ==========================================================================
    with export_tabs[0]:
        st.markdown("### Raw Observation Data")
        st.caption("Full DataFrame with all observations (lat, lon, time, variables)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if has_slcci:
                slcci_df = getattr(slcci_data, 'df', None)
                if slcci_df is not None and not slcci_df.empty:
                    st.markdown(f"**SLCCI**: {len(slcci_df):,} observations")
                    st.caption(f"Columns: {', '.join(slcci_df.columns[:8])}...")
                    csv_slcci = slcci_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download SLCCI Raw Data (CSV)",
                        data=csv_slcci,
                        file_name=f"slcci_raw_{config.selected_gate or 'data'}.csv",
                        mime="text/csv",
                        key="export_slcci_raw"
                    )
                else:
                    st.info("SLCCI: No raw data available")
            else:
                st.info("SLCCI: Not loaded")
        
        with col2:
            if has_cmems:
                cmems_df = getattr(cmems_data, 'df', None)
                if cmems_df is not None and not cmems_df.empty:
                    st.markdown(f"**CMEMS**: {len(cmems_df):,} observations")
                    st.caption(f"Columns: {', '.join(cmems_df.columns[:8])}...")
                    csv_cmems = cmems_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CMEMS Raw Data (CSV)",
                        data=csv_cmems,
                        file_name=f"cmems_raw_{config.selected_gate or 'data'}.csv",
                        mime="text/csv",
                        key="export_cmems_raw"
                    )
                else:
                    st.info("CMEMS: No raw data available")
            else:
                st.info("CMEMS: Not loaded")
    
    # ==========================================================================
    # TAB 2: TIME SERIES EXPORT
    # ==========================================================================
    with export_tabs[1]:
        st.markdown("### Monthly Time Series")
        st.caption("Slope, geostrophic velocity, and other derived time series")
        
        # Build combined time series DataFrame
        ts_data = []
        
        if has_slcci:
            time_array = getattr(slcci_data, 'time_array', None)
            slope_series = getattr(slcci_data, 'slope_series', None)
            v_geo = getattr(slcci_data, 'v_geostrophic_series', None)
            
            if time_array is not None and slope_series is not None:
                for i, t in enumerate(time_array):
                    row = {
                        'time': pd.Timestamp(t),
                        'source': 'SLCCI',
                        'slope_m_100km': slope_series[i] if i < len(slope_series) else np.nan,
                    }
                    if v_geo is not None and i < len(v_geo):
                        row['v_geostrophic_m_s'] = v_geo[i]
                    ts_data.append(row)
        
        if has_cmems:
            time_array = getattr(cmems_data, 'time_array', None)
            slope_series = getattr(cmems_data, 'slope_series', None)
            v_geo = getattr(cmems_data, 'v_geostrophic_series', None)
            
            if time_array is not None and slope_series is not None:
                for i, t in enumerate(time_array):
                    row = {
                        'time': pd.Timestamp(t),
                        'source': 'CMEMS',
                        'slope_m_100km': slope_series[i] if i < len(slope_series) else np.nan,
                    }
                    if v_geo is not None and i < len(v_geo):
                        row['v_geostrophic_m_s'] = v_geo[i]
                    ts_data.append(row)
        
        if ts_data:
            ts_df = pd.DataFrame(ts_data)
            ts_df = ts_df.sort_values(['source', 'time'])
            
            st.dataframe(ts_df.head(20), width='stretch')
            st.caption(f"Showing first 20 of {len(ts_df)} rows")
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv_ts = ts_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Time Series (Long Format)",
                    data=csv_ts,
                    file_name=f"timeseries_long_{config.selected_gate or 'data'}.csv",
                    mime="text/csv",
                    key="export_ts_long"
                )
            
            with col2:
                # Pivot to wide format for easier analysis
                try:
                    ts_wide = ts_df.pivot(index='time', columns='source', values='slope_m_100km')
                    ts_wide = ts_wide.reset_index()
                    csv_wide = ts_wide.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Time Series (Wide Format)",
                        data=csv_wide,
                        file_name=f"timeseries_wide_{config.selected_gate or 'data'}.csv",
                        mime="text/csv",
                        key="export_ts_wide"
                    )
                except Exception:
                    st.caption("Wide format not available (need both datasets)")
        else:
            st.warning("No time series data available")
    
    # ==========================================================================
    # TAB 3: STATISTICS EXPORT
    # ==========================================================================
    with export_tabs[2]:
        st.markdown("### Summary Statistics")
        
        stats_data = []
        
        # SLCCI stats
        if has_slcci:
            slcci_slope = getattr(slcci_data, 'slope_series', None)
            slcci_vgeo = getattr(slcci_data, 'v_geostrophic_series', None)
            strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
            pass_num = getattr(slcci_data, 'pass_number', None)
            
            if slcci_slope is not None:
                valid = slcci_slope[~np.isnan(slcci_slope)]
                if len(valid) > 0:
                    stats_data.append({
                        'Source': 'SLCCI',
                        'Variable': 'Slope (m/100km)',
                        'Mean': f"{np.mean(valid):.4f}",
                        'Std': f"{np.std(valid):.4f}",
                        'Min': f"{np.min(valid):.4f}",
                        'Max': f"{np.max(valid):.4f}",
                        'N': len(valid),
                        'Strait': strait_name,
                        'Pass/Track': pass_num or 'N/A'
                    })
            
            if slcci_vgeo is not None:
                valid = slcci_vgeo[~np.isnan(slcci_vgeo)]
                if len(valid) > 0:
                    stats_data.append({
                        'Source': 'SLCCI',
                        'Variable': 'V_geo (m/s)',
                        'Mean': f"{np.mean(valid):.4f}",
                        'Std': f"{np.std(valid):.4f}",
                        'Min': f"{np.min(valid):.4f}",
                        'Max': f"{np.max(valid):.4f}",
                        'N': len(valid),
                        'Strait': strait_name,
                        'Pass/Track': pass_num or 'N/A'
                    })
        
        # CMEMS stats
        if has_cmems:
            cmems_slope = getattr(cmems_data, 'slope_series', None)
            cmems_vgeo = getattr(cmems_data, 'v_geostrophic_series', None)
            strait_name = getattr(cmems_data, 'strait_name', 'Unknown')
            pass_num = getattr(cmems_data, 'pass_number', None)
            
            if cmems_slope is not None:
                valid = cmems_slope[~np.isnan(cmems_slope)]
                if len(valid) > 0:
                    stats_data.append({
                        'Source': 'CMEMS',
                        'Variable': 'Slope (m/100km)',
                        'Mean': f"{np.mean(valid):.4f}",
                        'Std': f"{np.std(valid):.4f}",
                        'Min': f"{np.min(valid):.4f}",
                        'Max': f"{np.max(valid):.4f}",
                        'N': len(valid),
                        'Strait': strait_name,
                        'Pass/Track': pass_num or 'N/A'
                    })
            
            if cmems_vgeo is not None:
                valid = cmems_vgeo[~np.isnan(cmems_vgeo)]
                if len(valid) > 0:
                    stats_data.append({
                        'Source': 'CMEMS',
                        'Variable': 'V_geo (m/s)',
                        'Mean': f"{np.mean(valid):.4f}",
                        'Std': f"{np.std(valid):.4f}",
                        'Min': f"{np.min(valid):.4f}",
                        'Max': f"{np.max(valid):.4f}",
                        'N': len(valid),
                        'Strait': strait_name,
                        'Pass/Track': pass_num or 'N/A'
                    })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, width='stretch')
            
            csv_stats = stats_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Statistics (CSV)",
                data=csv_stats,
                file_name=f"statistics_{config.selected_gate or 'data'}.csv",
                mime="text/csv",
                key="export_stats"
            )
        
        # Comparison metrics (if both datasets available)
        if has_slcci and has_cmems:
            st.markdown("### Comparison Metrics")
            
            slcci_slope = getattr(slcci_data, 'slope_series', None)
            cmems_slope = getattr(cmems_data, 'slope_series', None)
            slcci_time = getattr(slcci_data, 'time_array', None)
            cmems_time = getattr(cmems_data, 'time_array', None)
            
            if all(x is not None for x in [slcci_slope, cmems_slope, slcci_time, cmems_time]):
                # Align time series
                try:
                    slcci_series = pd.Series(slcci_slope, index=pd.to_datetime(slcci_time))
                    cmems_series = pd.Series(cmems_slope, index=pd.to_datetime(cmems_time))
                    
                    # Find common time range
                    common_idx = slcci_series.index.intersection(cmems_series.index)
                    
                    if len(common_idx) > 5:
                        slcci_aligned = slcci_series.loc[common_idx]
                        cmems_aligned = cmems_series.loc[common_idx]
                        
                        # Remove NaN pairs
                        mask = ~(slcci_aligned.isna() | cmems_aligned.isna())
                        slcci_clean = slcci_aligned[mask]
                        cmems_clean = cmems_aligned[mask]
                        
                        if len(slcci_clean) > 5:
                            # Calculate metrics
                            corr = np.corrcoef(slcci_clean, cmems_clean)[0, 1]
                            bias = np.mean(cmems_clean - slcci_clean)
                            rmse = np.sqrt(np.mean((cmems_clean - slcci_clean)**2))
                            mae = np.mean(np.abs(cmems_clean - slcci_clean))
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Correlation (r)", f"{corr:.3f}")
                            col2.metric("Bias (CMEMS-SLCCI)", f"{bias:.4f}")
                            col3.metric("RMSE", f"{rmse:.4f}")
                            col4.metric("MAE", f"{mae:.4f}")
                            
                            st.caption(f"Based on {len(slcci_clean)} overlapping monthly values")
                            
                            # Export comparison data
                            comp_df = pd.DataFrame({
                                'time': common_idx[mask],
                                'SLCCI_slope': slcci_clean.values,
                                'CMEMS_slope': cmems_clean.values,
                                'difference': (cmems_clean - slcci_clean).values
                            })
                            csv_comp = comp_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Comparison Data (CSV)",
                                data=csv_comp,
                                file_name=f"comparison_{config.selected_gate or 'data'}.csv",
                                mime="text/csv",
                                key="export_comparison"
                            )
                        else:
                            st.info("Not enough overlapping data points for comparison")
                    else:
                        st.info("No overlapping time periods found")
                except Exception as e:
                    st.warning(f"Could not compute comparison: {e}")
        else:
            st.info("Load both SLCCI and CMEMS to see comparison metrics")
    
    # ==========================================================================
    # TAB 4: PLOT EXPORT
    # ==========================================================================
    with export_tabs[3]:
        st.markdown("### Export Plots")
        
        st.info("""
        **Method 1: Camera Icon (Recommended)**
        - Hover over any Plotly chart in the app
        - Click the 📷 camera icon in the top-right toolbar
        - PNG will download automatically
        
        **Method 2: Generate PNG Below**
        - Select a plot type below
        - Click "Generate PNG"
        - Download the generated image
        """)
        
        plot_type = st.selectbox(
            "Select Plot to Export",
            ["Slope Time Series", "Geostrophic Velocity", "DOT Profile"],
            key="export_plot_type"
        )
        
        if st.button("🖼️ Generate PNG", key="generate_png"):
            fig = None
            
            if plot_type == "Slope Time Series":
                fig = go.Figure()
                
                if has_slcci:
                    time_arr = getattr(slcci_data, 'time_array', None)
                    slope = getattr(slcci_data, 'slope_series', None)
                    if time_arr is not None and slope is not None:
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(time_arr),
                            y=slope,
                            mode='lines+markers',
                            name='SLCCI',
                            line=dict(color='darkorange', width=2)
                        ))
                
                if has_cmems:
                    time_arr = getattr(cmems_data, 'time_array', None)
                    slope = getattr(cmems_data, 'slope_series', None)
                    if time_arr is not None and slope is not None:
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(time_arr),
                            y=slope,
                            mode='lines+markers',
                            name='CMEMS',
                            line=dict(color='steelblue', width=2)
                        ))
                
                fig.update_layout(
                    title=f"Slope Time Series - {config.selected_gate or 'Data'}",
                    xaxis_title="Time",
                    yaxis_title="Slope (m/100km)",
                    
                    height=500,
                    width=900
                )
            
            elif plot_type == "Geostrophic Velocity":
                fig = go.Figure()
                
                if has_slcci:
                    time_arr = getattr(slcci_data, 'time_array', None)
                    v_geo = getattr(slcci_data, 'v_geostrophic_series', None)
                    if time_arr is not None and v_geo is not None:
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(time_arr),
                            y=v_geo * 100,  # cm/s
                            mode='lines+markers',
                            name='SLCCI',
                            line=dict(color='darkorange', width=2)
                        ))
                
                if has_cmems:
                    time_arr = getattr(cmems_data, 'time_array', None)
                    v_geo = getattr(cmems_data, 'v_geostrophic_series', None)
                    if time_arr is not None and v_geo is not None:
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(time_arr),
                            y=v_geo * 100,  # cm/s
                            mode='lines+markers',
                            name='CMEMS',
                            line=dict(color='steelblue', width=2)
                        ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title=f"Geostrophic Velocity - {config.selected_gate or 'Data'}",
                    xaxis_title="Time",
                    yaxis_title="Velocity (cm/s)",
                    
                    height=500,
                    width=900
                )
            
            elif plot_type == "DOT Profile":
                fig = go.Figure()
                
                if has_slcci:
                    x_km = getattr(slcci_data, 'x_km', None)
                    profile = getattr(slcci_data, 'profile_mean', None)
                    if x_km is not None and profile is not None:
                        fig.add_trace(go.Scatter(
                            x=x_km,
                            y=profile,
                            mode='lines',
                            name='SLCCI',
                            line=dict(color='darkorange', width=2)
                        ))
                
                if has_cmems:
                    x_km = getattr(cmems_data, 'x_km', None)
                    profile = getattr(cmems_data, 'profile_mean', None)
                    if x_km is not None and profile is not None:
                        fig.add_trace(go.Scatter(
                            x=x_km,
                            y=profile,
                            mode='lines',
                            name='CMEMS',
                            line=dict(color='steelblue', width=2)
                        ))
                
                fig.update_layout(
                    title=f"Mean DOT Profile - {config.selected_gate or 'Data'}",
                    xaxis_title="Distance (km)",
                    yaxis_title="DOT (m)",
                    
                    height=500,
                    width=900
                )
            
            if fig is not None and len(fig.data) > 0:
                # Show the plot
                st.plotly_chart(fig, width='stretch', key="_render_export_tab_1")
                
                # Generate PNG bytes
                try:
                    import io
                    img_bytes = fig.to_image(format="png", width=900, height=500, scale=2)
                    
                    st.download_button(
                        label="📥 Download PNG",
                        data=img_bytes,
                        file_name=f"{plot_type.lower().replace(' ', '_')}_{config.selected_gate or 'data'}.png",
                        mime="image/png",
                        key="download_png"
                    )
                except Exception as e:
                    st.warning(f"PNG export requires kaleido: `pip install kaleido`")
                    st.caption(f"Error: {e}")
            else:
                st.warning("No data available for this plot type")


def _render_comparison_stats(slcci_data, cmems_data, variable_name: str, unit: str):
    """Render comparison statistics expander."""
    with st.expander(f"{variable_name} Statistics Comparison"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**SLCCI (orange)**")
            if slcci_data is not None:
                valid = slcci_data[~np.isnan(slcci_data)]
                if len(valid) > 0:
                    st.metric("Mean", f"{np.mean(valid):.4f} {unit}")
                    st.metric("Std Dev", f"{np.std(valid):.4f} {unit}")
                    st.metric("N Points", len(valid))
                else:
                    st.warning("No valid data")
            else:
                st.warning("No data")
        
        with col2:
            st.markdown(f"**CMEMS (blue)**")
            if cmems_data is not None:
                valid = cmems_data[~np.isnan(cmems_data)]
                if len(valid) > 0:
                    st.metric("Mean", f"{np.mean(valid):.4f} {unit}")
                    st.metric("Std Dev", f"{np.std(valid):.4f} {unit}")
                    st.metric("N Points", len(valid))
                else:
                    st.warning("No valid data")
            else:
                st.warning("No data")


# ==============================================================================
# CORRELATION ANALYSIS (NEW!)
# ==============================================================================
def _render_correlation_analysis(slcci_data, cmems_data, config: AppConfig):
    """
    Render correlation analysis between SLCCI and CMEMS data.
    
    Shows scatter plot of slope values with correlation metrics.
    """
    st.subheader("📈 Correlation Analysis: SLCCI vs CMEMS")
    
    # Get slope data
    slcci_slope = getattr(slcci_data, 'slope_series', None)
    slcci_time = getattr(slcci_data, 'time_array', None)
    cmems_slope = getattr(cmems_data, 'slope_series', None)
    cmems_time = getattr(cmems_data, 'time_array', None)
    
    if slcci_slope is None or cmems_slope is None:
        st.warning("Both SLCCI and CMEMS slope data required for correlation analysis.")
        return
    
    # Convert to pandas for alignment
    if slcci_time is not None:
        slcci_series = pd.Series(slcci_slope, index=pd.to_datetime(slcci_time))
    else:
        st.warning("SLCCI time array not available.")
        return
    
    if cmems_time is not None:
        cmems_series = pd.Series(cmems_slope, index=pd.to_datetime(cmems_time))
    else:
        st.warning("CMEMS time array not available.")
        return
    
    # Align by month (YYYY-MM)
    slcci_monthly = slcci_series.groupby(slcci_series.index.to_period('M')).mean()
    cmems_monthly = cmems_series.groupby(cmems_series.index.to_period('M')).mean()
    
    # Find common periods
    common_periods = slcci_monthly.index.intersection(cmems_monthly.index)
    
    if len(common_periods) < 3:
        st.warning(f"Not enough common periods for correlation. Found {len(common_periods)} common months.")
        st.info("Try loading data with overlapping time ranges.")
        return
    
    # Extract aligned values
    slcci_aligned = slcci_monthly.loc[common_periods].values
    cmems_aligned = cmems_monthly.loc[common_periods].values
    
    # Remove NaN
    mask = ~np.isnan(slcci_aligned) & ~np.isnan(cmems_aligned)
    slcci_clean = slcci_aligned[mask]
    cmems_clean = cmems_aligned[mask]
    
    if len(slcci_clean) < 3:
        st.warning("Not enough valid data points for correlation after removing NaN.")
        return
    
    # Calculate correlation metrics
    correlation = np.corrcoef(slcci_clean, cmems_clean)[0, 1]
    r_squared = correlation ** 2
    
    # Calculate RMSE and bias
    diff = slcci_clean - cmems_clean
    rmse = np.sqrt(np.mean(diff ** 2))
    bias = np.mean(diff)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=cmems_clean,
        y=slcci_clean,
        mode='markers',
        marker=dict(
            size=10,
            color='rgba(100, 149, 237, 0.7)',  # Cornflower blue
            line=dict(width=1, color='darkblue')
        ),
        name='Monthly Mean Slope',
        hovertemplate='CMEMS: %{x:.4f}<br>SLCCI: %{y:.4f}<extra></extra>'
    ))
    
    # 1:1 line
    min_val = min(cmems_clean.min(), slcci_clean.min())
    max_val = max(cmems_clean.max(), slcci_clean.max())
    margin = (max_val - min_val) * 0.1
    
    fig.add_trace(go.Scatter(
        x=[min_val - margin, max_val + margin],
        y=[min_val - margin, max_val + margin],
        mode='lines',
        line=dict(dash='dash', color='gray', width=1),
        name='1:1 Line'
    ))
    
    # Linear regression line
    slope_reg, intercept_reg = np.polyfit(cmems_clean, slcci_clean, 1)
    x_line = np.array([min_val - margin, max_val + margin])
    y_line = slope_reg * x_line + intercept_reg
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        line=dict(color='red', width=2),
        name=f'Linear Fit (y = {slope_reg:.2f}x + {intercept_reg:.4f})'
    ))
    
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    fig.update_layout(
        title=f"Slope Correlation: {strait_name}",
        xaxis_title="CMEMS Slope (m/100km)",
        yaxis_title="SLCCI Slope (m/100km)",
        height=500,
        
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_correlation_analysis_1")
    
    # Metrics
    st.markdown("### Correlation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Correlation (r)", f"{correlation:.3f}")
    with col2:
        st.metric("R²", f"{r_squared:.3f}")
    with col3:
        st.metric("RMSE", f"{rmse:.4f} m/100km")
    with col4:
        st.metric("Bias (SLCCI-CMEMS)", f"{bias:.4f} m/100km")
    
    # Interpretation
    with st.expander("📖 Interpretation"):
        if r_squared > 0.7:
            quality = "**Strong**"
            color = "green"
        elif r_squared > 0.4:
            quality = "**Moderate**"
            color = "orange"
        else:
            quality = "**Weak**"
            color = "red"
        
        st.markdown(f"""
        **Correlation Quality**: {quality}
        
        - **R² = {r_squared:.3f}**: {r_squared*100:.1f}% of variance explained
        - **Bias = {bias:.4f}**: {'SLCCI higher' if bias > 0 else 'CMEMS higher'} on average
        - **RMSE = {rmse:.4f}**: Typical difference between datasets
        - **N = {len(slcci_clean)}**: Number of monthly periods compared
        
        **Note**: Correlation is computed on monthly-averaged slope values to reduce noise.
        """)
    
    # Show data table
    with st.expander("📋 Data Table"):
        comparison_df = pd.DataFrame({
            'Period': [str(p) for p in common_periods[mask]],
            'SLCCI Slope': slcci_clean,
            'CMEMS Slope': cmems_clean,
            'Difference': diff
        })
        st.dataframe(comparison_df, width='stretch')


# ==============================================================================
# DIFFERENCE PLOT (NEW!)
# ==============================================================================
def _render_difference_plot(slcci_data, cmems_data, config: AppConfig):
    """
    Render difference plot showing SLCCI - CMEMS over time.
    
    Useful for identifying systematic biases and temporal patterns.
    """
    st.subheader("📊 Difference Analysis: SLCCI - CMEMS")
    
    # Get slope data
    slcci_slope = getattr(slcci_data, 'slope_series', None)
    slcci_time = getattr(slcci_data, 'time_array', None)
    cmems_slope = getattr(cmems_data, 'slope_series', None)
    cmems_time = getattr(cmems_data, 'time_array', None)
    
    if slcci_slope is None or cmems_slope is None:
        st.warning("Both SLCCI and CMEMS slope data required for difference analysis.")
        return
    
    # Convert to pandas for alignment
    if slcci_time is not None:
        slcci_series = pd.Series(slcci_slope, index=pd.to_datetime(slcci_time))
    else:
        st.warning("SLCCI time array not available.")
        return
    
    if cmems_time is not None:
        cmems_series = pd.Series(cmems_slope, index=pd.to_datetime(cmems_time))
    else:
        st.warning("CMEMS time array not available.")
        return
    
    # Align by month
    slcci_monthly = slcci_series.groupby(slcci_series.index.to_period('M')).mean()
    cmems_monthly = cmems_series.groupby(cmems_series.index.to_period('M')).mean()
    
    # Find common periods
    common_periods = slcci_monthly.index.intersection(cmems_monthly.index)
    
    if len(common_periods) < 2:
        st.warning("Not enough common periods for difference analysis.")
        return
    
    # Calculate difference
    diff_series = slcci_monthly.loc[common_periods] - cmems_monthly.loc[common_periods]
    
    # Remove NaN
    diff_clean = diff_series.dropna()
    
    if len(diff_clean) < 2:
        st.warning("Not enough valid data points after removing NaN.")
        return
    
    # Create time series plot
    fig = go.Figure()
    
    # Difference line with fill
    x_dates = diff_clean.index.to_timestamp()
    y_vals = diff_clean.values
    
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=y_vals,
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(100, 149, 237, 0.3)',
        line=dict(color='steelblue', width=2),
        marker=dict(size=6),
        name='SLCCI - CMEMS'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    # Mean bias line
    mean_diff = np.mean(y_vals)
    fig.add_hline(
        y=mean_diff, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Mean Bias: {mean_diff:.4f}",
        annotation_position="top right"
    )
    
    # ±1 std bands
    std_diff = np.std(y_vals)
    fig.add_hrect(
        y0=mean_diff - std_diff,
        y1=mean_diff + std_diff,
        fillcolor="rgba(255, 0, 0, 0.1)",
        line_width=0,
    )
    
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    fig.update_layout(
        title=f"Slope Difference Time Series: {strait_name}",
        xaxis_title="Date",
        yaxis_title="Difference (SLCCI - CMEMS) [m/100km]",
        height=450,
        
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_difference_plot_1")
    
    # Statistics
    st.markdown("### Difference Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Bias", f"{mean_diff:.4f} m/100km")
    with col2:
        st.metric("Std Dev", f"{std_diff:.4f} m/100km")
    with col3:
        st.metric("Max Diff", f"{np.max(y_vals):.4f} m/100km")
    with col4:
        st.metric("Min Diff", f"{np.min(y_vals):.4f} m/100km")
    
    # Interpretation
    with st.expander("📖 Interpretation"):
        bias_direction = "SLCCI shows higher slopes" if mean_diff > 0 else "CMEMS shows higher slopes"
        
        st.markdown(f"""
        **Systematic Bias**: {bias_direction} on average ({abs(mean_diff):.4f} m/100km)
        
        **Possible Causes**:
        - Different DOT calculation methods (SLCCI: corssh-geoid vs CMEMS: sla+mdt)
        - Different satellite coverage (SLCCI: J2 only vs CMEMS: J1+J2+J3)
        - Different temporal sampling
        - Processing differences (orbit, corrections)
        
        **N = {len(diff_clean)}** monthly periods analyzed
        """)


# ==============================================================================
# MULTI-DATASET COMPARISON TABS (Universal)
# ==============================================================================

def _render_multi_comparison_tabs(loaded_datasets: dict, config: AppConfig):
    """
    Render comparison tabs for multiple datasets (2+).
    
    Supports any combination of: SLCCI, CMEMS L3, CMEMS L4, DTUSpace
    """
    dataset_names = [DATASET_NAMES.get(k, k) for k in loaded_datasets.keys()]
    
    st.markdown(f"### 🔀 Multi-Dataset Comparison: {' vs '.join(dataset_names)}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Slope Timeline",
        "📊 DOT Profile",
        "🗺️ Spatial Overview",
        "🌊 Geostrophic Velocity",
        "📥 Export"
    ])
    
    with tab1:
        _render_multi_slope_comparison(loaded_datasets, config)
    with tab2:
        _render_multi_dot_comparison(loaded_datasets, config)
    with tab3:
        _render_multi_spatial_overview(loaded_datasets, config)
    with tab4:
        _render_multi_geostrophic_comparison(loaded_datasets, config)
    with tab5:
        _render_multi_export(loaded_datasets, config)


def _render_multi_slope_comparison(loaded_datasets: dict, config: AppConfig):
    """Render slope timeline comparison for multiple datasets."""
    st.subheader("SSH Slope Timeline Comparison")
    
    # Options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        show_trend = st.checkbox("Show trend lines", value=True, key="multi_slope_trend")
    with col2:
        unit = st.selectbox("Units", ["m/100km", "cm/km"], key="multi_slope_unit")
    with col3:
        resample = st.selectbox("Resample", ["Monthly", "Yearly", "None"], key="multi_slope_resample")
    
    fig = go.Figure()
    
    for dataset_key, data in loaded_datasets.items():
        color = DATASET_COLORS.get(dataset_key, "gray")
        name = DATASET_NAMES.get(dataset_key, dataset_key)
        
        # Get slope data (handle different data structures)
        slope = getattr(data, 'slope_series', None)
        time_arr = getattr(data, 'time_array', None)
        
        if slope is None:
            continue
        
        valid_mask = ~np.isnan(slope)
        if np.sum(valid_mask) == 0:
            continue
        
        y_vals = slope * 100 if unit == "cm/km" else slope
        x_vals = time_arr if time_arr is not None else np.arange(len(slope))
        
        valid_x = [x_vals[i] for i in range(len(x_vals)) if valid_mask[i]]
        valid_y = [y_vals[i] for i in range(len(y_vals)) if valid_mask[i]]
        
        # Resample if requested
        if resample == "Yearly" and len(valid_x) > 12:
            df_temp = pd.DataFrame({'time': valid_x, 'slope': valid_y})
            df_temp['time'] = pd.to_datetime(df_temp['time'])
            df_temp = df_temp.set_index('time').resample('YE').mean().reset_index()
            valid_x = df_temp['time'].tolist()
            valid_y = df_temp['slope'].tolist()
        
        # Main trace
        fig.add_trace(go.Scatter(
            x=valid_x, y=valid_y,
            mode="markers+lines",
            name=name,
            marker=dict(size=6, color=color),
            line=dict(width=2, color=color),
            legendgroup=dataset_key
        ))
        
        # Trend line
        if show_trend and len(valid_y) > 2:
            x_numeric = np.arange(len(valid_y))
            z = np.polyfit(x_numeric, valid_y, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=valid_x, y=p(x_numeric),
                mode="lines",
                name=f"{name} Trend",
                line=dict(dash="dash", color=color, width=1),
                legendgroup=dataset_key,
                showlegend=False
            ))
    
    unit_label = "cm/km" if unit == "cm/km" else "m/100km"
    fig.update_layout(
        title="SSH Slope Timeline - Multi-Dataset Comparison",
        xaxis_title="Time",
        yaxis_title=f"Slope ({unit_label})",
        height=500,
        
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_multi_slope_comparison_1")
    
    # Statistics table
    st.markdown("### 📊 Summary Statistics")
    stats_data = []
    for dataset_key, data in loaded_datasets.items():
        slope = getattr(data, 'slope_series', None)
        if slope is None:
            continue
        valid_slope = slope[~np.isnan(slope)]
        if len(valid_slope) == 0:
            continue
        
        stats_data.append({
            "Dataset": DATASET_NAMES.get(dataset_key, dataset_key),
            "Mean (m/100km)": f"{np.mean(valid_slope):.4f}",
            "Std (m/100km)": f"{np.std(valid_slope):.4f}",
            "Min": f"{np.min(valid_slope):.4f}",
            "Max": f"{np.max(valid_slope):.4f}",
            "N": len(valid_slope)
        })
    
    if stats_data:
        st.dataframe(pd.DataFrame(stats_data), width='stretch')


def _render_multi_dot_comparison(loaded_datasets: dict, config: AppConfig):
    """Render DOT profile comparison for multiple datasets."""
    st.subheader("DOT Profile Comparison")
    
    fig = go.Figure()
    
    for dataset_key, data in loaded_datasets.items():
        color = DATASET_COLORS.get(dataset_key, "gray")
        name = DATASET_NAMES.get(dataset_key, dataset_key)
        
        profile_mean = getattr(data, 'profile_mean', None)
        x_km = getattr(data, 'x_km', None)
        
        if profile_mean is None or x_km is None:
            continue
        
        valid_mask = ~np.isnan(profile_mean)
        if np.sum(valid_mask) == 0:
            continue
        
        fig.add_trace(go.Scatter(
            x=x_km[valid_mask],
            y=profile_mean[valid_mask],
            mode="lines+markers",
            name=name,
            line=dict(width=2, color=color),
            marker=dict(size=4, color=color)
        ))
    
    fig.update_layout(
        title="Mean DOT Profile Along Gate",
        xaxis_title="Distance along gate (km)",
        yaxis_title="DOT (m)",
        height=450,
        
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_multi_dot_comparison_1")
    
    # Profile statistics
    with st.expander("📈 Profile Statistics"):
        for dataset_key, data in loaded_datasets.items():
            profile_mean = getattr(data, 'profile_mean', None)
            x_km = getattr(data, 'x_km', None)
            if profile_mean is None:
                continue
            valid = profile_mean[~np.isnan(profile_mean)]
            name = DATASET_NAMES.get(dataset_key, dataset_key)
            st.markdown(f"**{name}**: Mean={np.mean(valid):.4f}m, Range={np.ptp(valid):.4f}m, Gradient={np.mean(np.diff(valid))*1000:.2f}mm/km")


def _render_multi_spatial_overview(loaded_datasets: dict, config: AppConfig):
    """Render spatial overview for multiple datasets."""
    st.subheader("Spatial Overview")
    
    # Select dataset to show
    dataset_options = list(loaded_datasets.keys())
    selected = st.selectbox(
        "Select dataset for spatial map",
        dataset_options,
        format_func=lambda x: DATASET_NAMES.get(x, x),
        key="multi_spatial_select"
    )
    
    data = loaded_datasets[selected]
    
    # Try to get DataFrame
    df = getattr(data, 'df', None)
    gate_lon = getattr(data, 'gate_lon_pts', None)
    gate_lat = getattr(data, 'gate_lat_pts', None)
    
    if df is not None and 'lon' in df.columns and 'lat' in df.columns:
        # Sample for performance
        if len(df) > 10000:
            df_plot = df.sample(10000)
        else:
            df_plot = df
        
        fig = px.scatter_mapbox(
            df_plot,
            lat="lat",
            lon="lon",
            color="dot" if "dot" in df.columns else None,
            color_continuous_scale="RdBu_r",
            zoom=4,
            height=500,
            title=f"{DATASET_NAMES.get(selected, selected)} - Observation Locations"
        )
        fig.update_layout(mapbox_style="carto-positron")
        
        # Add gate line
        if gate_lon is not None and gate_lat is not None:
            fig.add_trace(go.Scattermapbox(
                lon=gate_lon,
                lat=gate_lat,
                mode="lines",
                name="Gate",
                line=dict(width=3, color="red")
            ))
        
        st.plotly_chart(fig, width='stretch', key="_render_multi_spatial_overview_1")
    else:
        st.info(f"No spatial data available for {DATASET_NAMES.get(selected, selected)}")
    
    # Show gate coordinates for all datasets
    st.markdown("### Gate Coordinates (all datasets)")
    for dk, d in loaded_datasets.items():
        gate_lon = getattr(d, 'gate_lon_pts', None)
        gate_lat = getattr(d, 'gate_lat_pts', None)
        if gate_lon is not None:
            st.caption(f"**{DATASET_NAMES.get(dk, dk)}**: lon=[{gate_lon.min():.2f}, {gate_lon.max():.2f}], lat=[{gate_lat.min():.2f}, {gate_lat.max():.2f}]")


def _render_multi_geostrophic_comparison(loaded_datasets: dict, config: AppConfig):
    """Render geostrophic velocity comparison."""
    st.subheader("Geostrophic Velocity Comparison")
    
    fig = go.Figure()
    
    for dataset_key, data in loaded_datasets.items():
        color = DATASET_COLORS.get(dataset_key, "gray")
        name = DATASET_NAMES.get(dataset_key, dataset_key)
        
        v_geo = getattr(data, 'v_geostrophic_series', None)
        time_arr = getattr(data, 'time_array', None)
        
        if v_geo is None:
            continue
        
        valid_mask = ~np.isnan(v_geo)
        if np.sum(valid_mask) == 0:
            continue
        
        x_vals = time_arr if time_arr is not None else np.arange(len(v_geo))
        
        fig.add_trace(go.Scatter(
            x=[x_vals[i] for i in range(len(x_vals)) if valid_mask[i]],
            y=[v_geo[i] for i in range(len(v_geo)) if valid_mask[i]],
            mode="markers+lines",
            name=name,
            marker=dict(size=5, color=color),
            line=dict(width=2, color=color)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title="Cross-Gate Geostrophic Velocity",
        xaxis_title="Time",
        yaxis_title="Velocity (m/s)",
        height=450,
        
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_multi_geostrophic_comparison_1")
    
    # Convert to Sv
    st.markdown("### 🌊 Transport Estimation")
    st.info("For transport (Sv), multiply velocity by cross-section area. Gate depth profile required.")


def _render_multi_correlation(loaded_datasets: dict, config: AppConfig):
    """Render correlation analysis between datasets."""
    st.subheader("Correlation Analysis")
    
    if len(loaded_datasets) < 2:
        st.warning("Need at least 2 datasets for correlation analysis")
        return
    
    # Build correlation matrix from slope series
    slope_data = {}
    for dk, data in loaded_datasets.items():
        slope = getattr(data, 'slope_series', None)
        time_arr = getattr(data, 'time_array', None)
        if slope is not None and time_arr is not None:
            # Convert to DataFrame with time index
            df_temp = pd.DataFrame({
                'time': pd.to_datetime(time_arr),
                'slope': slope
            }).set_index('time')
            slope_data[DATASET_NAMES.get(dk, dk)] = df_temp['slope']
    
    if len(slope_data) < 2:
        st.warning("Not enough datasets with slope data for correlation")
        return
    
    # Combine into single DataFrame (align by time)
    df_combined = pd.DataFrame(slope_data)
    
    # Correlation matrix
    corr_matrix = df_combined.corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".3f"
    )
    fig.update_layout(
        title="Slope Correlation Matrix",
        height=400
    )
    st.plotly_chart(fig, width='stretch', key="_render_multi_correlation_1")
    
    # Pairwise scatter plots
    st.markdown("### Pairwise Scatter Plots")
    dataset_names = list(slope_data.keys())
    
    if len(dataset_names) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_dataset = st.selectbox("X-axis", dataset_names, key="corr_x")
        with col2:
            y_dataset = st.selectbox("Y-axis", [d for d in dataset_names if d != x_dataset], key="corr_y")
        
        # Get common times
        x_data = df_combined[x_dataset].dropna()
        y_data = df_combined[y_dataset].dropna()
        common_idx = x_data.index.intersection(y_data.index)
        
        if len(common_idx) > 2:
            x_vals = x_data.loc[common_idx].values
            y_vals = y_data.loc[common_idx].values
            
            # Scatter plot with regression
            fig_scatter = px.scatter(
                x=x_vals, y=y_vals,
                labels={'x': f"{x_dataset} (m/100km)", 'y': f"{y_dataset} (m/100km)"},
                trendline="ols"
            )
            fig_scatter.update_layout(
                title=f"{x_dataset} vs {y_dataset} (N={len(common_idx)})",
                height=400
            )
            st.plotly_chart(fig_scatter, width='stretch', key="_render_multi_correlation_2")
            
            # Statistics
            from scipy import stats as scipy_stats
            r, p = scipy_stats.pearsonr(x_vals, y_vals)
            rmse = np.sqrt(np.mean((x_vals - y_vals)**2))
            bias = np.mean(x_vals - y_vals)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pearson R", f"{r:.3f}")
            col2.metric("P-value", f"{p:.2e}")
            col3.metric("RMSE", f"{rmse:.4f}")
            col4.metric("Bias", f"{bias:.4f}")
        else:
            st.warning("Not enough overlapping data points")


def _render_multi_export(loaded_datasets: dict, config: AppConfig):
    """Render export options for multiple datasets."""
    st.subheader("Export Data")
    
    # Dataset selector
    export_datasets = st.multiselect(
        "Select datasets to export",
        list(loaded_datasets.keys()),
        default=list(loaded_datasets.keys()),
        format_func=lambda x: DATASET_NAMES.get(x, x),
        key="export_datasets"
    )
    
    if not export_datasets:
        st.warning("Select at least one dataset")
        return
    
    # Export format
    export_format = st.radio(
        "Export format",
        ["CSV (separate files)", "CSV (merged)", "Excel"],
        horizontal=True,
        key="export_format"
    )
    
    if st.button("📥 Generate Export", key="export_btn"):
        with st.spinner("Preparing export..."):
            if export_format == "CSV (separate files)":
                for dk in export_datasets:
                    data = loaded_datasets[dk]
                    slope = getattr(data, 'slope_series', None)
                    time_arr = getattr(data, 'time_array', None)
                    
                    if slope is not None:
                        df_export = pd.DataFrame({
                            'time': time_arr,
                            'slope_m_100km': slope,
                            'v_geostrophic_m_s': getattr(data, 'v_geostrophic_series', np.nan)
                        })
                        
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            f"📥 Download {DATASET_NAMES.get(dk, dk)}",
                            csv,
                            f"{dk}_timeseries.csv",
                            "text/csv",
                            key=f"dl_{dk}"
                        )
            
            elif export_format == "CSV (merged)":
                # Merge all datasets by time
                merged_data = {}
                for dk in export_datasets:
                    data = loaded_datasets[dk]
                    slope = getattr(data, 'slope_series', None)
                    time_arr = getattr(data, 'time_array', None)
                    name = DATASET_NAMES.get(dk, dk)
                    
                    if slope is not None and time_arr is not None:
                        for i, t in enumerate(time_arr):
                            t_str = str(t)[:10] if hasattr(t, '__str__') else str(t)
                            if t_str not in merged_data:
                                merged_data[t_str] = {'time': t_str}
                            merged_data[t_str][f'{name}_slope'] = slope[i] if i < len(slope) else np.nan
                
                df_merged = pd.DataFrame(list(merged_data.values()))
                csv = df_merged.to_csv(index=False)
                st.download_button(
                    "📥 Download Merged CSV",
                    csv,
                    "comparison_merged.csv",
                    "text/csv"
                )
            
            elif export_format == "Excel":
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for dk in export_datasets:
                        data = loaded_datasets[dk]
                        slope = getattr(data, 'slope_series', None)
                        time_arr = getattr(data, 'time_array', None)
                        
                        if slope is not None:
                            df_export = pd.DataFrame({
                                'time': time_arr,
                                'slope_m_100km': slope,
                                'v_geostrophic_m_s': getattr(data, 'v_geostrophic_series', np.nan)
                            })
                            sheet_name = DATASET_NAMES.get(dk, dk)[:31]  # Excel limit
                            df_export.to_excel(writer, sheet_name=sheet_name, index=False)
                
                st.download_button(
                    "📥 Download Excel",
                    output.getvalue(),
                    "comparison_data.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        st.success("✅ Export ready!")


# ==============================================================================
# CMEMS L4 TABS (Gridded via API)
# ==============================================================================

def _render_cmems_l4_tabs(cmems_l4_data, config: AppConfig):
    """Render tabs for CMEMS L4 gridded data."""
    tab1, tab2, tab3, tab4, tab5, tab_vt, tab6, tab_sal, tab7, tab8 = st.tabs([
        "📈 Slope Timeline",
        "📊 DOT Profile",
        "🗺️ Spatial Map",
        "📅 Monthly Analysis",
        "🌊 Geostrophic Velocity",
        "� Volume Transport",
        "�💧 Freshwater Transport",
        "🧪 Salinity Profile",
        "🧂 Salt Flux",
        "📥 Export"
    ])
    
    ds_info = _get_unified_dataset_info(cmems_l4_data, 'cmems_l4')
    
    with tab1:
        _render_dtu_slope_timeline(cmems_l4_data, config)
    with tab2:
        _render_dtu_dot_profile(cmems_l4_data, config)
    with tab3:
        _render_cmems_l4_spatial(cmems_l4_data, config)
    with tab4:
        _render_gridded_monthly_analysis(cmems_l4_data, config)
    with tab5:
        _render_geostrophic_velocity_tab_cmems_l4(cmems_l4_data, config)
    with tab_vt:
        _render_volume_transport_tab_cmems_l4(cmems_l4_data, config)
    with tab6:
        render_freshwater_transport_clean(cmems_l4_data, config, ds_info)
    with tab_sal:
        render_salinity_profile_tab(cmems_l4_data, config, ds_info)
    with tab7:
        render_salt_flux_clean(cmems_l4_data, config, ds_info)
    with tab8:
        _render_cmems_l4_export_tab(cmems_l4_data, config)


def _render_cmems_l4_spatial(cmems_l4_data, config: AppConfig):
    """Render spatial map for CMEMS L4 gridded data."""
    st.subheader("CMEMS L4 Spatial Coverage")
    
    strait_name = getattr(cmems_l4_data, 'strait_name', 'Unknown')
    time_range = getattr(cmems_l4_data, 'time_range', ('', ''))
    n_obs = getattr(cmems_l4_data, 'n_observations', 0)
    
    st.markdown(f"**Gate**: {strait_name}")
    st.markdown(f"**Period**: {time_range[0][:10] if time_range[0] else '?'} to {time_range[1][:10] if time_range[1] else '?'}")
    st.markdown(f"**Observations**: {n_obs:,}")
    
    # Get gate coordinates
    gate_lon = getattr(cmems_l4_data, 'gate_lon_pts', None)
    gate_lat = getattr(cmems_l4_data, 'gate_lat_pts', None)
    
    if gate_lon is not None and gate_lat is not None:
        # Create map centered on gate
        center_lon = (gate_lon.min() + gate_lon.max()) / 2
        center_lat = (gate_lat.min() + gate_lat.max()) / 2
        
        fig = go.Figure()
        
        # Add gate line
        fig.add_trace(go.Scattermapbox(
            lon=gate_lon,
            lat=gate_lat,
            mode="lines+markers",
            name="Gate",
            line=dict(width=4, color="red"),
            marker=dict(size=6, color="red")
        ))
        
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lon=center_lon, lat=center_lat),
                zoom=5
            ),
            height=500,
            title=f"CMEMS L4 Gate: {strait_name}"
        )
        
        st.plotly_chart(fig, width='stretch', key="_render_cmems_l4_spatial_1")
    else:
        st.warning("No gate coordinates available")


# ==============================================================================
# GEOSTROPHIC VELOCITY TAB (CMEMS L4) - CONFRONTO v_perp vs v_geo
# ==============================================================================
def _render_geostrophic_velocity_tab_cmems_l4(cmems_l4_data, config: AppConfig):
    """
    Render Geostrophic Velocity tab for CMEMS L4.
    
    Shows comparison between:
    1. v_perp: from ugos/vgos using formula v(θ) = v_N×cos(θ) + v_E×sin(θ)
    2. v_geo: from DOT slope using formula v = -g/f × (dη/dx)
    
    Layout:
    1. Spatial Profile Along Gate (monthly menu + bin slider, dual x-axis km/deg)
    2. Time series comparison v_perp vs v_geo (SINGLE PLOT)
    """
    st.subheader("🌊 Geostrophic Velocity Comparison")
    
    strait_name = getattr(cmems_l4_data, 'strait_name', 'Unknown')
    ugos_matrix = getattr(cmems_l4_data, 'ugos_matrix', None)
    vgos_matrix = getattr(cmems_l4_data, 'vgos_matrix', None)
    gate_lon = getattr(cmems_l4_data, 'gate_lon_pts', None)
    gate_lat = getattr(cmems_l4_data, 'gate_lat_pts', None)
    x_km = getattr(cmems_l4_data, 'x_km', None)
    time_array = getattr(cmems_l4_data, 'time_array', None)
    slope_series = getattr(cmems_l4_data, 'slope_series', None)
    
    # Check velocity data
    if ugos_matrix is None or vgos_matrix is None:
        st.warning("⚠️ Velocity data (ugos/vgos) not available.")
        st.info("""
        **To enable velocity comparison:**
        1. Go to sidebar → CMEMS L4 Variables
        2. Select **ugos** and **vgos** 
        3. Reload the data
        """)
        return
    
    st.success(f"✅ Data: {ugos_matrix.shape[0]} gate points × {ugos_matrix.shape[1]} time steps")
    
    # =========================================================================
    # CONTROLS - Use unique keys with strait name to avoid conflicts
    # =========================================================================
    st.markdown("### ⚙️ Settings")
    
    # Generate unique key prefix
    key_prefix = f"geovel_{strait_name.replace(' ', '_')}"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get native resolution from data
        native_res_km = getattr(cmems_l4_data, 'native_resolution_km', 3.0)
        effective_spacing_km = getattr(cmems_l4_data, 'effective_spacing_km', 3.0)
        
        # Round up to nearest km for slider
        min_bin_km = max(1, int(np.ceil(native_res_km)))
        
        bin_size_km = st.slider(
            "Spatial averaging (km)",
            min_value=min_bin_km,
            max_value=50,
            value=max(min_bin_km, st.session_state.get(f'{key_prefix}_bin', min_bin_km)),
            step=1,
            key=f"{key_prefix}_bin_slider",
            help=f"Native CMEMS resolution: {native_res_km:.1f} km at {strait_name} latitude"
        )
        
        # Show resolution info
        st.caption(f"🔬 Native res: {native_res_km:.1f} km | Effective spacing: {effective_spacing_km:.1f} km")
    
    with col2:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        selected_month = st.selectbox(
            "Month to display",
            options=list(range(1, 13)),
            format_func=lambda m: month_names[m-1],
            index=0,
            key=f"{key_prefix}_month"
        )
    
    with col3:
        show_v_geo = st.checkbox("Show v_geo (from slope)", value=True, key=f"{key_prefix}_show_vgeo")
    
    # =========================================================================
    # COMPUTE DATA
    # =========================================================================
    if st.button("🧮 Compute Velocities", type="primary", use_container_width=True, key=f"{key_prefix}_compute"):
        with st.spinner("Computing perpendicular velocity..."):
            try:
                from src.services.transport_service import (
                    compute_perpendicular_velocity,
                    compute_monthly_along_gate_profile,
                    compute_perpendicular_velocity_uncertainty
                )
                
                # 1. Compute v_perp from ugos/vgos (with direction info)
                v_perp, v_info = compute_perpendicular_velocity(
                    ugos_matrix, vgos_matrix, gate_lon, gate_lat,
                    gate_name=strait_name,
                    return_info=True
                )
                
                # 2. Monthly along-gate profiles for v_perp
                monthly_v_perp = compute_monthly_along_gate_profile(
                    x_km, v_perp, time_array, bin_size_km
                )
                
                # 3. If we have slope data, compute v_geo
                if slope_series is not None:
                    g = 9.81
                    OMEGA = 7.2921e-5
                    mean_lat = np.mean(gate_lat)
                    f = 2 * OMEGA * np.sin(np.deg2rad(mean_lat))
                    
                    # slope is m/100km, convert to m/m
                    slope_m_m = slope_series / 100000.0
                    v_geo_ts = -g / f * slope_m_m  # (n_time,)
                    
                    st.session_state[f'{key_prefix}_v_geo_ts'] = v_geo_ts
                
                # 4. Compute velocity uncertainty if err_ugosa/err_vgosa available
                err_ugosa = getattr(cmems_l4_data, 'err_ugosa_matrix', None)
                err_vgosa = getattr(cmems_l4_data, 'err_vgosa_matrix', None)
                
                sigma_v_perp = None
                if err_ugosa is not None and err_vgosa is not None:
                    sigma_v_perp = compute_perpendicular_velocity_uncertainty(
                        err_ugosa, err_vgosa, gate_lon, gate_lat, gate_name=strait_name
                    )
                    st.session_state[f'{key_prefix}_sigma_v_perp'] = sigma_v_perp
                    logger.info(f"Velocity uncertainty computed: mean σ_v_perp = {np.nanmean(sigma_v_perp)*100:.2f} cm/s")
                else:
                    st.session_state[f'{key_prefix}_sigma_v_perp'] = None
                
                # Store results
                st.session_state[f'{key_prefix}_v_perp'] = v_perp
                st.session_state[f'{key_prefix}_monthly_v_perp'] = monthly_v_perp
                st.session_state[f'{key_prefix}_x_km'] = x_km
                st.session_state[f'{key_prefix}_gate_lon'] = gate_lon
                st.session_state[f'{key_prefix}_time_array'] = time_array
                st.session_state[f'{key_prefix}_bin'] = bin_size_km
                st.session_state[f'{key_prefix}_v_info'] = v_info
                
                normal_dir_label = v_info.get('normal_direction', '?')
                st.success(f"✅ Velocities computed! INTO-Arctic direction ≈ **{normal_dir_label}**")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                return
    
    # =========================================================================
    # CHECK IF DATA COMPUTED
    # =========================================================================
    if f'{key_prefix}_v_perp' not in st.session_state:
        st.info("👆 Click 'Compute Velocities' to start")
        return
    
    # Retrieve from session state
    v_perp = st.session_state[f'{key_prefix}_v_perp']
    monthly_v_perp = st.session_state[f'{key_prefix}_monthly_v_perp']
    stored_x_km = st.session_state[f'{key_prefix}_x_km']
    stored_gate_lon = st.session_state[f'{key_prefix}_gate_lon']
    stored_time_array = st.session_state.get(f'{key_prefix}_time_array', time_array)  # FIX: retrieve time_array
    v_geo_ts = st.session_state.get(f'{key_prefix}_v_geo_ts', None)
    v_info = st.session_state.get(f'{key_prefix}_v_info', {})
    sigma_v_perp = st.session_state.get(f'{key_prefix}_sigma_v_perp', None)
    
    # Show direction convention
    if v_info:
        normal_dir = v_info.get('normal_direction', '?')
        mean_v = v_info.get('mean_v_perp', 0)
        pos_meaning = v_info.get('positive_means', 'Into Arctic')
        neg_meaning = v_info.get('negative_means', 'Out of Arctic')
        
        direction_map = {
            'N': '🔼 Northward',
            'S': '🔽 Southward',
            'E': '➡️ Eastward',
            'W': '⬅️ Westward',
        }
        
        dir_emoji = direction_map.get(normal_dir, '❓')
        mean_label = pos_meaning if mean_v > 0 else neg_meaning
        
        st.info(f"""
        **📐 Sign Convention for this gate:**
        - INTO-Arctic unit vector points **{dir_emoji}** ({normal_dir})
        - **Positive v_perp** = Flow {pos_meaning}
        - **Negative v_perp** = Flow {neg_meaning}
        - Mean v_perp = **{mean_v*100:.2f} cm/s** ({mean_label})
        """)
    
    # Recompute if bin size changed
    stored_bin = st.session_state.get(f'{key_prefix}_bin', 5)
    if stored_bin != bin_size_km:
        from src.services.transport_service import compute_monthly_along_gate_profile
        monthly_v_perp = compute_monthly_along_gate_profile(stored_x_km, v_perp, stored_time_array, bin_size_km)
        st.session_state[f'{key_prefix}_monthly_v_perp'] = monthly_v_perp
        st.session_state[f'{key_prefix}_bin'] = bin_size_km
    
    # =========================================================================
    # 1. VELOCITY PROFILE ALONG GATE (with dual x-axis: km + deg)
    # =========================================================================
    st.markdown("### 📊 Velocity Profile Along Gate")
    st.caption(f"v_perp from ugos/vgos for **{month_names[selected_month-1]}** (averaged over all years)")
    
    # Uncertainty toggle
    show_uncertainty = st.checkbox("Show uncertainty (±σ)", value=True, key=f"{key_prefix}_show_unc_profile")
    
    bin_centers, bin_means, bin_stds = monthly_v_perp.get(selected_month, (np.array([]), np.array([]), np.array([])))
    
    if len(bin_centers) > 0:
        # Calculate longitude for bin centers (interpolate)
        bin_lon = np.interp(bin_centers, stored_x_km, stored_gate_lon)
        
        # Create figure with manual dual x-axis (Plotly doesn't support secondary_x in make_subplots)
        fig_profile = go.Figure()
        
        # v_perp profile (cm/s) on primary x-axis (km)
        fig_profile.add_trace(go.Scatter(
            x=bin_centers,
            y=bin_means * 100,  # m/s to cm/s
            mode='lines+markers',
            name='v_perp (ugos/vgos)',
            line=dict(color='#1E3A5F', width=2.5),
            marker=dict(size=7, color='#1E3A5F'),
            error_y=dict(type='data', array=bin_stds * 100, visible=True, color='rgba(30,58,95,0.3)'),
            xaxis='x'
        ))
        
        # Add uncertainty (±σ_v_perp) as dashed boundary lines + light fill
        if show_uncertainty and sigma_v_perp is not None:
            from src.services.transport_service import bin_along_gate
            time_pd_unc = pd.to_datetime(stored_time_array)
            month_mask_unc = time_pd_unc.month == selected_month
            if np.any(month_mask_unc):
                sigma_month = np.nanmean(sigma_v_perp[:, month_mask_unc], axis=1)  # (n_pts,)
                _, sigma_bin_means, _ = bin_along_gate(stored_x_km, sigma_month, bin_size_km)
                
                upper_unc = (bin_means + sigma_bin_means) * 100
                lower_unc = (bin_means - sigma_bin_means) * 100
                
                # Upper bound (thin dashed line)
                fig_profile.add_trace(go.Scatter(
                    x=bin_centers, y=upper_unc,
                    mode='lines', line=dict(width=1, dash='dot', color='rgba(230,126,34,0.6)'),
                    showlegend=False, hoverinfo='skip',
                ))
                # Lower bound with very subtle fill
                fig_profile.add_trace(go.Scatter(
                    x=bin_centers, y=lower_unc,
                    mode='lines', line=dict(width=1, dash='dot', color='rgba(230,126,34,0.6)'),
                    fill='tonexty', fillcolor='rgba(230,126,34,0.08)',
                    name='±σ (CMEMS formal mapping error)',
                    hoverinfo='skip',
                ))
        
        fig_profile.add_hline(y=0, line_color="#7F8C8D", line_width=1, line_dash="dash")
        
        fig_profile.update_layout(
            title=dict(text=f"Perpendicular Velocity Along Gate — {month_names[selected_month-1]}", font=dict(size=16)),
            yaxis_title="Velocity (cm/s)",
            height=420,
            
            
            font=dict(family="Inter, sans-serif", size=12),
            xaxis=dict(
                title="Distance along gate (km)",
                gridcolor='#E8E8E8', 
                gridwidth=1,
                side='bottom'
            ),
            xaxis2=dict(
                title="Longitude (°)",
                overlaying='x',
                side='top',
                range=[bin_lon.min(), bin_lon.max()] if len(bin_lon) > 0 else None,
                showgrid=False
            ),
            yaxis=dict(gridcolor='#E8E8E8', gridwidth=1),
            margin=dict(l=60, r=40, t=80, b=50),
        )
        
        st.plotly_chart(fig_profile, width='stretch', key="_render_geostrophic_velocity_tab_cmems_l4_1")
        
        # Mean velocity for this month
        mean_v = np.nanmean(bin_means) * 100
        if sigma_v_perp is not None:
            mean_sigma = np.nanmean(sigma_v_perp) * 100
            snr = abs(mean_v) / mean_sigma if mean_sigma > 0 else float('inf')
            st.info(
                f"**{month_names[selected_month-1]} Mean v_perp**: {mean_v:.2f} ± {mean_sigma:.2f} cm/s  \n"
                f"Signal/Noise ratio: **{snr:.1f}**  •  "
                f"σ from CMEMS L4 formal mapping errors (conservative upper bound)"
            )
        else:
            st.info(f"**{month_names[selected_month-1]} Mean v_perp**: {mean_v:.2f} cm/s")
        
        # =====================================================================
        # TIMELAPSE ANIMATION
        # =====================================================================
        st.markdown("---")
        enable_timelapse = st.checkbox(
            "🎬 Enable Monthly Timelapse Animation",
            value=False,
            key=f"{key_prefix}_velocity_timelapse",
            help="Animate through all 12 months to see seasonal evolution"
        )
        
        if enable_timelapse:
            st.markdown("#### 🎬 Velocity Profile Timelapse")
            st.caption("Press ▶️ Play to animate through all months, or use the slider to select a specific month")
            
            fig_timelapse = _create_velocity_timelapse(
                monthly_v_perp, 
                stored_x_km, 
                stored_gate_lon,
                title_prefix="Perpendicular Velocity Along Gate"
            )
            st.plotly_chart(fig_timelapse, width='stretch', key="_render_geostrophic_velocity_tab_cmems_l4_2")
            
            # Show monthly summary stats
            with st.expander("📊 Monthly Statistics Summary"):
                monthly_stats = []
                for m in range(1, 13):
                    bc, bm, bs = monthly_v_perp.get(m, (np.array([]), np.array([]), np.array([])))
                    if len(bm) > 0:
                        monthly_stats.append({
                            'Month': month_names[m-1],
                            'Mean v_perp (cm/s)': f"{np.nanmean(bm)*100:.2f}",
                            'Max (cm/s)': f"{np.nanmax(bm)*100:.2f}",
                            'Min (cm/s)': f"{np.nanmin(bm)*100:.2f}",
                            'Std (cm/s)': f"{np.nanmean(bs)*100:.2f}"
                        })
                if monthly_stats:
                    st.dataframe(pd.DataFrame(monthly_stats), width='stretch', hide_index=True)
    else:
        st.warning(f"No data available for {month_names[selected_month-1]}")
    
    # =========================================================================
    # 2. COMPARISON: v_perp vs v_geo (SINGLE PLOT - both on same axes)
    # =========================================================================
    st.markdown("### 📈 Time Series: v_perp vs v_geo")
    
    time_pd = pd.to_datetime(stored_time_array)  # FIX: use stored_time_array
    
    # Compute mean v_perp per time step (average along gate)
    v_perp_mean_ts = np.nanmean(v_perp, axis=0) * 100  # cm/s
    
    fig_ts = go.Figure()
    
    # v_perp line
    fig_ts.add_trace(go.Scatter(
        x=time_pd, y=v_perp_mean_ts,
        mode='lines', name='v_perp (ugos/vgos)',
        line=dict(color='#1E3A5F', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>v_perp: %{y:.2f} cm/s<extra></extra>'
    ))
    
    # Uncertainty band on time series (±σ_v_perp, gate-averaged)
    if show_uncertainty and sigma_v_perp is not None:
        sigma_mean_ts = np.nanmean(sigma_v_perp, axis=0) * 100  # cm/s
        upper_ts = v_perp_mean_ts + sigma_mean_ts
        lower_ts = v_perp_mean_ts - sigma_mean_ts
        
        fig_ts.add_trace(go.Scatter(
            x=time_pd, y=upper_ts,
            mode='lines', line=dict(width=0.5, dash='dot', color='rgba(230,126,34,0.4)'),
            showlegend=False, hoverinfo='skip',
        ))
        fig_ts.add_trace(go.Scatter(
            x=time_pd, y=lower_ts,
            mode='lines', line=dict(width=0.5, dash='dot', color='rgba(230,126,34,0.4)'),
            fill='tonexty', fillcolor='rgba(230,126,34,0.08)',
            name='±σ (CMEMS formal error)',
            hoverinfo='skip',
        ))
    
    # v_geo line (if enabled and available)
    if show_v_geo and v_geo_ts is not None:
        fig_ts.add_trace(go.Scatter(
            x=time_pd, y=v_geo_ts * 100,  # m/s to cm/s
            mode='lines', name='v_geo (DOT slope)',
            line=dict(color='#E07B53', width=2),
            hovertemplate='%{x|%Y-%m-%d}<br>v_geo: %{y:.2f} cm/s<extra></extra>'
        ))
    
    fig_ts.add_hline(y=0, line_dash="dash", line_color="#7F8C8D", line_width=1)
    
    title = "Geostrophic Velocity Comparison" if (show_v_geo and v_geo_ts is not None) else "Geostrophic Velocity: v_perp"
    fig_ts.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Time",
        yaxis_title="Velocity (cm/s)",
        height=450,
        
        
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor='#E8E8E8', gridwidth=1),
        yaxis=dict(gridcolor='#E8E8E8', gridwidth=1),
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            bgcolor='rgba(255,255,255,0.9)', bordercolor='#E8E8E8', borderwidth=1
        ),
        margin=dict(l=60, r=40, t=60, b=50),
    )
    
    st.plotly_chart(fig_ts, width='stretch', key="_render_geostrophic_velocity_tab_cmems_l4_3")
    
    # Statistics comparison
    if show_v_geo and v_geo_ts is not None:
        n_stat_cols = 4 if sigma_v_perp is not None else 3
        stat_cols = st.columns(n_stat_cols)
        with stat_cols[0]:
            st.markdown("**v_perp (ugos/vgos)**")
            st.metric("Mean", f"{np.nanmean(v_perp_mean_ts):.2f} cm/s")
            st.metric("Std", f"{np.nanstd(v_perp_mean_ts):.2f} cm/s")
        with stat_cols[1]:
            st.markdown("**v_geo (DOT slope)**")
            st.metric("Mean", f"{np.nanmean(v_geo_ts)*100:.2f} cm/s")
            st.metric("Std", f"{np.nanstd(v_geo_ts)*100:.2f} cm/s")
        with stat_cols[2]:
            diff = v_perp_mean_ts - v_geo_ts * 100
            st.markdown("**Difference**")
            st.metric("Mean diff", f"{np.nanmean(diff):.2f} cm/s")
            st.metric("RMSE", f"{np.sqrt(np.nanmean(diff**2)):.2f} cm/s")
        if sigma_v_perp is not None:
            with stat_cols[3]:
                sigma_mean_all = np.nanmean(sigma_v_perp) * 100
                st.markdown("**σ_v_perp (uncertainty)**")
                st.metric("Mean σ", f"{sigma_mean_all:.2f} cm/s")
                st.metric("σ / |v|", f"{sigma_mean_all / max(abs(np.nanmean(v_perp_mean_ts)), 0.01):.1%}")
    else:
        n_stat_cols = 3 if sigma_v_perp is not None else 2
        stat_cols = st.columns(n_stat_cols)
        with stat_cols[0]:
            st.metric("Mean v_perp", f"{np.nanmean(v_perp_mean_ts):.2f} cm/s")
        with stat_cols[1]:
            st.metric("Std v_perp", f"{np.nanstd(v_perp_mean_ts):.2f} cm/s")
        if sigma_v_perp is not None:
            with stat_cols[2]:
                sigma_mean_all = np.nanmean(sigma_v_perp) * 100
                st.metric("Mean σ_v_perp", f"{sigma_mean_all:.2f} cm/s")
    

# ==============================================================================
# VOLUME TRANSPORT TAB (CMEMS L4)
# ==============================================================================
def _render_volume_transport_tab_cmems_l4(cmems_l4_data, config: AppConfig):
    """
    Render Volume Transport tab for CMEMS L4.
    
    Layout:
    1. Bathymetry Profile (IN CIMA) - con linea rossa a 250m
    2. Volume Transport Along-Gate - menu mese + slider km
    3. Statistics + Export
    
    Uses GEBCO bathymetry with 250m cap for transport calculation.
    """
    st.subheader("🚢 Volume Transport Calculation")
    
    strait_name = getattr(cmems_l4_data, 'strait_name', 'Unknown')
    ugos_matrix = getattr(cmems_l4_data, 'ugos_matrix', None)
    vgos_matrix = getattr(cmems_l4_data, 'vgos_matrix', None)
    gate_lon = getattr(cmems_l4_data, 'gate_lon_pts', None)
    gate_lat = getattr(cmems_l4_data, 'gate_lat_pts', None)
    x_km = getattr(cmems_l4_data, 'x_km', None)
    time_array = getattr(cmems_l4_data, 'time_array', None)
    
    # Check velocity data
    if ugos_matrix is None or vgos_matrix is None:
        st.warning("⚠️ Velocity data (ugos/vgos) not available.")
        st.info("""
        **To enable Volume Transport:**
        1. Go to sidebar → CMEMS L4 Variables
        2. Select **ugos** and **vgos** 
        3. Reload the data
        """)
        return
    
    st.success(f"✅ Velocity data loaded: {ugos_matrix.shape[0]} gate points × {ugos_matrix.shape[1]} time steps")
    
    # Namespace per gate — evita contaminazione tra gate diversi nel session_state
    vt_key = f"vt_{strait_name.replace(' ', '_').lower()}"
    
    # =========================================================================
    # CONTROLS
    # =========================================================================
    st.markdown("### ⚙️ Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        depth_cap = st.number_input(
            "Depth Cap (m)",
            min_value=50,
            max_value=1000,
            value=250,
            step=50,
            key=f"{vt_key}_depth_cap",
            help="Maximum depth for transport calculation (GEBCO capped at this value)"
        )
    
    with col2:
        # Adaptive minimum based on CMEMS native resolution
        native_res_km = getattr(cmems_l4_data, 'native_resolution_km', 3.0)
        effective_spacing_km = getattr(cmems_l4_data, 'effective_spacing_km', native_res_km)
        min_bin_km = max(1, int(np.ceil(native_res_km)))
        
        bin_size_km = st.slider(
            "Spatial averaging (km)",
            min_value=min_bin_km,
            max_value=50,
            value=max(5, min_bin_km),
            step=1,
            key=f"{vt_key}_bin_size",
            help="Average transport over bins of this width"
        )
        st.caption(f"🔬 Native res: {native_res_km:.1f} km | Effective: {effective_spacing_km:.1f} km")
    
    with col3:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        selected_month = st.selectbox(
            "Month to display",
            options=list(range(1, 13)),
            format_func=lambda m: month_names[m-1],
            index=0,
            key=f"{vt_key}_month_select"
        )
    
    # =========================================================================
    # LOAD/COMPUTE DATA
    # =========================================================================
    if st.button("🧮 Compute Transport", type="primary", width='stretch'):
        with st.spinner("Loading bathymetry and computing transport..."):
            try:
                # 1. Load GEBCO bathymetry (with caching, NO cap for display)
                from src.services.gebco_service import get_bathymetry_cache
                
                cache = get_bathymetry_cache()
                depth_profile_full = cache.get_or_compute(
                    gate_name=strait_name,
                    gate_lons=gate_lon,
                    gate_lats=gate_lat,
                    gebco_path=config.gebco_nc_path,
                    depth_cap=None  # Full bathymetry for display
                )
                
                # 2. Apply cap for transport calculation
                depth_profile_capped = np.minimum(depth_profile_full, depth_cap)
                
                # 3. Compute perpendicular velocity
                from src.services.transport_service import (
                    compute_perpendicular_velocity,
                    compute_segment_widths,
                    bin_along_gate,
                    compute_monthly_along_gate_profile,
                    compute_perpendicular_velocity_uncertainty,
                    compute_volume_transport_uncertainty,
                    SVERDRUP
                )
                
                v_perp = compute_perpendicular_velocity(ugos_matrix, vgos_matrix, gate_lon, gate_lat, gate_name=strait_name)
                
                # 4. Compute segment widths
                widths = compute_segment_widths(gate_lon, gate_lat, x_km)
                
                # 5. Compute transport per point per time step
                # Q(x, t) = v_perp(x, t) × h(x) × Δx
                # Shape: (n_pts, n_time)
                transport_per_point = v_perp * depth_profile_capped[:, np.newaxis] * widths[:, np.newaxis]
                transport_per_point_sv = transport_per_point / SVERDRUP
                
                # 6. Total transport time series
                # Use nansum but protect against all-NaN timesteps (nansum returns 0 for all-NaN)
                transport_total_sv = np.nansum(transport_per_point_sv, axis=0)
                all_nan_mask = np.all(np.isnan(transport_per_point_sv), axis=0)
                transport_total_sv[all_nan_mask] = np.nan
                
                # 7. Monthly along-gate profiles
                monthly_profiles = compute_monthly_along_gate_profile(
                    x_km, transport_per_point_sv, time_array, bin_size_km
                )
                
                # 8. Compute transport uncertainty if err_ugosa/err_vgosa available
                err_ugosa = getattr(cmems_l4_data, 'err_ugosa_matrix', None)
                err_vgosa = getattr(cmems_l4_data, 'err_vgosa_matrix', None)
                
                sigma_Q_sv = None
                if err_ugosa is not None and err_vgosa is not None:
                    sigma_v_perp = compute_perpendicular_velocity_uncertainty(
                        err_ugosa, err_vgosa, gate_lon, gate_lat, gate_name=strait_name
                    )
                    sigma_Q_m3s = compute_volume_transport_uncertainty(
                        sigma_v_perp, depth_profile_capped, x_km, gate_lon, gate_lat
                    )
                    sigma_Q_sv = sigma_Q_m3s / SVERDRUP
                    logger.info(f"Transport uncertainty: mean σ_Q = {np.nanmean(sigma_Q_sv):.4f} Sv")
                
                # Store in session state — namespaced by gate to avoid cross-gate contamination
                st.session_state[f'{vt_key}_depth_full'] = depth_profile_full
                st.session_state[f'{vt_key}_depth_capped'] = depth_profile_capped
                st.session_state[f'{vt_key}_v_perp'] = v_perp
                st.session_state[f'{vt_key}_transport_per_point_sv'] = transport_per_point_sv
                st.session_state[f'{vt_key}_transport_total_sv'] = transport_total_sv
                st.session_state[f'{vt_key}_monthly_profiles'] = monthly_profiles
                st.session_state[f'{vt_key}_x_km'] = x_km
                st.session_state[f'{vt_key}_time_array'] = time_array
                st.session_state[f'{vt_key}_gate_lon'] = gate_lon
                st.session_state[f'{vt_key}_stored_depth_cap'] = depth_cap
                st.session_state[f'{vt_key}_stored_bin_size'] = bin_size_km
                st.session_state[f'{vt_key}_sigma_Q_sv'] = sigma_Q_sv
                
                if sigma_Q_sv is not None:
                    st.success(f"✅ Transport computed! (with uncertainty: mean σ_Q = {np.nanmean(sigma_Q_sv):.4f} Sv)")
                else:
                    st.success("✅ Transport computed!")
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                return
    
    # =========================================================================
    # CHECK IF DATA COMPUTED
    # =========================================================================
    if f'{vt_key}_depth_full' not in st.session_state:
        st.info("👆 Click 'Compute Transport' to start")
        return
    
    # Retrieve from session state — namespaced by gate
    depth_full = st.session_state[f'{vt_key}_depth_full']
    depth_capped = st.session_state[f'{vt_key}_depth_capped']
    transport_per_point_sv = st.session_state[f'{vt_key}_transport_per_point_sv']
    transport_total_sv = st.session_state[f'{vt_key}_transport_total_sv']
    monthly_profiles = st.session_state[f'{vt_key}_monthly_profiles']
    stored_depth_cap = st.session_state[f'{vt_key}_stored_depth_cap']
    time_array = st.session_state.get(f'{vt_key}_time_array', None)
    x_km = st.session_state.get(f'{vt_key}_x_km', x_km)
    gate_lon = st.session_state.get(f'{vt_key}_gate_lon', gate_lon)
    sigma_Q_sv = st.session_state.get(f'{vt_key}_sigma_Q_sv', None)
    
    # =========================================================================
    # 1. BATHYMETRY PROFILE (IN CIMA) - with dual x-axis (km + deg)
    # =========================================================================
    st.markdown("### 🌊 Bathymetry Profile")
    st.caption(f"GEBCO bathymetry along gate. Red line = {stored_depth_cap}m depth cap for transport.")
    
    # Create figure with manual dual x-axis
    fig_bathy = go.Figure()
    
    # Fill area for bathymetry (real depth) - primary x-axis (km)
    fig_bathy.add_trace(go.Scatter(
        x=x_km,
        y=-depth_full,  # Negative to show below sea level
        fill='tozeroy',
        fillcolor='rgba(30, 58, 95, 0.4)',
        line=dict(color='#1E3A5F', width=2),
        name='GEBCO Depth',
        hovertemplate='%{x:.1f} km<br>Depth: %{y:.0f} m<extra></extra>',
        xaxis='x'
    ))
    
    # Sea level line
    fig_bathy.add_hline(y=0, line_color="#3498DB", line_width=2, 
                        annotation_text="Sea Level", annotation_position="top left")
    
    # Depth cap line (RED)
    fig_bathy.add_hline(y=-stored_depth_cap, line_color="#E74C3C", line_width=2, line_dash="dash",
                        annotation_text=f"Cap: {stored_depth_cap}m", 
                        annotation_position="bottom right")
    
    fig_bathy.update_layout(
        title=dict(text=f"{strait_name} — Cross-Section Bathymetry", font=dict(size=16)),
        yaxis_title="Depth (m)",
        height=380,
        
        
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(
            title="Distance along gate (km)",
            gridcolor='#E8E8E8',
            side='bottom'
        ),
        xaxis2=dict(
            title="Longitude (°)",
            overlaying='x',
            side='top',
            range=[gate_lon.min(), gate_lon.max()] if len(gate_lon) > 0 else None,
            showgrid=False
        ),
        yaxis=dict(gridcolor='#E8E8E8', range=[min(-depth_full.max() * 1.1, -stored_depth_cap * 1.5), 50]),
        margin=dict(l=60, r=40, t=80, b=50),
    )
    
    st.plotly_chart(fig_bathy, width='stretch', key="_render_volume_transport_tab_cmems_l4_1")
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max Depth", f"{depth_full.max():.0f} m")
    with col2:
        st.metric("Mean Depth", f"{depth_full.mean():.0f} m")
    with col3:
        st.metric("Depth Cap", f"{stored_depth_cap} m")
    with col4:
        pct_capped = (depth_full > stored_depth_cap).sum() / len(depth_full) * 100
        st.metric("% Capped", f"{pct_capped:.1f}%")
    
    # =========================================================================
    # 2. VOLUME TRANSPORT ALONG-GATE (Monthly Profile) - with dual x-axis
    # =========================================================================
    st.markdown("### 📊 Volume Transport Along Gate")
    st.caption(f"Monthly mean transport profile for **{month_names[selected_month-1]}** (averaged over all years)")
    
    # Recompute if bin size changed
    current_bin = st.session_state.get(f'{vt_key}_stored_bin_size', 5)
    if current_bin != bin_size_km:
        from src.services.transport_service import compute_monthly_along_gate_profile
        monthly_profiles = compute_monthly_along_gate_profile(
            x_km, transport_per_point_sv, time_array, bin_size_km
        )
        st.session_state[f'{vt_key}_monthly_profiles'] = monthly_profiles
        st.session_state[f'{vt_key}_stored_bin_size'] = bin_size_km
    
    # Get profile for selected month
    bin_centers, bin_means, bin_stds = monthly_profiles.get(selected_month, (np.array([]), np.array([]), np.array([])))
    
    if len(bin_centers) > 0:
        # Calculate longitude for bin centers (interpolate)
        bin_lon = np.interp(bin_centers, x_km, gate_lon)
        
        # Create figure with manual dual x-axis
        fig_profile = go.Figure()
        
        # Bar chart with elegant colors
        colors = ['#3498DB' if v >= 0 else '#E74C3C' for v in bin_means]
        
        fig_profile.add_trace(go.Bar(
            x=bin_centers,
            y=bin_means,
            marker_color=colors,
            name=f'{month_names[selected_month-1]} Mean',
            error_y=dict(type='data', array=bin_stds, visible=True, color='rgba(0,0,0,0.3)'),
            hovertemplate='%{x:.1f} km<br>Transport: %{y:.4f} ×10⁶ m³/s<extra></extra>',
            xaxis='x'
        ))
        
        fig_profile.add_hline(y=0, line_color="#7F8C8D", line_width=1)
        
        fig_profile.update_layout(
            title=dict(text=f"Transport Along Gate — {month_names[selected_month-1]}", font=dict(size=16)),
            yaxis_title="Transport (×10⁶ m³/s)",
            height=420,
            
            
            font=dict(family="Inter, sans-serif", size=12),
            xaxis=dict(
                title="Distance along gate (km)",
                gridcolor='#E8E8E8',
                side='bottom'
            ),
            xaxis2=dict(
                title="Longitude (°)",
                overlaying='x',
                side='top',
                range=[bin_lon.min(), bin_lon.max()] if len(bin_lon) > 0 else None,
                showgrid=False
            ),
            yaxis=dict(gridcolor='#E8E8E8'),
            bargap=0.15,
            margin=dict(l=60, r=40, t=80, b=50),
        )
        
        st.plotly_chart(fig_profile, width='stretch', key="_render_volume_transport_tab_cmems_l4_2")
        
        # Monthly total for this month
        total_month = np.nansum(bin_means)
        total_m3s = total_month * 1e6
        st.info(f"**{month_names[selected_month-1]} Total Transport**: {total_m3s:.2e} m³/s ({total_month:.3f} ×10⁶ m³/s)")
        
        # =====================================================================
        # TIMELAPSE ANIMATION
        # =====================================================================
        st.markdown("---")
        enable_transport_timelapse = st.checkbox(
            "🎬 Enable Monthly Timelapse Animation",
            value=False,
            key=f"{vt_key}_timelapse",
            help="Animate through all 12 months to see seasonal transport evolution"
        )
        
        if enable_transport_timelapse:
            st.markdown("#### 🎬 Volume Transport Timelapse")
            st.caption("Press ▶️ Play to animate through all months, or use the slider to select a specific month")
            
            fig_transport_timelapse = _create_transport_timelapse(
                monthly_profiles,
                x_km,
                gate_lon,
                title_prefix="Volume Transport Along Gate"
            )
            st.plotly_chart(fig_transport_timelapse, width='stretch', key="_render_volume_transport_tab_cmems_l4_3")
            
            # Show monthly summary stats
            with st.expander("📊 Monthly Transport Summary"):
                transport_stats = []
                for m in range(1, 13):
                    bc, bm, bs = monthly_profiles.get(m, (np.array([]), np.array([]), np.array([])))
                    if len(bm) > 0:
                        total_sv = np.nansum(bm)
                        transport_stats.append({
                            'Month': month_names[m-1],
                            'Total (×10⁶ m³/s)': f"{total_sv:.4f}",
                            'Total (m³/s)': f"{total_sv*1e6:.2e}",
                            'Max bin': f"{np.nanmax(bm):.4f}",
                            'Min bin': f"{np.nanmin(bm):.4f}"
                        })
                if transport_stats:
                    st.dataframe(pd.DataFrame(transport_stats), width='stretch', hide_index=True)
    else:
        st.warning(f"No data available for {month_names[selected_month-1]}")
    
    # =========================================================================
    # 3. TOTAL TRANSPORT TIME SERIES
    # =========================================================================
    st.markdown("### 📈 Total Transport Time Series")
    
    # Uncertainty toggle
    show_vt_unc = st.checkbox("Show uncertainty (±σ_Q)", value=True, key=f"{vt_key}_show_unc_ts")
    
    # Check time_array is available
    if time_array is None:
        st.warning("⚠️ Time array not available. Please re-compute transport.")
        return
    
    time_pd = pd.to_datetime(time_array)
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=time_pd,
        y=transport_total_sv,
        mode='lines',
        name='Total Transport',
        line=dict(color='#1E3A5F', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>Transport: %{y:.3f} ×10⁶ m³/s<extra></extra>'
    ))
    
    # Add uncertainty band (±σ_Q) if available
    if show_vt_unc and sigma_Q_sv is not None:
        upper_Q = transport_total_sv + sigma_Q_sv
        lower_Q = transport_total_sv - sigma_Q_sv
        
        fig_ts.add_trace(go.Scatter(
            x=time_pd, y=upper_Q,
            mode='lines', line=dict(width=0.5, dash='dot', color='rgba(230,126,34,0.4)'),
            showlegend=False, hoverinfo='skip',
        ))
        fig_ts.add_trace(go.Scatter(
            x=time_pd, y=lower_Q,
            mode='lines', line=dict(width=0.5, dash='dot', color='rgba(230,126,34,0.4)'),
            fill='tonexty', fillcolor='rgba(230,126,34,0.08)',
            name='±σ_Q (CMEMS formal error)',
            hoverinfo='skip',
        ))
    
    mean_transport = np.nanmean(transport_total_sv)
    fig_ts.add_hline(y=0, line_dash="dash", line_color="#7F8C8D", line_width=1)
    fig_ts.add_hline(y=mean_transport, line_dash="dot", line_color="#E74C3C", line_width=1.5,
                     annotation_text=f"Mean: {mean_transport:.3f}")
    
    fig_ts.update_layout(
        title=dict(text=f"{strait_name} — Volume Transport Time Series", font=dict(size=16)),
        xaxis_title="Time",
        yaxis_title="Transport (×10⁶ m³/s)",
        height=420,
        
        
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor='#E8E8E8'),
        yaxis=dict(gridcolor='#E8E8E8'),
        margin=dict(l=60, r=40, t=60, b=50),
    )
    
    st.plotly_chart(fig_ts, width='stretch', key="_render_volume_transport_tab_cmems_l4_4")
    
    # =========================================================================
    # 4. STATISTICS + EXPORT
    # =========================================================================
    st.markdown("### 📊 Statistics")
    
    n_stat_cols = 5 if sigma_Q_sv is not None else 4
    stat_cols = st.columns(n_stat_cols)
    with stat_cols[0]:
        st.metric("Mean Transport", f"{np.nanmean(transport_total_sv):.3f} ×10⁶ m³/s")
    with stat_cols[1]:
        st.metric("Std Dev", f"{np.nanstd(transport_total_sv):.3f} ×10⁶ m³/s")
    with stat_cols[2]:
        st.metric("Min", f"{np.nanmin(transport_total_sv):.3f} ×10⁶ m³/s")
    with stat_cols[3]:
        st.metric("Max", f"{np.nanmax(transport_total_sv):.3f} ×10⁶ m³/s")
    if sigma_Q_sv is not None:
        with stat_cols[4]:
            st.metric("Mean σ_Q", f"{np.nanmean(sigma_Q_sv):.3f} ×10⁶ m³/s")
    
    st.caption("1 ×10⁶ m³/s = 1 Sv (Sverdrup) | Positive = into Arctic (physics-based dot product)")
    
    # Export
    with st.expander("📥 Export Transport Data"):
        export_df = pd.DataFrame({
            'time': time_pd,
            'transport_sv': transport_total_sv,
        })
        
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            "Download Time Series (CSV)",
            data=csv_data,
            file_name=f"volume_transport_{strait_name.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )


# ==============================================================================
# CMEMS L4 TAB: MASS TRANSPORT (Salt + Freshwater)
# ==============================================================================

def _render_mass_transport_tab_cmems_l4(cmems_l4_data, config: AppConfig):
    """
    Render Mass Transport tab for CMEMS L4.
    
    Computes and displays:
    - Salt Water Mass Transport: Sm = Σ ρ × (S/1000) × v_perp × H × Δx  [kg/s]
    - Freshwater Transport: Fw = Σ v_perp × (1 - S/S_ref) × H × Δx  [m³/s]
    
    Also provides Excel export for multi-gate mass transport.
    
    Constants:
        ρ = 1025 kg/m³ (constant)
        S_ref = 34.8 PSU
        Salinity = 35.0 PSU (spatially uniform assumption)
    """
    st.subheader("📦 Mass Transport (Salt + Freshwater)")
    
    strait_name = getattr(cmems_l4_data, 'strait_name', 'Unknown')
    ugos_matrix = getattr(cmems_l4_data, 'ugos_matrix', None)
    vgos_matrix = getattr(cmems_l4_data, 'vgos_matrix', None)
    gate_lon = getattr(cmems_l4_data, 'gate_lon_pts', None)
    gate_lat = getattr(cmems_l4_data, 'gate_lat_pts', None)
    x_km = getattr(cmems_l4_data, 'x_km', None)
    time_array = getattr(cmems_l4_data, 'time_array', None)
    
    # Check velocity data
    if ugos_matrix is None or vgos_matrix is None:
        st.warning("⚠️ Velocity data (ugos/vgos) not available.")
        st.info("""
        **To enable Mass Transport:**
        1. Go to sidebar → CMEMS L4 Variables
        2. Select **ugos** and **vgos** 
        3. Reload the data
        """)
        return
    
    st.success(f"✅ Velocity data loaded: {ugos_matrix.shape[0]} gate points × {ugos_matrix.shape[1]} time steps")
    
    # =========================================================================
    # CONTROLS
    # =========================================================================
    st.markdown("### ⚙️ Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        depth_cap = st.number_input(
            "Depth Cap (m)",
            min_value=50,
            max_value=1000,
            value=250,
            step=50,
            key="mt_depth_cap",
            help="Maximum depth for transport integration"
        )
    
    with col2:
        salinity_input = st.number_input(
            "Salinity (PSU)",
            min_value=20.0,
            max_value=40.0,
            value=35.0,
            step=0.1,
            key="mt_salinity",
            help="Assumed uniform salinity (simplification)"
        )
    
    with col3:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        selected_month = st.selectbox(
            "Month to display",
            options=list(range(1, 13)),
            format_func=lambda m: month_names[m-1],
            index=0,
            key="mt_month_select"
        )
    
    st.info(f"**Constants:** ρ = 1025 kg/m³ (constant), S_ref = 34.8 PSU | "
            f"**Note:** Constant density, NOT from SSS dataset")
    
    # =========================================================================
    # COMPUTE BUTTON
    # =========================================================================
    if st.button("🧮 Compute Mass Transport", type="primary", use_container_width=True):
        with st.spinner("Computing mass transport..."):
            try:
                from src.services.transport_service import (
                    compute_perpendicular_velocity,
                    compute_segment_widths,
                )
                from src.services.gebco_service import get_bathymetry_cache
                from src.services.mass_transport_service import MassTransportService
                
                progress_bar = st.progress(0, text="Computing perpendicular velocity...")
                
                # 1. Compute perpendicular velocity
                v_perp = compute_perpendicular_velocity(
                    ugos_matrix, vgos_matrix, gate_lon, gate_lat, gate_name=strait_name
                )
                progress_bar.progress(20, text="Loading bathymetry...")
                
                # 2. Load bathymetry
                cache = get_bathymetry_cache()
                depth_profile = cache.get_or_compute(
                    gate_name=strait_name,
                    gate_lons=gate_lon,
                    gate_lats=gate_lat,
                    gebco_path=config.gebco_nc_path,
                    depth_cap=None  # Get full, cap below
                )
                
                # Cap depth
                depth_capped = np.minimum(depth_profile, depth_cap)
                progress_bar.progress(40, text="Computing segment widths...")
                
                # 3. Compute segment widths
                widths = compute_segment_widths(gate_lon, gate_lat, x_km)
                progress_bar.progress(60, text="Computing mass transport...")
                
                # 4. Compute mass transport
                mt_service = MassTransportService()
                mt_result = mt_service.compute(
                    v_perp=v_perp,
                    depth_profile=depth_capped,
                    segment_widths=widths,
                    salinity=salinity_input,
                    time_array=time_array,
                    gate_name=strait_name,
                    depth_cap=float(depth_cap)
                )
                progress_bar.progress(80, text="Aggregating monthly data...")
                
                # 5. Aggregate monthly
                monthly_data = mt_service.aggregate_monthly(mt_result)
                
                progress_bar.progress(100, text="Done!")
                
                # Store in session state
                st.session_state['mt_result'] = mt_result
                st.session_state['mt_monthly'] = monthly_data
                st.session_state['mt_v_perp'] = v_perp
                st.session_state['mt_depth_capped'] = depth_capped
                st.session_state['mt_depth_profile'] = depth_profile
                st.session_state['mt_strait_name'] = strait_name
                st.session_state['mt_salinity'] = salinity_input
                
                st.success("✅ Mass transport computed!")
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                return
    
    # =========================================================================
    # CHECK IF DATA COMPUTED
    # =========================================================================
    if 'mt_result' not in st.session_state:
        st.info("👆 Click 'Compute Mass Transport' to start")
        return
    
    mt_result = st.session_state['mt_result']
    monthly_data = st.session_state['mt_monthly']
    
    time_pd = pd.to_datetime(time_array)
    
    # =========================================================================
    # 1. SALT TRANSPORT TIME SERIES
    # =========================================================================
    st.markdown("### 🧂 Salt Water Mass Transport")
    
    salt_scaled = mt_result.salt_transport / 1e7  # ×10⁷ kg/s
    
    fig_salt = go.Figure()
    fig_salt.add_trace(go.Scatter(
        x=time_pd,
        y=salt_scaled,
        mode='lines',
        name='Salt Transport',
        line=dict(color='#9B59B6', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>Sm: %{y:.2f} ×10⁷ kg/s<extra></extra>'
    ))
    
    mean_salt = np.nanmean(salt_scaled)
    fig_salt.add_hline(y=0, line_dash="dash", line_color="#7F8C8D", line_width=1)
    fig_salt.add_hline(y=mean_salt, line_dash="dot", line_color="#E74C3C", line_width=1.5,
                       annotation_text=f"Mean: {mean_salt:.2f}")
    
    fig_salt.update_layout(
        title=dict(text=f"{strait_name} — Salt Water Mass Transport", font=dict(size=16)),
        xaxis_title="Time",
        yaxis_title="Salt Transport Sm (×10⁷ kg/s)",
        height=400,
         
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor='#E8E8E8'),
        yaxis=dict(gridcolor='#E8E8E8'),
        margin=dict(l=60, r=40, t=60, b=50),
    )
    st.plotly_chart(fig_salt, use_container_width=True, key="_render_mass_transport_tab_cmems_l4_1")
    
    # =========================================================================
    # 2. FRESHWATER TRANSPORT TIME SERIES
    # =========================================================================
    st.markdown("### 💧 Freshwater Transport")
    
    fw_scaled = mt_result.freshwater_transport / 1e3  # ×10³ m³/s = mSv
    
    fig_fw = go.Figure()
    fig_fw.add_trace(go.Scatter(
        x=time_pd,
        y=fw_scaled,
        mode='lines',
        name='Freshwater Transport',
        line=dict(color='#3498DB', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>Fw: %{y:.2f} ×10³ m³/s<extra></extra>'
    ))
    
    mean_fw = np.nanmean(fw_scaled)
    fig_fw.add_hline(y=0, line_dash="dash", line_color="#7F8C8D", line_width=1)
    fig_fw.add_hline(y=mean_fw, line_dash="dot", line_color="#E74C3C", line_width=1.5,
                     annotation_text=f"Mean: {mean_fw:.2f}")
    
    fig_fw.update_layout(
        title=dict(text=f"{strait_name} — Freshwater Transport (S_ref = {mt_result.s_ref} PSU)", font=dict(size=16)),
        xaxis_title="Time",
        yaxis_title="Freshwater Transport Fw (×10³ m³/s)",
        height=400,
         
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor='#E8E8E8'),
        yaxis=dict(gridcolor='#E8E8E8'),
        margin=dict(l=60, r=40, t=60, b=50),
    )
    st.plotly_chart(fig_fw, use_container_width=True, key="_render_mass_transport_tab_cmems_l4_2")
    
    # =========================================================================
    # 3. STATISTICS
    # =========================================================================
    st.markdown("### 📊 Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Salt Transport", f"{np.nanmean(mt_result.salt_transport):.2e} kg/s")
    with col2:
        st.metric("Std Salt Transport", f"{np.nanstd(mt_result.salt_transport):.2e} kg/s")
    with col3:
        st.metric("Mean FW Transport", f"{np.nanmean(mt_result.freshwater_transport):.2e} m³/s")
    with col4:
        st.metric("Std FW Transport", f"{np.nanstd(mt_result.freshwater_transport):.2e} m³/s")
    
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Salinity Used", f"{mt_result.salinity_used:.1f} PSU")
    with col2:
        st.metric("Density (ρ)", f"{mt_result.rho:.0f} kg/m³")
    with col3:
        st.metric("S_ref", f"{mt_result.s_ref:.1f} PSU")
    with col4:
        st.metric("Depth Cap", f"{mt_result.depth_cap:.0f} m")
    
    # =========================================================================
    # 5. EXCEL EXPORT (multi-gate)
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📥 Excel Export (Multi-Gate)")
    st.caption("Export mass transport data for ALL previously computed gates to a single Excel file.")
    
    # Store this gate's monthly data in a persistent dict
    if 'mt_all_gates' not in st.session_state:
        st.session_state['mt_all_gates'] = {}
    
    # Always update current gate
    st.session_state['mt_all_gates'][strait_name] = monthly_data
    
    # Show available gates
    available_gates = list(st.session_state['mt_all_gates'].keys())
    st.markdown(f"**Gates available for export:** {', '.join(available_gates)}")
    st.caption("💡 Run Mass Transport on each gate to include it in the export. "
               "Switch gates in the sidebar and compute to add more.")
    
    if st.button("📊 Export Mass Transport Excel", type="secondary", use_container_width=True):
        try:
            from src.services.mass_transport_excel import export_mass_transport_excel
            
            excel_bytes = export_mass_transport_excel(
                gate_results=st.session_state['mt_all_gates'],
                notes=f"Exported from NICO/ARCFRESH. Salinity={mt_result.salinity_used} PSU (uniform). "
                      f"Depth cap={mt_result.depth_cap} m.",
                s_ref=mt_result.s_ref,
                rho=mt_result.rho
            )
            
            if excel_bytes:
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_bytes,
                    file_name=f"mass_transport_arctic_gates.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )
                st.success(f"✅ Excel ready: {len(available_gates)} gate(s), "
                          f"{len(monthly_data)} months for current gate")
            else:
                st.warning("No data to export")
                
        except ImportError as e:
            st.error(f"❌ Missing dependency: {e}")
            st.info("Install openpyxl: `pip install openpyxl`")
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
            import traceback
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
    
    # =========================================================================
    # 6. CSV EXPORT (single gate, for backward compatibility)
    # =========================================================================
    with st.expander("📥 Export Single Gate CSV"):
        export_df = pd.DataFrame({
            'time': time_pd,
            'salt_transport_kg_s': mt_result.salt_transport,
            'salt_transport_1e7_kg_s': salt_scaled,
            'freshwater_transport_m3_s': mt_result.freshwater_transport,
            'freshwater_transport_1e3_m3_s': fw_scaled,
        })
        
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            "Download Time Series (CSV)",
            data=csv_data,
            file_name=f"mass_transport_{strait_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key="mt_csv_export"
        )


# ==============================================================================
# CMEMS L4 TAB: SALT FLUX (LEGACY - kept for reference)
# ==============================================================================

def _render_salt_flux_tab_cmems_l4(cmems_l4_data, config: AppConfig):
    """
    Render Salt Flux tab for CMEMS L4.
    
    Combines:
    - CMEMS L4 velocity (ugos/vgos) 
    - SSS data (salinity, density) from CMEMS SSS dataset
    - GEBCO bathymetry (capped at depth_cap)
    
    Formula: F_salt = Σ ρ(s,t) × S(s,t)/1000 × u_normal(s,t) × H_eff(s) × ds
    """
    st.subheader("🧂 Salt Flux Calculation")
    
    strait_name = getattr(cmems_l4_data, 'strait_name', 'Unknown')
    ugos_matrix = getattr(cmems_l4_data, 'ugos_matrix', None)
    vgos_matrix = getattr(cmems_l4_data, 'vgos_matrix', None)
    gate_lon = getattr(cmems_l4_data, 'gate_lon_pts', None)
    gate_lat = getattr(cmems_l4_data, 'gate_lat_pts', None)
    x_km = getattr(cmems_l4_data, 'x_km', None)
    time_array = getattr(cmems_l4_data, 'time_array', None)
    
    # Check velocity data
    if ugos_matrix is None or vgos_matrix is None:
        st.warning("⚠️ Velocity data (ugos/vgos) not available.")
        st.info("""
        **To enable Salt Flux:**
        1. Go to sidebar → CMEMS L4 Variables
        2. Select **ugos** and **vgos** 
        3. Reload the data
        """)
        return
    
    st.success(f"✅ Velocity data loaded: {ugos_matrix.shape[0]} gate points × {ugos_matrix.shape[1]} time steps")
    
    # =========================================================================
    # CONTROLS
    # =========================================================================
    st.markdown("### ⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        depth_cap = st.number_input(
            "Depth Cap (m)",
            min_value=50,
            max_value=1000,
            value=250,
            step=50,
            key="salt_flux_depth_cap",
            help="Maximum depth for flux calculation (salinity assumed constant below this)"
        )
    
    with col2:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        selected_month = st.selectbox(
            "Month to display",
            options=list(range(1, 13)),
            format_func=lambda m: month_names[m-1],
            index=0,
            key="salt_flux_month_select"
        )
    
    # =========================================================================
    # COMPUTE BUTTON
    # =========================================================================
    if st.button("🧮 Compute Salt Flux", type="primary", use_container_width=True):
        with st.spinner("Loading SSS data and computing salt flux..."):
            try:
                # Get gate path from session state (saved when CMEMS L4 was loaded)
                from app.components.sidebar import _get_gate_shapefile, _get_parent_gate_id
                
                selected_gate = st.session_state.get("selected_gate")
                if not selected_gate:
                    st.error("❌ No gate selected. Please load CMEMS L4 data first.")
                    return
                
                gate_path = _get_gate_shapefile(selected_gate)
                if not gate_path:
                    st.error(f"❌ Could not find shapefile for gate: {selected_gate}")
                    return
                
                # 1. Load SSS data
                from src.services.sss_service import SSSService, SSSConfig
                
                time_pd = pd.to_datetime(time_array)
                start_date = str(time_pd.min().date())
                end_date = str(time_pd.max().date())
                
                sss_config = SSSConfig(
                    gate_path=gate_path,
                    time_start=start_date,
                    time_end=end_date
                )
                
                sss_service = SSSService()
                
                progress_bar = st.progress(0, text="Loading SSS data...")
                
                def sss_progress(pct, msg):
                    progress_bar.progress(int(pct * 50), text=msg)
                
                sss_data = sss_service.load_gate_data(
                    config=sss_config,
                    progress_callback=sss_progress
                )
                
                if sss_data is None:
                    st.error("❌ Failed to load SSS data")
                    return
                
                progress_bar.progress(50, text="SSS loaded, computing flux...")
                
                # 2. Load bathymetry
                from src.services.gebco_service import get_bathymetry_cache
                
                cache = get_bathymetry_cache()
                depth_profile = cache.get_or_compute(
                    gate_name=strait_name,
                    gate_lons=gate_lon,
                    gate_lats=gate_lat,
                    gebco_path=config.gebco_nc_path,
                    depth_cap=None  # Get full depth, cap inside salt flux
                )
                
                # 3. Compute salt flux
                from src.services.salt_flux_service import SaltFluxService
                
                flux_service = SaltFluxService()
                salt_flux_data = flux_service.compute_salt_flux(
                    cmems_data=cmems_l4_data,
                    sss_data=sss_data,
                    depth_array=depth_profile,
                    depth_cap=float(depth_cap)
                )
                
                progress_bar.progress(100, text="Done!")
                
                # Store in session state
                st.session_state['sf_sss_data'] = sss_data
                st.session_state['sf_flux_data'] = salt_flux_data
                st.session_state['sf_depth_profile'] = depth_profile
                st.session_state['sf_depth_cap'] = depth_cap
                st.session_state['sf_strait_name'] = strait_name
                
                st.success("✅ Salt flux computed!")
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                return
    
    # =========================================================================
    # CHECK IF DATA COMPUTED
    # =========================================================================
    if 'sf_flux_data' not in st.session_state:
        st.info("👆 Click 'Compute Salt Flux' to start")
        return
    
    # Retrieve from session state
    sss_data = st.session_state['sf_sss_data']
    salt_flux_data = st.session_state['sf_flux_data']
    depth_profile = st.session_state['sf_depth_profile']
    stored_depth_cap = st.session_state['sf_depth_cap']
    
    # =========================================================================
    # 1. SALINITY & DENSITY PROFILES
    # =========================================================================
    st.markdown("### 🌡️ Salinity & Density Along Gate")
    st.caption("Mean values across the time period")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salinity profile
        sos_mean = np.nanmean(sss_data.sos_matrix, axis=1)
        sos_std = np.nanstd(sss_data.sos_matrix, axis=1)
        
        fig_sal = go.Figure()
        fig_sal.add_trace(go.Scatter(
            x=x_km,
            y=sos_mean,
            mode='lines',
            name='Salinity',
            line=dict(color='#3498DB', width=2),
            hovertemplate='%{x:.1f} km<br>S: %{y:.2f} PSU<extra></extra>'
        ))
        fig_sal.add_trace(go.Scatter(
            x=np.concatenate([x_km, x_km[::-1]]),
            y=np.concatenate([sos_mean + sos_std, (sos_mean - sos_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.2)',
            line=dict(color='rgba(52, 152, 219, 0)'),
            name='±1σ',
            hoverinfo='skip'
        ))
        fig_sal.update_layout(
            title="Salinity Profile",
            xaxis_title="Distance (km)",
            yaxis_title="Salinity (PSU)",
            height=300,
            
            margin=dict(l=50, r=20, t=40, b=40)
        )
        st.plotly_chart(fig_sal, use_container_width=True, key="_render_salt_flux_tab_cmems_l4_1")
    
    with col2:
        # Density profile — constant ρ = 1025 kg/m³
        RHO_CONST = 1025.0
        dos_mean = np.full_like(x_km, RHO_CONST)
        dos_std = np.zeros_like(x_km)
        
        fig_den = go.Figure()
        fig_den.add_trace(go.Scatter(
            x=x_km,
            y=dos_mean,
            mode='lines',
            name='Density',
            line=dict(color='#E74C3C', width=2),
            hovertemplate='%{x:.1f} km<br>ρ: %{y:.2f} kg/m³<extra></extra>'
        ))
        fig_den.add_trace(go.Scatter(
            x=np.concatenate([x_km, x_km[::-1]]),
            y=np.concatenate([dos_mean + dos_std, (dos_mean - dos_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(231, 76, 60, 0)'),
            name='±1σ',
            hoverinfo='skip'
        ))
        fig_den.update_layout(
            title="Density Profile",
            xaxis_title="Distance (km)",
            yaxis_title="Density (kg/m³)",
            height=300,
            
            margin=dict(l=50, r=20, t=40, b=40)
        )
        st.plotly_chart(fig_den, use_container_width=True, key="_render_salt_flux_tab_cmems_l4_2")
    
    # =========================================================================
    # 2. SALT FLUX TIME SERIES
    # =========================================================================
    st.markdown("### 📈 Salt Flux Time Series")
    
    time_pd = pd.to_datetime(time_array)
    flux_series = salt_flux_data.flux_series
    
    # Convert to more readable units (10^7 kg/s)
    flux_scaled = flux_series / 1e7
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=time_pd,
        y=flux_scaled,
        mode='lines',
        name='Salt Flux',
        line=dict(color='#9B59B6', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>Flux: %{y:.2f} ×10⁷ kg/s<extra></extra>'
    ))
    
    mean_flux = np.nanmean(flux_scaled)
    fig_ts.add_hline(y=0, line_dash="dash", line_color="#7F8C8D", line_width=1)
    fig_ts.add_hline(y=mean_flux, line_dash="dot", line_color="#E74C3C", line_width=1.5,
                     annotation_text=f"Mean: {mean_flux:.2f}")
    
    fig_ts.update_layout(
        title=dict(text=f"{strait_name} — Salt Flux Time Series", font=dict(size=16)),
        xaxis_title="Time",
        yaxis_title="Salt Flux (×10⁷ kg/s)",
        height=420,
        
        
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor='#E8E8E8'),
        yaxis=dict(gridcolor='#E8E8E8'),
        margin=dict(l=60, r=40, t=60, b=50),
    )
    
    st.plotly_chart(fig_ts, use_container_width=True, key="_render_salt_flux_tab_cmems_l4_3")
    
    # =========================================================================
    # 3. STATISTICS
    # =========================================================================
    st.markdown("### 📊 Statistics")
    
    # Calculate equivalent Sverdrup (for comparison with volume transport)
    mean_salinity = np.nanmean(sss_data.sos_matrix)
    mean_density = 1025.0  # Constant density ρ = 1025 kg/m³
    sv_equivalent = np.nanmean(flux_series) / (mean_density * mean_salinity / 1000) / 1e6
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Flux", f"{np.nanmean(flux_series):.2e} kg/s")
    with col2:
        st.metric("Std Dev", f"{np.nanstd(flux_series):.2e} kg/s")
    with col3:
        st.metric("~Sv Equivalent", f"{sv_equivalent:.2f} Sv")
    with col4:
        ice_mean = np.nanmean(sss_data.ice_matrix) if sss_data.ice_matrix is not None else 0
        st.metric("Mean Ice Fraction", f"{ice_mean:.1%}")
    
    # Salinity/density stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Salinity", f"{mean_salinity:.2f} PSU")
    with col2:
        st.metric("Mean Density", f"{mean_density:.1f} kg/m³")
    with col3:
        st.metric("Depth Cap", f"{stored_depth_cap} m")
    with col4:
        mean_vel = np.nanmean(salt_flux_data.velocity_mean) * 100  # cm/s
        st.metric("Mean Velocity", f"{mean_vel:.1f} cm/s")
    
    # =========================================================================
    # 5. SALT FLUX ALONG GATE (Monthly Profile)
    # =========================================================================
    st.markdown("### 🗺️ Salt Flux Along Gate")
    st.caption(f"Monthly mean salt flux profile for **{month_names[selected_month-1]}** (averaged over all years)")
    
    try:
        from src.services.transport_service import (
            compute_perpendicular_velocity,
            compute_monthly_salt_flux_profile,
            bin_along_gate
        )
        
        # Compute v_perp if not already
        v_perp = compute_perpendicular_velocity(
            ugos_matrix, vgos_matrix, gate_lon, gate_lat, gate_name=strait_name
        )
        
        # Get depth profile capped
        depth_capped = np.minimum(depth_profile, stored_depth_cap)
        
        # Use mean salinity/density from SSS data
        mean_sal = np.nanmean(sss_data.sos_matrix)
        mean_den = 1025.0  # Constant density ρ = 1025 kg/m³
        
        # Compute monthly profiles with bin_size slider
        bin_size_km = st.slider(
            "Spatial averaging (km)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key="sf_along_gate_bin"
        )
        
        monthly_flux_profile = compute_monthly_salt_flux_profile(
            x_km=x_km,
            v_perp=v_perp,
            depth_profile=depth_capped,
            time_array=time_array,
            salinity=mean_sal,
            density=mean_den,
            bin_size_km=bin_size_km
        )
        
        # Get selected month data
        bin_centers, bin_means, bin_stds = monthly_flux_profile[selected_month]
        
        if len(bin_centers) > 0:
            # Scale for readability (10^4 kg/(m·s))
            bin_means_scaled = bin_means / 1e4
            bin_stds_scaled = bin_stds / 1e4
            
            # Create bar chart like Volume Transport
            colors = ['#9B59B6' if v >= 0 else '#E74C3C' for v in bin_means_scaled]
            
            fig_along = go.Figure()
            fig_along.add_trace(go.Bar(
                x=bin_centers,
                y=bin_means_scaled,
                marker_color=colors,
                name='Salt Flux',
                error_y=dict(
                    type='data',
                    array=bin_stds_scaled,
                    visible=True,
                    color='gray',
                    thickness=1,
                    width=2
                ),
                hovertemplate='%{x:.1f} km<br>Flux: %{y:.3f} ×10⁴ kg/(m·s)<extra></extra>'
            ))
            
            fig_along.add_hline(y=0, line_color='gray', line_width=1)
            
            fig_along.update_layout(
                title=dict(
                    text=f"Salt Flux Along Gate — {month_names[selected_month-1]}",
                    font=dict(size=16)
                ),
                xaxis_title="Distance along gate (km)",
                yaxis_title="Salt Flux (×10⁴ kg/(m·s))",
                height=420,
                
                
                font=dict(family="Inter, sans-serif", size=12),
                xaxis=dict(gridcolor='#E8E8E8'),
                yaxis=dict(gridcolor='#E8E8E8'),
                bargap=0.15,
            )
            
            st.plotly_chart(fig_along, use_container_width=True, key="_render_salt_flux_tab_cmems_l4_4")
            
            # Show total monthly flux
            total_flux = np.nansum(bin_means) * bin_size_km * 1000  # integrate over width
            st.info(f"**Total flux in {month_names[selected_month-1]}:** {total_flux:.2e} kg/s "
                    f"(≈ {total_flux/1e7:.2f} ×10⁷ kg/s)")
        else:
            st.warning(f"No data available for {month_names[selected_month-1]}")
            
    except Exception as e:
        st.warning(f"Could not compute along-gate profile: {e}")
    
    # =========================================================================
    # 6. EXPORT
    # =========================================================================
    with st.expander("📥 Export Salt Flux Data"):
        export_df = pd.DataFrame({
            'time': time_pd,
            'salt_flux_kg_s': flux_series,
            'salt_flux_1e7_kg_s': flux_scaled,
        })
        
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            "Download Time Series (CSV)",
            data=csv_data,
            file_name=f"salt_flux_{strait_name.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )


# ==============================================================================
# CMEMS-ONLY TABS
# ==============================================================================
def _render_cmems_tabs(cmems_data, config: AppConfig):
    """Render tabs for CMEMS data only - uses unified functions."""
    # Get dataset info
    ds_info = _get_unified_dataset_info(cmems_data, "cmems_l4")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        f"{ds_info['emoji']} Slope Timeline",
        f"{ds_info['emoji']} DOT Profile",
        f"{ds_info['emoji']} Spatial Map",
        f"{ds_info['emoji']} Monthly Analysis",
        f"{ds_info['emoji']} Geostrophic Velocity",
        "📥 Export"
    ])
    
    with tab1:
        _render_unified_slope_timeline(cmems_data, config, ds_info)
    with tab2:
        _render_unified_dot_profile(cmems_data, config, ds_info)
    with tab3:
        _render_unified_spatial_map(cmems_data, config, ds_info)
    with tab4:
        _render_unified_monthly_analysis(cmems_data, config, ds_info)
    with tab5:
        _render_unified_geostrophic_velocity(cmems_data, config, ds_info)
    with tab6:
        _render_unified_export_tab(cmems_data, config, ds_info)


# ==============================================================================
# DTUSpace TABS (ISOLATED - does not share code with SLCCI/CMEMS)
# ==============================================================================

def _render_dtu_tabs(dtu_data, config: AppConfig):
    """
    Render tabs for DTUSpace gridded data - uses unified functions.
    """
    # Get dataset info
    ds_info = _get_unified_dataset_info(dtu_data, "dtu")
    
    tab1, tab2, tab3, tab4, tab5, tab_vt, tab6, tab_sal, tab7, tab8 = st.tabs([
        f"{ds_info['emoji']} Slope Timeline",
        f"{ds_info['emoji']} DOT Profile",
        f"{ds_info['emoji']} Spatial Map",
        f"{ds_info['emoji']} Monthly Analysis",
        f"{ds_info['emoji']} Geostrophic Velocity",
        "🚢 Volume Transport",
        "💧 Freshwater Transport",
        "🧪 Salinity Profile",
        "🧂 Salt Flux",
        "📥 Export"
    ])
    
    with tab1:
        _render_unified_slope_timeline(dtu_data, config, ds_info)
    with tab2:
        _render_unified_dot_profile(dtu_data, config, ds_info)
    with tab3:
        _render_unified_spatial_map(dtu_data, config, ds_info)
    with tab4:
        _render_unified_monthly_analysis(dtu_data, config, ds_info)
    with tab5:
        _render_unified_geostrophic_velocity(dtu_data, config, ds_info)
    with tab_vt:
        _render_volume_transport_tab_cmems_l4(dtu_data, config)
    with tab6:
        render_freshwater_transport_clean(dtu_data, config, ds_info)
    with tab_sal:
        render_salinity_profile_tab(dtu_data, config, ds_info)
    with tab7:
        render_salt_flux_clean(dtu_data, config, ds_info)
    with tab8:
        _render_unified_export_tab(dtu_data, config, ds_info)


# ==============================================================================
# DTU TAB 1: SLOPE TIMELINE [LEGACY]
# ==============================================================================

def _get_gridded_dataset_info(data):
    """Get dataset type info (DTU or CMEMS L4) from PassData object."""
    data_source = getattr(data, 'data_source', '')
    dataset_name = getattr(data, 'dataset_name', '')
    
    # Detect CMEMS L4 by data_source
    if 'cmems' in data_source.lower() or 'cmems_l4' in dataset_name.lower():
        return {
            'emoji': '🟣',
            'name': dataset_name or 'CMEMS L4',
            'color': '#9B59B6',  # Purple for CMEMS L4
            'type': 'cmems_l4'
        }
    else:
        return {
            'emoji': '🟢',
            'name': dataset_name or 'DTUSpace v4',
            'color': COLOR_DTU,  # Green for DTU
            'type': 'dtu'
        }


def _render_dtu_slope_timeline(dtu_data, config: AppConfig):
    """
    Render DTUSpace/CMEMS L4 slope timeline.
    
    From DTUSpace_plotter notebook Panel 1:
    - X-axis: time_array (monthly dates)
    - Y-axis: slope_series (m/100km)
    """
    # Get dataset info dynamically
    ds_info = _get_gridded_dataset_info(dtu_data)
    
    st.subheader(f"{ds_info['emoji']} {ds_info['name']} - Slope Timeline")
    
    slope_series = getattr(dtu_data, 'slope_series', None)
    time_array = getattr(dtu_data, 'time_array', None)
    strait_name = getattr(dtu_data, 'strait_name', 'Unknown')
    dataset_name = ds_info['name']
    start_year = getattr(dtu_data, 'start_year', None)
    end_year = getattr(dtu_data, 'end_year', None)
    time_range = getattr(dtu_data, 'time_range', None)
    
    # Get year range from time_range if not available
    if start_year is None and time_range:
        start_year = time_range[0][:4] if time_range[0] else '?'
    if end_year is None and time_range:
        end_year = time_range[1][:4] if time_range[1] else '?'
    
    if slope_series is None or time_array is None:
        st.error("❌ No slope data available")
        return
    
    # Check for valid data
    valid_mask = ~np.isnan(slope_series)
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        st.warning("⚠️ All slope values are NaN")
        return
    
    # Display info - add (West) or (East) if it's a divided gate
    gate_suffix = ""
    if "west" in strait_name.lower():
        gate_suffix = " (West)"
    elif "east" in strait_name.lower():
        gate_suffix = " (East)"
    st.info(f"📊 **{dataset_name}** | {strait_name}{gate_suffix} | {start_year}–{end_year}")
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        show_trend = st.checkbox("Show trend line", value=True, key=f"{ds_info['type']}_slope_trend")
    with col2:
        unit = st.selectbox("Units", ["m/100km", "cm/km"], key=f"{ds_info['type']}_slope_unit")
    
    # Convert units
    if unit == "cm/km":
        y_vals = slope_series * 100
        y_label = "Slope (cm/km)"
    else:
        y_vals = slope_series
        y_label = "Slope (m/100km)"
    
    # Create figure
    fig = go.Figure()
    
    # Convert time to pandas datetime for plotting
    time_pd = pd.to_datetime(time_array)
    
    # Plot valid values only
    valid_x = time_pd[valid_mask]
    valid_y = y_vals[valid_mask]
    
    fig.add_trace(go.Scatter(
        x=valid_x,
        y=valid_y,
        mode="markers+lines",
        name="DOT Slope",
        marker=dict(size=6, color=ds_info['color']),
        line=dict(width=2, color=ds_info['color'])
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.8)
    
    # Trend line
    if show_trend and len(valid_y) > 2:
        x_numeric = np.arange(len(valid_y))
        z = np.polyfit(x_numeric, valid_y, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=valid_x,
            y=p(x_numeric),
            mode="lines",
            name=f"Trend ({z[0]:.4f}/month)",
            line=dict(dash="dash", color="darkgreen", width=1.5)
        ))
    
    fig.update_layout(
        title=f"{dataset_name} - {strait_name}<br><sup>Monthly DOT Slope ({start_year}–{end_year})</sup>",
        xaxis_title="Date",
        yaxis_title=y_label,
        height=500,
        
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_dtu_slope_timeline_1")
    
    # Statistics
    with st.expander("📊 Statistics"):
        valid_slopes = slope_series[valid_mask]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{np.mean(valid_slopes):.4f} m/100km")
        with col2:
            st.metric("Std Dev", f"{np.std(valid_slopes):.4f} m/100km")
        with col3:
            st.metric("Min", f"{np.min(valid_slopes):.4f} m/100km")
        with col4:
            st.metric("Max", f"{np.max(valid_slopes):.4f} m/100km")
        
        st.caption(f"Valid time steps: {n_valid}/{len(slope_series)}")


# ==============================================================================
# DTU TAB 2: DOT PROFILE [LEGACY]
# ==============================================================================

def _render_dtu_dot_profile(dtu_data, config: AppConfig):
    """
    Render DTUSpace/CMEMS L4 mean DOT profile across gate.
    
    From DTUSpace_plotter notebook Panel 2:
    - X-axis: x_km (Distance along gate in km) or longitude
    - Y-axis: profile_mean (Mean DOT in m/cm/mm)
    - With WEST/EAST labels
    """
    # Get dataset info dynamically
    ds_info = _get_gridded_dataset_info(dtu_data)
    
    st.subheader(f"{ds_info['emoji']} {ds_info['name']} - Mean DOT Profile")
    
    profile_mean = getattr(dtu_data, 'profile_mean', None)
    x_km = getattr(dtu_data, 'x_km', None)
    gate_lon_pts = getattr(dtu_data, 'gate_lon_pts', None)
    dot_matrix = getattr(dtu_data, 'dot_matrix', None)
    strait_name = getattr(dtu_data, 'strait_name', 'Unknown')
    dataset_name = ds_info['name']
    
    if profile_mean is None or x_km is None:
        st.error("❌ No profile data available")
        return
    
    # Check for valid data
    valid_mask = ~np.isnan(profile_mean)
    if not np.any(valid_mask):
        st.warning("⚠️ All DOT values are NaN")
        return
    
    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        view_mode = st.radio(
            "View mode",
            ["Mean Profile", "Individual Time Steps"],
            horizontal=True,
            key=f"{ds_info['type']}_dot_view_mode"
        )
    with col2:
        x_axis_mode = st.selectbox("X-axis", ["Distance (km)", "Longitude (°)"], key=f"{ds_info['type']}_dot_xaxis")
    with col3:
        y_units = st.selectbox("Y units", ["m", "cm", "mm"], key=f"{ds_info['type']}_dot_yunits")
    
    show_std = st.checkbox("Show ±1 Std Dev", value=True, key=f"{ds_info['type']}_dot_std")
    
    # Y scaling
    y_scale = {"m": 1.0, "cm": 100.0, "mm": 1000.0}[y_units]
    
    # X values
    if x_axis_mode == "Distance (km)":
        x_vals = x_km
        x_label = "Distance along gate (km)"
    else:
        x_vals = gate_lon_pts if gate_lon_pts is not None else x_km
        x_label = "Longitude (°)"
    
    fig = go.Figure()
    
    if view_mode == "Mean Profile":
        # Plot mean profile
        fig.add_trace(go.Scatter(
            x=x_vals[valid_mask],
            y=profile_mean[valid_mask] * y_scale,
            mode="lines",
            name="Mean DOT",
            line=dict(color=ds_info['color'], width=2)
        ))
        
        # Add std band if requested
        if show_std and dot_matrix is not None:
            profile_std = np.nanstd(dot_matrix, axis=1) * y_scale
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_vals[valid_mask], x_vals[valid_mask][::-1]]),
                y=np.concatenate([
                    (profile_mean[valid_mask] * y_scale + profile_std[valid_mask]),
                    (profile_mean[valid_mask] * y_scale - profile_std[valid_mask])[::-1]
                ]),
                fill='toself',
                fillcolor='rgba(46, 139, 87, 0.2)',  # seagreen with alpha
                line=dict(color='rgba(0,0,0,0)'),
                name='±1 Std Dev'
            ))
    
    else:  # Individual Time Steps
        if dot_matrix is None:
            st.warning("No time step data available")
            return
        
        n_time = dot_matrix.shape[1]
        time_array = getattr(dtu_data, 'time_array', None)
        
        # Let user select time steps
        max_select = min(10, n_time)
        selected = st.multiselect(
            "Select time steps",
            options=list(range(n_time)),
            default=list(range(min(5, n_time))),
            format_func=lambda i: str(pd.Timestamp(time_array[i]).strftime('%Y-%m')) if time_array is not None else f"Step {i}",
            key=f"{ds_info['type']}_time_steps",
            max_selections=max_select
        )
        
        if not selected:
            st.info("Select at least one time step")
            return
        
        # Use a green color scale
        colors = px.colors.sequential.Greens[2:]  # Skip lightest greens
        
        for i, idx in enumerate(selected):
            profile = dot_matrix[:, idx]
            mask = ~np.isnan(profile)
            if np.any(mask):
                color = colors[i % len(colors)]
                label = str(pd.Timestamp(time_array[idx]).strftime('%Y-%m')) if time_array is not None else f"Step {idx}"
                fig.add_trace(go.Scatter(
                    x=x_vals[mask],
                    y=profile[mask] * y_scale,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1.5)
                ))
    
    # Add WEST/EAST labels (like DTUSpace_plotter notebook)
    y_max = np.nanmax(profile_mean[valid_mask]) * y_scale
    y_min = np.nanmin(profile_mean[valid_mask]) * y_scale
    y_text = y_max - 0.05 * (y_max - y_min)
    
    fig.add_annotation(
        x=x_vals[valid_mask].min(),
        y=y_text,
        text="WEST",
        showarrow=False,
        font=dict(size=12, color="black", weight="bold"),
        xanchor="left"
    )
    fig.add_annotation(
        x=x_vals[valid_mask].max(),
        y=y_text,
        text="EAST",
        showarrow=False,
        font=dict(size=12, color="black", weight="bold"),
        xanchor="right"
    )
    
    fig.update_layout(
        title=f"{dataset_name} - {strait_name}<br><sup>Mean DOT Profile Across Gate</sup>",
        xaxis_title=x_label,
        yaxis_title=f"DOT ({y_units})",
        yaxis_tickformat=".3f",  # 3 decimal places like notebook
        height=500,
        
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_dtu_dot_profile_1")
    
    # Stats
    with st.expander("📊 Profile Statistics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean DOT", f"{np.nanmean(profile_mean):.4f} m")
        with col2:
            st.metric("DOT Range", f"{np.nanmax(profile_mean) - np.nanmin(profile_mean):.4f} m")
        with col3:
            st.metric("Gate Length", f"{x_km.max():.1f} km")


# ==============================================================================
# DTU TAB 3: SPATIAL MAP (GRIDDED - uses pcolormesh style) [LEGACY]
# ==============================================================================

def _render_dtu_spatial_map(dtu_data, config: AppConfig):
    """
    Render DTUSpace spatial map with mean DOT.
    
    Unlike SLCCI/CMEMS (scatter), DTU uses a gridded pcolormesh-style display.
    """
    st.subheader("🟢 DTUSpace - Spatial Map")
    
    dot_mean_grid = getattr(dtu_data, 'dot_mean_grid', None)
    gate_lon_pts = getattr(dtu_data, 'gate_lon_pts', None)
    gate_lat_pts = getattr(dtu_data, 'gate_lat_pts', None)
    lat_grid = getattr(dtu_data, 'lat_grid', None)
    lon_grid = getattr(dtu_data, 'lon_grid', None)
    map_extent = getattr(dtu_data, 'map_extent', {})
    strait_name = getattr(dtu_data, 'strait_name', 'Unknown')
    dataset_name = getattr(dtu_data, 'dataset_name', 'DTUSpace v4')
    
    if dot_mean_grid is None:
        st.error("❌ No gridded DOT data available")
        return
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        show_gate = st.checkbox("Show gate line", value=True, key="dtu_map_gate")
    with col2:
        colorscale = st.selectbox(
            "Colorscale",
            ["viridis", "RdBu_r", "Plasma", "Cividis"],
            key="dtu_map_colorscale"
        )
    
    # Get grid data
    if hasattr(dot_mean_grid, 'values'):
        z_data = dot_mean_grid.values
    else:
        z_data = dot_mean_grid
    
    # Compute colorbar limits (5th-95th percentile like notebook)
    vmin = np.nanpercentile(z_data, 5)
    vmax = np.nanpercentile(z_data, 95)
    
    # Create heatmap figure
    fig = go.Figure()
    
    # Add gridded DOT as heatmap
    fig.add_trace(go.Heatmap(
        x=lon_grid,
        y=lat_grid,
        z=z_data,
        colorscale=colorscale,
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(title="DOT (m)"),
        name="Mean DOT"
    ))
    
    # Add gate line
    if show_gate and gate_lon_pts is not None and gate_lat_pts is not None:
        fig.add_trace(go.Scatter(
            x=gate_lon_pts,
            y=gate_lat_pts,
            mode="lines",
            name="Gate",
            line=dict(color="red", width=3)
        ))
    
    fig.update_layout(
        title=f"{dataset_name} - {strait_name}<br><sup>Mean DOT (Gridded)</sup>",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=600,
        
        yaxis=dict(scaleanchor="x", scaleratio=1)  # Equal aspect ratio
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_dtu_spatial_map_1")
    
    # Stats
    with st.expander("📊 Grid Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Grid Size", f"{lat_grid.shape[0]} × {lon_grid.shape[0]}")
        with col2:
            st.metric("DOT Range", f"{np.nanmax(z_data) - np.nanmin(z_data):.3f} m")
        with col3:
            st.metric("Mean DOT", f"{np.nanmean(z_data):.4f} m")
        with col4:
            st.metric("Coverage", f"{(~np.isnan(z_data)).sum() / z_data.size * 100:.1f}%")


# ==============================================================================
# GRIDDED MONTHLY ANALYSIS (for DTU and CMEMS L4)
# ==============================================================================

def _render_gridded_monthly_analysis(data, config: AppConfig):
    """
    Render 12-month DOT analysis for gridded datasets (DTU, CMEMS L4).
    Shows DOT profile vs distance/longitude for each month with linear regression.
    Includes R² and slope statistics.
    """
    ds_info = _get_gridded_dataset_info(data)
    
    st.subheader(f"{ds_info['emoji']} {ds_info['name']} - Monthly Analysis")
    
    # Get required data
    dot_matrix = getattr(data, 'dot_matrix', None)  # (n_gate_pts, n_time)
    time_array = getattr(data, 'time_array', None)
    x_km = getattr(data, 'x_km', None)
    gate_lon_pts = getattr(data, 'gate_lon_pts', None)
    strait_name = getattr(data, 'strait_name', 'Unknown')
    
    if dot_matrix is None or time_array is None or x_km is None:
        st.error("❌ Missing data for monthly analysis (dot_matrix, time_array, x_km)")
        return
    
    # Convert time to pandas for month extraction
    time_pd = pd.to_datetime(time_array)
    months = time_pd.month
    
    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_regression = st.checkbox("Show linear regression", value=True, key=f"{ds_info['type']}_monthly_reg")
    with col2:
        x_axis_mode = st.selectbox("X-axis", ["Distance (km)", "Longitude (°)"], key=f"{ds_info['type']}_monthly_xaxis")
    with col3:
        y_units = st.selectbox("Y units", ["m", "cm", "mm"], key=f"{ds_info['type']}_monthly_yunits")
    
    # Y-axis scaling
    y_scale = {"m": 1.0, "cm": 100.0, "mm": 1000.0}[y_units]
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=[f"{month_names[i]} ({i+1})" for i in range(12)],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    # X-axis values
    if x_axis_mode == "Distance (km)":
        x_vals = x_km
        x_label = "Distance (km)"
    else:
        x_vals = gate_lon_pts if gate_lon_pts is not None else x_km
        x_label = "Longitude (°)"
    
    slopes_info = []
    
    for month in range(1, 13):
        row = (month - 1) // 4 + 1
        col = (month - 1) % 4 + 1
        
        # Get time indices for this month
        month_mask = months == month
        if not np.any(month_mask):
            continue
        
        # Average DOT profile for this month
        dot_month = dot_matrix[:, month_mask]
        dot_mean = np.nanmean(dot_month, axis=1)
        
        # Valid data mask
        mask = np.isfinite(x_vals) & np.isfinite(dot_mean)
        if np.sum(mask) < 2:
            continue
        
        x_valid = x_vals[mask]
        y_valid = dot_mean[mask] * y_scale
        
        # Scatter
        fig.add_trace(
            go.Scatter(
                x=x_valid, y=y_valid, mode='markers',
                marker=dict(size=4, color=ds_info['color'], opacity=0.6),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Regression
        if show_regression and len(x_valid) > 2:
            try:
                from scipy import stats as scipy_stats
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_valid, y_valid)
                r_squared = r_value ** 2
                
                # For display, convert slope to meaningful units
                if x_axis_mode == "Distance (km)":
                    # slope is in y_units/km, convert to m/100km for standard comparison
                    slope_m_100km = (slope / y_scale) * 100  # m/100km
                    slope_display = f"{slope:.4f} {y_units}/km"
                else:
                    # slope is in y_units/degree
                    slope_display = f"{slope:.4f} {y_units}/°"
                    slope_m_100km = slope / y_scale  # approximate
                
                slopes_info.append({
                    'month': month,
                    'name': month_names[month-1],
                    'slope': slope,
                    'slope_m_100km': slope_m_100km,
                    'r_squared': r_squared,
                    'n_time': np.sum(month_mask),
                    'n_points': len(x_valid)
                })
                
                # Regression line
                x_line = np.linspace(x_valid.min(), x_valid.max(), 50)
                y_line = slope * x_line + intercept
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line, y=y_line, mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False,
                        hovertemplate=f"R²={r_squared:.3f}<br>slope={slope:.4f}"
                    ),
                    row=row, col=col
                )
            except Exception as e:
                pass
    
    fig.update_layout(
        title=f"{ds_info['name']} - {strait_name} - Monthly Mean DOT Profile",
        height=700,
        
        showlegend=False
    )
    
    # Axis labels
    for i in range(1, 13):
        row = (i - 1) // 4 + 1
        col = (i - 1) % 4 + 1
        if row == 3:
            fig.update_xaxes(title_text=x_label, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=f"DOT ({y_units})", row=row, col=col)
    
    st.plotly_chart(fig, width='stretch', key="_render_gridded_monthly_analysis_1")
    
    # Summary table with R² and slope
    if slopes_info:
        with st.expander("📊 Monthly Slopes & R² Summary"):
            slopes_df = pd.DataFrame(slopes_info)
            
            # Format columns for display
            display_df = pd.DataFrame({
                'Month': slopes_df['name'],
                f'Slope ({y_units}/{"km" if x_axis_mode == "Distance (km)" else "°"})': slopes_df['slope'].apply(lambda x: f"{x:.4f}"),
                'Slope (m/100km)': slopes_df['slope_m_100km'].apply(lambda x: f"{x:.4f}"),
                'R²': slopes_df['r_squared'].apply(lambda x: f"{x:.3f}"),
                'N time steps': slopes_df['n_time'],
                'N points': slopes_df['n_points']
            })
            
            st.dataframe(display_df, width='stretch', hide_index=True)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Slope", f"{slopes_df['slope_m_100km'].mean():.4f} m/100km")
            with col2:
                st.metric("Std Dev", f"{slopes_df['slope_m_100km'].std():.4f} m/100km")
            with col3:
                st.metric("Mean R²", f"{slopes_df['r_squared'].mean():.3f}")
            with col4:
                st.metric("Months with Data", len(slopes_df))


# ==============================================================================
# DTU TAB 4: GEOSTROPHIC VELOCITY [LEGACY]
# ==============================================================================

def _render_dtu_geostrophic_velocity(dtu_data, config: AppConfig):
    """
    Render DTUSpace/CMEMS L4 geostrophic velocity.
    
    Uses pre-computed v_geostrophic_series from DTUService/CMEMSL4Service.
    """
    # Get dataset info dynamically
    ds_info = _get_gridded_dataset_info(dtu_data)
    
    st.subheader(f"{ds_info['emoji']} {ds_info['name']} - Geostrophic Velocity")
    
    v_geo = getattr(dtu_data, 'v_geostrophic_series', None)
    time_array = getattr(dtu_data, 'time_array', None)
    mean_lat = getattr(dtu_data, 'mean_latitude', 70.0)
    coriolis_f = getattr(dtu_data, 'coriolis_f', 1e-4)
    strait_name = getattr(dtu_data, 'strait_name', 'Unknown')
    dataset_name = ds_info['name']
    
    if v_geo is None or len(v_geo) == 0:
        st.warning("⚠️ No geostrophic velocity data available. Make sure the service computes v_geostrophic_series.")
        st.info(f"""
        **Debug Info:**
        - data_source: `{getattr(dtu_data, 'data_source', 'N/A')}`
        - slope_series available: `{getattr(dtu_data, 'slope_series', None) is not None}`
        - mean_latitude: `{mean_lat}`
        - coriolis_f: `{coriolis_f}`
        """)
        return
    
    st.info(f"📍 Computing at lat={mean_lat:.2f}° (f={coriolis_f:.2e} s⁻¹)")
    
    # Convert to pandas for easier handling
    time_pd = pd.to_datetime(time_array)
    v_series = pd.Series(v_geo, index=time_pd)
    
    # Time series plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_pd,
        y=v_geo * 100,  # Convert m/s to cm/s
        mode="lines+markers",
        name="v_geostrophic",
        line=dict(color=ds_info['color'], width=2),
        marker=dict(size=6, color=ds_info['color'])
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{dataset_name} - {strait_name}<br><sup>Geostrophic Velocity Time Series</sup>",
        xaxis_title="Time",
        yaxis_title="Geostrophic Velocity (cm/s)",
        height=450,
        
    )
    
    st.plotly_chart(fig, width='stretch', key="_render_dtu_geostrophic_velocity_1")
    
    # Statistics
    with st.expander("📊 Geostrophic Velocity Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{v_geo.mean() * 100:.2f} cm/s")
        with col2:
            st.metric("Std Dev", f"{v_geo.std() * 100:.2f} cm/s")
        with col3:
            st.metric("Max", f"{v_geo.max() * 100:.2f} cm/s")
        with col4:
            st.metric("Min", f"{v_geo.min() * 100:.2f} cm/s")
    
    # Physical interpretation
    with st.expander("📖 Physical Interpretation"):
        mean_v = v_geo.mean() * 100
        st.markdown(f"""
        **Geostrophic Balance Formula:**
        
        v = -g/f × (dη/dx)
        
        Where:
        - g = 9.81 m/s² (gravity)
        - f = 2Ω sin(lat) = {coriolis_f:.2e} s⁻¹ (Coriolis at {mean_lat:.1f}°)
        - dη/dx = DOT slope along gate
        
        **Results:**
        - Mean geostrophic velocity: **{mean_v:.2f} cm/s**
        - Positive values → flow in one direction
        - Negative values → flow in opposite direction
        """)


# ==============================================================================
# CMEMS L4 ADVANCED EXPORT TAB
# ==============================================================================

def _render_cmems_l4_export_tab(cmems_l4_data, config: AppConfig):
    """
    Export tab for CMEMS L4 — simplified, clear checkboxes.
    
    Exports:
    - CSV: raw time-series (velocity, FW transport, salt flux)
    - PNG: key plots at chosen DPI
    """
    st.subheader("📥 Export Data")

    strait_name = getattr(cmems_l4_data, 'strait_name', 'Unknown')
    time_array = getattr(cmems_l4_data, 'time_array', None)

    if time_array is None:
        st.error("❌ No time data available for export")
        return

    time_array_np = np.asarray(time_array).ravel()
    if len(time_array_np) == 0:
        st.error("❌ No time data available for export")
        return

    time_pd = pd.to_datetime(time_array_np)
    start_year = time_pd.min().year
    end_year = time_pd.max().year

    st.info(f"🟣 **{strait_name}** · {start_year}–{end_year} · {len(time_pd):,} obs")

    # ----- data availability -----
    ugos = getattr(cmems_l4_data, 'ugos_matrix', None)
    vgos = getattr(cmems_l4_data, 'vgos_matrix', None)
    gate_lon = getattr(cmems_l4_data, 'gate_lon_pts', None)
    gate_lat = getattr(cmems_l4_data, 'gate_lat_pts', None)
    x_km = getattr(cmems_l4_data, 'x_km', None)
    has_vel = ugos is not None and vgos is not None

    # FW / Salt flux from session
    key_fw = f"fw_{strait_name.replace(' ', '_').lower()}"
    key_sf = f"sf_{strait_name.replace(' ', '_').lower()}"
    key_gv = f"geovel_{strait_name.replace(' ', '_')}"
    fw_done = st.session_state.get(f'{key_fw}_done', False)
    sf_done = st.session_state.get(f'{key_sf}_done', False)
    vel_done = f'{key_gv}_v_perp' in st.session_state

    # ----- status badges -----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.success("✅ Velocity") if has_vel else st.warning("⚠️ No velocity")
    with c2:
        st.success("✅ Vel profiles computed") if vel_done else st.info("ℹ️ Vel profiles not computed")
    with c3:
        st.success("✅ FW Transport computed") if fw_done else st.info("ℹ️ FW Transport not computed")
    with c4:
        st.success("✅ Salt Flux computed") if sf_done else st.info("ℹ️ Salt Flux not computed")

    st.divider()

    # ====================== CHECKBOXES ======================
    st.markdown("### Select items to export")

    col_csv, col_img = st.columns(2)

    with col_csv:
        st.markdown("**📊 CSV Data**")
        exp_vel_csv = st.checkbox("Perpendicular Velocity time-series", value=has_vel, key="exp2_vel")
        exp_fw_csv = st.checkbox("Freshwater Transport time-series", value=fw_done, key="exp2_fw")
        exp_sf_csv = st.checkbox("Salt Flux time-series", value=sf_done, key="exp2_sf")

    with col_img:
        st.markdown("**🖼️ Images**")
        exp_map = st.checkbox("Geographic Map (gate location)", value=True, key="exp2_map")
        exp_bathy = st.checkbox("Bathymetry Profile", value=True, key="exp2_bathy")
        exp_slope_ts = st.checkbox("Slope Timeline", value=has_vel, key="exp2_slope_ts")
        exp_vel_ts = st.checkbox("Velocity Timeline", value=has_vel, key="exp2_vel_ts")
        exp_vel_4x3 = st.checkbox("Velocity Profiles 4×3 (monthly)", value=vel_done, key="exp2_vel4x3", disabled=not vel_done)
        exp_dot_4x3 = st.checkbox("DOT Monthly Profiles 3×4", value=has_vel, key="exp2_dot4x3", disabled=not has_vel)
        exp_fw_ts = st.checkbox("FW Transport Timeline", value=fw_done, key="exp2_fw_ts", disabled=not fw_done)
        exp_sf_ts = st.checkbox("Salt Flux Timeline", value=sf_done, key="exp2_sf_ts", disabled=not sf_done)
        exp_fw_4x3 = st.checkbox("FW Transport Along Gate (4×3)", value=fw_done, key="exp2_fw4x3", disabled=not fw_done)
        exp_sf_4x3 = st.checkbox("Salt Flux Along Gate (4×3)", value=sf_done, key="exp2_sf4x3", disabled=not sf_done)
        st.markdown("**📊 Publication Plots (matplotlib + cmocean)**")
        _has_combined = has_vel  # VT always available if vel loaded
        exp_combined = st.checkbox(
            "Combined Transport (VT + FWT + SF)",
            value=_has_combined and (fw_done or sf_done),
            key="exp2_combined",
            help="Publication-quality 3-panel timeline with uncertainty bands, no rolling mean"
        )
        exp_vt_pub = st.checkbox("Volume Transport (publication)", value=has_vel, key="exp2_vt_pub")
        exp_fw_pub = st.checkbox("FW Transport (publication)", value=fw_done, key="exp2_fw_pub", disabled=not fw_done)
        exp_sf_pub = st.checkbox("Salt Flux (publication)", value=sf_done, key="exp2_sf_pub", disabled=not sf_done)
        export_dpi = st.selectbox("Image DPI", [150, 300, 600], index=1, key="exp2_dpi")

    st.divider()

    # ====================== PREVIEW: Combined Transport Plot ======================
    if has_vel and st.button("👁️ Preview Combined Transport Plot", key="exp2_preview"):
        with st.spinner("Generating preview…"):
            try:
                from src.services.publication_plots import plot_combined_transport_timeline
                from src.services.transport_service import (
                    compute_perpendicular_velocity as _prev_cpv,
                    compute_segment_widths as _prev_csw,
                    compute_perpendicular_velocity_uncertainty as _prev_cpu,
                )
                from src.services.gebco_service import get_bathymetry_cache as _prev_gbc

                _prev_v_perp = _prev_cpv(ugos, vgos, gate_lon, gate_lat, gate_name=strait_name)
                _prev_dx = _prev_csw(gate_lon, gate_lat, x_km)
                _prev_bc = _prev_gbc()
                _prev_Hf = _prev_bc.get_or_compute(
                    gate_name=strait_name, gate_lons=gate_lon, gate_lats=gate_lat,
                    gebco_path=config.gebco_nc_path, depth_cap=None
                )
                _prev_H = np.minimum(_prev_Hf, 250.0)
                _prev_nt = _prev_v_perp.shape[1]
                _prev_vt = np.zeros(_prev_nt)
                for _t in range(_prev_nt):
                    _v = _prev_v_perp[:, _t]
                    _ok = np.isfinite(_v) & np.isfinite(_prev_H)
                    _prev_vt[_t] = np.nansum(_v[_ok] * _prev_H[_ok] * _prev_dx[_ok]) / 1e6 if _ok.any() else np.nan

                _prev_fw = st.session_state.get(f'{key_fw}_fw') if fw_done else None
                _prev_sf = st.session_state.get(f'{key_sf}_salt_flux') if sf_done else None

                # Uncertainty
                _prev_eu = getattr(cmems_l4_data, 'err_ugosa_matrix', None)
                _prev_ev = getattr(cmems_l4_data, 'err_vgosa_matrix', None)
                _prev_sigma = None
                if _prev_eu is not None and _prev_ev is not None:
                    _prev_svp = _prev_cpu(_prev_eu, _prev_ev, gate_lon, gate_lat, gate_name=strait_name)
                    _prev_sigma = np.zeros(_prev_nt)
                    for _t in range(_prev_nt):
                        _sv = _prev_svp[:, _t]
                        _ok = np.isfinite(_sv) & np.isfinite(_prev_H)
                        if _ok.any():
                            _prev_sigma[_t] = np.sqrt(np.nansum((_sv[_ok] * _prev_H[_ok] * _prev_dx[_ok])**2)) / 1e6

                _prev_gk = strait_name.lower().replace(' ', '_').replace('-', '_')
                _prev_meta = {
                    'lon_range': f"{gate_lon.min():.2f}°–{gate_lon.max():.2f}°E" if gate_lon is not None else "?",
                    'lat_range': f"{gate_lat.min():.2f}°–{gate_lat.max():.2f}°N" if gate_lat is not None else "?",
                    'gate_length_km': f"{float(x_km.max()):.0f}" if x_km is not None else "?",
                    'date_start': time_pd.min().strftime('%Y-%m-%d'),
                    'date_end': time_pd.max().strftime('%Y-%m-%d'),
                }
                _prev_png = plot_combined_transport_timeline(
                    time_array=time_array_np,
                    vt_sv=_prev_vt,
                    fw_m3s=_prev_fw,
                    sm_kgs=_prev_sf,
                    sigma_vt=_prev_sigma,
                    gate_name=_prev_gk,
                    gate_display=strait_name,
                    dpi=150,
                    metadata=_prev_meta,
                )
                st.image(_prev_png, caption=f"Combined Transport — {strait_name} (matplotlib + cmocean)",
                         use_container_width=True)
            except Exception as e:
                st.error(f"❌ Preview failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.divider()

    # ====================== GENERATE ======================
    if st.button("📦 Generate Export ZIP", type="primary", use_container_width=True, key="exp2_go"):
        if not has_vel:
            st.error("❌ Load velocity data first (ugos + vgos)")
            return

        with st.spinner("Generating export…"):
            try:
                from src.services.transport_service import compute_perpendicular_velocity, compute_segment_widths
                files: dict[str, str | bytes] = {}
                gate_safe = strait_name.lower().replace(' ', '_').replace('-', '_')

                # --- v_perp ---
                v_perp = compute_perpendicular_velocity(ugos, vgos, gate_lon, gate_lat, gate_name=strait_name)

                # === CSV: velocity ===
                if exp_vel_csv and v_perp is not None:
                    mean_v = np.nanmean(v_perp, axis=0)
                    df_v = pd.DataFrame({
                        'date': time_pd.strftime('%Y-%m-%d'),
                        'v_perp_mean_m_s': mean_v,
                    })
                    files[f"csv/{gate_safe}_velocity.csv"] = df_v.to_csv(index=False)

                # === CSV: freshwater transport ===
                if exp_fw_csv and fw_done:
                    fw_arr = st.session_state.get(f'{key_fw}_fw', np.array([]))
                    fw_time = st.session_state.get(f'{key_fw}_time')
                    fw_params = st.session_state.get(f'{key_fw}_params', {})
                    t_col = pd.to_datetime(np.asarray(fw_time).ravel()).strftime('%Y-%m-%d') if fw_time is not None else range(len(fw_arr))
                    df_fw = pd.DataFrame({
                        'date': t_col,
                        'fw_transport_m3_s': fw_arr,
                        'fw_transport_mSv': fw_arr / 1e3,
                    })
                    # Add header comment via first row
                    header = f"# Freshwater Transport — {strait_name}, S_A={fw_params.get('s_ref', 34.8)} PSU, H={fw_params.get('depth', 250)} m\n"
                    files[f"csv/{gate_safe}_freshwater_transport.csv"] = header + df_fw.to_csv(index=False)

                # === CSV: salt flux ===
                if exp_sf_csv and sf_done:
                    sf_arr = st.session_state.get(f'{key_sf}_salt_flux', np.array([]))
                    sf_time = st.session_state.get(f'{key_sf}_time')
                    sf_params = st.session_state.get(f'{key_sf}_params', {})
                    t_col = pd.to_datetime(np.asarray(sf_time).ravel()).strftime('%Y-%m-%d') if sf_time is not None else range(len(sf_arr))
                    df_sf = pd.DataFrame({
                        'date': t_col,
                        'salt_flux_kg_s': sf_arr,
                        'salt_flux_Gg_s': sf_arr / 1e9,
                    })
                    header = f"# Salt Flux — {strait_name}, rho=1025 kg/m³, H={sf_params.get('depth', 250)} m\n"
                    files[f"csv/{gate_safe}_salt_flux.csv"] = header + df_sf.to_csv(index=False)

                # === Images ===
                if exp_map and gate_lon is not None:
                    try:
                        from src.services.export_service import export_spatial_map as _exp_map
                        gate_length = float(x_km.max()) if x_km is not None else None
                        files[f"images/{gate_safe}_map.png"] = _exp_map(gate_lon, gate_lat, strait_name, gate_length, export_dpi)
                    except Exception as e:
                        st.warning(f"⚠️ Map: {e}")

                if exp_bathy:
                    try:
                        from src.services.gebco_service import get_bathymetry_cache
                        from src.services.export_service import export_bathymetry_profile_clean as _exp_bathy
                        cache = get_bathymetry_cache()
                        depth = cache.get_or_compute(
                            gate_name=strait_name, gate_lons=gate_lon, gate_lats=gate_lat,
                            gebco_path=config.gebco_nc_path, depth_cap=None
                        )
                        files[f"images/{gate_safe}_bathymetry.png"] = _exp_bathy(
                            depth, x_km, strait_name, gate_lon, gate_lat, 250.0, export_dpi
                        )
                    except Exception as e:
                        st.warning(f"⚠️ Bathymetry: {e}")

                # === Timeline plots ===
                _dpi_scale = export_dpi / 100

                # --- Build descriptive metadata for plot annotations ---
                _date_start = time_pd.min().strftime('%Y-%m-%d')
                _date_end = time_pd.max().strftime('%Y-%m-%d')
                _n_pts = ugos.shape[0] if ugos is not None else 0
                _gate_len_km = f"{float(x_km.max()):.0f}" if x_km is not None else "?"
                _lon_range = f"{gate_lon.min():.2f}°–{gate_lon.max():.2f}°E" if gate_lon is not None else "?"
                _lat_range = f"{gate_lat.min():.2f}°–{gate_lat.max():.2f}°N" if gate_lat is not None else "?"
                _vel_source = "CMEMS L4 (SEALEVEL_GLO_PHY_L4_MY_008_047) — ugos/vgos"
                _sal_source = "ESA CCI Sea Surface Salinity v5.5 (satellite, ~2010–2023)"
                _bathy_source = "GEBCO 2023"

                # === Slope Timeline ===
                if exp_slope_ts:
                    try:
                        slope_series = getattr(cmems_l4_data, 'slope_series', None)
                        dot_matrix = getattr(cmems_l4_data, 'dot_matrix', None)
                        if slope_series is not None:
                            fig_slope = go.Figure()
                            fig_slope.add_trace(go.Scatter(
                                x=time_pd, y=slope_series, mode='lines+markers',
                                line=dict(color='mediumpurple', width=1.5),
                                marker=dict(size=4), name='DOT Slope',
                            ))
                            fig_slope.add_hline(y=0, line_color="#7F8C8D", line_width=1)
                            fig_slope.update_layout(
                                title=dict(text=f"DOT Slope Time Series — {strait_name}", font=dict(size=14)),
                                xaxis_title="Time", yaxis_title="Slope (m/100km)",
                                height=500, width=1400,
                                margin=dict(t=100, l=70, r=40, b=60),
                            )
                            files[f"images/{gate_safe}_slope_timeline.png"] = fig_slope.to_image(
                                format="png", width=1400, height=500, scale=_dpi_scale)
                        elif dot_matrix is not None:
                            st.warning("⚠️ Slope Timeline: slope_series not computed — run DOT tab first")
                    except Exception as e:
                        st.warning(f"⚠️ Slope timeline: {e}")

                # === Velocity 4×3 Monthly Profiles ===
                if exp_vel_4x3 and vel_done:
                    try:
                        gv_monthly = st.session_state.get(f'{key_gv}_monthly_v_perp')
                        gv_x = st.session_state.get(f'{key_gv}_x_km', x_km)
                        gv_time = st.session_state.get(f'{key_gv}_time_array', time_array)
                        gv_v_perp = st.session_state.get(f'{key_gv}_v_perp', v_perp)
                        gv_v_geo = st.session_state.get(f'{key_gv}_v_geo_ts', None)

                        if gv_monthly is not None:
                            # Plotly interactive 4×3 grid
                            _month_names_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                            fig_v4x3 = make_subplots(rows=3, cols=4,
                                                      subplot_titles=_month_names_short,
                                                      horizontal_spacing=0.06,
                                                      vertical_spacing=0.10)
                            _gv_t_pd = pd.to_datetime(np.asarray(gv_time).ravel())
                            for m_idx in range(12):
                                row = m_idx // 4 + 1
                                col_idx = m_idx % 4 + 1
                                m = m_idx + 1
                                bc, bm, bs = gv_monthly.get(m, (np.array([]), np.array([]), np.array([])))
                                if len(bc) > 0:
                                    fig_v4x3.add_trace(go.Scatter(
                                        x=bc, y=bm * 100, mode='lines',
                                        line=dict(color='#1E3A5F', width=2),
                                        showlegend=False,
                                        error_y=dict(type='data', array=bs * 100,
                                                     visible=True, color='rgba(30,58,95,0.3)'),
                                    ), row=row, col=col_idx)
                                    fig_v4x3.add_hline(y=0, line_color="#999", line_width=0.8,
                                                       row=row, col=col_idx)
                            _gv_period = f"{_gv_t_pd.min().year}–{_gv_t_pd.max().year}"
                            fig_v4x3.update_layout(
                                title=dict(
                                    text=f"{strait_name} — Monthly Velocity Profiles (v⊥)<br>"
                                         f"<sup>CMEMS L4 · {_gv_period} · {_gate_len_km} km</sup>",
                                    font=dict(size=14),
                                ),
                                height=900, width=1400,
                                showlegend=False,
                                margin=dict(t=100, l=60, r=30, b=50),
                            )
                            for i in range(1, 13):
                                fig_v4x3.update_yaxes(title_text="cm/s" if (i - 1) % 4 == 0 else "",
                                                       row=(i - 1) // 4 + 1, col=(i - 1) % 4 + 1)
                                if i > 8:
                                    fig_v4x3.update_xaxes(title_text="km",
                                                           row=(i - 1) // 4 + 1, col=(i - 1) % 4 + 1)
                            files[f"images/{gate_safe}_velocity_profiles_4x3.png"] = fig_v4x3.to_image(
                                format="png", width=1400, height=900, scale=_dpi_scale)

                        # Also export matplotlib version via export_service
                        if gv_v_perp is not None and gv_time is not None:
                            try:
                                from src.services.export_service import export_monthly_velocity_profiles_grid
                                _gv_t_pd2 = pd.to_datetime(np.asarray(gv_time).ravel())
                                _gv_v_geo_1d = gv_v_geo if gv_v_geo is not None else None
                                files[f"images/{gate_safe}_velocity_profiles_4x3_hq.png"] = \
                                    export_monthly_velocity_profiles_grid(
                                        v_perp=gv_v_perp,
                                        x_km=gv_x,
                                        time_array=gv_time,
                                        gate_name=strait_name,
                                        dataset='cmems_l4',
                                        gate_lon=gate_lon,
                                        gate_lat=gate_lat,
                                        v_geo=_gv_v_geo_1d,
                                        dpi=export_dpi,
                                    )
                            except Exception as e_mpl:
                                st.warning(f"⚠️ Velocity 4×3 HQ: {e_mpl}")
                    except Exception as e:
                        st.warning(f"⚠️ Velocity 4×3: {e}")

                # === DOT Monthly Profiles 3×4 ===
                if exp_dot_4x3:
                    try:
                        dot_matrix = getattr(cmems_l4_data, 'dot_matrix', None)
                        if dot_matrix is not None:
                            _dot_t_pd = pd.to_datetime(np.asarray(time_array).ravel())
                            _month_names_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                            fig_dot4x3 = make_subplots(rows=3, cols=4,
                                                        subplot_titles=_month_names_short,
                                                        horizontal_spacing=0.06,
                                                        vertical_spacing=0.10)
                            for m_idx in range(12):
                                row = m_idx // 4 + 1
                                col_idx = m_idx % 4 + 1
                                m = m_idx + 1
                                mask = _dot_t_pd.month == m
                                if mask.sum() > 0:
                                    dot_m = dot_matrix[:, mask]
                                    dot_mean = np.nanmean(dot_m, axis=1)
                                    dot_std = np.nanstd(dot_m, axis=1)
                                    _xkm = x_km if x_km is not None else np.arange(len(dot_mean))
                                    fig_dot4x3.add_trace(go.Scatter(
                                        x=_xkm, y=dot_mean, mode='lines',
                                        line=dict(color='#1E3A5F', width=2),
                                        showlegend=False,
                                        error_y=dict(type='data', array=dot_std,
                                                     visible=True, color='rgba(30,58,95,0.3)'),
                                    ), row=row, col=col_idx)
                            _dot_period = f"{_dot_t_pd.min().year}–{_dot_t_pd.max().year}"
                            fig_dot4x3.update_layout(
                                title=dict(
                                    text=f"{strait_name} — Monthly DOT Profiles<br>"
                                         f"<sup>CMEMS L4 · {_dot_period} · {_gate_len_km} km</sup>",
                                    font=dict(size=14),
                                ),
                                height=900, width=1400,
                                showlegend=False,
                                margin=dict(t=100, l=60, r=30, b=50),
                            )
                            for i in range(1, 13):
                                fig_dot4x3.update_yaxes(title_text="DOT (m)" if (i - 1) % 4 == 0 else "",
                                                         row=(i - 1) // 4 + 1, col=(i - 1) % 4 + 1)
                                if i > 8:
                                    fig_dot4x3.update_xaxes(title_text="km",
                                                             row=(i - 1) // 4 + 1, col=(i - 1) % 4 + 1)
                            files[f"images/{gate_safe}_dot_monthly_profiles_3x4.png"] = fig_dot4x3.to_image(
                                format="png", width=1400, height=900, scale=_dpi_scale)

                            # Also HQ matplotlib version
                            try:
                                from src.services.export_service import export_monthly_dot_profiles_grid
                                files[f"images/{gate_safe}_dot_monthly_profiles_3x4_hq.png"] = \
                                    export_monthly_dot_profiles_grid(
                                        dot_matrix=dot_matrix,
                                        x_km=x_km if x_km is not None else np.arange(dot_matrix.shape[0]),
                                        time_array=time_array,
                                        gate_name=strait_name,
                                        dataset='cmems_l4',
                                        gate_lon=gate_lon,
                                        gate_lat=gate_lat,
                                        dpi=export_dpi,
                                    )
                            except Exception as e_mpl:
                                st.warning(f"⚠️ DOT 3×4 HQ: {e_mpl}")
                        else:
                            st.warning("⚠️ DOT Monthly Profiles: dot_matrix not available in loaded data")
                    except Exception as e:
                        st.warning(f"⚠️ DOT 3×4: {e}")

                if exp_vel_ts and v_perp is not None:
                    try:
                        mean_v = np.nanmean(v_perp, axis=0) * 100  # cm/s
                        fig_vts = go.Figure()
                        fig_vts.add_trace(go.Scatter(
                            x=time_pd, y=mean_v, mode='lines',
                            line=dict(color='#1E3A5F', width=2), name='v_perp',
                        ))
                        fig_vts.add_hline(y=float(np.nanmean(mean_v)), line_dash="dash",
                                          line_color="#E74C3C", line_width=1,
                                          annotation_text=f"Mean: {np.nanmean(mean_v):.2f} cm/s")
                        fig_vts.add_hline(y=0, line_color="gray", line_width=1)
                        _vel_subtitle = (
                            f"Gate: {_lon_range}, {_lat_range} ({_gate_len_km} km, {_n_pts} pts) · "
                            f"Period: {_date_start} to {_date_end} ({len(time_pd)} months)<br>"
                            f"Source: {_vel_source} · v⊥ = v_N·cos θ + v_E·sin θ (gate-normal projection)"
                        )
                        fig_vts.update_layout(
                            title=dict(text=f"Cross-Gate Perpendicular Geostrophic Velocity — {strait_name}", font=dict(size=14)),
                            xaxis_title="Time", yaxis_title="v⊥ (cm/s)",
                            height=550, width=1400,  
                            margin=dict(t=120, l=70, r=40, b=60),
                        )
                        fig_vts.add_annotation(
                            text=_vel_subtitle, xref="paper", yref="paper",
                            x=0, y=1.12, showarrow=False, font=dict(size=10, color="#555"),
                            xanchor="left", yanchor="top",
                        )
                        files[f"images/{gate_safe}_velocity_timeline.png"] = fig_vts.to_image(
                            format="png", width=1400, height=550, scale=_dpi_scale)
                    except Exception as e:
                        st.warning(f"⚠️ Velocity timeline: {e}")

                if exp_fw_ts and fw_done:
                    try:
                        fw_arr = st.session_state.get(f'{key_fw}_fw', np.array([]))
                        fw_time = st.session_state.get(f'{key_fw}_time')
                        fw_params = st.session_state.get(f'{key_fw}_params', {})
                        fw_msv = fw_arr / 1e3
                        fw_t_pd = pd.to_datetime(np.asarray(fw_time).ravel()) if fw_time is not None else np.arange(len(fw_arr))
                        _fw_sref = fw_params.get('s_ref', 34.8)
                        _fw_depth = fw_params.get('depth', 250)
                        _fw_sal_mean_raw = fw_params.get('sss_mean', None)
                        _fw_sal_mean = f"{_fw_sal_mean_raw:.2f}" if isinstance(_fw_sal_mean_raw, (int, float)) and not np.isnan(_fw_sal_mean_raw) else "N/A"
                        _fw_date_start = fw_t_pd.min().strftime('%Y-%m-%d') if hasattr(fw_t_pd, 'min') else '?'
                        _fw_date_end = fw_t_pd.max().strftime('%Y-%m-%d') if hasattr(fw_t_pd, 'max') else '?'
                        fig_fts = go.Figure()
                        fig_fts.add_trace(go.Scatter(
                            x=fw_t_pd, y=fw_msv, mode='lines',
                            line=dict(color='#3498DB', width=2), name='FW Transport',
                        ))
                        fig_fts.add_hline(y=float(np.nanmean(fw_msv)), line_dash="dash",
                                          line_color="#E74C3C", line_width=1,
                                          annotation_text=f"Mean: {np.nanmean(fw_msv):.2f} mSv")
                        fig_fts.add_hline(y=0, line_color="gray", line_width=1)
                        _fw_subtitle = (
                            f"Gate: {_lon_range}, {_lat_range} ({_gate_len_km} km) · "
                            f"Period: {_fw_date_start} to {_fw_date_end}<br>"
                            f"Fw = Σ v⊥ · (1 − S̄/S_A) · H(x) · Δx   |   S_A = {_fw_sref} PSU, "
                            f"H cap = {_fw_depth} m, mean S̄ = {_fw_sal_mean} PSU<br>"
                            f"Velocity: {_vel_source} · Salinity: {_sal_source} · Bathymetry: {_bathy_source}"
                        )
                        fig_fts.update_layout(
                            title=dict(text=f"Freshwater Transport Time Series — {strait_name}", font=dict(size=14)),
                            xaxis_title="Time", yaxis_title="Freshwater Transport (mSv)",
                            height=550, width=1400,  
                            margin=dict(t=140, l=70, r=40, b=60),
                        )
                        fig_fts.add_annotation(
                            text=_fw_subtitle, xref="paper", yref="paper",
                            x=0, y=1.15, showarrow=False, font=dict(size=10, color="#555"),
                            xanchor="left", yanchor="top",
                        )
                        files[f"images/{gate_safe}_fw_transport_timeline.png"] = fig_fts.to_image(
                            format="png", width=1400, height=550, scale=_dpi_scale)
                    except Exception as e:
                        st.warning(f"⚠️ FW timeline: {e}")

                if exp_sf_ts and sf_done:
                    try:
                        sf_arr = st.session_state.get(f'{key_sf}_salt_flux', np.array([]))
                        sf_time = st.session_state.get(f'{key_sf}_time')
                        sf_params = st.session_state.get(f'{key_sf}_params', {})
                        sf_ggs = sf_arr / 1e9
                        sf_t_pd = pd.to_datetime(np.asarray(sf_time).ravel()) if sf_time is not None else np.arange(len(sf_arr))
                        _sf_rho = sf_params.get('rho', 1025)
                        _sf_depth = sf_params.get('depth', 250)
                        _sf_sal_mean_raw = sf_params.get('sss_mean', '?')
                        _sf_sal_mean = f"{_sf_sal_mean_raw:.2f}" if isinstance(_sf_sal_mean_raw, (int, float)) and not np.isnan(_sf_sal_mean_raw) else '?'
                        _sf_date_start = sf_t_pd.min().strftime('%Y-%m-%d') if hasattr(sf_t_pd, 'min') else '?'
                        _sf_date_end = sf_t_pd.max().strftime('%Y-%m-%d') if hasattr(sf_t_pd, 'max') else '?'
                        fig_sts = go.Figure()
                        fig_sts.add_trace(go.Scatter(
                            x=sf_t_pd, y=sf_ggs, mode='lines',
                            line=dict(color='#9B59B6', width=2), name='Salt Flux',
                        ))
                        fig_sts.add_hline(y=float(np.nanmean(sf_ggs)), line_dash="dash",
                                          line_color="#E74C3C", line_width=1,
                                          annotation_text=f"Mean: {np.nanmean(sf_ggs):.3f} Gg/s")
                        fig_sts.add_hline(y=0, line_color="gray", line_width=1)
                        _sf_subtitle = (
                            f"Gate: {_lon_range}, {_lat_range} ({_gate_len_km} km) · "
                            f"Period: {_sf_date_start} to {_sf_date_end}<br>"
                            f"Sm = Σ ρ · (S̄/1000) · v⊥ · H(x) · Δx   |   ρ = {_sf_rho:.0f} kg/m³, "
                            f"H cap = {_sf_depth} m, mean S̄ = {_sf_sal_mean} PSU<br>"
                            f"Velocity: {_vel_source} · Salinity: {_sal_source} · Bathymetry: {_bathy_source}"
                        )
                        fig_sts.update_layout(
                            title=dict(text=f"Salt Flux Time Series — {strait_name}", font=dict(size=14)),
                            xaxis_title="Time", yaxis_title="Salt Flux (Gg/s)",
                            height=550, width=1400,  
                            margin=dict(t=140, l=70, r=40, b=60),
                        )
                        fig_sts.add_annotation(
                            text=_sf_subtitle, xref="paper", yref="paper",
                            x=0, y=1.15, showarrow=False, font=dict(size=10, color="#555"),
                            xanchor="left", yanchor="top",
                        )
                        files[f"images/{gate_safe}_salt_flux_timeline.png"] = fig_sts.to_image(
                            format="png", width=1400, height=550, scale=_dpi_scale)
                    except Exception as e:
                        st.warning(f"⚠️ Salt Flux timeline: {e}")

                # === 4×3 Along-gate plots ===
                if exp_fw_4x3 and fw_done:
                    try:
                        from .transport_plots import plot_transport_along_gate_4x3
                        fw_v = st.session_state.get(f'{key_fw}_v_perp')
                        fw_x = st.session_state.get(f'{key_fw}_x_km')
                        fw_t = st.session_state.get(f'{key_fw}_time')
                        fw_H = st.session_state.get(f'{key_fw}_H_profile')
                        fw_s = st.session_state.get(f'{key_fw}_sss_interp')
                        fw_dx = st.session_state.get(f'{key_fw}_dx')
                        fw_sr = st.session_state.get(f'{key_fw}_sref_val', 34.8)
                        _fw_p = st.session_state.get(f'{key_fw}_params', {})
                        _fw_t_pd = pd.to_datetime(np.asarray(fw_t).ravel()) if fw_t is not None else None
                        _fw_meta = {
                            'lon_range': _lon_range, 'lat_range': _lat_range,
                            'gate_length_km': _gate_len_km,
                            'date_start': _fw_t_pd.min().strftime('%Y-%m-%d') if _fw_t_pd is not None else '?',
                            'date_end': _fw_t_pd.max().strftime('%Y-%m-%d') if _fw_t_pd is not None else '?',
                            'vel_source': _vel_source, 'sal_source': _sal_source,
                            'bathy_source': _bathy_source,
                            'depth_cap': _fw_p.get('depth', 250),
                            'sss_mean': _fw_p.get('sss_mean', np.nan),
                        }
                        fig_fw = plot_transport_along_gate_4x3(
                            v_perp=fw_v, x_km=fw_x, time_array=fw_t,
                            H_profile=fw_H, sss_interp=fw_s, dx=fw_dx,
                            strait_name=strait_name, mode='freshwater', s_ref=fw_sr,
                            metadata=_fw_meta,
                        )
                        if fig_fw is not None:
                            files[f"images/{gate_safe}_fw_transport_4x3.png"] = fig_fw.to_image(
                                format="png", width=1400, height=1200, scale=export_dpi / 100
                            )
                    except Exception as e:
                        st.warning(f"⚠️ FW 4×3: {e}")

                if exp_sf_4x3 and sf_done:
                    try:
                        from .transport_plots import plot_transport_along_gate_4x3
                        sf_v = st.session_state.get(f'{key_sf}_v_perp')
                        sf_x = st.session_state.get(f'{key_sf}_x_km')
                        sf_t = st.session_state.get(f'{key_sf}_time')
                        sf_H = st.session_state.get(f'{key_sf}_H_profile')
                        sf_s = st.session_state.get(f'{key_sf}_sss_interp')
                        sf_dx = st.session_state.get(f'{key_sf}_dx')
                        _sf_p = st.session_state.get(f'{key_sf}_params', {})
                        _sf_t_pd = pd.to_datetime(np.asarray(sf_t).ravel()) if sf_t is not None else None
                        _sf_meta = {
                            'lon_range': _lon_range, 'lat_range': _lat_range,
                            'gate_length_km': _gate_len_km,
                            'date_start': _sf_t_pd.min().strftime('%Y-%m-%d') if _sf_t_pd is not None else '?',
                            'date_end': _sf_t_pd.max().strftime('%Y-%m-%d') if _sf_t_pd is not None else '?',
                            'vel_source': _vel_source, 'sal_source': _sal_source,
                            'bathy_source': _bathy_source,
                            'depth_cap': _sf_p.get('depth', 250),
                            'rho': _sf_p.get('rho', 1025),
                            'sss_mean': _sf_p.get('sss_mean', np.nan),
                        }
                        fig_sf = plot_transport_along_gate_4x3(
                            v_perp=sf_v, x_km=sf_x, time_array=sf_t,
                            H_profile=sf_H, sss_interp=sf_s, dx=sf_dx,
                            strait_name=strait_name, mode='salt',
                            metadata=_sf_meta,
                        )
                        if fig_sf is not None:
                            files[f"images/{gate_safe}_salt_flux_4x3.png"] = fig_sf.to_image(
                                format="png", width=1400, height=1200, scale=export_dpi / 100
                            )
                    except Exception as e:
                        st.warning(f"⚠️ Salt Flux 4×3: {e}")

                # === Publication Plots (matplotlib + cmocean) ===
                _pub_meta = {
                    'lon_range': _lon_range, 'lat_range': _lat_range,
                    'gate_length_km': _gate_len_km,
                    'date_start': _date_start, 'date_end': _date_end,
                    'vel_source': _vel_source, 'sal_source': _sal_source,
                    'bathy_source': _bathy_source,
                }

                # --- Combined Transport Timeline (VT + FWT + SF) ---
                if exp_combined:
                    try:
                        from src.services.publication_plots import plot_combined_transport_timeline

                        # Get VT from v_perp + bathymetry
                        from src.services.transport_service import compute_segment_widths as _csw
                        from src.services.gebco_service import get_bathymetry_cache as _gbc
                        _dx = _csw(gate_lon, gate_lat, x_km)
                        _bathy_cache = _gbc()
                        _H_full = _bathy_cache.get_or_compute(
                            gate_name=strait_name, gate_lons=gate_lon, gate_lats=gate_lat,
                            gebco_path=config.gebco_nc_path, depth_cap=None
                        )
                        _H = np.minimum(_H_full, 250.0)
                        _n_pts_v, _n_time_v = v_perp.shape
                        _vt_sv = np.zeros(_n_time_v)
                        for _t in range(_n_time_v):
                            _v = v_perp[:, _t]
                            _ok = np.isfinite(_v) & np.isfinite(_H)
                            _vt_sv[_t] = np.nansum(_v[_ok] * _H[_ok] * _dx[_ok]) / 1e6 if _ok.any() else np.nan

                        # FWT array
                        _fw_arr = st.session_state.get(f'{key_fw}_fw') if fw_done else None

                        # SF array
                        _sf_arr = st.session_state.get(f'{key_sf}_salt_flux') if sf_done else None

                        # Uncertainty
                        _err_u = getattr(cmems_l4_data, 'err_ugosa_matrix', None)
                        _err_v = getattr(cmems_l4_data, 'err_vgosa_matrix', None)
                        _sigma_vt = None
                        if _err_u is not None and _err_v is not None:
                            from src.services.transport_service import compute_perpendicular_velocity_uncertainty
                            _sigma_vp = compute_perpendicular_velocity_uncertainty(
                                _err_u, _err_v, gate_lon, gate_lat, gate_name=strait_name
                            )
                            _sigma_vt = np.zeros(_n_time_v)
                            for _t in range(_n_time_v):
                                _sv = _sigma_vp[:, _t]
                                _ok = np.isfinite(_sv) & np.isfinite(_H)
                                if _ok.any():
                                    _sigma_vt[_t] = np.sqrt(np.nansum((_sv[_ok] * _H[_ok] * _dx[_ok])**2)) / 1e6

                        _gate_key = strait_name.lower().replace(' ', '_').replace('-', '_')
                        files[f"images/{gate_safe}_combined_transport.png"] = plot_combined_transport_timeline(
                            time_array=time_array_np,
                            vt_sv=_vt_sv,
                            fw_m3s=_fw_arr,
                            sm_kgs=_sf_arr,
                            sigma_vt=_sigma_vt,
                            gate_name=_gate_key,
                            gate_display=strait_name,
                            dpi=export_dpi,
                            metadata=_pub_meta,
                        )
                    except Exception as e:
                        st.warning(f"⚠️ Combined Transport: {e}")
                        import traceback
                        logger.error(f"Combined Transport export error: {traceback.format_exc()}")

                # --- Publication VT Timeline ---
                if exp_vt_pub:
                    try:
                        from src.services.publication_plots import plot_single_transport_timeseries

                        # Reuse VT computed above, or compute fresh
                        if '_vt_sv' not in dir():
                            from src.services.transport_service import compute_segment_widths as _csw2
                            from src.services.gebco_service import get_bathymetry_cache as _gbc2
                            _dx2 = _csw2(gate_lon, gate_lat, x_km)
                            _bc2 = _gbc2()
                            _Hf2 = _bc2.get_or_compute(
                                gate_name=strait_name, gate_lons=gate_lon, gate_lats=gate_lat,
                                gebco_path=config.gebco_nc_path, depth_cap=None
                            )
                            _H2 = np.minimum(_Hf2, 250.0)
                            _vt_sv = np.zeros(v_perp.shape[1])
                            for _t in range(v_perp.shape[1]):
                                _v = v_perp[:, _t]
                                _ok = np.isfinite(_v) & np.isfinite(_H2)
                                _vt_sv[_t] = np.nansum(_v[_ok] * _H2[_ok] * _dx2[_ok]) / 1e6 if _ok.any() else np.nan

                        _gate_key = strait_name.lower().replace(' ', '_').replace('-', '_')
                        files[f"images/{gate_safe}_volume_transport_pub.png"] = plot_single_transport_timeseries(
                            time_array=time_array_np,
                            values=_vt_sv,
                            ylabel='VT [Sv]',
                            color='#1E3A5F',
                            gate_name=_gate_key,
                            sigma=_sigma_vt if '_sigma_vt' in dir() else None,
                            title=f"{strait_name} — Volume Transport",
                            dpi=export_dpi,
                        )
                    except Exception as e:
                        st.warning(f"⚠️ VT publication: {e}")

                # --- Publication FWT Timeline ---
                if exp_fw_pub and fw_done:
                    try:
                        from src.services.publication_plots import plot_single_transport_timeseries
                        _fw_pub_arr = st.session_state.get(f'{key_fw}_fw', np.array([]))
                        _fw_pub_time = st.session_state.get(f'{key_fw}_time', time_array_np)
                        _gate_key = strait_name.lower().replace(' ', '_').replace('-', '_')
                        files[f"images/{gate_safe}_fw_transport_pub.png"] = plot_single_transport_timeseries(
                            time_array=np.asarray(_fw_pub_time).ravel(),
                            values=_fw_pub_arr / 1e3,  # → mSv
                            ylabel='FWT [mSv]',
                            color='#3498DB',
                            gate_name=_gate_key,
                            title=f"{strait_name} — Freshwater Transport",
                            dpi=export_dpi,
                        )
                    except Exception as e:
                        st.warning(f"⚠️ FWT publication: {e}")

                # --- Publication SF Timeline ---
                if exp_sf_pub and sf_done:
                    try:
                        from src.services.publication_plots import plot_single_transport_timeseries
                        _sf_pub_arr = st.session_state.get(f'{key_sf}_salt_flux', np.array([]))
                        _sf_pub_time = st.session_state.get(f'{key_sf}_time', time_array_np)
                        _gate_key = strait_name.lower().replace(' ', '_').replace('-', '_')
                        files[f"images/{gate_safe}_salt_flux_pub.png"] = plot_single_transport_timeseries(
                            time_array=np.asarray(_sf_pub_time).ravel(),
                            values=_sf_pub_arr / 1e9,  # → Gg/s
                            ylabel='SF [Gg/s]',
                            color='#9B59B6',
                            gate_name=_gate_key,
                            title=f"{strait_name} — Salt Flux",
                            dpi=export_dpi,
                        )
                    except Exception as e:
                        st.warning(f"⚠️ SF publication: {e}")

                # === README.txt ===
                from datetime import datetime
                _fw_p_r = st.session_state.get(f'{key_fw}_params', {})
                _sf_p_r = st.session_state.get(f'{key_sf}_params', {})
                readme_lines = [
                    f"ARCFRESH Export — {strait_name}",
                    f"{'=' * 50}",
                    f"",
                    f"Gate:       {strait_name}",
                    f"Longitude:  {_lon_range}",
                    f"Latitude:   {_lat_range}",
                    f"Length:     {_gate_len_km} km  ({_n_pts} points)",
                    f"Period:     {_date_start} to {_date_end}  ({len(time_pd)} months)",
                    f"Export:     {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    f"DPI:        {export_dpi}",
                    f"",
                    f"DATA SOURCES",
                    f"{'-' * 50}",
                    f"Velocity:   {_vel_source}",
                    f"Salinity:   {_sal_source}",
                    f"Bathymetry: {_bathy_source}",
                    f"",
                    f"PARAMETERS",
                    f"{'-' * 50}",
                ]
                if fw_done:
                    readme_lines += [
                        f"Freshwater Transport:",
                        f"  Formula:  Fw = Σ v⊥ · (1 − S̄/S_A) · H(x) · Δx",
                        f"  S_A:      {_fw_p_r.get('s_ref', 34.8)} PSU",
                        f"  H cap:    {_fw_p_r.get('depth', 250)} m",
                        f"  Mean S̄:   {_fw_p_r.get('sss_mean', '?')} PSU",
                        f"",
                    ]
                if sf_done:
                    readme_lines += [
                        f"Salt Flux:",
                        f"  Formula:  Sm = Σ ρ · (S̄/1000) · v⊥ · H(x) · Δx",
                        f"  ρ:        {_sf_p_r.get('rho', 1025)} kg/m³",
                        f"  H cap:    {_sf_p_r.get('depth', 250)} m",
                        f"  Mean S̄:   {_sf_p_r.get('sss_mean', '?')} PSU",
                        f"",
                    ]
                readme_lines += [
                    f"FILES",
                    f"{'-' * 50}",
                ]
                for f_key in sorted(files):
                    sz = len(files[f_key]) if isinstance(files[f_key], (bytes, str)) else 0
                    readme_lines.append(f"  {f_key}  ({sz / 1024:.1f} KB)")
                readme_lines += [
                    f"",
                    f"DESCRIPTION OF IMAGE FILES",
                    f"{'-' * 50}",
                ]
                _img_descriptions = {
                    '_map.png': 'Geographic map showing the gate location with coastlines and bathymetric context.',
                    '_bathymetry.png': 'Bathymetric depth profile along the gate (GEBCO 2023). Red line = depth cap used for transport.',
                    '_slope_timeline.png': (
                        'Monthly time series of DOT slope along the gate (m/100km). '
                        'Computed via linear regression of the along-gate DOT profile. '
                        'Proxy for geostrophic velocity magnitude.'
                    ),
                    '_velocity_timeline.png': (
                        'Monthly time series of cross-gate perpendicular geostrophic velocity v⊥ (cm/s). '
                        'Computed from CMEMS L4 ugos/vgos via gate-normal projection. Red dashed = temporal mean.'
                    ),
                    '_velocity_profiles_4x3.png': (
                        '4×3 grid (Jan–Dec) of perpendicular velocity profiles along the gate. '
                        'Each subplot: monthly climatology of v⊥ (cm/s) averaged over all years. '
                        'Error bars = ±1 std. Positive = into Arctic.'
                    ),
                    '_velocity_profiles_4x3_hq.png': (
                        'High-resolution matplotlib version of the velocity 4×3 grid. '
                        'Includes v_perp (blue) and v_geo from DOT slope (orange dashed). '
                        'Suitable for publications.'
                    ),
                    '_dot_monthly_profiles_3x4.png': (
                        '3×4 grid (Jan–Dec) of mean DOT profiles along the gate. '
                        'Each subplot: monthly climatology ± std shaded. Blue = mean DOT (m).'
                    ),
                    '_dot_monthly_profiles_3x4_hq.png': (
                        'High-resolution matplotlib version of the DOT monthly 3×4 grid. '
                        'Includes slope (mm/km) and R² annotations per month.'
                    ),
                    '_fw_transport_timeline.png': (
                        'Monthly time series of freshwater transport (mSv). '
                        'Fw = Σ v⊥·(1−S̄/S_A)·H(x)·Δx integrated along gate. '
                        'Positive = into Arctic. Red dashed = temporal mean.'
                    ),
                    '_salt_flux_timeline.png': (
                        'Monthly time series of salt flux (Gg/s). '
                        'Sm = Σ ρ·(S̄/1000)·v⊥·H(x)·Δx integrated along gate. '
                        'Positive = into Arctic. Red dashed = temporal mean.'
                    ),
                    '_fw_transport_4x3.png': (
                        '4×3 grid (Jan–Dec) of freshwater transport bar charts along the gate. '
                        'Each subplot: monthly climatology averaged over all years. '
                        'Blue bars = positive (into Arctic), red bars = negative (out of Arctic). '
                        'Σ annotation = total integrated transport for that month.'
                    ),
                    '_salt_flux_4x3.png': (
                        '4×3 grid (Jan–Dec) of salt flux bar charts along the gate. '
                        'Each subplot: monthly climatology averaged over all years. '
                        'Blue bars = positive (into Arctic), red bars = negative (out of Arctic). '
                        'Σ annotation = total integrated salt flux for that month.'
                    ),
                    '_combined_transport.png': (
                        'Publication-quality combined transport timeline (matplotlib + cmocean). '
                        '3-panel plot: Volume Transport (Sv), Freshwater Transport (mSv), Salt Flux (Gg/s). '
                        'Raw daily data with ±σ uncertainty band on VT. No rolling mean. '
                        'Mean reference line per panel. '
                        'Sign convention: positive = INTO Arctic (dot product with gate-normal INTO-Arctic vector).'
                    ),
                    '_volume_transport_pub.png': (
                        'Publication-quality VT time series (matplotlib). '
                        'Raw data + ±σ uncertainty band + mean reference. No rolling mean.'
                    ),
                    '_fw_transport_pub.png': (
                        'Publication-quality FWT time series (matplotlib). '
                        'Raw data + mean reference. Fw = Σ v⊥·(1−S̄/S_A)·H·Δx.'
                    ),
                    '_salt_flux_pub.png': (
                        'Publication-quality SF time series (matplotlib). '
                        'Raw data + mean reference. Sm = Σ ρ·(S̄/1000)·v⊥·H·Δx.'
                    ),
                }
                for f_key in sorted(files):
                    if f_key.endswith('.png'):
                        fname = f_key.split('/')[-1]
                        desc = None
                        for suffix, d in _img_descriptions.items():
                            if fname.endswith(suffix):
                                desc = d
                                break
                        if desc:
                            readme_lines.append(f"  {fname}:")
                            readme_lines.append(f"    {desc}")
                            readme_lines.append("")
                files["README.txt"] = "\n".join(readme_lines)

                # === ZIP ===
                from src.services.export_service import create_export_zip
                ts = datetime.now().strftime("%Y%m%d")
                folder = f"arcfresh_{gate_safe}_{start_year}-{end_year}_{ts}"
                zip_bytes = create_export_zip(files, folder)

                n_csv = sum(1 for f in files if f.endswith('.csv'))
                n_png = sum(1 for f in files if f.endswith('.png'))
                st.success(f"✅ Ready: {n_csv} CSV + {n_png} PNG")

                st.download_button(
                    "📥 Download ZIP", zip_bytes,
                    f"{folder}.zip", "application/zip",
                    type="primary", use_container_width=True
                )

                with st.expander("📂 Contents"):
                    for f in sorted(files):
                        sz = len(files[f]) if isinstance(files[f], (bytes, str)) else 0
                        st.text(f"  {f}  ({sz / 1024:.1f} KB)")

            except Exception as e:
                st.error(f"❌ Export failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# ==============================================================================
# DTU TAB 5: EXPORT [LEGACY]
# ==============================================================================

def _render_dtu_export_tab(dtu_data, config: AppConfig):
    """Render export tab for DTUSpace/CMEMS L4 data."""
    # Get dataset info dynamically
    ds_info = _get_gridded_dataset_info(dtu_data)
    
    st.subheader(f"📤 Export {ds_info['name']} Data")
    
    # Info
    strait_name = getattr(dtu_data, 'strait_name', 'Unknown')
    dataset_name = ds_info['name']
    
    st.info(f"{ds_info['emoji']} Exporting **{dataset_name}** data for **{strait_name}**")
    
    # Create tabs for different export types
    export_tabs = st.tabs(["📊 Synthetic Data", "📈 Time Series", "📉 Statistics"])
    
    # ==========================================================================
    # TAB 1: SYNTHETIC DATA EXPORT
    # ==========================================================================
    with export_tabs[0]:
        st.markdown("### Synthetic Observation Data")
        st.caption("DTUSpace is gridded - this creates synthetic 'observations' along the gate")
        
        df = getattr(dtu_data, 'df', None)
        
        if df is not None and not df.empty:
            st.markdown(f"**Rows**: {len(df):,} (gate points × time steps)")
            st.caption(f"Columns: {', '.join(df.columns)}")
            
            # Preview
            st.dataframe(df.head(20), width='stretch')
            
            # Download
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download DTUSpace Synthetic Data (CSV)",
                data=csv_data,
                file_name=f"dtuspace_synthetic_{strait_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key="export_dtu_synthetic"
            )
        else:
            st.warning("No synthetic data available")
    
    # ==========================================================================
    # TAB 2: TIME SERIES EXPORT
    # ==========================================================================
    with export_tabs[1]:
        st.markdown("### Monthly Time Series")
        
        time_array = getattr(dtu_data, 'time_array', None)
        slope_series = getattr(dtu_data, 'slope_series', None)
        v_geo = getattr(dtu_data, 'v_geostrophic_series', None)
        
        if time_array is not None and slope_series is not None:
            ts_df = pd.DataFrame({
                'time': pd.to_datetime(time_array),
                'slope_m_100km': slope_series,
            })
            
            if v_geo is not None:
                ts_df['v_geostrophic_m_s'] = v_geo
                ts_df['v_geostrophic_cm_s'] = v_geo * 100
            
            ts_df['source'] = 'DTUSpace'
            ts_df['strait'] = strait_name
            
            st.dataframe(ts_df.head(20), width='stretch')
            st.caption(f"Showing first 20 of {len(ts_df)} rows")
            
            csv_ts = ts_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Time Series (CSV)",
                data=csv_ts,
                file_name=f"dtuspace_timeseries_{strait_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key="export_dtu_timeseries"
            )
        else:
            st.warning("No time series data available")
    
    # ==========================================================================
    # TAB 3: STATISTICS EXPORT
    # ==========================================================================
    with export_tabs[2]:
        st.markdown("### Summary Statistics")
        
        slope_series = getattr(dtu_data, 'slope_series', None)
        v_geo = getattr(dtu_data, 'v_geostrophic_series', None)
        start_year = getattr(dtu_data, 'start_year', 2006)
        end_year = getattr(dtu_data, 'end_year', 2017)
        x_km = getattr(dtu_data, 'x_km', None)
        
        stats_data = []
        
        if slope_series is not None:
            valid = slope_series[~np.isnan(slope_series)]
            if len(valid) > 0:
                stats_data.append({
                    'Source': 'DTUSpace',
                    'Variable': 'Slope (m/100km)',
                    'Mean': f"{np.mean(valid):.4f}",
                    'Std': f"{np.std(valid):.4f}",
                    'Min': f"{np.min(valid):.4f}",
                    'Max': f"{np.max(valid):.4f}",
                    'N_valid': len(valid),
                    'N_total': len(slope_series),
                    'Strait': strait_name,
                    'Period': f"{start_year}-{end_year}"
                })
        
        if v_geo is not None:
            valid = v_geo[~np.isnan(v_geo)]
            if len(valid) > 0:
                stats_data.append({
                    'Source': 'DTUSpace',
                    'Variable': 'V_geo (m/s)',
                    'Mean': f"{np.mean(valid):.6f}",
                    'Std': f"{np.std(valid):.6f}",
                    'Min': f"{np.min(valid):.6f}",
                    'Max': f"{np.max(valid):.6f}",
                    'N_valid': len(valid),
                    'N_total': len(v_geo),
                    'Strait': strait_name,
                    'Period': f"{start_year}-{end_year}"
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, width='stretch')
            
            # Additional info
            if x_km is not None and len(x_km) > 0:
                st.metric("Gate Length", f"{x_km.max():.1f} km")
            
            csv_stats = stats_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Statistics (CSV)",
                data=csv_stats,
                file_name=f"dtuspace_stats_{strait_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key="export_dtu_stats"
            )
        else:
            st.warning("No statistics available")



def _render_salt_flux_tab(data, config: AppConfig, ds_info: dict):
    """
    Salt Flux / Mass Transport tab - WORKING VERSION
    """
    import streamlit as st
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.header("🧂 Mass Transport (Salt + Freshwater)")
    
    # Get data
    strait_name = getattr(data, 'strait_name', 'Unknown')
    v_geo_perp = getattr(data, 'v_geo_perp', None)
    gate_lons = getattr(data, 'gate_lons', None)
    gate_lats = getattr(data, 'gate_lats', None)
    x_km = getattr(data, 'x_km', None)
    time_array = getattr(data, 'time', None)
    
    # Check if we have velocity data
    if v_geo_perp is None:
        st.error("❌ No velocity data available!")
        st.info("💡 Please go to 'Geostrophic Velocity' tab and compute velocities first")
        return
    
    # Show data info
    if v_geo_perp is not None:
        n_points = v_geo_perp.shape[0]
        n_time = v_geo_perp.shape[1] if v_geo_perp.ndim > 1 else 1
        st.success(f"✅ Velocity data loaded: {n_points} points × {n_time} time steps")
    else:
        n_points = 0
        n_time = 0
    
    # Simple controls (NO session_state conflicts!)
    st.subheader("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        depth_val = st.slider(
            "Depth Cap (m)",
            min_value=50,
            max_value=500,
            value=250,
            step=50,
            key="salt_flux_depth_slider"
        )
    
    with col2:
        sal_val = st.slider(
            "Salinity (PSU)",
            min_value=30.0,
            max_value=36.0,
            value=35.0,
            step=0.1,
            key="salt_flux_salinity_slider"
        )
    
    st.info("Constants: ρ = 1025 kg/m³, S_ref = 34.8 PSU")
    
    # Compute button
    if st.button("�� Compute Mass Transport", type="primary", key="salt_flux_compute_btn"):
        
        with st.spinner("Computing..."):
            try:
                # Constants
                RHO = 1025.0
                S_REF = 34.8
                
                # Simple width calculation
                if x_km is not None and len(x_km) > 1:
                    dx = np.diff(x_km)
                    dx = np.append(dx, dx[-1]) * 1000  # km to m
                else:
                    dx = np.ones(n_points) * 1000  # 1 km default
                
                # Calculate transport
                salt_transport = []
                fw_transport = []
                
                for t in range(n_time):
                    v = v_geo_perp[:, t] if v_geo_perp.ndim > 1 else v_geo_perp
                    
                    # Salt: ρ × S × v × H × dx
                    salt = np.nansum(RHO * (sal_val/1000) * v * depth_val * dx)
                    salt_transport.append(salt)
                    
                    # Freshwater: v × (1 - S/S_ref) × H × dx
                    fw = np.nansum(v * (1 - sal_val/S_REF) * depth_val * dx)
                    fw_transport.append(fw)
                
                salt_transport = np.array(salt_transport)
                fw_transport = np.array(fw_transport)
                
                # Store results
                st.session_state['salt_flux_results_salt'] = salt_transport
                st.session_state['salt_flux_results_fw'] = fw_transport
                st.session_state['salt_flux_results_time'] = time_array
                st.session_state['salt_flux_computed'] = True
                
                st.success("✅ Computed!")
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # Show results
    if st.session_state.get('salt_flux_computed', False):
        
        salt = st.session_state.get('salt_flux_results_salt', [])
        fw = st.session_state.get('salt_flux_results_fw', [])
        times = st.session_state.get('salt_flux_results_time', None)
        
        if len(salt) > 0:
            st.divider()
            st.subheader("📊 Results")
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Salt Transport", f"{np.mean(salt)/1e9:.2f} Gg/s")
            with col2:
                st.metric("Mean Freshwater", f"{np.mean(fw)/1e6:.3f} Sv")
            
            # Plot
            fig = make_subplots(
                rows=2, cols=1, 
                subplot_titles=("Salt Transport", "Freshwater Transport"),
                vertical_spacing=0.15
            )
            
            if times is not None:
                x = pd.to_datetime(times)
            else:
                x = list(range(len(salt)))
            
            fig.add_trace(
                go.Scatter(x=x, y=np.array(salt)/1e9, name="Salt", line=dict(color="orange", width=2)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=x, y=np.array(fw)/1e6, name="FW", line=dict(color="blue", width=2)),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Salt Transport (Gg/s)", row=1, col=1)
            fig.update_yaxes(title_text="Freshwater (Sv)", row=2, col=1)
            fig.update_layout(height=600, showlegend=False, hovermode='x unified')
            
            st.plotly_chart(fig, use_container_width=True, key="_render_salt_flux_tab_1")
