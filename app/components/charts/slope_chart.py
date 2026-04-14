"""
Slope Timeline Charts - Unified rendering for all dataset types.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats as scipy_stats
from typing import Optional, Dict, Any

from .utils import DATASET_COLORS, DATASET_NAMES, get_pass_data_attributes


def render_slope_timeline(
    pass_data,
    dataset_key: str = "slcci",
    config: Optional[Any] = None,
    show_controls: bool = True,
) -> None:
    """
    Render slope timeline for any dataset type.
    
    Args:
        pass_data: PassData object with slope_series and time_array
        dataset_key: Dataset identifier (slcci, cmems, cmems_l4, dtu)
        config: AppConfig object
        show_controls: Whether to show UI controls
    """
    # Get attributes
    attrs = get_pass_data_attributes(pass_data)
    slope_series = attrs.get("slope_series")
    time_array = attrs.get("time_array")
    strait_name = attrs.get("strait_name", "Unknown")
    pass_number = attrs.get("pass_number", 0)
    data_source = attrs.get("data_source", DATASET_NAMES.get(dataset_key, dataset_key))
    
    if slope_series is None:
        st.error("No slope_series in PassData.")
        return
    
    # Check for valid data
    valid_mask = ~np.isnan(slope_series)
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        st.warning("All slope values are NaN. The data may not intersect the gate.")
        return
    
    # Build time axis
    if time_array is not None and len(time_array) > 0:
        x_vals = time_array
        x_label = "Date"
    else:
        x_vals = np.arange(len(slope_series))
        x_label = "Index"
    
    # Controls
    show_trend = True
    unit = "m/100km"
    
    if show_controls:
        col1, col2 = st.columns([2, 1])
        with col1:
            show_trend = st.checkbox("Show trend line", value=True, key=f"slope_trend_{dataset_key}")
        with col2:
            unit = st.selectbox("Units", ["m/100km", "cm/km"], key=f"slope_unit_{dataset_key}")
    
    # Convert units
    if unit == "cm/km":
        y_vals = slope_series * 100
        y_label = "Slope (cm/km)"
    else:
        y_vals = slope_series
        y_label = "Slope (m/100km)"
    
    # Get color
    color = DATASET_COLORS.get(dataset_key, "steelblue")
    
    # Create figure
    fig = go.Figure()
    
    # Plot only valid values
    valid_x = [x_vals[i] for i in range(len(x_vals)) if valid_mask[i]]
    valid_y = [y_vals[i] for i in range(len(y_vals)) if valid_mask[i]]
    
    fig.add_trace(go.Scatter(
        x=valid_x,
        y=valid_y,
        mode="markers+lines",
        name="SSH Slope",
        marker=dict(size=6, color=color),
        line=dict(width=1, color=color)
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.8)
    
    # Trend line
    if show_trend and len(valid_y) > 2:
        x_numeric = np.arange(len(valid_y))
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_numeric, valid_y)
        p = np.poly1d([slope, intercept])
        r_squared = r_value ** 2
        fig.add_trace(go.Scatter(
            x=valid_x,
            y=p(x_numeric),
            mode="lines",
            name=f"Trend (R²={r_squared:.3f})",
            line=dict(dash="dash", color="red", width=2)
        ))
    
    # Title
    if pass_number:
        title = f"{strait_name} - Pass {pass_number} - {data_source}"
    else:
        title = f"{strait_name} - {data_source}"
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    st.plotly_chart(fig, use_container_width=True, key="slope_chart_single")
    
    # Statistics
    with st.expander("Statistics"):
        valid_slopes = slope_series[valid_mask]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{np.mean(valid_slopes):.4f}")
        with col2:
            st.metric("Std Dev", f"{np.std(valid_slopes):.4f}")
        with col3:
            st.metric("Valid Points", f"{n_valid}/{len(slope_series)}")
        with col4:
            if pass_number:
                st.metric("Pass", pass_number)
            else:
                st.metric("Source", data_source[:8])


def render_multi_slope(
    datasets: Dict[str, Any],
    config: Optional[Any] = None,
) -> None:
    """
    Render slope comparison across multiple datasets.
    
    Args:
        datasets: Dict mapping dataset_key to PassData
        config: AppConfig object
    """
    if not datasets:
        st.warning("No datasets loaded for comparison.")
        return
    
    # Controls
    col1, col2 = st.columns([2, 1])
    with col1:
        show_trend = st.checkbox("Show trend lines", value=False, key="multi_slope_trend")
    with col2:
        unit = st.selectbox("Units", ["m/100km", "cm/km"], key="multi_slope_unit")
    
    fig = go.Figure()
    
    stats_data = []
    
    for dataset_key, pass_data in datasets.items():
        attrs = get_pass_data_attributes(pass_data)
        slope_series = attrs.get("slope_series")
        time_array = attrs.get("time_array")
        
        if slope_series is None:
            continue
        
        valid_mask = ~np.isnan(slope_series)
        if not np.any(valid_mask):
            continue
        
        # Build time axis
        if time_array is not None and len(time_array) > 0:
            x_vals = time_array
        else:
            x_vals = np.arange(len(slope_series))
        
        # Convert units
        if unit == "cm/km":
            y_vals = slope_series * 100
        else:
            y_vals = slope_series
        
        # Get color and name
        color = DATASET_COLORS.get(dataset_key, "gray")
        name = DATASET_NAMES.get(dataset_key, dataset_key)
        
        # Filter valid
        valid_x = [x_vals[i] for i in range(len(x_vals)) if valid_mask[i]]
        valid_y = [y_vals[i] for i in range(len(y_vals)) if valid_mask[i]]
        
        fig.add_trace(go.Scatter(
            x=valid_x,
            y=valid_y,
            mode="markers+lines",
            name=name,
            marker=dict(size=5, color=color),
            line=dict(width=1.5, color=color)
        ))
        
        # Trend
        if show_trend and len(valid_y) > 2:
            x_numeric = np.arange(len(valid_y))
            slope, intercept, r_value, _, _ = scipy_stats.linregress(x_numeric, valid_y)
            p = np.poly1d([slope, intercept])
            fig.add_trace(go.Scatter(
                x=valid_x,
                y=p(x_numeric),
                mode="lines",
                name=f"{name} trend",
                line=dict(dash="dash", color=color, width=1),
                showlegend=False
            ))
        
        # Stats
        valid_slopes = slope_series[valid_mask]
        stats_data.append({
            "Dataset": name,
            "Mean": f"{np.mean(valid_slopes):.4f}",
            "Std": f"{np.std(valid_slopes):.4f}",
            "N": np.sum(valid_mask)
        })
    
    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.8)
    
    y_label = "Slope (cm/km)" if unit == "cm/km" else "Slope (m/100km)"
    
    fig.update_layout(
        title="SSH Slope Comparison",
        xaxis_title="Date",
        yaxis_title=y_label,
        height=500,
        
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    st.plotly_chart(fig, use_container_width=True, key="slope_chart_comparison")
    
    # Stats table
    if stats_data:
        with st.expander("Comparison Statistics"):
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
