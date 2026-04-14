"""
DOT Profiles Tab
================
Compare DOT profiles across cycles.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from .sidebar import AppConfig
from src.analysis.slope import bin_by_longitude


def render_profiles_tab(datasets: list, cycle_info: list, config: AppConfig):
    """Render DOT profiles comparison tab."""
    
    st.subheader("🌊 DOT Profiles Comparison")
    
    # Compute profiles for all cycles
    profiles = []
    
    for i, ds in enumerate(datasets):
        cycle_num = cycle_info[i]["cycle"] if i < len(cycle_info) else i + 1
        
        profile = _compute_profile(ds, config, cycle_num)
        if profile:
            profiles.append(profile)
    
    if not profiles:
        st.warning("No profiles available.")
        return
    
    # Cycle selector
    available_cycles = [p["cycle"] for p in profiles]
    selected_cycles = st.multiselect(
        "Select Cycles to Compare",
        available_cycles,
        default=available_cycles[:min(5, len(available_cycles))],
    )
    
    if not selected_cycles:
        st.info("Select at least one cycle to display.")
        return
    
    # Filter selected profiles
    selected_profiles = [p for p in profiles if p["cycle"] in selected_cycles]
    
    # Create plot
    fig = _create_profile_plot(selected_profiles)
    st.plotly_chart(fig, use_container_width=True, key="profiles_tab_chart")
    
    # Statistics table
    st.subheader("📊 Profile Statistics")
    
    stats_data = []
    for p in selected_profiles:
        dot_mean = p["dot"]
        stats_data.append({
            "Cycle": p["cycle"],
            "DOT Mean (m)": f"{np.nanmean(dot_mean):.4f}",
            "DOT Std (m)": f"{np.nanstd(dot_mean):.4f}",
            "DOT Range (m)": f"{np.nanmax(dot_mean) - np.nanmin(dot_mean):.4f}",
            "Lon Range": f"{p['lon'].min():.2f}° - {p['lon'].max():.2f}°",
        })
    
    st.dataframe(stats_data, use_container_width=True)


def _compute_profile(ds, config: AppConfig, cycle_num: int) -> dict | None:
    """Compute DOT profile for a single cycle."""
    
    if "corssh" not in ds.data_vars:
        return None
    if config.mss_var not in ds.data_vars:
        return None
    
    # Compute DOT
    dot = ds["corssh"] - ds[config.mss_var]
    
    lon = ds["longitude"].values.flatten()
    dot_vals = dot.values.flatten()
    
    # Bin by longitude
    bin_centers, bin_means, _, _ = bin_by_longitude(lon, dot_vals, config.bin_size)
    
    if len(bin_centers) < 2:
        return None
    
    return {
        "cycle": cycle_num,
        "lon": bin_centers,
        "dot": bin_means,
    }


def _create_profile_plot(profiles: list) -> go.Figure:
    """Create multi-profile comparison plot."""
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set1
    
    for i, profile in enumerate(profiles):
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=profile["lon"],
            y=profile["dot"],
            mode="lines",
            name=f"Cycle {profile['cycle']}",
            line=dict(color=color, width=2),
        ))
    
    fig.update_layout(
        title="DOT Profiles by Longitude",
        xaxis_title="Longitude (°)",
        yaxis_title="DOT (m)",
        height=500,
        showlegend=True,
    )
    
    return fig
