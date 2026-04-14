"""
Shared Transport Plotting Utilities
================================================================================
Contains the along-gate transport bar chart used by both:
  - salt_flux_clean.py (Salt Flux tab)
  - freshwater_transport_clean.py (Freshwater Transport tab)

Style: bar chart with blue (positive/into Arctic) and red (negative/out of Arctic)
bars, distance along gate on x-axis. White background, black text.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

# Constants used in transport formulas
RHO_SEAWATER = 1025.0   # kg/m³
S_REF_DEFAULT = 34.8     # PSU

_MONTH_NAMES = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
]
_MONTH_NAMES_SHORT = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
]

# Colors
_COLOR_POS = '#4A90D9'      # blue  — into Arctic
_COLOR_NEG = '#D94A4A'      # red   — out of Arctic
_EDGE_POS  = '#2C5F8A'
_EDGE_NEG  = '#8A2C2C'

# Shared white-background layout defaults
_LAYOUT_BASE = dict(
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(family='Inter, sans-serif', color='#111111'),
)


def plot_transport_along_gate(
    v_perp: np.ndarray,
    x_km: Optional[np.ndarray],
    time_array,
    H_profile: np.ndarray,
    sss_interp: np.ndarray,
    dx: np.ndarray,
    strait_name: str,
    key_prefix: str,
    mode: str = 'salt',
    s_ref: float = S_REF_DEFAULT,
):
    """Along-gate transport bar chart for a selected month.

    Blue bars = positive (into Arctic), red bars = negative (out of Arctic).

    Args:
        v_perp:     (n_points, n_time) perpendicular velocity [m/s]
        x_km:       (n_points,) distance along gate [km]
        time_array:  datetime-like array of length n_time
        H_profile:  (n_points,) effective depth at each gate point [m]
        sss_interp: (n_points, n_time) depth-averaged salinity [PSU]
        dx:         (n_points,) segment widths [m]
        strait_name: gate display name
        key_prefix:  Streamlit widget key prefix
        mode:       'salt'  → Sm = ρ × (S̄/1000) × v × H × Δx  [kg/s]
                    'freshwater' → Fw = v × (1 − S̄/S_A) × H × Δx  [m³/s]
        s_ref:      reference salinity (freshwater mode only)
    """
    selected_month = st.selectbox(
        "Select month",
        options=list(range(1, 13)),
        format_func=lambda m: _MONTH_NAMES[m - 1],
        index=0,
        key=f"{key_prefix}_along_month",
    )

    # --- Parse time & filter to selected month ---
    try:
        time_pd = pd.to_datetime(np.asarray(time_array).ravel())
    except Exception:
        st.warning("Cannot parse time array for monthly grouping")
        return

    n_points = v_perp.shape[0]
    month_mask = time_pd.month == selected_month
    month_indices = np.where(month_mask)[0]
    n_month = len(month_indices)

    if n_month == 0:
        st.warning(f"No data for {_MONTH_NAMES[selected_month - 1]}")
        return

    # Distance axis
    dist = x_km if (x_km is not None and len(x_km) == n_points) else np.arange(n_points)

    # --- Vectorised per-point transport for every timestep of this month ---
    # v_sel, s_sel: (n_points, n_month)
    v_sel = v_perp[:, month_indices] if v_perp.ndim > 1 else np.tile(v_perp[:, None], (1, n_month))
    s_sel = sss_interp[:, month_indices] if sss_interp.ndim > 1 else np.tile(sss_interp[:, None], (1, n_month))

    if mode == 'salt':
        transport = RHO_SEAWATER * (s_sel / 1000.0) * v_sel * H_profile[:, None] * dx[:, None]
    else:
        transport = v_sel * (1.0 - s_sel / s_ref) * H_profile[:, None] * dx[:, None]

    mean_t = np.nanmean(transport, axis=1)
    total  = np.nansum(mean_t)

    # --- Auto-scale units ---
    scale, unit_display, total_str = _auto_scale(mean_t, total, mode)

    mean_s = mean_t / scale

    # --- Build figure ---
    colors = [_COLOR_POS if v >= 0 else _COLOR_NEG for v in mean_s]
    edges  = [_EDGE_POS  if v >= 0 else _EDGE_NEG  for v in mean_s]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dist,
        y=mean_s,
        marker_color=colors,
        marker_line_color=edges,
        marker_line_width=0.5,
        hovertemplate=(
            'Distance: %{x:.1f} km<br>'
            f'Transport: %{{y:.4f}} {unit_display}<br>'
            '<extra></extra>'
        ),
        showlegend=False,
    ))

    fig.add_hline(y=0, line_color='#999999', line_width=1)

    fig.update_layout(
        title=dict(
            text=f"Transport Along Gate — {_MONTH_NAMES[selected_month - 1]}",
            font=dict(size=18, family='Inter, sans-serif'),
        ),
        xaxis_title="Distance along gate (km)",
        yaxis_title=f"Transport ({unit_display})",
        height=480,
        
        
        font=dict(family='Inter, sans-serif', size=13),
        xaxis=dict(gridcolor='#E0E0E0', gridwidth=1),
        yaxis=dict(gridcolor='#E0E0E0', gridwidth=1),
        margin=dict(l=70, r=30, t=80, b=60),
        annotations=[
            dict(
                text=f"{_MONTH_NAMES[selected_month - 1]} Total: {total_str}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=12, color='#333'),
                bgcolor='white', bordercolor='#999',
                borderwidth=1, borderpad=4,
            )
        ],
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_along_gate_chart")
    st.caption(
        f"Blue = positive (into Arctic) | Red = negative (out of Arctic) | "
        f"Monthly mean over {n_month} years | Sign convention: physics-based dot product (+ = into Arctic)"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auto_scale(mean_transport: np.ndarray, total: float, mode: str):
    """Choose a human-friendly scale and return (scale, unit_display, total_str)."""
    max_abs = np.nanmax(np.abs(mean_transport)) if len(mean_transport) > 0 else 1.0

    if mode == 'salt':
        base_unit = "kg/s"
        if max_abs > 1e6:
            scale, unit_display = 1e6, "×10⁶ kg/s"
        elif max_abs > 1e3:
            scale, unit_display = 1e3, "×10³ kg/s"
        else:
            scale, unit_display = 1.0, "kg/s"
        total_str = f"{total:.3e} kg/s ({total / 1e9:.3f} Gg/s)"
    else:
        base_unit = "m³/s"
        if max_abs > 1e6:
            scale, unit_display = 1e6, "×10⁶ m³/s"
        elif max_abs > 1e3:
            scale, unit_display = 1e3, "×10³ m³/s"
        else:
            scale, unit_display = 1.0, "m³/s"
        total_str = f"{total:.3e} m³/s ({total / 1e3:.3f} mSv)"

    return scale, unit_display, total_str


def plot_transport_along_gate_4x3(
    v_perp: np.ndarray,
    x_km: Optional[np.ndarray],
    time_array,
    H_profile: np.ndarray,
    sss_interp: np.ndarray,
    dx: np.ndarray,
    strait_name: str,
    mode: str = 'salt',
    s_ref: float = S_REF_DEFAULT,
    metadata: Optional[dict] = None,
) -> Optional[go.Figure]:
    """Along-gate transport bar chart as 4×3 grid (all 12 months).

    Returns a plotly Figure (does NOT call st.plotly_chart).

    Args:
        v_perp:     (n_points, n_time) perpendicular velocity [m/s]
        x_km:       (n_points,) distance along gate [km]
        time_array:  datetime-like array of length n_time
        H_profile:  (n_points,) effective depth at each gate point [m]
        sss_interp: (n_points, n_time) depth-averaged salinity [PSU]
        dx:         (n_points,) segment widths [m]
        strait_name: gate display name
        mode:       'salt' or 'freshwater'
        s_ref:      reference salinity (freshwater mode only)
        metadata:   optional dict with keys like 'lon_range', 'lat_range',
                    'gate_length_km', 'date_start', 'date_end', 'vel_source',
                    'sal_source', 'bathy_source', 'depth_cap', 'rho', 'sss_mean'
    """
    try:
        time_pd = pd.to_datetime(np.asarray(time_array).ravel())
    except Exception:
        return None

    n_points = v_perp.shape[0]
    dist = x_km if (x_km is not None and len(x_km) == n_points) else np.arange(n_points)

    # --- Pre-compute transport for all 12 months ---
    month_data = {}  # month(1-12) -> (mean_transport, n_years, total)
    global_max = 0.0

    for m in range(1, 13):
        month_mask = time_pd.month == m
        month_indices = np.where(month_mask)[0]
        n_m = len(month_indices)

        if n_m == 0:
            month_data[m] = (np.zeros(n_points), 0, 0.0)
            continue

        v_sel = v_perp[:, month_indices] if v_perp.ndim > 1 else np.tile(v_perp[:, None], (1, n_m))
        s_sel = sss_interp[:, month_indices] if sss_interp.ndim > 1 else np.tile(sss_interp[:, None], (1, n_m))

        if mode == 'salt':
            transport = RHO_SEAWATER * (s_sel / 1000.0) * v_sel * H_profile[:, None] * dx[:, None]
        else:
            transport = v_sel * (1.0 - s_sel / s_ref) * H_profile[:, None] * dx[:, None]

        mean_t = np.nanmean(transport, axis=1)
        total = np.nansum(mean_t)
        month_data[m] = (mean_t, n_m, total)

        local_max = np.nanmax(np.abs(mean_t))
        if local_max > global_max:
            global_max = local_max

    # --- Choose a common scale across all months ---
    if mode == 'salt':
        if global_max > 1e6:
            scale, unit_display = 1e6, "×10⁶ kg/s"
        elif global_max > 1e3:
            scale, unit_display = 1e3, "×10³ kg/s"
        else:
            scale, unit_display = 1.0, "kg/s"
    else:
        if global_max > 1e6:
            scale, unit_display = 1e6, "×10⁶ m³/s"
        elif global_max > 1e3:
            scale, unit_display = 1e3, "×10³ m³/s"
        else:
            scale, unit_display = 1.0, "m³/s"

    y_lim = global_max / scale * 1.15  # symmetric y-axis with 15% margin

    # --- Build 4×3 subplot figure ---
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=[_MONTH_NAMES[m] for m in range(12)],
        vertical_spacing=0.06,
        horizontal_spacing=0.06,
    )

    for m in range(1, 13):
        row = (m - 1) // 3 + 1
        col = (m - 1) % 3 + 1

        mean_t, n_m, total = month_data[m]
        mean_s = mean_t / scale

        colors = [_COLOR_POS if v >= 0 else _COLOR_NEG for v in mean_s]
        edges = [_EDGE_POS if v >= 0 else _EDGE_NEG for v in mean_s]

        # Format total
        if mode == 'salt':
            total_label = f"{total / 1e9:.3f} Gg/s"
        else:
            total_label = f"{total / 1e3:.3f} mSv"

        fig.add_trace(
            go.Bar(
                x=dist,
                y=mean_s,
                marker_color=colors,
                marker_line_color=edges,
                marker_line_width=0.3,
                showlegend=False,
                hovertemplate=(
                    'Dist: %{x:.1f} km<br>'
                    f'Transport: %{{y:.4f}} {unit_display}<br>'
                    '<extra></extra>'
                ),
            ),
            row=row, col=col,
        )

        # Zero line
        fig.add_hline(y=0, line_color='#999999', line_width=0.5, row=row, col=col)

        # Total annotation inside subplot
        fig.add_annotation(
            text=f"Σ = {total_label}",
            xref=f"x{m}" if m > 1 else "x",
            yref=f"y{m}" if m > 1 else "y",
            x=0.02, xanchor='left',
            y=y_lim * 0.9, yanchor='top',
            showarrow=False,
            font=dict(size=9, color='#333'),
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='#999', borderwidth=0.5, borderpad=2,
        )

        # Symmetric y-axis
        fig.update_yaxes(range=[-y_lim, y_lim], row=row, col=col)

    mode_label = "Salt Flux" if mode == 'salt' else "Freshwater Transport"

    # Build descriptive title with metadata
    if metadata:
        _m = metadata
        _subtitle_parts = []
        if _m.get('lon_range') and _m.get('lat_range'):
            _subtitle_parts.append(f"Gate: {_m['lon_range']}, {_m['lat_range']}")
        if _m.get('gate_length_km'):
            _subtitle_parts[-1] += f" ({_m['gate_length_km']} km)"
        if _m.get('date_start') and _m.get('date_end'):
            _subtitle_parts.append(f"Period: {_m['date_start']} to {_m['date_end']}")
        if mode == 'salt':
            formula_parts = [f"Sm = Σ ρ·(S̄/1000)·v⊥·H(x)·Δx"]
            if _m.get('rho'):
                formula_parts.append(f"ρ = {_m['rho']:.0f} kg/m³")
        else:
            formula_parts = [f"Fw = Σ v⊥·(1−S̄/S_A)·H(x)·Δx"]
            formula_parts.append(f"S_A = {s_ref} PSU")
        if _m.get('depth_cap'):
            formula_parts.append(f"H cap = {_m['depth_cap']} m")
        if _m.get('sss_mean'):
            formula_parts.append(f"mean S̄ = {_m['sss_mean']:.2f} PSU")
        _subtitle_parts.append(" | ".join(formula_parts))
        source_parts = []
        if _m.get('vel_source'):
            source_parts.append(f"Velocity: {_m['vel_source']}")
        if _m.get('sal_source'):
            source_parts.append(f"Salinity: {_m['sal_source']}")
        if _m.get('bathy_source'):
            source_parts.append(f"Bathymetry: {_m['bathy_source']}")
        if source_parts:
            _subtitle_parts.append(" · ".join(source_parts))
        _subtitle = "<br>".join(_subtitle_parts)
        _full_title = f"{mode_label} Along Gate — {strait_name} (Monthly Climatology)"
        _subtitle_annotation = _subtitle
    else:
        _full_title = f"{mode_label} Along Gate — {strait_name} (Monthly Climatology)"
        _subtitle_annotation = None

    fig.update_layout(
        title=dict(
            text=_full_title,
            font=dict(size=15, family='Inter, sans-serif'),
        ),
        height=1200,
        width=1400,
        
        
        font=dict(family='Inter, sans-serif', size=11),
        margin=dict(l=60, r=30, t=130, b=40),
    )

    if _subtitle_annotation:
        fig.add_annotation(
            text=_subtitle_annotation, xref="paper", yref="paper",
            x=0, y=1.06, showarrow=False, font=dict(size=10, color="#555"),
            xanchor="left", yanchor="top",
        )

    # Common axis labels (only left column y-axis, bottom row x-axis)
    for row in range(1, 5):
        fig.update_yaxes(title_text=f"({unit_display})" if row == 2 else "",
                         row=row, col=1)
    for col in range(1, 4):
        fig.update_xaxes(title_text="Distance (km)" if col == 2 else "",
                         row=4, col=col)

    return fig
