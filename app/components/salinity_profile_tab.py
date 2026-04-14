"""
Salinity Profile Tab — CCI SSS v5.5
================================================================================
Displays monthly climatology profiles of Sea Surface Salinity along the gate.

Dataset: ESA CCI SSS v5.5 (SMOS satellite, ~2010–2023)
    /straits/netcdf/{gate}_SSS_CCIv5.5.nc

Shape: (time, nb_prof) — monthly surface salinity along gate profile
Plot: 4×3 grid (Jan–Dec), each subplot shows mean ± std along-gate profile.

This tab is INDEPENDENT of velocity data — it only needs the gate name.
"""

import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

SALINITY_NC_DIR = Path("/Users/nicolocaron/Desktop/ARCFRESH/straits/netcdf")

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# CCI SSS v5.5 file map (gate name → filename)
SSS_CCI_FILE_MAP = {
    'fram strait': 'fram_strait_S3_pass_481_SSS_CCIv5.5.nc',
    'davis strait': 'davis_strait_SSS_CCIv5.5.nc',
    'denmark strait': 'denmark_strait_TPJ_pass_246_SSS_CCIv5.5.nc',
    'bering strait': 'bering_strait_TPJ_pass_076_SSS_CCIv5.5.nc',
    'barents sea opening': 'barents_sea_opening_S3_pass_481_SSS_CCIv5.5.nc',
    'barents opening': 'barents_sea_opening_S3_pass_481_SSS_CCIv5.5.nc',
    'norwegian sea boundary': 'norwegian_sea_boundary_TPJ_pass_220_SSS_CCIv5.5.nc',
    'nares strait': 'nares_strait_SSS_CCIv5.5.nc',
}

# Colors
_COLOR_MEAN = '#1565C0'      # Dark blue — mean line
_COLOR_STD = '#2196F3'       # Light blue — std band
_COLOR_HMEAN = '#E53935'     # Red — horizontal mean line


def render_salinity_profile_tab(data: Any, config: Any, ds_info: dict):
    """
    Salinity Profile Tab — CCI SSS v5.5 monthly climatology along gate.
    
    Independent of velocity data. Only requires gate name from `data`.
    """
    st.header("🧪 Salinity Profile — CCI SSS v5.5")
    
    # =========================================================================
    # 0. BANNER — CCI temporal coverage warning
    # =========================================================================
    st.warning(
        "⚠️ **CCI SSS v5.5 covers ~2010–2023 only** (ESA SMOS satellite). "
        "Data outside this range is not available regardless of the CMEMS time window selected. "
        "Months with sea-ice coverage (e.g. winter in Nares Strait) may have no valid data."
    )
    
    # =========================================================================
    # 1. GET GATE NAME
    # =========================================================================
    strait_name = getattr(data, 'strait_name', None)
    if strait_name is None:
        strait_name = getattr(data, 'gate_name', 'Unknown')
    
    # =========================================================================
    # 2. LOAD CCI SSS DATA
    # =========================================================================
    cci_result = _load_cci_sss(strait_name)
    
    if cci_result is None:
        st.error(f"❌ No CCI SSS v5.5 data available for gate: **{strait_name}**")
        st.info(f"Expected file in: `{SALINITY_NC_DIR}`")
        return
    
    sss = cci_result['sss']          # (time, nb_prof)
    dates = cci_result['dates']      # DatetimeIndex
    lon = cci_result['lon']          # (nb_prof,)
    lat = cci_result['lat']          # (nb_prof,)
    
    # Along-gate distance (km)
    dist_km = _compute_along_gate_distance(lon, lat)
    
    # =========================================================================
    # 3. INFO METRICS
    # =========================================================================
    n_time, n_prof = sss.shape
    valid_pct = (~np.isnan(sss)).sum() / sss.size * 100
    
    st.success(
        f"✅ **CCI SSS v5.5** loaded: {n_prof} profile points × {n_time} months "
        f"({dates[0].strftime('%Y-%m')} → {dates[-1].strftime('%Y-%m')})"
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gate", strait_name)
    with col2:
        st.metric("Mean SSS", f"{np.nanmean(sss):.2f} PSU")
    with col3:
        st.metric("Coverage", f"{valid_pct:.0f}%")
    with col4:
        st.metric("Months", f"{n_time}")
    
    # =========================================================================
    # 4. PLOT 4×3 MONTHLY CLIMATOLOGY (Plotly)
    # =========================================================================
    st.subheader("📊 Monthly Mean SSS Along Gate (4×3)")
    
    fig = _create_monthly_4x3_plotly(sss, dates, dist_km, strait_name)
    
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"sal_prof_{strait_name.replace(' ', '_').lower()}_4x3_chart")
        
        # Export PNG
        try:
            img_bytes = fig.to_image(format="png", width=1400, height=1200, scale=2)
            safe_name = strait_name.replace(' ', '_').lower()
            st.download_button(
                "📥 Download 4×3 Salinity Profile PNG",
                img_bytes,
                f"{safe_name}_cci_sss_monthly_profile_4x3.png",
                "image/png",
                key=f"sal_prof_{safe_name}_4x3_png",
            )
        except Exception:
            st.info("💡 Install **kaleido** (`pip install kaleido`) for PNG export.")
    
    # =========================================================================
    # 5. OVERALL TIME SERIES (monthly gate-mean SSS)
    # =========================================================================
    st.subheader("📈 Gate-Mean SSS Time Series")
    
    gate_mean_ts = np.nanmean(sss, axis=1)  # (n_time,)
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=dates,
        y=gate_mean_ts,
        mode='lines+markers',
        name='Gate-mean SSS',
        line=dict(color=_COLOR_MEAN, width=2),
        marker=dict(size=3),
    ))
    fig_ts.add_hline(
        y=np.nanmean(gate_mean_ts),
        line_dash="dash", line_color=_COLOR_HMEAN, line_width=1,
        annotation_text=f"μ = {np.nanmean(gate_mean_ts):.2f} PSU",
        annotation_position="top left",
    )
    fig_ts.update_layout(
        title=f"CCI SSS v5.5 — {strait_name} — Monthly Gate-Mean Surface Salinity",
        xaxis_title="Time",
        yaxis_title="SSS (PSU)",
        height=400,
        
        xaxis=dict(gridcolor='#eee'),
        yaxis=dict(gridcolor='#eee'),
    )
    st.plotly_chart(fig_ts, use_container_width=True, key=f"sal_prof_{strait_name.replace(' ', '_').lower()}_ts_chart")
    
    # =========================================================================
    # 7. EXPORT CSV
    # =========================================================================
    st.subheader("📥 Export Data")
    
    # Build export dataframe: each row = one month, columns = profile points
    export_rows = []
    for t in range(n_time):
        row = {
            'date': dates[t].strftime('%Y-%m'),
            'year': dates[t].year,
            'month': dates[t].month,
            'gate_mean_sss': np.nanmean(sss[t, :]),
            'valid_pct': (~np.isnan(sss[t, :])).sum() / n_prof * 100,
        }
        export_rows.append(row)
    
    export_df = pd.DataFrame(export_rows)
    csv = export_df.to_csv(index=False)
    
    safe_name = strait_name.replace(' ', '_').lower()
    st.download_button(
        "📥 Download CCI SSS Time Series CSV",
        csv,
        f"{safe_name}_cci_sss_v5.5_timeseries.csv",
        "text/csv",
        key=f"sal_prof_{safe_name}_csv",
    )
    
    st.info(
        f"**Dataset:** ESA CCI SSS v5.5 (SMOS satellite)\n\n"
        f"**Coverage:** {dates[0].strftime('%Y-%m')} → {dates[-1].strftime('%Y-%m')} "
        f"({n_time} months, {valid_pct:.0f}% non-NaN)\n\n"
        f"**Spatial:** {n_prof} profile points along gate, "
        f"distance = {dist_km[-1]:.0f} km\n\n"
        f"**Variable:** Sea Surface Salinity (SSS) in PSU"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _find_file(strait_name: str) -> Optional[str]:
    """Find CCI SSS filename by fuzzy-matching gate name."""
    strait_lower = strait_name.lower().strip().replace('_', ' ')
    filename = SSS_CCI_FILE_MAP.get(strait_lower)
    if filename is None:
        for key, fname in SSS_CCI_FILE_MAP.items():
            if key in strait_lower or strait_lower in key:
                filename = fname
                break
    return filename


def _load_cci_sss(strait_name: str) -> Optional[dict]:
    """Load CCI SSS v5.5 data for a gate. Returns None if not available."""
    cci_file = _find_file(strait_name)
    if cci_file is None:
        logger.warning(f"No CCI SSS file mapping for gate: {strait_name}")
        return None
    
    filepath = SALINITY_NC_DIR / cci_file
    if not filepath.exists():
        logger.warning(f"CCI SSS file not found: {filepath}")
        return None
    
    try:
        ds = xr.open_dataset(filepath)
        sss = ds['sss'].values            # (time, nb_prof)
        dates = pd.to_datetime(ds['date'].values)
        lon = ds['longitude'].values       # (nb_prof,)
        lat = ds['latitude'].values        # (nb_prof,)
        ds.close()
        
        valid_pct = (~np.isnan(sss)).sum() / sss.size * 100
        if valid_pct < 1.0:
            logger.warning(f"CCI SSS for {strait_name} has only {valid_pct:.1f}% valid data")
            return None
        
        logger.info(f"Loaded CCI SSS for {strait_name}: {sss.shape}, {valid_pct:.1f}% valid")
        return {
            'sss': sss,
            'dates': dates,
            'lon': lon.astype(float),
            'lat': lat.astype(float),
        }
    except Exception as e:
        logger.error(f"Error loading CCI SSS: {e}")
        return None


def _compute_along_gate_distance(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Compute cumulative along-gate distance in km."""
    dist = np.zeros(len(lon))
    for i in range(1, len(lon)):
        dlat = lat[i] - lat[i - 1]
        dlon = (lon[i] - lon[i - 1]) * np.cos(np.deg2rad((lat[i] + lat[i - 1]) / 2))
        dist[i] = dist[i - 1] + np.sqrt(dlat**2 + dlon**2) * 111.32
    return dist


def _create_monthly_4x3_plotly(
    sss: np.ndarray,
    dates: pd.DatetimeIndex,
    dist_km: np.ndarray,
    strait_name: str,
) -> Optional[go.Figure]:
    """
    Create a 4×3 Plotly subplot figure with monthly mean SSS profiles.
    
    Each subplot: mean ± std along-gate profile for one month,
    averaged over all available years.
    """
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=[f"{MONTH_NAMES[m]}" for m in range(12)],
        vertical_spacing=0.06,
        horizontal_spacing=0.05,
    )
    
    # Compute global y-range
    all_means = []
    for m in range(1, 13):
        mask = dates.month == m
        if mask.sum() > 0:
            monthly = np.nanmean(sss[mask, :], axis=0)
            all_means.append(monthly)
    
    if not all_means:
        return None
    
    all_vals = np.concatenate(all_means)
    ymin = np.nanmin(all_vals) - 0.3
    ymax = np.nanmax(all_vals) + 0.3
    if np.isnan(ymin) or np.isnan(ymax):
        ymin, ymax = 30, 36
    
    for m in range(1, 13):
        row = (m - 1) // 3 + 1
        col = (m - 1) % 3 + 1
        
        mask = dates.month == m
        n_months = mask.sum()
        
        if n_months == 0:
            # Empty subplot — add invisible trace
            fig.add_trace(
                go.Scatter(x=[0], y=[0], mode='markers', marker=dict(opacity=0),
                           showlegend=False),
                row=row, col=col,
            )
            fig.layout.annotations[m - 1].text = f"{MONTH_NAMES[m - 1]} — No data"
            continue
        
        monthly_sss = sss[mask, :]  # (n_years, nb_prof)
        mean_profile = np.nanmean(monthly_sss, axis=0)
        std_profile = np.nanstd(monthly_sss, axis=0)
        valid_pct = (~np.isnan(monthly_sss)).sum() / monthly_sss.size * 100
        gate_mean = np.nanmean(mean_profile)
        
        # Update subtitle
        fig.layout.annotations[m - 1].text = (
            f"{MONTH_NAMES[m - 1]} (n={n_months}, {valid_pct:.0f}% valid)"
        )
        
        # Std band (upper)
        fig.add_trace(
            go.Scatter(
                x=dist_km, y=mean_profile + std_profile,
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip',
            ),
            row=row, col=col,
        )
        # Std band (lower, filled to upper)
        fig.add_trace(
            go.Scatter(
                x=dist_km, y=mean_profile - std_profile,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(33, 150, 243, 0.2)',
                showlegend=False, hoverinfo='skip',
            ),
            row=row, col=col,
        )
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=dist_km, y=mean_profile,
                mode='lines', line=dict(color=_COLOR_MEAN, width=2),
                showlegend=False,
                hovertemplate='Dist: %{x:.1f} km<br>SSS: %{y:.2f} PSU<extra></extra>',
            ),
            row=row, col=col,
        )
        # Horizontal mean
        fig.add_hline(
            y=gate_mean, row=row, col=col,
            line_dash="dash", line_color=_COLOR_HMEAN, line_width=0.8,
            annotation_text=f"μ={gate_mean:.2f}",
            annotation_font_size=9,
            annotation_position="bottom right",
        )
        
        # Set axis range
        fig.update_yaxes(range=[ymin, ymax], row=row, col=col, gridcolor='#eee')
        fig.update_xaxes(gridcolor='#eee', row=row, col=col)
    
    # Y-axis labels (leftmost column)
    for r in range(1, 5):
        fig.update_yaxes(title_text="SSS (PSU)", title_font_size=9, row=r, col=1)
    # X-axis labels (bottom row)
    for c in range(1, 4):
        fig.update_xaxes(title_text="Distance (km)", title_font_size=9, row=4, col=c)
    
    fig.update_layout(
        title=dict(
            text=(
                f"CCI SSS v5.5 — {strait_name} — Monthly Mean Surface Salinity<br>"
                f"<sub>{dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}, "
                f"{len(dist_km)} points, {len(dates)} months</sub>"
            ),
            font_size=15,
        ),
        height=900,
        width=1200,
        
        showlegend=False,
    )
    
    return fig
