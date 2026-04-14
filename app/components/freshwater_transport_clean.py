"""
Freshwater Transport Tab - VERSIONE DEFINITIVA
================================================================================
Calcola il Freshwater Transport:
    Fw = Σ v_perp(x,t) × (1 - SSS(x,t)/S_A) × H(x) × Δx  [m³/s]

Usa dati di salinità SUPERFICIALE da CCI SSS v5.5:
    /straits/netcdf/{gate}_SSS_CCIv5.5.nc

CCI SSS shape: (time, nb_prof) — monthly sea surface salinity
La salinità superficiale viene considerata costante lungo la profondità
(uniforme da 0 a H), interpolata spazio-temporalmente lungo il gate.

NO ISAS PSAL fallback — solo CCI SSS v5.5. Se CCI non è disponibile,
il freshwater transport non può essere calcolato.

Sign convention: positive v_perp = into Arctic (dot product with INTO-Arctic vector)

S_A = reference salinity (default 34.8 PSU, configurabile)
Unità: mSv (milli-Sverdup = 10³ m³/s)
"""

import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

# Constants
S_REF_DEFAULT = 34.8  # PSU - reference salinity for Arctic
SALINITY_NC_DIR = Path("/Users/nicolocaron/Desktop/ARCFRESH/straits/netcdf")

# Primary: CCI SSS v5.5 (monthly time series, surface only)
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

# NOTE: ISAS PSAL fallback removed — CCI SSS v5.5 only


def render_freshwater_transport_clean(data: Any, config: Any, ds_info: dict):
    """
    Freshwater Transport Tab.

    Formula: Fw = Σ v_perp(x,t) × (1 - SSS(x,t)/S_A) × H(x) × Δx  [m³/s]
    
    Uses CCI SSS v5.5 (surface salinity, constant along depth).
    Sign convention: positive v_perp = into Arctic (dot product with INTO-Arctic vector).
    """
    st.header("🌊 Freshwater Transport")

    strait_name = getattr(data, 'strait_name', 'Unknown')

    # =========================================================================
    # 1. CHECK VELOCITY DATA + APPLY SIGN FLIP
    # =========================================================================
    v_perp = _get_velocity_data(data)
    _time = getattr(data, 'time_array', None)
    if _time is None:
        _time = getattr(data, 'time', None)
    time_array = _time
    x_km = getattr(data, 'x_km', None)
    gate_lon = getattr(data, 'gate_lon_pts', None)
    gate_lat = getattr(data, 'gate_lat_pts', None)

    if v_perp is None:
        st.error("❌ No velocity data available!")
        st.info("💡 Go to **Geostrophic Velocity** tab first and compute velocities.")
        return

    # Sign flip is now applied centrally inside compute_perpendicular_velocity()
    # when gate_name is provided. _get_velocity_data passes strait_name.
    # No manual flip needed here.

    n_points = v_perp.shape[0]
    n_time = v_perp.shape[1] if v_perp.ndim > 1 else 1

    # =========================================================================
    # 2. LOAD CCI SSS v5.5 (no ISAS fallback)
    # =========================================================================
    salinity_result = _load_salinity(strait_name)

    if salinity_result is None:
        st.error(f"❌ No CCI SSS v5.5 data available for gate: **{strait_name}**")
        st.warning(
            "⚠️ This tab uses **only CCI SSS v5.5** (no ISAS PSAL fallback). "
            "CCI SSS covers ~2010–2023. If your gate has no CCI coverage, "
            "freshwater transport cannot be computed."
        )
        return

    sal_source = salinity_result['source']
    sss_data = salinity_result['sss']       # (time, nb_prof)
    sss_time = salinity_result['time']
    sss_lon = salinity_result['lon']
    sss_lat = salinity_result['lat']
    is_cci = salinity_result['is_cci']

    # =========================================================================
    # 3. INFO PANEL
    # =========================================================================
    st.success(f"✅ Velocity: {n_points} points × {n_time} timesteps (positive = into Arctic)")
    st.success(f"✅ Salinity: {sss_data.shape[1]} points × {sss_data.shape[0]} months — CCI SSS v5.5 (surface)")
    st.info(
        "⚠️ **CCI SSS v5.5 covers ~2010–2023 only.** "
        "Velocity timesteps outside CCI range will have NaN freshwater transport."
    )

    # --- Spatial coverage diagnostic ---
    _check_cci_gate_alignment(sss_lon, sss_lat, gate_lon, gate_lat, sss_data, strait_name)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gate", strait_name)
    with col2:
        st.metric("Mean SSS", f"{np.nanmean(sss_data):.2f} PSU")
    with col3:
        nan_pct = np.isnan(sss_data).sum() / sss_data.size * 100
        st.metric("SSS Coverage", f"{100 - nan_pct:.0f}%")
    with col4:
        st.metric("S_A (ref)", f"{S_REF_DEFAULT:.1f} PSU")

    # =========================================================================
    # 4. SETTINGS
    # =========================================================================
    st.subheader("⚙️ Settings")

    key_prefix = f"fw_{strait_name.replace(' ', '_').lower()}"

    col1, col2, col3 = st.columns(3)

    with col1:
        depth_cap = st.number_input(
            "Depth Cap H (m)",
            min_value=50, max_value=500, value=250, step=50,
            key=f"{key_prefix}_depth",
            help="Integration depth H for freshwater transport"
        )

    with col2:
        s_ref = st.number_input(
            "Reference Salinity S_A (PSU)",
            min_value=30.0, max_value=40.0, value=S_REF_DEFAULT, step=0.1,
            key=f"{key_prefix}_sref",
            help="Reference salinity for freshwater anomaly (1 − S/S_A)"
        )

    with col3:
        with st.expander("📊 SSS Profile"):
            _plot_sss_profile_cci(sss_data, sss_lon, sss_time, is_cci)

    st.info("**Formula:** Fw = Σ v_perp × (1 − SSS/S_A) × H × Δx  [m³/s]  — SSS = CCI surface salinity (uniform with depth)")

    # =========================================================================
    # 5. COMPUTE
    # =========================================================================
    if st.button("🧮 Compute Freshwater Transport", type="primary",
                 use_container_width=True, key=f"{key_prefix}_compute"):

        with st.spinner("Computing freshwater transport (loading bathymetry + SSS)..."):
            try:
                # --- Load GEBCO bathymetry ---
                from src.services.gebco_service import get_bathymetry_cache
                cache = get_bathymetry_cache()
                depth_profile_full = cache.get_or_compute(
                    gate_name=strait_name,
                    gate_lons=gate_lon,
                    gate_lats=gate_lat,
                    gebco_path=config.gebco_nc_path,
                    depth_cap=None
                )
                # H(x) = min(depth_cap, bathymetry(x))
                H_profile = np.minimum(depth_profile_full, depth_cap)

                st.info(f"🏔️ Bathymetry: mean={np.mean(depth_profile_full):.0f}m, "
                        f"H(x) capped at {depth_cap}m → effective mean H={np.mean(H_profile):.0f}m")

                # Safely convert time_array to numeric for matching
                vel_time_safe = _safe_time_values(time_array)

                # --- Interpolate SSS onto velocity grid (CCI only) ---
                sss_interp = _interpolate_sss_cci(
                    sss_data, sss_time, sss_lon, sss_lat,
                    vel_time_safe, gate_lon, gate_lat,
                    n_points, n_time
                )
                sal_label = "CCI SSS v5.5 (surface)"

                if sss_interp is None:
                    st.error("❌ Could not interpolate salinity to velocity grid")
                    return

                # Report NaN timesteps (no ISAS fallback — CCI only)
                nan_timesteps = np.all(np.isnan(sss_interp), axis=0)
                n_missing = int(nan_timesteps.sum())
                if n_missing > 0:
                    st.warning(
                        f"⚠️ {n_missing}/{n_time} velocity timesteps have no CCI SSS coverage "
                        f"(outside CCI range 2010–2023 or sea-ice gaps). "
                        f"Freshwater transport will be NaN for those timesteps."
                    )

                st.info(f"📍 SSS interpolated ({sal_label}): shape={sss_interp.shape}, mean={np.nanmean(sss_interp):.2f} PSU")

                dx = _compute_segment_widths(x_km, n_points, gate_lon=gate_lon, gate_lat=gate_lat)

                # Fw = Σ v_perp × (1 - S̄/S_A) × H(x) × Δx
                fw = np.zeros(n_time)
                for t in range(n_time):
                    v = v_perp[:, t] if v_perp.ndim > 1 else v_perp
                    s = sss_interp[:, t] if sss_interp.ndim > 1 else sss_interp
                    fw_val = v * (1.0 - s / s_ref) * H_profile * dx
                    if np.all(np.isnan(fw_val)):
                        fw[t] = np.nan  # Preserve NaN for missing timesteps
                    else:
                        fw[t] = np.nansum(fw_val)

                # Compute freshwater transport uncertainty if err_ugosa/err_vgosa available
                # σ_Fw(t) = √( Σᵢ (σ_v_perp(i,t) × |1 - S(i,t)/S_A| × H(i) × Δx(i))² )
                err_ugosa = getattr(data, 'err_ugosa_matrix', None)
                err_vgosa = getattr(data, 'err_vgosa_matrix', None)
                sigma_fw = None
                
                if err_ugosa is not None and err_vgosa is not None:
                    from src.services.transport_service import compute_perpendicular_velocity_uncertainty
                    sigma_v_perp = compute_perpendicular_velocity_uncertainty(
                        err_ugosa, err_vgosa, gate_lon, gate_lat, gate_name=strait_name
                    )
                    sigma_fw = np.zeros(n_time)
                    for t in range(n_time):
                        sv = sigma_v_perp[:, t] if sigma_v_perp.ndim > 1 else sigma_v_perp
                        s = sss_interp[:, t] if sss_interp.ndim > 1 else sss_interp
                        fw_factor = np.abs(1.0 - s / s_ref)
                        valid = np.isfinite(sv) & np.isfinite(fw_factor)
                        if np.any(valid):
                            sigma_fw[t] = np.sqrt(np.nansum(
                                (sv[valid] * fw_factor[valid] * H_profile[valid] * dx[valid]) ** 2
                            ))
                        else:
                            sigma_fw[t] = np.nan
                    logger.info(f"FW uncertainty computed: mean σ_Fw = {np.nanmean(sigma_fw):.1f} m³/s")

                # Store
                st.session_state[f'{key_prefix}_fw'] = fw
                st.session_state[f'{key_prefix}_time'] = time_array
                st.session_state[f'{key_prefix}_sss_interp'] = sss_interp
                st.session_state[f'{key_prefix}_H_profile'] = H_profile
                st.session_state[f'{key_prefix}_v_perp'] = v_perp
                st.session_state[f'{key_prefix}_x_km'] = x_km
                st.session_state[f'{key_prefix}_dx'] = dx
                st.session_state[f'{key_prefix}_sref_val'] = s_ref
                st.session_state[f'{key_prefix}_done'] = True
                st.session_state[f'{key_prefix}_sigma_fw'] = sigma_fw
                st.session_state[f'{key_prefix}_params'] = {
                    'depth': depth_cap,
                    's_ref': s_ref,
                    'sss_mean': float(np.nanmean(sss_interp)),
                    'sal_source': sal_label,
                    'sign_convention': 'dot product with INTO-Arctic vector',
                }

                st.success("✅ Freshwater transport computed!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # =========================================================================
    # 6. DISPLAY RESULTS
    # =========================================================================
    if not st.session_state.get(f'{key_prefix}_done', False):
        st.info("👆 Click 'Compute Freshwater Transport' to start")
        return

    fw = st.session_state.get(f'{key_prefix}_fw', np.array([]))
    stored_time = st.session_state.get(f'{key_prefix}_time', None)
    params = st.session_state.get(f'{key_prefix}_params', {})

    if len(fw) == 0:
        return

    st.divider()

    # Convert to mSv
    fw_msv = fw / 1e3

    # =========================================================================
    # 7. METRICS
    # =========================================================================
    st.subheader("📊 Summary Statistics")

    sigma_fw = st.session_state.get(f'{key_prefix}_sigma_fw', None)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Mean FW Transport", f"{np.nanmean(fw_msv):.2f} mSv")
    with col2:
        st.metric("Std Dev", f"{np.nanstd(fw_msv):.2f} mSv")
    with col3:
        st.metric("Min", f"{np.nanmin(fw_msv):.2f} mSv")
    with col4:
        st.metric("Max", f"{np.nanmax(fw_msv):.2f} mSv")
    with col5:
        if sigma_fw is not None:
            st.metric("Mean σ_Fw", f"{np.nanmean(sigma_fw)/1e3:.2f} mSv")
        else:
            st.metric("σ_Fw", "N/A")

    # =========================================================================
    # 7b. ALONG-TRACK FW TRANSPORT PROFILE (per-month bar chart)
    # =========================================================================
    st.subheader("📊 Freshwater Transport Along Gate")
    
    sss_interp = st.session_state.get(f'{key_prefix}_sss_interp', None)
    H_profile = st.session_state.get(f'{key_prefix}_H_profile', None)
    stored_v_perp = st.session_state.get(f'{key_prefix}_v_perp', v_perp)
    stored_x_km = st.session_state.get(f'{key_prefix}_x_km', x_km)
    stored_dx = st.session_state.get(f'{key_prefix}_dx', None)
    
    if sss_interp is not None and H_profile is not None and stored_time is not None:
        if stored_dx is None:
            stored_dx = _compute_segment_widths(stored_x_km, stored_v_perp.shape[0],
                                                gate_lon=gate_lon, gate_lat=gate_lat)
        _plot_transport_along_gate(
            v_perp=stored_v_perp,
            x_km=stored_x_km,
            time_array=stored_time,
            H_profile=H_profile,
            sss_interp=sss_interp,
            dx=stored_dx,
            strait_name=strait_name,
            key_prefix=key_prefix,
            mode='freshwater',
            s_ref=params.get('s_ref', S_REF_DEFAULT),
        )

    # =========================================================================
    # 8. TIME SERIES
    # =========================================================================
    st.subheader("📈 Freshwater Transport Time Series")

    # Uncertainty toggle
    show_fw_unc = st.checkbox("Show uncertainty (±σ_Fw)", value=True, key=f"{key_prefix}_show_unc")

    x_axis = _safe_time_axis(stored_time, len(fw))

    fig = go.Figure()

    # ±σ_Fw uncertainty band (if available)
    if show_fw_unc and sigma_fw is not None:
        sigma_fw_msv = sigma_fw / 1e3
        fig.add_trace(go.Scatter(
            x=x_axis, y=fw_msv + sigma_fw_msv,
            mode='lines', line=dict(width=0.5, dash='dot', color='rgba(230,126,34,0.4)'),
            showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=x_axis, y=fw_msv - sigma_fw_msv,
            mode='lines', line=dict(width=0.5, dash='dot', color='rgba(230,126,34,0.4)'),
            fill='tonexty', fillcolor='rgba(230,126,34,0.08)',
            name='±σ_Fw (CMEMS formal error)', hoverinfo='skip',
        ))

    fig.add_trace(go.Scatter(
        x=x_axis, y=fw_msv,
        mode='lines', name='FW Transport',
        line=dict(color='#3498DB', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>Fw: %{y:.2f} mSv<extra></extra>'
    ))
    fig.add_hline(y=float(np.nanmean(fw_msv)), line_dash="dot",
                  line_color="#E74C3C", line_width=1.5,
                  annotation_text=f"Mean: {np.nanmean(fw_msv):.2f} mSv")
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(
        title=f"Freshwater Transport — {strait_name}  (S_A = {params.get('s_ref', S_REF_DEFAULT)} PSU)",
        xaxis_title="Time", yaxis_title="Fw (mSv = 10³ m³/s)",
        height=500, hovermode='x unified',
         
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_ts_chart")

    # =========================================================================
    # 10. EXPORT
    # =========================================================================
    st.divider()
    st.subheader("📥 Export Data")

    time_col = _safe_time_col(x_axis, len(fw))

    export_df = pd.DataFrame({
        'Time': time_col,
        'FW_Transport_m3_s': fw,
        'FW_Transport_mSv': fw_msv,
    })
    if sigma_fw is not None:
        export_df['Sigma_Fw_m3_s'] = sigma_fw
        export_df['Sigma_Fw_mSv'] = sigma_fw / 1e3
    csv = export_df.to_csv(index=False)

    st.download_button(
        "📥 Download Freshwater Transport CSV", csv,
        f"freshwater_transport_{strait_name.replace(' ', '_').lower()}.csv",
        "text/csv", key=f"{key_prefix}_download"
    )

    # --- 4×3 Along-Gate Grid ---
    st.subheader("📊 FW Transport Along Gate — All Months (4×3)")
    if sss_interp is not None and H_profile is not None and stored_time is not None:
        from .transport_plots import plot_transport_along_gate_4x3
        fig_4x3 = plot_transport_along_gate_4x3(
            v_perp=stored_v_perp, x_km=stored_x_km, time_array=stored_time,
            H_profile=H_profile, sss_interp=sss_interp, dx=stored_dx,
            strait_name=strait_name, mode='freshwater',
            s_ref=params.get('s_ref', S_REF_DEFAULT),
        )
        if fig_4x3 is not None:
            st.plotly_chart(fig_4x3, use_container_width=True, key=f"{key_prefix}_4x3_chart")
            try:
                img_bytes = fig_4x3.to_image(format="png", width=1400, height=1200, scale=2)
                safe_name = strait_name.replace(' ', '_').lower()
                st.download_button(
                    "📥 Download FW Transport 4×3 PNG",
                    img_bytes,
                    f"{safe_name}_fw_transport_along_gate_4x3.png",
                    "image/png",
                    key=f"{key_prefix}_4x3_png",
                )
            except Exception:
                st.info("💡 Install **kaleido** (`pip install kaleido`) for PNG export.")

    st.info(f"""
    **Parameters:**
    - Depth cap H: {params.get('depth', 250)} m
    - S_A (reference): {params.get('s_ref', S_REF_DEFAULT)} PSU
    - Mean SSS: {params.get('sss_mean', 35.0):.2f} PSU
    - Salinity source: {params.get('sal_source', 'CCI SSS v5.5')}
    - Sign convention: positive = into Arctic ({params.get('sign_convention', 'dot product with INTO-Arctic vector')})

    **Formula:** Fw = Σ v_perp × (1 − SSS/S_A) × H × Δx  [m³/s]  — SSS = surface salinity (uniform with depth)
    """)


# =============================================================================
# SHARED HELPERS (salinity loading, interpolation, velocity)
# =============================================================================

def _find_file(file_map: dict, strait_name: str) -> Optional[str]:
    """Find filename in a map by fuzzy-matching the gate name."""
    strait_lower = strait_name.lower().strip().replace('_', ' ')
    filename = file_map.get(strait_lower)
    if filename is None:
        for key, fname in file_map.items():
            if key in strait_lower or strait_lower in key:
                filename = fname
                break
    return filename


def _load_salinity(strait_name: str) -> Optional[dict]:
    """Load salinity data — CCI SSS v5.5 ONLY (no ISAS fallback).
    
    Returns dict with:
        sss: array — (time, nb_prof)
        time: DatetimeIndex
        lon, lat: (nb_prof,)
        source: str — description of data source
        is_cci: True (always)
    Returns None if CCI SSS is not available or has <1% valid data.
    """
    cci_file = _find_file(SSS_CCI_FILE_MAP, strait_name)
    if cci_file is None:
        logger.warning(f"No CCI SSS file mapped for {strait_name}")
        return None

    filepath = SALINITY_NC_DIR / cci_file
    if not filepath.exists():
        logger.warning(f"CCI SSS file not found: {filepath}")
        return None

    try:
        ds = xr.open_dataset(filepath)
        sss = ds['sss'].values  # (time, nb_prof)
        valid_pct = (~np.isnan(sss)).sum() / sss.size * 100
        if valid_pct < 1.0:
            logger.warning(f"CCI SSS for {strait_name} only {valid_pct:.1f}% valid — too sparse")
            ds.close()
            return None
        result = {
            'sss': sss,
            'time': pd.to_datetime(ds['date'].values),
            'lon': ds['longitude'].values.astype(float),
            'lat': ds['latitude'].values.astype(float),
            'source': f"CCI SSS v5.5 ({valid_pct:.0f}% valid)",
            'is_cci': True,
        }
        ds.close()
        logger.info(f"Loaded CCI SSS for {strait_name}: {sss.shape}, {valid_pct:.1f}% valid")
        return result
    except Exception as e:
        logger.error(f"Error loading CCI SSS: {e}")
        return None



def _interpolate_sss_cci(cci_sss, cci_time, cci_lon, cci_lat,
                          vel_time, gate_lon, gate_lat,
                          n_points, n_time):
    """Interpolate CCI SSS monthly surface data to the velocity grid.
    
    CCI SSS is surface-only — no depth averaging needed.
    The surface value is assumed constant along depth.
    
    Steps:
    1. Match each velocity timestep to the (year, month) in CCI data
    2. Spatially interpolate the CCI profile onto the velocity gate points
    
    Returns: (n_points, n_time) surface salinity on velocity grid
    """
    from scipy.interpolate import interp1d

    # Normalised distance along CCI profile
    cci_dist = np.zeros(len(cci_lon))
    for i in range(1, len(cci_lon)):
        cci_dist[i] = cci_dist[i-1] + np.sqrt(
            (cci_lon[i] - cci_lon[i-1])**2 + (cci_lat[i] - cci_lat[i-1])**2
        )
    cci_dist_norm = cci_dist / cci_dist[-1] if cci_dist[-1] > 0 else np.linspace(0, 1, len(cci_lon))

    # Normalised distance along velocity gate
    if gate_lon is not None and len(gate_lon) > 1:
        vel_dist = np.zeros(n_points)
        for i in range(1, n_points):
            dx = gate_lon[i] - gate_lon[i-1]
            dy = gate_lat[i] - gate_lat[i-1] if gate_lat is not None else 0
            vel_dist[i] = vel_dist[i-1] + np.sqrt(dx**2 + dy**2)
        vel_dist_norm = vel_dist / vel_dist[-1] if vel_dist[-1] > 0 else np.linspace(0, 1, n_points)
    else:
        vel_dist_norm = np.linspace(0, 1, n_points)

    # Build (year, month) → CCI index lookup
    cci_time_pd = pd.to_datetime(np.asarray(cci_time).ravel())
    cci_lookup = {}
    for idx, t in enumerate(cci_time_pd):
        cci_lookup[(t.year, t.month)] = idx

    vel_time_pd = pd.to_datetime(np.asarray(vel_time).ravel())
    sss_interp = np.full((n_points, n_time), np.nan)

    for t in range(n_time):
        yr, mo = vel_time_pd[t].year, vel_time_pd[t].month
        cci_idx = cci_lookup.get((yr, mo))
        if cci_idx is None:
            continue  # No CCI data for this month — leave NaN (will be filled by fallback)

        sss_profile = cci_sss[cci_idx, :]  # (n_prof,)
        valid = ~np.isnan(sss_profile)

        if np.sum(valid) < 2:
            if np.sum(valid) == 1:
                sss_interp[:, t] = sss_profile[valid][0]
            continue

        try:
            f = interp1d(cci_dist_norm[valid], sss_profile[valid],
                         kind='linear', bounds_error=False,
                         fill_value=np.nan)
            sss_interp[:, t] = f(vel_dist_norm)
        except Exception:
            sss_interp[:, t] = np.nanmean(sss_profile[valid])

    return sss_interp


def _safe_time_values(time_array):
    """Convert time_array to a list of python datetimes (safe for arithmetic)."""
    try:
        ta = pd.to_datetime(np.asarray(time_array).ravel())
        return ta
    except Exception:
        return time_array


def _safe_time_axis(stored_time, n):
    """Build a safe x-axis from stored_time."""
    if stored_time is None:
        return list(range(n))
    try:
        return pd.to_datetime(np.asarray(stored_time).ravel())
    except Exception:
        return list(range(n))


def _safe_time_col(x_axis, n):
    """Build a column for CSV export."""
    try:
        return pd.to_datetime(x_axis).strftime('%Y-%m-%d').tolist()
    except Exception:
        return list(range(n))




def _get_velocity_data(data) -> Optional[np.ndarray]:
    """Get perpendicular velocity from the data object.
    
    Uses the centralized compute_perpendicular_velocity from transport_service.py
    with gate_name so that the dot product with INTO-Arctic vector is applied automatically.
    Positive v_perp = into Arctic.
    """
    from src.services.transport_service import compute_perpendicular_velocity
    
    ugos = getattr(data, 'ugos_matrix', None)
    vgos = getattr(data, 'vgos_matrix', None)
    gate_lon = getattr(data, 'gate_lon_pts', None)
    gate_lat = getattr(data, 'gate_lat_pts', None)
    strait_name = getattr(data, 'strait_name', None)
    
    if all(x is not None for x in [ugos, vgos, gate_lon, gate_lat]):
        return compute_perpendicular_velocity(
            ugos, vgos, gate_lon, gate_lat, gate_name=strait_name
        )
    # Fallback: use pre-computed v_geo_perp only if ugos/vgos not available
    v_perp = getattr(data, 'v_geo_perp', None)
    if v_perp is not None:
        logger.warning("Using pre-computed v_geo_perp — sign flip NOT guaranteed")
        return v_perp
    return None


def _compute_segment_widths(x_km, n_points, gate_lon=None, gate_lat=None):
    """Compute segment widths in meters — delegates to transport_service.
    
    Uses transport_service.compute_segment_widths (canonical implementation)
    when gate_lon/gate_lat are provided. Falls back to x_km-only central
    difference otherwise (same algorithm, kept for backward compatibility).
    """
    if gate_lon is not None and gate_lat is not None and x_km is not None:
        from src.services.transport_service import compute_segment_widths as _csw
        return _csw(gate_lon, gate_lat, x_km)
    # Fallback: x_km only (no gate coordinates)
    if x_km is not None and len(x_km) > 1:
        dx = np.zeros(n_points)
        for i in range(n_points):
            if i == 0:
                dx[i] = (x_km[1] - x_km[0]) * 1000
            elif i == n_points - 1:
                dx[i] = (x_km[i] - x_km[i - 1]) * 1000
            else:
                dx[i] = (x_km[i + 1] - x_km[i - 1]) / 2 * 1000
        dx = np.abs(dx)
    else:
        dx = np.ones(n_points) * 1000
    return dx


def _plot_transport_along_gate(v_perp, x_km, time_array, H_profile, sss_interp,
                               dx, strait_name, key_prefix, mode='freshwater',
                               s_ref=None):
    """Delegate to shared transport_plots module."""
    from .transport_plots import plot_transport_along_gate
    if s_ref is None:
        s_ref = S_REF_DEFAULT
    plot_transport_along_gate(
        v_perp=v_perp, x_km=x_km, time_array=time_array,
        H_profile=H_profile, sss_interp=sss_interp, dx=dx,
        strait_name=strait_name, key_prefix=key_prefix,
        mode=mode, s_ref=s_ref,
    )


def _gate_length_km(lons, lats):
    """Compute total along-gate length in km from lon/lat arrays."""
    if lons is None or lats is None or len(lons) < 2:
        return None
    dlat = np.diff(lats) * 111.0
    dlon = np.diff(lons) * 111.0 * np.cos(np.deg2rad(np.array(lats[:-1])))
    return float(np.sum(np.sqrt(dlat**2 + dlon**2)))


def _check_cci_gate_alignment(cci_lons, cci_lats, gate_lons, gate_lats, sss_data, strait_name):
    """Show a spatial alignment diagnostic between the gate shapefile and CCI SSS track."""
    cci_n = len(cci_lons)
    cci_km = _gate_length_km(cci_lons, cci_lats)
    nan_pct = float(np.isnan(sss_data).sum() / sss_data.size * 100)

    gate_km = _gate_length_km(gate_lons, gate_lats) if gate_lons is not None else None
    gate_n = len(gate_lons) if gate_lons is not None else None

    lines = []

    if nan_pct >= 90:
        lines.append(
            f"🚫 **{strait_name} — CCI SSS coverage: {100-nan_pct:.0f}% valid** "
            f"({nan_pct:.0f}% NaN). Sea-ice / latitude limitations make freshwater transport "
            f"unreliable for this gate."
        )
    elif nan_pct >= 50:
        lines.append(
            f"⚠️ **{strait_name} — CCI SSS coverage: {100-nan_pct:.0f}% valid** "
            f"({nan_pct:.0f}% NaN). Many timesteps will have NaN transport "
            f"(sea-ice / high-latitude gaps)."
        )

    if gate_km is not None and cci_km is not None:
        ratio = cci_km / gate_km
        cci_sp = cci_km / (cci_n - 1) if cci_n > 1 else 0
        gate_sp = gate_km / (gate_n - 1) if gate_n > 1 else 0

        if abs(ratio - 1.0) > 0.10:
            lines.append(
                f"⚠️ **CCI track length ({cci_km:.0f} km) differs from gate shapefile "
                f"({gate_km:.0f} km) by {abs(ratio-1)*100:.0f}%.** "
                f"The CCI satellite ground track follows a curved orbit that is not identical "
                f"to the straight gate line. Spatial interpolation uses normalised distance (0→1), "
                f"which corrects for this — but the along-gate salinity assignment has positional "
                f"uncertainty of ±{abs(cci_km-gate_km)/2:.0f} km."
            )

        lines.append(
            f"ℹ️ **Resolution:** Gate shapefile {gate_n} pts @ {gate_sp:.1f} km/pt — "
            f"CCI SSS {cci_n} pts @ {cci_sp:.1f} km/pt. "
            f"Interpolated to velocity grid via normalised along-gate distance."
        )

    if lines:
        with st.expander("🔍 CCI SSS spatial alignment diagnostics", expanded=(nan_pct >= 50 or (gate_km is not None and cci_km is not None and abs(cci_km/gate_km - 1.0) > 0.10))):
            for line in lines:
                st.markdown(line)


def _plot_sss_profile_cci(sss_data, sss_lon, sss_time, is_cci):
    """Plot CCI SSS v5.5 surface salinity profile along the gate."""
    sss_mean = np.nanmean(sss_data, axis=0)
    sss_std = np.nanstd(sss_data, axis=0)
    n_months = sss_data.shape[0]

    # Use along-gate point index on x-axis to avoid broken axis for gates crossing 0° lon
    x_axis = np.arange(len(sss_lon))
    valid = ~np.isnan(sss_mean)

    fig = go.Figure()
    # ±1 std band
    if np.any(valid):
        _x_v = x_axis[valid]
        fig.add_trace(go.Scatter(
            x=np.concatenate([_x_v, _x_v[::-1]]),
            y=np.concatenate([sss_mean[valid] + sss_std[valid],
                               (sss_mean[valid] - sss_std[valid])[::-1]]),
            fill='toself', fillcolor='rgba(46,139,87,0.18)',
            line=dict(color='rgba(0,0,0,0)'), name='±1 Std (interannual)',
        ))
    # Mean SSS line — green circles
    fig.add_trace(go.Scatter(
        x=x_axis[valid], y=sss_mean[valid], mode='lines+markers',
        name='Climatological mean SSS',
        line=dict(color='#2E8B57', width=2),
        marker=dict(size=5, color='#2E8B57', symbol='circle'),
    ))
    # NaN points — grey circles
    if np.any(~valid):
        fig.add_trace(go.Scatter(
            x=x_axis[~valid],
            y=np.full(np.sum(~valid), np.nanmin(sss_mean[valid]) - 0.2 if np.any(valid) else 30.0),
            mode='markers', name=f'No coverage (n={np.sum(~valid)})',
            marker=dict(size=5, color='lightgray', symbol='circle',
                        line=dict(color='#999', width=0.5)),
        ))
    fig.update_layout(
        title="Surface Salinity Along Gate — CCI SSS v5.5 (satellite surface salinity)",
        xaxis_title="Along-gate point index",
        yaxis_title="Sea Surface Salinity (PSU)",
        height=300,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(color='#111111')),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#111111'),
        xaxis=dict(showgrid=True, gridcolor='#E0E0E0', zeroline=False,
                   title_font=dict(color='#111111'), tickfont=dict(color='#111111')),
        yaxis=dict(showgrid=True, gridcolor='#E0E0E0', zeroline=False,
                   title_font=dict(color='#111111'), tickfont=dict(color='#111111')),
    )
    st.plotly_chart(fig, use_container_width=True, key="fw_sss_profile_cci")
    st.caption(
        f"**Source:** ESA CCI Sea Surface Salinity v5.5 — "
        f"{n_months} monthly timesteps, {len(sss_lon)} along-gate points. "
        f"Mean ± 1σ (interannual variability). X-axis = along-gate point index."
    )
