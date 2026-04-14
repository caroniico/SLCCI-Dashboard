"""
Salt Flux Tab - VERSIONE DEFINITIVA
================================================================================
Calcola SOLO il Salt Flux (non Freshwater):
    Sm = Σ ρ × (SSS(x,t)/1000) × v_perp(x,t) × H(x) × Δx  [kg/s]

Usa dati di salinità SUPERFICIALE da CCI SSS v5.5:
    /straits/netcdf/{gate}_SSS_CCIv5.5.nc

CCI SSS shape: (time, nb_prof) — monthly sea surface salinity
La salinità superficiale viene considerata costante lungo la profondità.

NO ISAS PSAL fallback — solo CCI SSS v5.5. Se CCI non è disponibile,
il salt flux non può essere calcolato.

Sign convention: positive v_perp = into Arctic (dot product with INTO-Arctic vector)
Densità costante: ρ = 1025 kg/m³
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

# Constants
RHO_SEAWATER = 1025.0  # kg/m³ (costante)
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


def render_salt_flux_clean(data: Any, config: Any, ds_info: dict):
    """
    Salt Flux Tab - Versione DEFINITIVA.
    
    Calcola SOLO Salt Flux usando CCI SSS v5.5 (surface salinity).
    Sign convention: positive v_perp = into Arctic (dot product with INTO-Arctic vector).
    """
    st.header("🧂 Salt Flux")
    
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
            "salt flux cannot be computed."
        )
        return
    
    sal_source = salinity_result['source']
    sss_data = salinity_result['sss']
    sss_time = salinity_result['time']
    sss_lon = salinity_result['lon']
    sss_lat = salinity_result['lat']
    is_cci = salinity_result['is_cci']
    
    # =========================================================================
    # 3. INFO PANEL
    # =========================================================================
    st.success(f"✅ Data loaded: {n_points} gate points × {n_time} velocity timesteps (positive = into Arctic)")
    st.success(f"✅ Salinity: {sss_data.shape[1]} points × {sss_data.shape[0]} months — CCI SSS v5.5 (surface)")
    st.info(
        "⚠️ **CCI SSS v5.5 covers ~2010–2023 only.** "
        "Velocity timesteps outside CCI range will have NaN salt flux."
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
        st.metric("ρ (constant)", f"{RHO_SEAWATER:.0f} kg/m³")
    
    # =========================================================================
    # 4. SETTINGS
    # =========================================================================
    st.subheader("⚙️ Settings")
    
    key_prefix = f"sf_{strait_name.replace(' ', '_').lower()}"
    
    col1, col2 = st.columns(2)
    
    with col1:
        depth_cap = st.number_input(
            "Depth Cap (m)",
            min_value=50,
            max_value=500,
            value=250,
            step=50,
            key=f"{key_prefix}_depth",
            help="Integration depth H for salt flux calculation"
        )
    
    with col2:
        with st.expander("📊 SSS Profile Along Gate"):
            _plot_sss_profile_cci(sss_data, sss_lon, sss_time, is_cci)
    
    st.info("**Formula:** Sm = Σ ρ × (SSS/1000) × v_perp × H × Δx  [kg/s]  — SSS = CCI surface salinity (uniform with depth)")
    
    # =========================================================================
    # 5. COMPUTE SALT FLUX
    # =========================================================================
    if st.button("🧮 Compute Salt Flux", type="primary", use_container_width=True,
                 key=f"{key_prefix}_compute"):
        
        with st.spinner("Computing salt flux (loading bathymetry + SSS)..."):
            try:
                # --- Load GEBCO bathymetry ---
                from src.services.gebco_service import get_bathymetry_cache
                cache = get_bathymetry_cache()
                depth_profile_full = cache.get_or_compute(
                    gate_name=strait_name,
                    gate_lons=gate_lon,
                    gate_lats=gate_lat,
                    gebco_path=config.gebco_nc_path,
                    depth_cap=None  # Full bathymetry
                )
                # H(x) = min(depth_cap, bathymetry(x))  — point-by-point
                H_profile = np.minimum(depth_profile_full, depth_cap)
                
                st.info(f"🏔️ Bathymetry: mean={np.mean(depth_profile_full):.0f}m, "
                        f"H(x) capped at {depth_cap}m → effective mean H={np.mean(H_profile):.0f}m")
                
                # --- Interpolate SSS onto velocity grid (CCI only) ---
                sss_interp = _interpolate_sss_cci(
                    sss_data, sss_time, sss_lon, sss_lat,
                    time_array, gate_lon, gate_lat,
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
                        f"Salt flux will be NaN for those timesteps."
                    )
                
                st.info(f"📍 SSS interpolated ({sal_label}): shape={sss_interp.shape}, mean={np.nanmean(sss_interp):.2f} PSU")
                
                # Compute segment widths — usa gate_lon/lat per delegare a transport_service
                dx = _compute_segment_widths(x_km, n_points, gate_lon=gate_lon, gate_lat=gate_lat)
                
                # Compute salt flux: Sm = Σ ρ × (S̄/1000) × v_perp × H(x) × Δx
                salt_flux = np.zeros(n_time)
                
                for t in range(n_time):
                    v = v_perp[:, t] if v_perp.ndim > 1 else v_perp
                    s = sss_interp[:, t] if sss_interp.ndim > 1 else sss_interp
                    
                    flux_point = RHO_SEAWATER * (s / 1000.0) * v * H_profile * dx
                    if np.all(np.isnan(flux_point)):
                        salt_flux[t] = np.nan  # Preserve NaN for missing timesteps
                    else:
                        salt_flux[t] = np.nansum(flux_point)
                
                # Compute salt flux uncertainty if err_ugosa/err_vgosa available
                # σ_Sm(t) = √( Σᵢ (ρ × S(i,t)/1000 × σ_v_perp(i,t) × H(i) × Δx(i))² )
                err_ugosa = getattr(data, 'err_ugosa_matrix', None)
                err_vgosa = getattr(data, 'err_vgosa_matrix', None)
                sigma_sm = None
                
                if err_ugosa is not None and err_vgosa is not None:
                    from src.services.transport_service import compute_perpendicular_velocity_uncertainty
                    sigma_v_perp = compute_perpendicular_velocity_uncertainty(
                        err_ugosa, err_vgosa, gate_lon, gate_lat, gate_name=strait_name
                    )
                    sigma_sm = np.zeros(n_time)
                    for t in range(n_time):
                        sv = sigma_v_perp[:, t] if sigma_v_perp.ndim > 1 else sigma_v_perp
                        s = sss_interp[:, t] if sss_interp.ndim > 1 else sss_interp
                        valid = np.isfinite(sv) & np.isfinite(s)
                        if np.any(valid):
                            sigma_sm[t] = np.sqrt(np.nansum(
                                (RHO_SEAWATER * (s[valid] / 1000.0) * sv[valid] * H_profile[valid] * dx[valid]) ** 2
                            ))
                        else:
                            sigma_sm[t] = np.nan
                    logger.info(f"Salt flux uncertainty computed: mean σ_Sm = {np.nanmean(sigma_sm):.1f} kg/s")
                
                # Store results
                st.session_state[f'{key_prefix}_salt_flux'] = salt_flux
                st.session_state[f'{key_prefix}_time'] = time_array
                st.session_state[f'{key_prefix}_sss_interp'] = sss_interp
                st.session_state[f'{key_prefix}_H_profile'] = H_profile
                st.session_state[f'{key_prefix}_v_perp'] = v_perp
                st.session_state[f'{key_prefix}_x_km'] = x_km
                st.session_state[f'{key_prefix}_dx'] = dx
                st.session_state[f'{key_prefix}_done'] = True
                st.session_state[f'{key_prefix}_sigma_sm'] = sigma_sm
                st.session_state[f'{key_prefix}_params'] = {
                    'depth': depth_cap,
                    'sss_mean': np.nanmean(sss_interp),
                    'rho': RHO_SEAWATER,
                    'H_mean': float(np.mean(H_profile)),
                    'sal_source': sal_label,
                    'sign_convention': 'dot product with INTO-Arctic vector',
                }
                
                st.success("✅ Salt flux computed!")
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
        st.info("👆 Click 'Compute Salt Flux' to start")
        return
    
    salt_flux = st.session_state.get(f'{key_prefix}_salt_flux', np.array([]))
    stored_time = st.session_state.get(f'{key_prefix}_time', None)
    params = st.session_state.get(f'{key_prefix}_params', {})
    
    if len(salt_flux) == 0:
        return
    
    st.divider()
    
    # =========================================================================
    # 7. METRICS
    # =========================================================================
    st.subheader("📊 Summary Statistics")
    
    sigma_sm = st.session_state.get(f'{key_prefix}_sigma_sm', None)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Mean Salt Flux", f"{np.nanmean(salt_flux)/1e9:.3f} Gg/s")
    with col2:
        st.metric("Std Dev", f"{np.nanstd(salt_flux)/1e9:.3f} Gg/s")
    with col3:
        st.metric("Min", f"{np.nanmin(salt_flux)/1e9:.3f} Gg/s")
    with col4:
        st.metric("Max", f"{np.nanmax(salt_flux)/1e9:.3f} Gg/s")
    with col5:
        if sigma_sm is not None:
            st.metric("Mean σ_Sm", f"{np.nanmean(sigma_sm)/1e9:.3f} Gg/s")
        else:
            st.metric("σ_Sm", "N/A")
    
    # =========================================================================
    # 7b. ALONG-TRACK SALT FLUX PROFILE (per-month bar chart)
    # =========================================================================
    st.subheader("📊 Salt Flux Along Gate")
    
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
            mode='salt',
        )
    
    # =========================================================================
    # 8. TIME SERIES PLOT
    # =========================================================================
    st.subheader("📈 Salt Flux Time Series")

    # Uncertainty toggle
    show_sm_unc = st.checkbox("Show uncertainty (±σ_Sm)", value=True, key=f"{key_prefix}_show_unc")
    
    if stored_time is not None:
        try:
            x_axis = pd.to_datetime(np.asarray(stored_time).ravel())
        except Exception:
            x_axis = list(range(len(salt_flux)))
    else:
        x_axis = list(range(len(salt_flux)))
    
    fig = go.Figure()
    
    # ±σ_Sm uncertainty band (if available)
    if show_sm_unc and sigma_sm is not None:
        fig.add_trace(go.Scatter(
            x=x_axis, y=(salt_flux + sigma_sm) / 1e9,
            mode='lines', line=dict(width=0.5, dash='dot', color='rgba(230,126,34,0.4)'),
            showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=x_axis, y=(salt_flux - sigma_sm) / 1e9,
            mode='lines', line=dict(width=0.5, dash='dot', color='rgba(230,126,34,0.4)'),
            fill='tonexty', fillcolor='rgba(230,126,34,0.08)',
            name='±σ_Sm (CMEMS formal error)', hoverinfo='skip',
        ))
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=salt_flux / 1e9,
        mode='lines',
        name='Salt Flux',
        line=dict(color='#9B59B6', width=2)
    ))
    
    fig.add_hline(
        y=np.nanmean(salt_flux) / 1e9,
        line_dash="dash",
        line_color="#E74C3C",
        annotation_text=f"Mean: {np.nanmean(salt_flux)/1e9:.3f} Gg/s"
    )
    
    fig.add_hline(y=0, line_color="gray", line_width=1)
    
    fig.update_layout(
        title=f"Salt Flux - {strait_name}",
        xaxis_title="Time",
        yaxis_title="Salt Flux (Gg/s)",
        height=500,
        hovermode='x unified',
        
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_ts_chart")
    
    # =========================================================================
    # 10. EXPORT
    # =========================================================================
    st.divider()
    st.subheader("📥 Export Data")
    
    if isinstance(x_axis, pd.DatetimeIndex):
        time_col = x_axis.strftime('%Y-%m-%d').tolist()
    else:
        time_col = list(range(len(salt_flux)))
    
    export_df = pd.DataFrame({
        'Time': time_col,
        'Salt_Flux_kg_s': salt_flux,
        'Salt_Flux_Gg_s': salt_flux / 1e9,
    })
    if sigma_sm is not None:
        export_df['Sigma_Sm_kg_s'] = sigma_sm
        export_df['Sigma_Sm_Gg_s'] = sigma_sm / 1e9
    
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        "📥 Download Salt Flux CSV",
        csv,
        f"salt_flux_{strait_name.replace(' ', '_').lower()}.csv",
        "text/csv",
        key=f"{key_prefix}_download"
    )
    
    # --- 4×3 Along-Gate Grid ---
    st.subheader("📊 Salt Flux Along Gate — All Months (4×3)")
    if sss_interp is not None and H_profile is not None and stored_time is not None:
        from .transport_plots import plot_transport_along_gate_4x3
        fig_4x3 = plot_transport_along_gate_4x3(
            v_perp=stored_v_perp, x_km=stored_x_km, time_array=stored_time,
            H_profile=H_profile, sss_interp=sss_interp, dx=stored_dx,
            strait_name=strait_name, mode='salt',
        )
        if fig_4x3 is not None:
            st.plotly_chart(fig_4x3, use_container_width=True, key=f"{key_prefix}_4x3_chart")
            try:
                img_bytes = fig_4x3.to_image(format="png", width=1400, height=1200, scale=2)
                safe_name = strait_name.replace(' ', '_').lower()
                st.download_button(
                    "📥 Download Salt Flux 4×3 PNG",
                    img_bytes,
                    f"{safe_name}_salt_flux_along_gate_4x3.png",
                    "image/png",
                    key=f"{key_prefix}_4x3_png",
                )
            except Exception:
                st.info("💡 Install **kaleido** (`pip install kaleido`) for PNG export.")
    
    st.info(f"""
    **Parameters:**
    - Depth: {params.get('depth', 250)} m
    - Mean SSS: {params.get('sss_mean', 35.0):.2f} PSU
    - Salinity source: {params.get('sal_source', 'CCI SSS v5.5')}
    - ρ: {params.get('rho', RHO_SEAWATER):.0f} kg/m³
    - Sign convention: positive = into Arctic ({params.get('sign_convention', 'dot product with INTO-Arctic vector')})
    
    **Formula:** Sm = Σ ρ × (SSS/1000) × v_perp × H × Δx [kg/s]  — SSS = surface salinity (uniform with depth)
    """)


# =============================================================================
# HELPER FUNCTIONS
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
        is_cci: bool — always True
    
    Returns None if CCI SSS is not available for this gate.
    """
    # CCI SSS v5.5 only — no ISAS fallback
    cci_file = _find_file(SSS_CCI_FILE_MAP, strait_name)
    if cci_file is not None:
        filepath = SALINITY_NC_DIR / cci_file
        if filepath.exists():
            try:
                ds = xr.open_dataset(filepath)
                sss = ds['sss'].values  # (time, nb_prof)
                valid_pct = (~np.isnan(sss)).sum() / sss.size * 100
                if valid_pct >= 1.0:
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
                else:
                    logger.warning(f"CCI SSS for {strait_name} only {valid_pct:.1f}% valid — no data available")
                ds.close()
            except Exception as e:
                logger.error(f"Error loading CCI SSS: {e}")

    logger.warning(f"No CCI SSS v5.5 data available for {strait_name} — no ISAS fallback")
    return None


def _interpolate_sss_cci(cci_sss, cci_time, cci_lon, cci_lat,
                          vel_time, gate_lon, gate_lat,
                          n_points, n_time):
    """Interpolate CCI SSS monthly surface data to the velocity grid.
    
    CCI SSS is surface-only — no depth averaging needed.
    The surface value is assumed constant along depth.
    
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
            continue

        sss_profile = cci_sss[cci_idx, :]
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


def _get_velocity_data(data) -> Optional[np.ndarray]:
    """Get perpendicular velocity from data object.
    
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
                               dx, strait_name, key_prefix, mode='salt'):
    """Delegate to shared transport_plots module."""
    from .transport_plots import plot_transport_along_gate
    plot_transport_along_gate(
        v_perp=v_perp, x_km=x_km, time_array=time_array,
        H_profile=H_profile, sss_interp=sss_interp, dx=dx,
        strait_name=strait_name, key_prefix=key_prefix, mode=mode,
    )


def _gate_length_km(lons, lats):
    """Compute total along-gate length in km from lon/lat arrays."""
    if lons is None or lats is None or len(lons) < 2:
        return None
    dlat = np.diff(lats) * 111.0
    dlon = np.diff(lons) * 111.0 * np.cos(np.deg2rad(np.array(lats[:-1])))
    return float(np.sum(np.sqrt(dlat**2 + dlon**2)))


def _check_cci_gate_alignment(cci_lons, cci_lats, gate_lons, gate_lats, sss_data, strait_name):
    """Show a spatial alignment diagnostic between the gate shapefile and CCI SSS track.
    
    Warns if:
    - CCI along-track length differs > 10% from gate shapefile length  
    - CCI NaN fraction is high (poor coverage)
    - Point counts differ strongly (resolution mismatch)
    """
    cci_n = len(cci_lons)
    cci_km = _gate_length_km(cci_lons, cci_lats)
    nan_pct = float(np.isnan(sss_data).sum() / sss_data.size * 100)

    gate_km = _gate_length_km(gate_lons, gate_lats) if gate_lons is not None else None
    gate_n = len(gate_lons) if gate_lons is not None else None

    lines = []

    # NaN coverage warning
    if nan_pct >= 90:
        lines.append(
            f"🚫 **{strait_name} — CCI SSS coverage: {100-nan_pct:.0f}% valid** "
            f"({nan_pct:.0f}% NaN). Sea-ice / latitude limitations make salt flux "
            f"unreliable for this gate."
        )
    elif nan_pct >= 50:
        lines.append(
            f"⚠️ **{strait_name} — CCI SSS coverage: {100-nan_pct:.0f}% valid** "
            f"({nan_pct:.0f}% NaN). Many timesteps will have NaN salt flux "
            f"(sea-ice / high-latitude gaps)."
        )

    # Track length mismatch
    if gate_km is not None and cci_km is not None:
        ratio = cci_km / gate_km
        cci_sp = cci_km / (cci_n - 1) if cci_n > 1 else 0
        gate_sp = gate_km / (gate_n - 1) if gate_n > 1 else 0

        if abs(ratio - 1.0) > 0.10:
            lines.append(
                f"⚠️ **CCI track length ({cci_km:.0f} km) differs from gate shapefile "
                f"({gate_km:.0f} km) by {abs(ratio-1)*100:.0f}%.** "
                f"The CCI satellite ground track follows a curved orbit (S3 pass 481) "
                f"that is not identical to the straight gate line. "
                f"Spatial interpolation uses normalised distance (0→1), which corrects "
                f"for this — but the along-gate salinity assignment has positional uncertainty "
                f"of ±{abs(cci_km-gate_km)/2:.0f} km."
            )

        # Always show the resolution info
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
    # CCI SSS v5.5: (time, nb_prof) → climatological mean ± std
    sss_mean = np.nanmean(sss_data, axis=0)
    sss_std = np.nanstd(sss_data, axis=0)
    n_months = sss_data.shape[0]

    # Compute cumulative along-track distance (km) — avoids broken axis for gates crossing 0° lon
    sss_lat = getattr(sss_time, '_lat', None)  # fallback: use index if lat not available
    dist_km = np.zeros(len(sss_lon))
    # We only have lon here; use simple lon-only distance as proxy if lat unavailable
    # Better: caller should pass sss_lat — but for the expander plot lon-based is acceptable
    # Use index-based distance (equal spacing) as safe fallback
    x_axis = np.arange(len(sss_lon))  # index 0..N-1

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
        title=f"Surface Salinity Along Gate — CCI SSS v5.5 (satellite surface salinity)",
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
    st.plotly_chart(fig, use_container_width=True, key="sf_sss_profile_cci")
    st.caption(
        f"**Source:** ESA CCI Sea Surface Salinity v5.5 — "
        f"{n_months} monthly timesteps, {len(sss_lon)} along-gate points. "
        f"Mean ± 1σ (interannual variability). X-axis = along-gate point index."
    )