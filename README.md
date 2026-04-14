# 🛰️ SLCCI Satellite Altimetry Dashboard

Standalone Streamlit dashboard for ESA Sea Level CCI (SLCCI) along-track satellite altimetry analysis.

## 🌊 What it does

- Load **Jason-2 (J2)** along-track SLCCI data (NetCDF cycles)
- Compute **DOT** (Dynamic Ocean Topography) = corssh − geoid
- Compute **DOT slope** via linear fit → geostrophic velocity
- Estimate **volume / freshwater / salt transport** across Arctic gates
- Visualize everything with interactive Plotly charts

## 📁 Project Structure

```
SLCCI-Dashboard/
├── streamlit_app.py          # Entry point
├── app_slcci/                # Standalone SLCCI app
│   ├── main.py               # App orchestrator
│   ├── sidebar.py            # Gate selection + settings
│   ├── state.py              # Session state management
│   └── tabs.py               # Analysis tabs
├── app/                      # Shared components
│   ├── components/           # Charts, loaders, tabs
│   ├── state.py              # AppConfig dataclass
│   └── styles.py             # CSS styling
├── src/
│   ├── slcci/                # SLCCI data I/O
│   │   ├── loader.py         # NetCDF cycle reader
│   │   ├── dot.py            # DOT computation & slope
│   │   ├── geoid.py          # Geoid interpolation
│   │   ├── binning.py        # Longitude binning
│   │   ├── spatial.py        # Gate spatial filtering
│   │   └── models.py         # Data models
│   ├── physics/              # Geophysical computations
│   │   ├── geostrophy.py     # Geostrophic velocity
│   │   ├── transport.py      # Volume/FW/salt transport
│   │   ├── constants.py      # Physical constants
│   │   └── coordinates.py    # Coordinate transforms
│   ├── services/
│   │   └── slcci_service.py  # Service layer
│   └── core/
│       └── logging_config.py # Logging setup
├── gates/                    # Gate shapefiles (.shp)
├── config/                   # YAML configuration
└── requirements.txt
```

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/caroniico/SLCCI-Dashboard.git
cd SLCCI-Dashboard

# 2. Create venv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run streamlit_app.py --server.port 8502
```

## 📡 Required Data

You need local SLCCI NetCDF files:
- **J2 data**: `SLCCI_ALTDB_J2_CycleXXX_V2.nc` files in a directory
- **Geoid**: `TUM_ogmoc.nc` file

Set paths in the sidebar when the app starts.

## 🔬 Analysis Tabs

| Tab | Description |
|-----|-------------|
| 📈 Slope Timeline | DOT slope (m/100km) over time |
| 🌊 DOT Profile | Along-gate DOT profile |
| 🗺️ Spatial Map | Geographic view |
| 📅 Monthly Analysis | Seasonal patterns |
| 🌀 Geostrophic Velocity | v = −(g/f) × ∂DOT/∂x |
| 🚢 Volume Transport | Sv through gate |
| 💧 Freshwater Transport | mSv relative to 34.8 PSU |
| 🧪 Salinity Profile | CCI SSS v5.5 |
| 🧂 Salt Flux | Salt transport estimates |
| �� Export | Download results |

## 📜 License

Part of the ARCFRESH project — DTU Space.
