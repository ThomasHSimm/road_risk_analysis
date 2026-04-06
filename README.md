# Road Risk Analysis

Open-source road safety pipeline combining DfT STATS19 collision data, AADF traffic counts,
OS Open Roads network geometry, and OpenStreetMap attributes to produce **exposure-adjusted
risk scores for every road link in Yorkshire** — including the 85% of roads without traffic
counters.

**Current geography:** Yorkshire pilot (forces 12, 13, 14, 16) → Great Britain  
**Time range:** 2015–2024  
**Grain:** OS Open Roads link × year (705,672 links)

---

## What this builds

**Stage 1a — Traffic estimation**  
Predicts AADT (annual average daily traffic) for all 705k road links using a gradient
boosting model trained on 2,240 AADF count points. Fills coverage gaps on minor/unclassified
roads where DfT has no measured counts. CV R² 0.723 with 17 features including network
centrality, OSM speed limits, population density, and betweenness relative to road class.

**Stage 1b — Temporal profiles**  
Monthly traffic variation and weekday/weekend ratios from WebTRIS motorway sensors.
Seasonality indices per road corridor (M62, M1, A1M etc).

**Stage 2 — Collision risk model**  
Poisson GLM + XGBoost predicting collision counts per link per year.
Uses `log(AADT × length_km × 365 / 1e6)` as exposure offset so the model learns
*which roads are dangerous given their traffic* — not just which are busy.
Outputs risk percentiles and GLM residuals (excess risk above expected).

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd road-risk-analysis
pip install -e ".[dev]"

# 2. Download raw data — see data/README.md for links
#    Required: STATS19 CSV, AADF zip, OS Open Roads GeoPackage, MRDB, OSM pbf files

# 3. Convert OSM pbf files (download Yorkshire counties from Geofabrik first)
sudo apt install osmium-tool
for f in data/raw/osm/*.osm.pbf; do
    osmium cat "$f" -o "${f%.osm.pbf}.osm"
done

# 4. Run pipeline in order
python src/road_risk/clean.py
python src/road_risk/snap.py
python src/road_risk/join.py
python src/road_risk/network_features.py   # ~10 mins first run, cached after
python src/road_risk/network_features.py --osm   # adds OSM features (~15 mins)

python src/road_risk/model.py --stage traffic     # Stage 1a: AADT estimator
python src/road_risk/model.py --stage temporal    # Stage 1b: seasonal profiles
python src/road_risk/model.py --stage collision   # Stage 2: Poisson risk model
```

---

## Data Sources

| Source | Provider | Granularity | Coverage |
|---|---|---|---|
| STATS19 (collisions, vehicles, casualties) | DfT | Per incident | GB 1979– |
| AADF by direction | DfT | Count point / year | GB — major + some minor |
| OS Open Roads | Ordnance Survey | Road link geometry | GB |
| Major Road Database (MRDB) | DfT / OS | Road geometry | GB |
| WebTRIS / MIDAS | National Highways | Site / month | Motorways + trunk A-roads |
| OpenStreetMap | OSM contributors | Road edge | GB — speed, lanes, surface |
| LSOA population + area | ONS | LSOA 2021 | England & Wales |

See `data/README.md` for download instructions.

---

## Repo Structure

```
road-risk-analysis/
├── src/road_risk/           # Core pipeline modules
│   ├── ingest/              # Data ingestion (STATS19, AADF, WebTRIS, MRDB, OS Roads)
│   ├── config.py            # YAML loader, paths
│   ├── clean.py             # Coordinate validation, COVID flags
│   ├── snap.py              # Collision → road link snapping (weighted multi-criteria)
│   ├── join.py              # Build road_link × year feature table
│   ├── network_features.py  # Graph centrality, OSM attributes, population density
│   ├── features.py          # Feature engineering for model
│   └── model.py             # Stage 1a/1b/2 models
├── notebooks/
│   ├── 01_eda_stats19.ipynb
│   ├── 02_eda_joins.ipynb
│   ├── 03_model_results.ipynb       # Full model results, maps, SHAP
│   └── stats19_coordinate_issue.ipynb  # DfT data quality report (see below)
├── app/                     # Streamlit app (coming soon)
├── data/
│   ├── raw/                 # Source files — never modified, not in git
│   ├── processed/           # Cleaned parquets
│   ├── features/            # Model-ready feature tables
│   └── models/              # Saved model artefacts + risk scores
├── config/settings.yaml     # Police force codes, year ranges, paths
└── docs/
    └── data-quality-notes.md
```

---

## Key Results (Yorkshire pilot — April 2026)

| Metric | Value |
|---|---|
| Collisions loaded (2015–2024) | 102,361 |
| Collisions snapped to road links | 100,952 (98.6%) |
| Mean snap score | 0.860 |
| Mean snap distance | 16.6m |
| Road links with collisions | 49,247 |
| AADT estimator CV R² | 0.723 (±0.029) |
| Poisson GLM pseudo-R² | 0.083 |
| XGBoost pseudo-R² | 0.330 |
| Top 1% risk links | 21,171 link-years |

---

## Key Data Quality Notes

- **STATS19 force code bug (fixed April 2026)** — `config/settings.yaml` previously
  used police_force codes 4–7 (Lancashire, Merseyside, Greater Manchester, Cheshire)
  instead of the correct Yorkshire codes 12–16. This caused the entire pipeline to load
  NW England data. Now fixed and documented in `config/settings.yaml` with instructions
  to re-derive codes from the DfT data guide Excel.

- **Snap rate 98.6%** — achieved after force code fix. Previous 40.6% ceiling was
  because NW England collisions were snapping to NW England roads in the 20km buffer zone.

- **AADF coverage** — 3 years only (2019, 2021, 2023). AADT estimated via ML for
  all ~418k links without a nearby count point.

- **OSM attribute coverage** — speed limit 43.6%, lanes 5.5%, surface 12.8%.
  Sparse features median-imputed in GLM to retain training rows.

See `docs/data-quality-notes.md` for full detail.

---

## STATS19 Coordinate Issue (reported to DfT)

Analysis revealed that `location_easting_osgr` / `location_northing_osgr` fields for
Yorkshire forces have a systematic BNG grid square prefix error. The `latitude`/`longitude`
fields are correct and are used throughout this pipeline. See
`notebooks/stats19_coordinate_issue.ipynb` for a standalone reproducible report prepared
for the DfT road safety statistics team.

---

## Positioning

This pipeline produces **Safety Performance Functions (SPFs)** for the full road network
using open data — extending exposure-adjusted risk analysis to the 85% of roads where
DfT currently has no traffic counts.

Compatible with ESRI/ArcGIS workflows via GeoPackage output. PostGIS backend for app queries.

---

## Requirements

```
geopandas, pandas, numpy, scikit-learn, networkx, scipy, pyproj
statsmodels    # Stage 2 Poisson GLM
xgboost        # Stage 2 XGBoost
osmnx          # OSM network features
osmium-tool    # CLI — convert pbf to osm (apt install osmium-tool)
streamlit      # App (optional)
```