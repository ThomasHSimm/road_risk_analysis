# Road Risk Analysis

Open-source road safety pipeline combining DfT STATS19 collision data, AADF traffic counts,
and National Highways WebTRIS sensor data to produce **exposure-adjusted risk scores for
every road link in Yorkshire** — including the 85% of roads without traffic counters.

**Current geography:** Yorkshire pilot → Great Britain  
**Time range:** 2015–2024  
**Grain:** OS Open Roads link × year (705,672 links)

---

## What this builds

**Stage 1a — Traffic estimation**  
Predicts AADT (annual average daily traffic) for all 705k road links using a gradient
boosting model trained on 2,240 AADF count points. Fills coverage gaps on minor/unclassified
roads where DfT has no measured counts. Network centrality and population density features
give the model structural signal beyond road classification.

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
#    Required: STATS19, AADF, OS Open Roads, MRDB, LSOA centroids

# 3. Run pipeline in order
python src/road_risk/clean.py
python src/road_risk/snap.py
python src/road_risk/join.py
python src/road_risk/network_features.py   # ~5 mins, cached after first run
python src/road_risk/features.py

python src/road_risk/model.py --stage traffic     # Stage 1a: AADT estimator
python src/road_risk/model.py --stage temporal    # Stage 1b: seasonal profiles
python src/road_risk/model.py --stage collision   # Stage 2: Poisson risk model

# 4. Launch app (coming soon)
streamlit run app/main.py
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
| LSOA centroids + population | ONS | LSOA 2021 | England & Wales |

See `data/README.md` for download instructions.

---

## Repo Structure

```
road-risk-analysis/
├── src/road_risk/           # Core pipeline modules
│   ├── ingest/              # Data ingestion (STATS19, AADF, WebTRIS, MRDB, OS Roads)
│   ├── config.py            # YAML loader, paths
│   ├── clean.py             # Coordinate correction, validation, COVID flags
│   ├── snap.py              # Collision → road link snapping (weighted multi-criteria)
│   ├── join.py              # Build road_link × year feature table
│   ├── network_features.py  # Graph centrality + population density features
│   ├── features.py          # Feature engineering for model
│   └── model.py             # Stage 1a/1b/2 models
├── notebooks/               # EDA (01_eda_stats19, 02_eda_joins)
├── app/                     # Streamlit app (coming soon)
├── data/
│   ├── raw/                 # Source files — never modified, not in git
│   ├── processed/           # Cleaned parquets
│   ├── features/            # Model-ready feature tables
│   └── models/              # Saved model artefacts
├── config/settings.yaml
└── docs/
    ├── data-quality-notes.md
    └── quarto/
```

---

## Key Results (Yorkshire pilot)

| Metric | Value |
|---|---|
| Collisions loaded (2015–2024) | 101,567 |
| Collisions snapped to road links | 41,216 (40.6%) |
| Mean snap score | 0.878 |
| Mean snap distance | 14.5m |
| Road links with collisions | 22,672 |
| AADT estimator CV R² | 0.665 (baseline — improving with network features) |
| Estimated AADT range | 352–47,586 vehicles/day |

---

## Key Data Quality Notes

- **SD→SE BNG grid error** — ~9 West Yorkshire collisions corrected using LSOA-centroid
  detection. ~60k further unsnapped collisions have coordinates systematically placed in
  Lancashire — both grid reference and LSOA code wrong, unrecoverable.
- **Snap rate ceiling 40.6%** — remaining 60k collisions have genuinely wrong coordinates
  (visible as road-shaped patterns in Lancashire on scatter plot).
- **AADF coverage** — 3 years only (2019, 2021, 2023). Mean distance from collision links
  to nearest AADF count point: 22km. AADT estimated via ML for all uncounted links.
- **WebTRIS coverage** — 6–7 motorway corridors only (M62, M1, A1M etc), ~2,435 sensors.
  Dense per-sensor but narrow network coverage.

See `docs/data-quality-notes.md` for full detail.

---

## Positioning

This pipeline produces **Safety Performance Functions (SPFs)** for the full road network
using open data — extending exposure-adjusted risk analysis to the 85% of roads where
DfT currently has no traffic counts.

Compatible with ESRI/ArcGIS workflows via GeoPackage output. PostGIS backend for app queries.

---

## Requirements

```
geopandas, pandas, numpy, scikit-learn, networkx, scipy
statsmodels    # Stage 2 Poisson GLM
xgboost        # Stage 2 XGBoost
streamlit      # App (optional)
```