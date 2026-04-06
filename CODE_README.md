# src/road_risk — Module Tracker

Status of each module in the pipeline.

| Module | Status | Notes |
|---|---|---|
| `config.py` | ✅ Done | YAML loader, `_ROOT`, path helpers |
| `ingest/ingest_stats19.py` | ✅ Done | Loads 1979-latest CSVs, Yorkshire filter (forces 12/13/14/16), pre-filters vehicle/casualty by collision index |
| `ingest/ingest_aadf.py` | ✅ Done | Reads from zip, bidirectional aggregation, parquet cache |
| `ingest/ingest_webtris.py` | ✅ Done | WebTRIS API, annual reports, per-site-year chunk saves, Yorkshire active sites only |
| `ingest/ingest_openroads.py` | ✅ Done | OS Open Roads GeoPackage, Yorkshire bbox, road_name_clean + street_name_clean |
| `clean.py` | ✅ Done | LSOA validation, COVID flag, target year filters. Note: BNG easting/northing fields have known DfT error — lat/lon used throughout |
| `snap.py` | ✅ Done | Weighted multi-criteria snap + quick snap, densified geometry KD-tree, 98.6% match rate |
| `join.py` | ✅ Done | road_link × year table, AADF join, WebTRIS join, snap quality filter (score ≥ 0.6), STATS19 context aggregates |
| `network_features.py` | ✅ Done | Betweenness, degree, dist_to_major, pop_density, betweenness_relative, OSM speed/lanes/surface/lit |
| `features.py` | ✅ Done | Target vars, traffic features, road encoding, temporal, lag, confidence flags |
| `model.py` | ✅ Done | Stage 1a AADT estimator (CV R² 0.723), Stage 1b temporal profiles, Stage 2 Poisson GLM + XGBoost |
| `db.py` | ⬜ Not started | PostGIS loader |
| `app/` | ⬜ Not started | Streamlit app |

---

## Pipeline Run Order

```bash
# 1. Ingest — download raw files first (see data/README.md)
python src/road_risk/ingest/ingest_stats19.py
python src/road_risk/ingest/ingest_aadf.py
python src/road_risk/ingest/ingest_webtris.py   # slow — ~60 mins
python src/road_risk/ingest/ingest_openroads.py

# 2. Convert OSM pbf files (download county files from Geofabrik first)
#    https://download.geofabrik.de/europe/great-britain/england/
for f in data/raw/osm/*.osm.pbf; do
    osmium cat "$f" -o "${f%.osm.pbf}.osm"
done

# 3. Clean
python src/road_risk/clean.py

# 4. Snap collisions to road links
python src/road_risk/snap.py

# 5. Join — build road_link × year feature table
python src/road_risk/join.py

# 6. Network features (first run ~10 mins, cached after)
python src/road_risk/network_features.py
# Add OSM attributes (speed limit, lanes, surface, lit) — ~15 mins first run
python src/road_risk/network_features.py --osm

# 7. Models
python src/road_risk/model.py --stage traffic     # Stage 1a: AADT estimator
python src/road_risk/model.py --stage temporal    # Stage 1b: seasonal profiles
python src/road_risk/model.py --stage collision   # Stage 2: Poisson risk model
```

---

## Key Data Quality Findings

See `docs/data-quality-notes.md` for full detail. Summary:

- **STATS19 police force code bug (fixed April 2026)** — `config/settings.yaml`
  previously used codes 4–7 (Lancashire/Merseyside/GM/Cheshire) instead of 12–16
  (Yorkshire). All pipeline outputs before this fix used NW England data.
  Codes are now documented with a derivation snippet using the DfT data guide Excel.

- **STATS19 BNG coordinate field error (reported to DfT)** — `location_easting_osgr`
  and `location_northing_osgr` have a systematic grid square prefix error for Yorkshire
  forces. The `latitude`/`longitude` fields are correct (GPS-derived) and are used
  throughout. See `notebooks/stats19_coordinate_issue.ipynb` for the full reproducible
  report. The BNG fields are not used in this pipeline.

- **Snap rate 98.6%** at mean 16.6m — achieved after force code fix on correct
  Yorkshire data. A snap quality filter (score ≥ 0.6) removes a further 3% of
  ambiguous matches.

- **AADF coverage** — 3 years (2019, 2021, 2023), ~287k links matched within 2km.
  Remaining ~418k links have AADT estimated by Stage 1a model.

- **OSM coverage** — speed limit 43.6%, lanes 5.5%, surface 12.8%, lit 7.2%.
  Sparse features median-imputed in GLM.

---

## Hardcoded Values — Source Reference

| Value | File | Derivable from? |
|---|---|---|
| Police force codes 12/13/14/16 | `config/settings.yaml`, `ingest_stats19.py` | DfT data guide Excel — `police_force` field |
| HGV vehicle types {19,20,21} | `join.py` | DfT data guide Excel — `vehicle_type` field |
| Road class scores (1=Motorway etc) | `snap.py` | DfT data guide Excel — `first_road_class` field |
| Junction detail codes | `snap.py` | DfT data guide Excel — `junction_detail` field |
| COVID years {2020, 2021} | `clean.py`, `model.py` | Domain knowledge — not in Excel |
| Yorkshire bbox BNG | `ingest_openroads.py` | Spatial — not in Excel |

---

## Model Results Summary (April 2026)

**Stage 1a — AADT Estimator**
- CV R²: 0.723 (±0.029) | Training R²: 0.901
- 17 features: road_class_ord, hgv_proportion, pop_density, betweenness, dist_to_major,
  lat/lon, link_length, speed_limit_mph, lanes, betweenness_relative, lit, is_unpaved
- Applied to 705,672 links × 3 years

**Stage 2 — Collision Model**
- Poisson GLM pseudo-R²: 0.083 | XGBoost pseudo-R²: 0.330
- 1,910,145 training rows
- Key GLM effects: degree_mean +0.71, betweenness_relative +0.28,
  is_a_road -1.21, is_motorway -1.13, year_norm -0.035
- XGBoost top feature: estimated_aadt (0.305) — exposure dominates

---

## Data Sources

| Source | Location | Coverage |
|---|---|---|
| STATS19 collisions | `data/raw/stats19/` | Yorkshire 2015–2024 |
| AADF traffic counts | `data/raw/aadf/` | Yorkshire 2019, 2021, 2023 |
| WebTRIS sensor data | `data/raw/webtris/` | Yorkshire motorways/trunk 2019, 2021, 2023 |
| OS Open Roads | `data/raw/shapefiles/oproad_gb.gpkg` | Yorkshire + 20km buffer |
| MRDB | `data/raw/shapefiles/MRDB_2024_published.shp` | Yorkshire major roads |
| OSM pbf files | `data/raw/osm/*.osm` | 4 Yorkshire county files from Geofabrik |
| LSOA population + area | `data/raw/stats19/lsoa_*.csv` | England & Wales 2021 |
| DfT data guide Excel | `data/raw/stats19/dft-road-casualty-*-data-guide-2024.xlsx` | Code lookups |