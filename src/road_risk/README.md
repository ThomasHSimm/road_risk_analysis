# src/road_risk — Module Tracker

Status of each module in the pipeline.

| Module | Status | Notes |
|---|---|---|
| `config.py` | ✅ Done | YAML loader, `_ROOT`, path helpers |
| `ingest/ingest_stats19.py` | ✅ Done | Loads 1979-latest CSVs, Yorkshire filter, pre-filters vehicle/casualty by collision index |
| `ingest/ingest_aadf.py` | ✅ Done | Reads from zip, bidirectional aggregation, parquet cache |
| `ingest/ingest_webtris.py` | ✅ Done | pytris API, annual reports, per-site-year chunk saves, Yorkshire active sites only |
| `ingest/ingest_mrdb.py` | ✅ Done | MRDB shapefile loader — road geometry, WGS84 reproject |
| `ingest/ingest_openroads.py` | ✅ Done | OS Open Roads GeoPackage, Yorkshire bbox, road_name_clean + street_name_clean |
| `clean.py` | ✅ Done | SD→SE coordinate correction, LSOA validation, COVID flag, target year filters |
| `snap.py` | ✅ Done | Weighted multi-criteria snap + quick snap, densified geometry KD-tree |
| `join.py` | 🔄 Needs update | Update to use snap.py output instead of inline snap logic |
| `features.py` | ⬜ Not started | Feature engineering for model |
| `model.py` | ⬜ Not started | XGBoost/RandomForest, SHAP explainability |
| `db.py` | ⬜ Not started | PostGIS loader |
| `app/` | ⬜ Not started | Streamlit app |

---

## Pipeline Run Order

```bash
# 1. Ingest — download raw files first (see docs/data-sources/)
python src/road_risk/ingest/ingest_stats19.py
python src/road_risk/ingest/ingest_aadf.py
python src/road_risk/ingest/ingest_webtris.py   # slow — ~60 mins
python src/road_risk/ingest/ingest_mrdb.py
# ingest_openroads.py runs automatically on first join.py call

# 2. Clean
python src/road_risk/clean.py

# 3. Snap collisions to road links
python src/road_risk/snap.py

# 4. Join — build road_link × year feature table
python src/road_risk/join.py

# 5. Features (TODO)
# python src/road_risk/features.py

# 6. Model (TODO)
# python src/road_risk/model.py
```

---

## Key Data Quality Findings

See `docs/data-quality-notes.md` for full details. Summary:

- **SD→SE grid error** — ~60k West Yorkshire collisions have easting 100km too
  far west due to BNG grid letter error. Fixed in `clean.py`.
- **Road numbers unreliable** — STATS19 `first_road_number` has systematic
  errors. Used only as low-weight (10%) input in `snap.py`, not as primary
  join key.
- **Post-2015 coordinates reliable** — near 100% valid once SD→SE fix applied.
  Pipeline filters to 2015–2024.
- **Snap rate ~80%** after SD→SE fix, using densified geometry KD-tree at 25m
  interval with weighted multi-criteria scoring.

---

## Data Sources

| Source | Location | Coverage |
|---|---|---|
| STATS19 collisions | `data/raw/stats19/` | Yorkshire 2015–2024 |
| AADF traffic counts | `data/raw/aadf/` | Yorkshire 2019, 2021, 2023 |
| WebTRIS sensor data | `data/raw/webtris/` | Yorkshire motorways/trunk roads 2019, 2021, 2023 |
| OS Open Roads | `data/raw/shapefiles/oproad_gb.gpkg` | Yorkshire + 20km buffer |
| MRDB | `data/raw/shapefiles/MRDB_2024_published.shp` | Yorkshire major roads |
| LSOA centroids | `data/raw/stats19/lsoa_centroids.csv` | England & Wales 2021 |