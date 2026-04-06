# src/road_risk — Module Tracker

Status of each module in the pipeline.

| Module | Status | Notes |
|---|---|---|
| `config.py` | ✅ Done | YAML loader, `_ROOT`, path helpers |
| `ingest/ingest_stats19.py` | ✅ Done | Loads 1979-latest CSVs, Yorkshire filter, year filter, pre-filters vehicle/casualty by collision index |
| `ingest/ingest_aadf.py` | ✅ Done | Reads from zip, bidirectional aggregation, parquet cache |
| `ingest/ingest_webtris.py` | ✅ Done | pytris API, annual reports, per-site-year chunk saves, Yorkshire active sites only |
| `ingest/ingest_mrdb.py` | ✅ Done | MRDB shapefile loader, WGS84 reproject |
| `ingest/ingest_openroads.py` | ✅ Done | OS Open Roads GeoPackage, Yorkshire bbox, road_name_clean |
| `clean.py` | ✅ Done | SD→SE LSOA-based correction, LSOA validation, COVID flag, year filter 2015–2024, WebTRIS lat/lon fix |
| `snap.py` | ✅ Done | Weighted multi-criteria snap + quick snap, densified KD-tree at 25m, 40.6% match rate |
| `join.py` | ✅ Done | Loads snapped_weighted.parquet, builds road_link × year table, AADF + WebTRIS spatial joins |
| `network_features.py` | ✅ Done | Node degree, betweenness centrality (k=200), dist_to_major, population density (LSOA) |
| `features.py` | ✅ Done | Target vars, traffic features, road encoding, temporal, lag features, confidence flags |
| `model.py` | ✅ Done | Stage 1a AADT estimator, Stage 1b temporal profiles, Stage 2 Poisson GLM + XGBoost |
| `db.py` | ⬜ Not started | PostGIS loader for app queries |
| `app/` | ⬜ Not started | Streamlit map + risk dashboard |

---

## Pipeline Run Order

```bash
# 1. Ingest — download raw files first (see data/README.md)
python src/road_risk/ingest/ingest_stats19.py
python src/road_risk/ingest/ingest_aadf.py
python src/road_risk/ingest/ingest_webtris.py   # slow ~60 mins
python src/road_risk/ingest/ingest_mrdb.py
# ingest_openroads.py runs automatically on first join.py call

# 2. Clean
python src/road_risk/clean.py

# 3. Snap collisions to road links
python src/road_risk/snap.py                    # ~8 mins

# 4. Join — road_link × year feature table
python src/road_risk/join.py

# 5. Network features — graph centrality + population density
python src/road_risk/network_features.py        # ~5 mins, cached after first run

# 6. Feature engineering
python src/road_risk/features.py

# 7. Models
python src/road_risk/model.py --stage traffic     # Stage 1a: AADT estimator
python src/road_risk/model.py --stage temporal    # Stage 1b: seasonal profiles
python src/road_risk/model.py --stage collision   # Stage 2: Poisson risk model
```

---

## Key Data Quality Notes

| Issue | Detail | Resolution |
|---|---|---|
| SD→SE BNG grid error | Officers recorded SD grid square instead of SE for some West Yorkshire collisions | Fixed in `clean.py` using LSOA centroid as ground truth — 9 confirmed corrections |
| Coordinate ceiling | ~60k collisions have coordinates in Lancashire (wrong grid + wrong LSOA) | Unrecoverable — flagged `coords_valid=False`, excluded from snap |
| Snap rate 40.6% | 60k unsnapped = systematic coordinate error, not random | Ceiling, not a bug. Scatter shows road-shaped patterns in Lancashire |
| AADF coverage 3 years | Only 2019, 2021, 2023 ingested | 7/10 years have NaN traffic features — solved by Stage 1a AADT estimator |
| AADF 22km mean distance | Collision links are mostly minor roads, AADF mostly major roads | By design — AADT estimated via ML for all uncounted links |
| WebTRIS lat/lon missing | Site coordinates dropped during groupby aggregation in clean.py | Fixed — sites.parquet merged into webtris_clean after aggregation |
| join.py snap method filter | Filter excluded "weighted" snap method, producing empty output | Fixed — `["attribute", "spatial", "weighted"]` |
| link_length_km from AADF | Only 7 rows had link length — came from AADF not OpenRoads | Fixed — added to `or_meta` merge in `build_road_link_annual()` |

---

## Current Output Files

| File | Rows | Description |
|---|---|---|
| `data/processed/stats19/collision_clean.parquet` | 101,567 | Cleaned 2015–2024 Yorkshire collisions |
| `data/processed/stats19/snapped_weighted.parquet` | 101,567 | Collisions with road link snap (41,216 matched) |
| `data/processed/aadf/aadf_clean.parquet` | 5,260 | AADF count points × 3 years |
| `data/processed/webtris/webtris_clean.parquet` | 6,516 | WebTRIS sites × 3 years with lat/lon |
| `data/features/road_link_annual.parquet` | 35,996 | Road links with collisions × year |
| `data/features/network_features.parquet` | 705,672 | Network centrality + population per link |
| `data/features/model_features.parquet` | 35,996 | Engineered features for modelling |
| `data/models/aadt_estimates.parquet` | 2,117,016 | Estimated AADT for all links × 3 years |
| `data/models/risk_scores.parquet` | — | Collision risk scores (after Stage 2) |

---

## Confidence Tiers

| Tier | Outputs | Basis |
|---|---|---|
| **High** | Risk per road segment, comparisons within road type, major road analysis | Measured AADF + snapped collisions |
| **Medium** | Minor road risk estimates, network-derived features | Modelled AADT, sparser collision data |
| **Exploratory** | Vehicle-type risk across full network, fine-grained local conclusions | Inferred exposure, noisy denominators |

---

## Pending

- Download additional AADF years (2015–2018, 2020, 2022, 2024) to enable collision rate for all years
- Speed limit feature via OSM/osmnx
- `db.py` — PostGIS loader
- Streamlit app
- DVSA test route case study notebook
- Kaggle dataset (processed parquets) + `data/README.md` with download instructions