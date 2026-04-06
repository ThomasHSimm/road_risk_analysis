# TODO

Tracked here so nothing gets lost. Cross off as done.

---

## 🔴 High Priority — Core Pipeline

- [ ] Run `network_features.py` to completion and verify betweenness/degree/dist_to_major/pop_density outputs
- [ ] Retrain Stage 1a AADT model with network features — confirm CV R² improves from 0.665 and `road_class_ord` importance drops from 0.94
- [ ] Run Stage 2 collision model (`model.py --stage collision`) — get first Poisson GLM pseudo-R² and risk scores
- [ ] Download additional AADF years (2015–2018, 2020, 2022, 2024) from DfT road traffic statistics portal — enables collision rate for all 10 years
- [ ] Fix `pct_attribute_snapped` column in `road_link_annual` — always 0 because snap method name changed to "weighted". Remove or recalculate in `join.py`

---

## 🟡 Medium Priority — Model Improvements

- [ ] Add speed limit feature via OSM — `osmnx` `maxspeed` tag on edges. Low effort, decent signal for road design intent
- [ ] Add urban/rural LSOA classification — ONS publishes urban/rural flag per LSOA, easy join. Important for avoiding regional bias when scaling beyond Yorkshire
- [ ] Add land use features — OS OpenMap Local or OSM tags (residential/commercial/industrial). Higher signal but more work
- [ ] Validate AADT estimates on WebTRIS motorway corridors — compare predicted vs measured flow on M62/M1 as a sense check. Currently the model is evaluated only on AADF holdout
- [ ] Investigate snap rate ceiling — audit ~50 random unsnapped collisions manually. Confirm whether bias is toward junctions/complex roads (affects Stage 2 conclusions)
- [ ] Stage 1c vehicle mix model — predict % HGV/LGV/car per link from road type + network features. Enables vehicle-type risk analysis on uncounted roads

---

## 🟢 Infrastructure / Output

- [ ] `db.py` — PostGIS loader for all processed parquets + model outputs. Required for Streamlit app queries
- [ ] Streamlit app skeleton — basic map with road links coloured by risk percentile, sidebar filters for road type / year / severity
- [ ] GeoPackage export — produce ESRI-compatible output layer (link_id, geometry, estimated_aadt, risk_percentile, road_classification). Demonstrates ESRI integration story
- [ ] Kaggle dataset — upload processed parquets so others can skip ingest/clean/snap pipeline. Saves ~2 hrs of compute on first run
- [ ] `data/README.md` — download instructions for all large raw files not in git (STATS19 CSV, OS Open Roads GeoPackage, AADF zip)

---

## 🔵 Applications / Demonstrations

- [ ] DVSA test route notebook — fetch 3–5 Yorkshire test centre routes (community-mapped GPX), snap to OS Open Roads, score using risk_percentile. Show which centres have structurally harder routes
- [ ] Risk-normalised output table — "Top 1% highest-risk road segments controlling for traffic" as a clean output. The headline deliverable for any stakeholder conversation
- [ ] Seasonal risk analysis — combine Stage 2 risk scores with Stage 1b temporal profiles. Do high-risk roads have worse seasonal variation?

---

## ✅ Done

- [x] STATS19 ingestion (101,567 Yorkshire collisions 2015–2024)
- [x] AADF ingestion (5,260 rows, 2019/2021/2023)
- [x] WebTRIS ingestion (6,516 site × year rows, lat/lon fix)
- [x] MRDB ingestion (1,948 Yorkshire major road links)
- [x] OS Open Roads ingestion (705,672 links, Yorkshire bbox)
- [x] SD→SE BNG coordinate correction (LSOA-centroid detection, 9 confirmed)
- [x] LSOA validation (20 flagged suspect, 0.02%)
- [x] Weighted multi-criteria snap (41,216 matched, score 0.878, dist 14.5m)
- [x] `join.py` snap method filter bug fixed (was excluding "weighted" method → empty output)
- [x] `join.py` main() bug fixed (was re-running weak snap instead of loading parquet)
- [x] `join.py` WebTRIS lat/lon bug fixed (coordinates lost during groupby — merged from sites.parquet)
- [x] `road_link_annual.parquet` (35,996 rows × 37 cols, 22,672 links, 2015–2024)
- [x] `features.py` — target vars, traffic features, road encoding, temporal, lag, confidence flags
- [x] `network_features.py` — degree, betweenness, dist_to_major, pop_density (running)
- [x] `model.py` Stage 1a — AADT estimator (CV R² 0.665 baseline)
- [x] `model.py` Stage 1b — temporal profiles (WebTRIS seasonal indices)
- [x] `model.py` Stage 2 — Poisson GLM + XGBoost collision model (written, not yet run)
- [x] README.md — project overview, quick start, results
- [x] CODE_README.md — module tracker, pipeline order, data quality log
- [x] TODO.md — this file