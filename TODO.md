# TODO

Tracked here so nothing gets lost. Cross off as done.

---

## 🔴 High Priority

- [ ] Remove `_fix_sd_se_error()` from `clean.py` — function is now a no-op since
  the root cause was wrong force codes, not a BNG grid error. Dead code that will
  confuse future readers. Remove entirely.

- [ ] Fix temporal trend chart in `03_model_results.ipynb` — observed collision bars
  are invisible because the y-axis is dominated by the predicted × 1000 line. Either
  drop the ×1000 scaling or use a secondary y-axis.

- [ ] Fix middle panel in risk score distribution plot (section 5) — flat uniform
  distribution because it's plotting risk_percentile which is by construction uniform.
  Replace with collision count distribution on collision links instead.

- [ ] Download additional AADF years (2015–2018, 2020, 2022, 2024) — only 3 years
  currently (2019/2021/2023). More years would allow the collision model to use
  measured AADT for all 10 years rather than estimating it.

- [ ] Fix `pct_attribute_snapped` in `road_link_annual` — always 0 because snap
  method name changed to "weighted". Column is misleading, should be removed or
  recalculated as `pct_weighted_snapped`.

---

## 🟡 Medium Priority — Model

- [ ] Drop raw `betweenness` from GLM — coefficient is −8 which dominates the
  coefficient chart and is a multicollinearity symptom. Keep `betweenness_relative`
  which is the cleaner feature. XGBoost handles both fine via tree splits.

- [ ] Investigate `speed_limit_mph` and `lanes_imputed` positive coefficients in GLM
  — both flipped sign vs expectation. With 43.6% and 5.5% coverage respectively and
  median imputation for the rest, these may reflect imputation bias rather than
  genuine effects. Consider excluding from GLM or investigating coverage patterns.

- [ ] Validate AADT estimates on WebTRIS motorway corridors — compare predicted vs
  measured flow on M62/M1 as sense check. Currently evaluated only on AADF holdout.

- [ ] Add urban/rural ONS classification — easy LSOA join, important for avoiding
  regional bias when scaling beyond Yorkshire.

- [ ] Stage 1c vehicle mix model — predict % HGV/LGV/car per link from road type +
  network features. Enables vehicle-type risk analysis on uncounted roads.

---

## 🟢 Infrastructure / Output

- [ ] `db.py` — PostGIS loader for all processed parquets + model outputs. Required
  for Streamlit app queries.

- [ ] Streamlit app skeleton — map with road links coloured by risk percentile,
  sidebar filters for road type / year / severity.

- [ ] GeoPackage export — ESRI-compatible output layer (link_id, geometry,
  estimated_aadt, risk_percentile, road_classification). Demonstrates ESRI
  integration story for DfT/DVSA conversations.

- [ ] `data/README.md` — download instructions for all large raw files not in git
  (STATS19 CSV, OS Open Roads GeoPackage, AADF zip, OSM pbf files, MRDB).

- [ ] Kaggle dataset — upload processed parquets so others can skip ingest/clean/snap.

---

## 🔵 Applications / Demonstrations

- [ ] DVSA test route notebook — fetch 3–5 Yorkshire test centre routes (GPX or
  manual coordinates), snap to OS Open Roads, score using risk_percentile. Shows
  which centres have structurally harder routes per vehicle-km.

- [ ] Risk-normalised output table — "Top 1% highest-risk road segments controlling
  for traffic" as a clean publishable output.

- [ ] Seasonal risk analysis — combine Stage 2 risk scores with Stage 1b temporal
  profiles. Do high-risk roads have worse seasonal variation?


---

## ✅ Done

- [x] STATS19 ingestion (102,361 Yorkshire collisions 2015–2024, forces 12/13/14/16)
- [x] AADF ingestion (5,260 rows, 2019/2021/2023)
- [x] WebTRIS ingestion (6,516 site × year rows)
- [x] MRDB ingestion (1,948 Yorkshire major road links)
- [x] OS Open Roads ingestion (705,672 links, Yorkshire bbox)
- [x] **Force code bug fixed** — was loading Lancashire/NW England (codes 4–7),
      now correctly loads Yorkshire (codes 12/13/14/16). Pipeline re-run April 2026.
- [x] LSOA coordinate validation (146 flagged suspect, 0.1%)
- [x] Weighted multi-criteria snap (100,952 matched, 98.6%, score 0.860, dist 16.6m)
- [x] Snap quality filter in join.py (score ≥ 0.6, retains 97% of matches)
- [x] `road_link_annual.parquet` (84,146 rows × 37 cols, 49,247 links, 2015–2024)
- [x] `network_features.py` — degree, betweenness, betweenness_relative,
      dist_to_major, pop_density, speed_limit_mph, lanes, lit, is_unpaved
- [x] `model.py` Stage 1a — AADT estimator (CV R² 0.723, 17 features)
- [x] `model.py` Stage 1b — temporal profiles (WebTRIS seasonal indices)
- [x] `model.py` Stage 2 — Poisson GLM (pseudo-R² 0.083) + XGBoost (0.330)
- [x] `03_model_results.ipynb` — full model results notebook with maps
- [x] `stats19_coordinate_issue.ipynb` — standalone DfT data quality report
- [x] README.md, CODE_README.md, TODO.md — updated April 2026
- [x] Hardcoded values audit — documented in CODE_README with derivation sources
- [x] OSM features via osmnx + osmium (Geofabrik Yorkshire county pbf files)
- [x] STATS19 contextual aggregates in join.py (pct_urban, pct_dark, pct_junction)