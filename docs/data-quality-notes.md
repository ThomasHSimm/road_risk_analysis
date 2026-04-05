# Data Quality Notes

This document records data quality issues found during the pipeline build,
the evidence for each, and how they are handled in the code.

---

## STATS19 — Collision Coordinates

### SD→SE BNG Grid Letter Error

**Severity:** High — affects ~60,000 collisions (~60% of unmatched records)

**Symptom:**
West Yorkshire collisions (police_force = 4, West Yorkshire) with valid-looking
lat/lon coordinates that place them in Lancashire, typically 80-100km west of
their true location.

**Evidence:**
```
Collision lat/lon: 53.763697, -2.694727  (appears to be near Burnley, Lancashire)
Recorded LSOA: E01025226                 (West Yorkshire LSOA)
location_easting_osgr: 354300            (Lancashire range, should be ~454300)
location_northing_osgr: 429929           (correct for West Yorkshire)
Nearest OS Open Roads link: 5,584m away  (should be <100m)
```

**Root cause:**
British National Grid (BNG) uses two-letter 100km grid square prefixes.
Yorkshire sits mainly in grid squares `SE` and `TA`. Lancashire sits in `SD`.
A numeric grid reference like `354300 429929` is valid in both `SD` and `SE`
but refers to locations 100km apart. The officer or recording system used `SD`
when the correct square was `SE`, producing coordinates that appear plausible
(valid lat/lon within GB bounds) but are systematically wrong.

**Detection logic** (in `clean.py → _fix_sd_se_error()`):
- Yorkshire northing: `location_northing_osgr` between 390,000 and 540,000
- Lancashire easting: `location_easting_osgr` < 400,000
- Both conditions together indicate SD→SE error

**Fix:**
Add 100,000 to `location_easting_osgr`, re-derive `latitude` and `longitude`
from corrected BNG using pyproj. Corrected records flagged with
`coords_corrected = True`.

**Validation:**
After correction, coordinates land in lon range -1.7 to -0.8 (correct for
West Yorkshire). Confirmed by checking sample corrected records against their
recorded LSOA centroids.

**Code location:** `src/road_risk/clean.py` → `_fix_sd_se_error()`

---

### LSOA Coordinate Validation

**Purpose:** Catch any remaining coordinate errors not covered by the SD→SE fix,
and provide a principled quality flag for use in modelling.

**Method:**
Join each collision to its recorded `lsoa_of_accident_location` using ONS 2021
LSOA population-weighted centroids. Compute BNG distance between collision
coordinates and LSOA centroid. Flag collisions more than 10km from their LSOA
centroid as `coords_suspect = True`.

A collision being far from its LSOA centroid is strong evidence of a coordinate
error — the officer records the LSOA independently of the grid reference, so
disagreement between the two suggests the grid reference is wrong.

**Source file:** `data/raw/stats19/lsoa_centroids.csv`
Downloaded from ONS Open Geography Portal:
LSOA (Dec 2021) EW Population Weighted Centroids V4

**Output columns added:**
- `coords_corrected` (bool) — True if SD→SE correction was applied
- `coords_suspect` (bool) — True if >10km from LSOA centroid after correction
- `lsoa_dist_m` (float) — distance in metres from LSOA centroid
- `coords_valid` (bool) — False if outside GB bounds OR coords_suspect

**Code location:** `src/road_risk/clean.py` → `_validate_lsoa_coords()`

---

### Post-2015 Coordinate Quality

**Finding:** Post-2015 Yorkshire STATS19 data has near-100% valid coordinates
once the SD→SE error is corrected. Pre-2015 data has much higher rates of
missing/invalid coordinates because GPS recording was not universal.

**Decision:** Pipeline filters to 2015–2024 to avoid the pre-GPS coordinate
quality issues. This is set in `clean.py → main()`.

---

## STATS19 — Road Number Quality

### Road Numbers Are Unreliable for Spatial Joining

**Finding:**
STATS19 `first_road_number` contains systematic errors — e.g. a collision near
Burnley (West Yorkshire) recorded with road number 23, producing `road_name_clean = A23`
which is a Surrey/London road. Analysis showed 709 unique road names in Yorkshire
STATS19 but only 97 overlapping with Yorkshire OS Open Roads links.

**Root cause:**
Road numbers are manually entered by the attending officer. Errors include:
- Wrong road number (e.g. A23 in Yorkshire)
- Road numbers from adjacent force areas
- Numbers that exist in other regions but not Yorkshire

**Decision:**
Road number is used as a low-weight (10%) scoring input in `snap_weighted()`
rather than a primary join key. Weight is set to `W_NUMBER = 0.10` in
`src/road_risk/snap.py`. The DfT themselves do not perform road network
matching in published STATS19 data — coordinates are the primary spatial
reference.

**Reference:** DfT STATS19 Review (2021) roadmap explicitly lists road network
matching as a *future* priority, confirming it has not been done in official
publications.

**Code location:** `src/road_risk/snap.py` → `W_NUMBER = 0.10`

---

## STATS19 — Snap Rate

**Achieved snap rate:** ~80% (after SD→SE correction) of 2015–2024
Yorkshire collisions successfully snapped to OS Open Roads links.

**Method:** `snap_weighted()` in `src/road_risk/snap.py`
- KD-tree built on road geometry densified at 25m intervals (~3.9M points)
- Top K=20 candidate links within 500m search radius per collision
- Composite scoring across 4 dimensions: spatial distance (40%), road
  classification (25%), junction/form-of-way (25%), road number (10%)
- Mean snap distance for matched collisions: ~14.5m
- Mean composite score for matched collisions: 0.878 / 1.0

**Remaining unmatched (~20%):**
After SD→SE correction, residual unmatched collisions are likely to have other
coordinate errors (different systematic recording issues, or genuinely ambiguous
locations). These are retained in the dataset with `snap_method = 'unmatched'`
and excluded from road-link analysis but included in aggregate statistics.

---

## AADF — Count Point Coverage

**Finding:** AADF count points cover ~62% of OS Open Roads links within 2km.
Major roads (motorways, A roads) have near-complete coverage. Minor roads and
unclassified roads have sparse coverage.

**Handling:** AADF features are NaN for links beyond 2km from any count point.
The 2km cap is set in `src/road_risk/join.py → build_road_features()`.

---

## WebTRIS — Coverage

**Coverage:** National Highways network only — motorways and major trunk roads.
In Yorkshire: M1, M62, M18, M621, A1(M), A64(M) corridors.

**Finding:** Initial pull attempted all 19,518 GB sites. Filtered to 2,571
active Yorkshire sites after applying bounding box and Active status filter.

**Pull years:** 2019, 2021, 2023 (pre-COVID, COVID anomaly, recent normal).
Full 10-year pull was impractical (~25 hours). Three representative years
gives sufficient temporal coverage for the model.

---

## OS Open Roads — Yorkshire Bounding Box

**Issue:** Initial bbox clipped road links that serve collisions near the
Yorkshire boundary, particularly in the west (Lancashire border).

**Fix:** Bbox widened by 20km on all sides:
```python
# Before
YORKSHIRE_BBOX_BNG = (390000, 370000, 570000, 520000)
# After
YORKSHIRE_BBOX_BNG = (370000, 350000, 590000, 540000)
```

This increased link count from 457,884 to 705,672 and improved snap coverage
for boundary-area collisions.