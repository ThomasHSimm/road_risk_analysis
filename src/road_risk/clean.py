"""
clean.py
--------
Per-source cleaning functions for road risk pipeline.

Each function accepts the raw loaded DataFrame(s) and returns a cleaned
version ready for joining. Cleaning is kept separate from ingest so the
raw data is never overwritten and cleaning rules are testable in isolation.

Functions
---------
clean_stats19(data)   : drop historic cols, flag COVID, validate coords
clean_aadf(df)        : filter to target years, validate flows, add road_name_clean
clean_webtris(df)     : drop duplicates, aggregate monthly → annual
clean_mrdb(gdf)       : add link_id, derive road_name_clean for joining
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj

from road_risk.config import _ROOT, cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WebTRIS pull years — must match ingest_webtris.TARGET_YEARS
TARGET_YEARS = [2019, 2021, 2023]

# STATS19 collision columns that are historic/superseded — drop these
HISTORIC_COLS = [
    "junction_detail_historic",
    "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic",
    "carriageway_hazards_historic",
    "local_authority_highway_current",  # superseded by local_authority_highway
]

# STATS19 first_road_class code → road prefix for name reconstruction
ROAD_CLASS_PREFIX = {
    1: "M",   # Motorway
    2: "A",   # A(M) — motorway-standard A road
    3: "A",   # A road
    4: "B",   # B road
    5: "C",   # C road
    6: "",    # Unclassified
}

# GB bounding box for coordinate validation
GB_LAT = (49.9, 60.9)
GB_LON = (-8.2, 2.0)

# COVID years flag
COVID_YEARS = {2020, 2021}


# ---------------------------------------------------------------------------
# STATS19
# ---------------------------------------------------------------------------

def clean_stats19(
    data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Clean all three STATS19 tables.

    Changes applied
    ---------------
    Collisions:
      - Drop historic / superseded columns
      - Parse date → datetime, extract year/month/hour
      - Validate lat/lon within GB bounds (flag bad coords, don't drop)
      - Add is_covid boolean flag
      - Reconstruct road_name_clean from first_road_class + first_road_number
        (used as join key to MRDB/AADF in join.py Stage 1)

    Vehicles:
      - No structural changes — column names already normalised by ingest

    Casualties:
      - No structural changes

    Parameters
    ----------
    data : dict returned by load_stats19() — keys collision/vehicle/casualty

    Returns
    -------
    dict with same keys, cleaned DataFrames
    """
    cleaned = {}

    # --- Collisions ---------------------------------------------------------
    if "collision" in data:
        col = data["collision"].copy()
        logger.info(f"Cleaning collisions: {len(col):,} rows")

        # Drop historic columns (only those present — years vary)
        drop = [c for c in HISTORIC_COLS if c in col.columns]
        col = col.drop(columns=drop)
        logger.info(f"  Dropped {len(drop)} historic columns: {drop}")

        # Parse date
        if "date" in col.columns and not pd.api.types.is_datetime64_any_dtype(col["date"]):
            col["date"] = pd.to_datetime(col["date"], dayfirst=True, errors="coerce")

        # Derive time features
        if "date" in col.columns:
            col["month"] = col["date"].dt.month
            col["day_name"] = col["date"].dt.day_name()

        if "time" in col.columns:
            col["hour"] = pd.to_datetime(
                col["time"], format="%H:%M", errors="coerce"
            ).dt.hour

        # Use collision_year if available, else derive from date
        if "collision_year" not in col.columns and "date" in col.columns:
            col["collision_year"] = col["date"].dt.year

        # COVID flag
        if "collision_year" in col.columns:
            col["is_covid"] = col["collision_year"].isin(COVID_YEARS)
            logger.info(
                f"  COVID rows flagged: {col['is_covid'].sum():,} "
                f"({col['is_covid'].mean():.1%})"
            )

        # --- SD→SE grid letter correction ------------------------------------
        # ~60k West Yorkshire collisions have easting in Lancashire range
        # (330k-400k) due to SD grid square being recorded instead of SE.
        # Fix: add 100,000 to easting and re-derive lat/lon from corrected BNG.
        col = _fix_sd_se_error(col)

        # --- LSOA coordinate validation --------------------------------------
        # Cross-check collision coordinates against its recorded LSOA centroid.
        # Collisions >10km from their LSOA centroid have suspect coordinates.
        col = _validate_lsoa_coords(col)

        # Final coords_valid flag — False if outside GB bounds OR suspect
        if "latitude" in col.columns and "longitude" in col.columns:
            bad_lat = ~col["latitude"].between(*GB_LAT) | col["latitude"].isna()
            bad_lon = ~col["longitude"].between(*GB_LON) | col["longitude"].isna()
            bad_coords = bad_lat | bad_lon
            if "coords_suspect" in col.columns:
                bad_coords = bad_coords | col["coords_suspect"]
            col["coords_valid"] = ~bad_coords
            n_bad = bad_coords.sum()
            if n_bad:
                logger.warning(
                    f"  {n_bad:,} rows ({n_bad/len(col):.1%}) have invalid/suspect "
                    f"coordinates — flagged as coords_valid=False (not dropped)"
                )

        # Reconstruct road_name_clean for Stage 1 snap in join.py
        col = _add_road_name_clean(col)

        cleaned["collision"] = col
        logger.info(f"  Collisions cleaned: {len(col):,} rows × {col.shape[1]} cols")

    # --- Vehicles -----------------------------------------------------------
    if "vehicle" in data:
        veh = data["vehicle"].copy()
        # No structural changes needed — normalisation done in ingest
        cleaned["vehicle"] = veh
        logger.info(f"Vehicles: {len(veh):,} rows (no changes)")

    # --- Casualties ---------------------------------------------------------
    if "casualty" in data:
        cas = data["casualty"].copy()
        cleaned["casualty"] = cas
        logger.info(f"Casualties: {len(cas):,} rows (no changes)")

    return cleaned



# ---------------------------------------------------------------------------
# Coordinate correction helpers
# ---------------------------------------------------------------------------

# BNG → WGS84 transformer (shared across calls)
_BNG_TO_WGS84 = pyproj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# Yorkshire northing range in BNG
YORKSHIRE_NORTHING = (390000, 540000)

# Easting threshold below which SD→SE error is suspected
SD_SE_EASTING_THRESHOLD = 400000

# LSOA centroid file — place in data/raw/stats19/
LSOA_CENTROIDS_PATH = _ROOT / "data/raw/stats19/lsoa_centroids.csv"

# Distance threshold for LSOA validation (metres)
LSOA_DIST_THRESHOLD_M = 10000


def _fix_sd_se_error(col: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and fix the SD→SE BNG grid letter error in STATS19 coordinates.

    West Yorkshire collisions sometimes have easting values in the Lancashire
    range (~330k-400k) because the officer recorded grid square SD instead of
    SE. Both squares use the same numeric suffix, but SD is 100km further west.

    Detection uses LSOA centroid as ground truth:
      - Collision easting < 400k (candidate SD error)
      - AND the recorded LSOA centroid easting > 400k (LSOA is genuinely in Yorkshire)
    This avoids incorrectly correcting legitimate western Yorkshire collisions
    (Calderdale, Kirklees) where both the collision and LSOA centroid are < 400k.

    Fix: add 100,000 to easting, re-derive lat/lon from corrected BNG.
    """
    if "location_easting_osgr" not in col.columns or        "location_northing_osgr" not in col.columns:
        col["coords_corrected"] = False
        return col

    col["coords_corrected"] = False

    # Only proceed if LSOA centroids are available
    if not LSOA_CENTROIDS_PATH.exists():
        logger.warning(
            "  LSOA centroids not found — using easting threshold for SD→SE detection "
            "(may over-correct western Yorkshire collisions)"
        )
        yorkshire_northing = col["location_northing_osgr"].between(*YORKSHIRE_NORTHING)
        lancashire_easting = (
            col["location_easting_osgr"] < SD_SE_EASTING_THRESHOLD
        ) & col["location_easting_osgr"].notna()
        sd_se_mask = yorkshire_northing & lancashire_easting
    else:
        # Load LSOA centroids to use as ground truth
        lsoa = pd.read_csv(
            LSOA_CENTROIDS_PATH,
            usecols=["LSOA21CD", "x"],
            encoding="utf-8-sig",
        ).rename(columns={"x": "lsoa_e"})

        col_with_lsoa = col.merge(
            lsoa, left_on="lsoa_of_accident_location",
            right_on="LSOA21CD", how="left"
        )

        # SD→SE error: collision easting in Lancashire range,
        # but LSOA centroid is in Yorkshire (easting > 400k)
        collision_in_lancashire = (
            col_with_lsoa["location_easting_osgr"] < SD_SE_EASTING_THRESHOLD
        ) & col_with_lsoa["location_easting_osgr"].notna()

        lsoa_in_yorkshire = (
            col_with_lsoa["lsoa_e"] >= SD_SE_EASTING_THRESHOLD
        ) & col_with_lsoa["lsoa_e"].notna()

        sd_se_mask_values = (collision_in_lancashire & lsoa_in_yorkshire).values
        sd_se_mask = pd.Series(sd_se_mask_values, index=col.index)

        logger.info(
            f"  SD→SE detection using LSOA centroids: "
            f"{collision_in_lancashire.sum():,} candidates → "
            f"{sd_se_mask.sum():,} confirmed (LSOA centroid in Yorkshire)"
        )

    n = sd_se_mask.sum()
    col["coords_corrected"] = sd_se_mask

    if n == 0:
        logger.info("  SD→SE grid correction: no errors detected")
        return col

    corrected_e = col.loc[sd_se_mask, "location_easting_osgr"] + 100_000
    corrected_n = col.loc[sd_se_mask, "location_northing_osgr"]

    lon_c, lat_c = _BNG_TO_WGS84.transform(corrected_e.values, corrected_n.values)

    col.loc[sd_se_mask, "location_easting_osgr"] = corrected_e.values
    col.loc[sd_se_mask, "longitude"]             = lon_c
    col.loc[sd_se_mask, "latitude"]              = lat_c

    logger.info(
        f"  SD→SE grid correction: {n:,} collisions corrected "
        f"(easting +100km, lat/lon re-derived)"
    )
    return col


def _validate_lsoa_coords(col: pd.DataFrame) -> pd.DataFrame:
    """
    Validate collision coordinates against recorded LSOA population centroid.

    Collisions more than LSOA_DIST_THRESHOLD_M (10km) from their LSOA centroid
    are flagged as coords_suspect=True. These are likely to have coordinates
    that were not corrected by the SD→SE fix or have other systematic errors.

    If the LSOA centroids file is not found, validation is skipped with a warning.
    """
    col["coords_suspect"] = False

    if "lsoa_of_accident_location" not in col.columns:
        logger.warning("  lsoa_of_accident_location not found — LSOA validation skipped")
        return col

    if not LSOA_CENTROIDS_PATH.exists():
        logger.warning(
            f"  LSOA centroids file not found at {LSOA_CENTROIDS_PATH} — "
            "LSOA validation skipped. Download from ONS Open Geography Portal."
        )
        return col

    lsoa = pd.read_csv(
        LSOA_CENTROIDS_PATH,
        usecols=["LSOA21CD", "x", "y"],
        encoding="utf-8-sig",
    ).rename(columns={"x": "lsoa_e", "y": "lsoa_n"})

    col = col.merge(lsoa, left_on="lsoa_of_accident_location",
                    right_on="LSOA21CD", how="left")

    has_both = (
        col["location_easting_osgr"].notna() &
        col["lsoa_e"].notna()
    )
    dist = np.sqrt(
        (col.loc[has_both, "location_easting_osgr"] - col.loc[has_both, "lsoa_e"])**2 +
        (col.loc[has_both, "location_northing_osgr"] - col.loc[has_both, "lsoa_n"])**2
    )
    col.loc[has_both, "lsoa_dist_m"] = dist.values
    col.loc[has_both, "coords_suspect"] = dist.values > LSOA_DIST_THRESHOLD_M

    # Drop the centroid join columns — keep only dist and suspect flag
    col = col.drop(columns=["LSOA21CD", "lsoa_e", "lsoa_n"], errors="ignore")

    n_suspect = col["coords_suspect"].sum()
    n_corrected = col.get("coords_corrected", pd.Series(False, index=col.index)).sum()
    logger.info(
        f"  LSOA validation: {n_suspect:,} collisions flagged as coords_suspect "
        f"({n_suspect/len(col):.1%}) | "
        f"{n_corrected:,} already corrected by SD→SE fix"
    )
    return col


def _add_road_name_clean(col: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct road name string from first_road_class + first_road_number.

    Examples: class=1, number=62 → 'M62'
              class=3, number=64 → 'A64'
              class=4, number=1234 → 'B1234'
              class=6            → '' (unclassified — no name)

    The result is stored in road_name_clean and used as the Stage 1
    join key in join.py to snap collisions to MRDB/AADF road links
    without relying on spatial proximity alone.
    """
    if "first_road_class" not in col.columns or "first_road_number" not in col.columns:
        logger.warning(
            "first_road_class / first_road_number not found — "
            "road_name_clean will be empty; join.py will fall back to Stage 2 spatial"
        )
        col["road_name_clean"] = ""
        return col

    prefix = col["first_road_class"].map(ROAD_CLASS_PREFIX).fillna("")
    number = col["first_road_number"].fillna(0).astype(int).astype(str)
    number = number.replace("0", "")  # 0 means no road number

    col["road_name_clean"] = (prefix + number).where(
        col["first_road_class"].isin(ROAD_CLASS_PREFIX) & (number != ""),
        other="",
    )

    n_named = (col["road_name_clean"] != "").sum()
    logger.info(
        f"  road_name_clean: {n_named:,} / {len(col):,} collisions have a named road "
        f"({n_named/len(col):.1%}) — remainder will use Stage 2 spatial snap"
    )
    return col


# ---------------------------------------------------------------------------
# AADF
# ---------------------------------------------------------------------------

def clean_aadf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the AADF bidirectional aggregate.

    Changes applied
    ---------------
    - Filter to TARGET_YEARS [2019, 2021, 2023] to match WebTRIS pull
    - Validate flow columns are non-negative
    - Add road_name_clean (standardised road name for joining to MRDB)
    - Add is_covid flag

    Parameters
    ----------
    df : DataFrame from aggregate_bidirectional() in ingest_aadf.py

    Returns
    -------
    Cleaned DataFrame at count_point_id × year grain.
    """
    logger.info(f"Cleaning AADF: {len(df):,} rows")

    # Filter to target years
    before = len(df)
    df = df[df["year"].isin(TARGET_YEARS)].copy()
    logger.info(f"  Year filter → {len(df):,} rows (from {before:,})")

    # Validate flow columns non-negative
    flow_cols = ["all_motor_vehicles", "all_hgvs", "pedal_cycles"]
    for col in flow_cols:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg:
                logger.warning(f"  {col}: {n_neg} negative values → set to NaN")
                df.loc[df[col] < 0, col] = np.nan

    # Validate proportions are in [0, 1]
    prop_cols = ["hgv_proportion", "lgv_proportion", "cars_proportion", "heavy_vehicle_prop"]
    for col in prop_cols:
        if col in df.columns:
            n_bad = (~df[col].between(0, 1) & df[col].notna()).sum()
            if n_bad:
                logger.warning(f"  {col}: {n_bad} values outside [0,1] → set to NaN")
                df.loc[~df[col].between(0, 1), col] = np.nan

    # Standardise road name for Stage 1 join
    if "road_name" in df.columns:
        df["road_name_clean"] = (
            df["road_name"]
            .str.strip()
            .str.upper()
            .str.replace(r"\s+", "", regex=True)  # 'A 64' → 'A64'
        )
    else:
        df["road_name_clean"] = ""

    # COVID flag
    df["is_covid"] = df["year"].isin(COVID_YEARS)

    logger.info(
        f"AADF cleaned: {len(df):,} rows | "
        f"years: {sorted(df['year'].unique())} | "
        f"count points: {df['count_point_id'].nunique():,}"
    )
    return df


# ---------------------------------------------------------------------------
# WebTRIS
# ---------------------------------------------------------------------------

def clean_webtris(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and aggregate WebTRIS monthly data to annual grain.

    Changes applied
    ---------------
    - Drop duplicate columns: site_id (= siteid), _pull_year (= year)
    - Aggregate 12 monthly rows → 1 annual row per site per year
      (mean of flow metrics, mean of large vehicle percentage)
    - Keep only TARGET_YEARS [2019, 2021, 2023]
    - Add is_covid flag

    Parameters
    ----------
    df : Combined DataFrame from combine_raw() in ingest_webtris.py
         Grain: site × year × month (12 rows per site-year)

    Returns
    -------
    DataFrame at site_id × year grain.
    """
    logger.info(f"Cleaning WebTRIS: {len(df):,} rows")

    # Rename siteid → site_id for consistency
    if "siteid" in df.columns:
        df = df.rename(columns={"siteid": "site_id"})

    # Use _pull_year as authoritative year — always an int set by our code.
    # The API 'year' column may be string or wrong type; drop it after promoting.
    if "_pull_year" in df.columns:
        if "year" in df.columns:
            df = df.drop(columns=["year"])
        df = df.rename(columns={"_pull_year": "year"})
    elif "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Drop duplicate site_id (siteid already renamed above)
    if "site_id" in df.columns and df.columns.tolist().count("site_id") > 1:
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    logger.info(f"  Year column set — unique years: {sorted(df['year'].dropna().unique()) if 'year' in df.columns else 'missing'}")

    # Filter to target years
    year_col = "year" if "year" in df.columns else None
    if year_col:
        before = len(df)
        df = df[df[year_col].isin(TARGET_YEARS)].copy()
        logger.info(f"  Year filter → {len(df):,} rows (from {before:,})")

    # Identify flow and percentage columns
    flow_cols = [c for c in df.columns if c.startswith("adt") or c.startswith("awt")]
    pct_cols  = [c for c in flow_cols if "percentage" in c]
    vol_cols  = [c for c in flow_cols if "percentage" not in c]

    if not flow_cols:
        logger.warning("No adt/awt columns found in WebTRIS data — check column names")

    # Coerce flow/percentage columns to numeric — pytris returns them as strings
    for c in vol_cols + pct_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Aggregate monthly → annual
    # Volumes: mean daily flow across months (already a daily average)
    # Percentages: mean across months
    group_cols = ["site_id", "year"] if year_col else ["site_id"]
    non_numeric = [c for c in df.columns if c not in flow_cols + group_cols
                   and df[c].dtype == object and c != "monthname"]

    agg_dict = {c: "mean" for c in vol_cols + pct_cols if c in df.columns}

    # For non-numeric metadata — take first value (stable within site-year)
    for c in non_numeric:
        agg_dict[c] = "first"

    annual = df.groupby(group_cols).agg(agg_dict).reset_index()

    # Rename key columns for clarity
    rename_map = {}
    if "adt24hour" in annual.columns:
        rename_map["adt24hour"] = "mean_daily_flow"
    if "adt24largevehiclepercentage" in annual.columns:
        rename_map["adt24largevehiclepercentage"] = "large_vehicle_pct"
    if "awt24hour" in annual.columns:
        rename_map["awt24hour"] = "mean_weekday_flow"
    if "awt24largevehiclepercentage" in annual.columns:
        rename_map["awt24largevehiclepercentage"] = "large_vehicle_weekday_pct"
    annual = annual.rename(columns=rename_map)

    # COVID flag
    if "year" in annual.columns:
        annual["is_covid"] = annual["year"].isin(COVID_YEARS)

    logger.info(
        f"WebTRIS cleaned: {len(annual):,} rows "
        f"(site × year) | sites: {annual['site_id'].nunique():,} | "
        f"years: {sorted(annual['year'].unique()) if 'year' in annual.columns else 'n/a'}"
    )
    return annual


# ---------------------------------------------------------------------------
# MRDB
# ---------------------------------------------------------------------------

def clean_mrdb(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean the MRDB GeoDataFrame and prepare for spatial joining.

    Changes applied
    ---------------
    - Add link_id: stable integer ID for each road link
    - Add road_name_clean: standardised road name for Stage 1 attribute join
    - Validate geometry (drop null/invalid geometries)

    Note: the MRDB shapefile only contains CP_Number, RoadNumber, geometry.
    Road type and link length are sourced from AADF in join.py.

    Parameters
    ----------
    gdf : GeoDataFrame from load_mrdb() in ingest_mrdb.py

    Returns
    -------
    GeoDataFrame with link_id and road_name_clean added.
    """
    logger.info(f"Cleaning MRDB: {len(gdf):,} road links")

    gdf = gdf.copy()

    # Validate geometry
    null_geom = gdf.geometry.isna()
    if null_geom.any():
        logger.warning(f"  {null_geom.sum()} null geometries — dropping")
        gdf = gdf[~null_geom]

    invalid_geom = ~gdf.geometry.is_valid
    if invalid_geom.any():
        logger.info(f"  Fixing {invalid_geom.sum()} invalid geometries with buffer(0)")
        gdf.loc[invalid_geom, "geometry"] = gdf.loc[invalid_geom, "geometry"].buffer(0)

    # Stable integer link ID (index-based, 1-indexed)
    gdf = gdf.reset_index(drop=True)
    gdf["link_id"] = gdf.index + 1

    # Standardise road name: 'M62', 'A64', 'B1234' etc.
    road_col = "road_name" if "road_name" in gdf.columns else (
        "RoadNumber" if "RoadNumber" in gdf.columns else None
    )
    if road_col:
        gdf["road_name_clean"] = (
            gdf[road_col]
            .fillna("")
            .str.strip()
            .str.upper()
            .str.replace(r"\s+", "", regex=True)
        )
        # Ensure road_name column exists with standard name
        if road_col != "road_name":
            gdf = gdf.rename(columns={road_col: "road_name"})
    else:
        logger.warning("No road name column found in MRDB")
        gdf["road_name_clean"] = ""

    # count_point_id — normalise to string for joining to AADF
    cp_col = "count_point_id" if "count_point_id" in gdf.columns else (
        "CP_Number" if "CP_Number" in gdf.columns else None
    )
    if cp_col:
        gdf["count_point_id"] = gdf[cp_col].astype(str).str.strip()
        gdf["count_point_id"] = gdf["count_point_id"].replace("nan", np.nan)
        if cp_col != "count_point_id":
            gdf = gdf.drop(columns=[cp_col])
        n_with_cp = gdf["count_point_id"].notna().sum()
        logger.info(
            f"  count_point_id: {n_with_cp:,} / {len(gdf):,} links have a CP number"
        )

    logger.info(
        f"MRDB cleaned: {len(gdf):,} links | "
        f"road names: {gdf['road_name_clean'].nunique():,} unique"
    )
    return gdf


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_cleaned(
    data: dict | pd.DataFrame | gpd.GeoDataFrame,
    name: str,
    output_folder: str | Path = None,
) -> None:
    """
    Save cleaned data to data/processed/<name>/.

    Parameters
    ----------
    data : dict of DataFrames (stats19), single DataFrame (aadf/webtris),
           or GeoDataFrame (mrdb)
    name : source name — 'stats19', 'aadf', 'webtris', 'mrdb'
    output_folder : defaults to data/processed/<name>/
    """
    if output_folder is None:
        output_folder = _ROOT / cfg["paths"]["processed"] / name
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(data, dict):
        for table, df in data.items():
            path = out / f"{table}_clean.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"  Saved {name}/{table}_clean.parquet ({len(df):,} rows)")
    elif isinstance(data, gpd.GeoDataFrame):
        path = out / f"{name}_clean.parquet"
        data.to_parquet(path, index=False)
        logger.info(f"  Saved {name}_clean.parquet ({len(data):,} rows)")
    else:
        path = out / f"{name}_clean.parquet"
        data.to_parquet(path, index=False)
        logger.info(f"  Saved {name}_clean.parquet ({len(data):,} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Load all processed parquets, clean each source, and save to
    data/processed/<source>/  as *_clean.parquet files.
    """
    from road_risk.ingest.ingest_stats19 import load_stats19
    from road_risk.ingest.ingest_aadf import load_aadf, aggregate_bidirectional
    from road_risk.ingest.ingest_mrdb import load_mrdb
    import glob

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    # --- STATS19 ------------------------------------------------------------
    logger.info("=== STATS19 ===")
    stats19_raw = load_stats19(
        _ROOT / "data/raw/stats19",
        years=list(range(2015, 2025)),   # 2015-2024 only — pre-2015 has unreliable coords
    )
    stats19_clean = clean_stats19(stats19_raw)
    save_cleaned(stats19_clean, "stats19")

    # --- AADF ---------------------------------------------------------------
    logger.info("=== AADF ===")
    aadf_raw = load_aadf(_ROOT / "data/raw/aadf")
    aadf_agg = aggregate_bidirectional(aadf_raw)
    aadf_clean = clean_aadf(aadf_agg)
    save_cleaned(aadf_clean, "aadf")

    # --- WebTRIS ------------------------------------------------------------
    logger.info("=== WebTRIS ===")
    webtris_chunks = sorted(
        (_ROOT / "data/raw/webtris").glob("site_*_*.parquet")
    )
    if webtris_chunks:
        from road_risk.ingest.ingest_webtris import combine_raw
        webtris_raw = combine_raw(_ROOT / "data/raw/webtris")
        webtris_clean = clean_webtris(webtris_raw)

        # Attach site coordinates — lat/lon lives in sites.parquet,
        # not in the per-site traffic chunks that combine_raw() loads.
        sites_path = _ROOT / "data/raw/webtris/sites.parquet"
        if sites_path.exists():
            sites = pd.read_parquet(sites_path, columns=["site_id", "latitude", "longitude"])
            sites["site_id"] = sites["site_id"].astype(webtris_clean["site_id"].dtype)
            webtris_clean = webtris_clean.merge(sites, on="site_id", how="left")
            n_with_coords = webtris_clean["latitude"].notna().sum()
            logger.info(f"  Site coordinates attached: {n_with_coords:,} / {len(webtris_clean):,} rows")
        else:
            logger.warning("sites.parquet not found — WebTRIS lat/lon will be missing")

        save_cleaned(webtris_clean, "webtris")
    else:
        logger.warning("No WebTRIS chunks found — skipping")

    # --- MRDB ---------------------------------------------------------------
    logger.info("=== MRDB ===")
    mrdb_raw = load_mrdb(_ROOT / "data/raw/shapefiles")
    mrdb_clean = clean_mrdb(mrdb_raw)
    save_cleaned(mrdb_clean, "mrdb")

    logger.info("=== Cleaning complete ===")


if __name__ == "__main__":
    main()