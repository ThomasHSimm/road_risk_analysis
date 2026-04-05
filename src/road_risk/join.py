"""
join.py
-------
Spatial and attribute joins across all four data sources.

The final output is a road_link × year table — one row per MRDB road
link per year — with collision counts, traffic volume, and vehicle mix
aggregated onto each link.

Pipeline
--------
1. snap_collisions_to_roads()
      Stage 1 — attribute match: reconstruct road name from
      first_road_class + first_road_number, match to MRDB road_name_clean.
      Then nearest-neighbour *within that named road only*.

      Stage 2 — spatial fallback: pure nearest-neighbour for collisions
      that didn't match in Stage 1 (unclassified roads, bad road numbers).
      Applies a 100m distance cap; beyond that snap_distance_m is still
      recorded but snap_method = 'unmatched'.

2. build_road_features()
      Joins AADF count point data onto MRDB links via count_point_id
      (direct key join). WebTRIS sensor data joined to AADF count points
      via spatial nearest-neighbour (no shared key).

3. build_road_link_annual()
      Aggregates snapped collisions onto road links per year.
      Joins road features (AADF + WebTRIS) onto the collision aggregates.
      Returns the final analysis table at road_link × year grain.

Key join columns
----------------
OpenRoads road_name_clean  -> STATS19 road_name_clean  (Stage 1)
OpenRoads geometry         -> STATS19 lat/lon          (Stage 2 spatial)
AADF      lat/lon          -> OpenRoads link centroid  (spatial, nearest)
AADF      lat/lon          -> WebTRIS lat/lon          (spatial, nearest)
MRDB      used for road metadata only (road_type on major roads)
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from road_risk.config import _ROOT, cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CRS
WGS84 = "EPSG:4326"
BNG   = "EPSG:27700"   # British National Grid — metres, used for distance calcs

# Stage 2 spatial fallback distance cap (metres)
SPATIAL_CAP_M = 100

# Minimum collision count per link-year to include in model features
MIN_COLLISIONS = 1

# AADF columns to carry through to the road features table
AADF_FEATURE_COLS = [
    "count_point_id",
    "year",
    "road_name",
    "road_type",
    "latitude",
    "longitude",
    "link_length_km",
    "all_motor_vehicles",
    "all_hgvs",
    "hgv_proportion",
    "lgv_proportion",
    "cars_proportion",
    "heavy_vehicle_prop",
    "estimation_method",
    "is_covid",
]

# WebTRIS columns to carry through
WEBTRIS_FEATURE_COLS = [
    "site_id",
    "year",
    "mean_daily_flow",
    "large_vehicle_pct",
    "mean_weekday_flow",
    "large_vehicle_weekday_pct",
]


# ---------------------------------------------------------------------------
# 1. Snap collisions to road links
# ---------------------------------------------------------------------------

def snap_collisions_to_roads(
    collisions: pd.DataFrame,
    openroads: gpd.GeoDataFrame,
    spatial_cap_m: float = SPATIAL_CAP_M,
) -> gpd.GeoDataFrame:
    """
    Snap STATS19 collisions to OS Open Roads links using a two-stage approach.

    OS Open Roads covers ALL classified roads in GB, giving full coverage
    for collisions on B-roads and minor roads that MRDB misses.

    Stage 1 — Attribute match (high confidence)
        Match collision road_name_clean to OpenRoads road_name_clean.
        Then find the nearest link *on that named road* to the collision
        coordinates. Prevents snapping an M62 collision to a nearby A-road.

    Stage 2 — Spatial fallback
        For unmatched collisions (unclassified roads, missing road numbers),
        find the nearest OpenRoads link overall within spatial_cap_m.

    Parameters
    ----------
    collisions  : cleaned collision DataFrame from clean_stats19()
                  Must have: latitude, longitude, road_name_clean, coords_valid
    openroads   : OS Open Roads GeoDataFrame from load_openroads()
                  Must have: link_id, road_name_clean, geometry (WGS84)
    spatial_cap_m : distance cap for Stage 2 in metres (default 100m)

    Returns
    -------
    GeoDataFrame at collision grain with added columns:
      link_id          : matched OpenRoads link ID
      snap_distance_m  : distance from collision to snapped link (metres)
      snap_method      : 'attribute', 'spatial', or 'unmatched'
    """
    logger.info(
        f"Snapping {len(collisions):,} collisions to "
        f"{len(openroads):,} OS Open Roads links"
    )

    # --- Prepare collisions GeoDataFrame ------------------------------------
    # Only snap collisions with valid coordinates
    valid = collisions["coords_valid"].fillna(False) if "coords_valid" in collisions.columns \
        else pd.Series(True, index=collisions.index)

    coll_gdf = gpd.GeoDataFrame(
        collisions[valid].copy(),
        geometry=gpd.points_from_xy(
            collisions.loc[valid, "longitude"],
            collisions.loc[valid, "latitude"],
        ),
        crs=WGS84,
    )

    # Project to BNG for distance calculations in metres
    coll_bng = coll_gdf.to_crs(BNG)
    roads_bng = openroads.to_crs(BNG)

    # Output columns
    coll_bng["link_id"]         = pd.NA
    coll_bng["snap_distance_m"] = np.nan
    coll_bng["snap_method"]     = "unmatched"

    # --- Stage 1: Attribute match -------------------------------------------
    named = coll_bng[
        coll_bng["road_name_clean"].notna() &
        (coll_bng["road_name_clean"] != "")
    ]
    logger.info(f"  Stage 1: {len(named):,} collisions have a named road")

    stage1_matched = 0
    for road_name, group in named.groupby("road_name_clean"):
        road_links = roads_bng[roads_bng["road_name_clean"] == road_name]
        if road_links.empty:
            continue
        matched = _nearest_link(group, road_links)
        coll_bng.loc[matched.index, "link_id"]         = matched["link_id"].values
        coll_bng.loc[matched.index, "snap_distance_m"] = matched["snap_distance_m"].values
        coll_bng.loc[matched.index, "snap_method"]     = "attribute"
        stage1_matched += len(matched)

    logger.info(
        f"  Stage 1 matched: {stage1_matched:,} / {len(named):,} "
        f"({stage1_matched/max(len(named),1):.1%})"
    )

    # --- Stage 2: Spatial fallback ------------------------------------------
    unmatched = coll_bng[coll_bng["snap_method"] == "unmatched"]
    logger.info(f"  Stage 2: {len(unmatched):,} collisions for spatial fallback")

    if not unmatched.empty:
        matched2 = _nearest_link(unmatched, roads_bng)
        within_cap = matched2["snap_distance_m"] <= spatial_cap_m
        n_within = within_cap.sum()

        coll_bng.loc[matched2.index, "link_id"]         = matched2["link_id"].values
        coll_bng.loc[matched2.index, "snap_distance_m"] = matched2["snap_distance_m"].values
        coll_bng.loc[matched2[within_cap].index, "snap_method"] = "spatial"

        logger.info(
            f"  Stage 2 matched within {spatial_cap_m}m: {n_within:,} / {len(unmatched):,} "
            f"({n_within/max(len(unmatched),1):.1%})"
        )

    # --- Reproject back to WGS84 and add invalid-coord rows ----------------
    coll_out = coll_bng.to_crs(WGS84)

    # Append rows with invalid coordinates (unsnappable — no link_id)
    if (~valid).any():
        invalid_rows = collisions[~valid].copy()
        invalid_rows["link_id"]         = pd.NA
        invalid_rows["snap_distance_m"] = np.nan
        invalid_rows["snap_method"]     = "invalid_coords"
        invalid_gdf = gpd.GeoDataFrame(invalid_rows, geometry=[None]*len(invalid_rows), crs=WGS84)
        coll_out = pd.concat([coll_out, invalid_gdf], ignore_index=True)

    # Summary
    method_counts = coll_out["snap_method"].value_counts()
    logger.info(f"  Snap summary:\n{method_counts.to_string()}")

    return coll_out


def _nearest_link(
    points: gpd.GeoDataFrame,
    links: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    For each point, find the nearest link and return link_id + distance.

    Uses geopandas sjoin_nearest which builds an STRtree index internally —
    efficient for large datasets.

    Both inputs must be in the same projected CRS (BNG recommended for metres).

    Returns
    -------
    DataFrame indexed like `points` with columns: link_id, snap_distance_m
    """
    if points.empty or links.empty:
        return pd.DataFrame(
            {"link_id": pd.NA, "snap_distance_m": np.nan},
            index=points.index,
        )

    joined = gpd.sjoin_nearest(
        points[["geometry"]],
        links[["link_id", "geometry"]],
        how="left",
        distance_col="snap_distance_m",
    )

    # sjoin_nearest can produce duplicates if equidistant — keep first
    joined = joined[~joined.index.duplicated(keep="first")]

    return joined[["link_id", "snap_distance_m"]]


# ---------------------------------------------------------------------------
# 2. Build road features (AADF + WebTRIS per MRDB link per year)
# ---------------------------------------------------------------------------

def build_road_features(
    openroads: gpd.GeoDataFrame,
    aadf: pd.DataFrame,
    webtris: pd.DataFrame | None = None,
    aadf_snap_cap_m: float = 2000,
) -> pd.DataFrame:
    """
    Join AADF traffic features onto OS Open Roads links via spatial
    nearest-neighbour, then attach WebTRIS sensor features where available.

    AADF snap cap of 2km applied — links further than this from any count
    point get NaN traffic features rather than a meaningless distant match.

    Parameters
    ----------
    openroads      : OS Open Roads GeoDataFrame (link_id, geometry in WGS84)
    aadf           : cleaned AADF DataFrame (count_point_id, year, flow cols, lat/lon)
    webtris        : cleaned WebTRIS DataFrame (site_id, year, flow cols, lat/lon)
    aadf_snap_cap_m: max distance (metres) for AADF→road spatial join (default 2km)

    Returns
    -------
    DataFrame at link_id × year grain with traffic features.
    """
    logger.info(
        f"Building road features (OpenRoads × AADF × WebTRIS) — spatial joins"
    )

    # --- WebTRIS → AADF spatial join ----------------------------------------
    if webtris is not None and not webtris.empty:
        aadf = _attach_webtris_to_aadf(aadf, webtris)
        logger.info("  WebTRIS features attached to AADF count points")

    # Trim AADF to feature columns
    webtris_cols = [
        c for c in ["mean_daily_flow", "large_vehicle_pct",
                    "mean_weekday_flow", "large_vehicle_weekday_pct", "site_id"]
        if c in aadf.columns
    ]
    aadf_keep = [c for c in AADF_FEATURE_COLS + webtris_cols if c in aadf.columns]
    aadf_trim = aadf[aadf_keep].copy()

    # --- AADF → OpenRoads: spatial join per year ----------------------------
    logger.info(
        f"  Spatial AADF join: {len(openroads):,} links × "
        f"{aadf_trim['year'].nunique()} years (cap: {aadf_snap_cap_m}m)"
    )

    aadf_gdf = gpd.GeoDataFrame(
        aadf_trim,
        geometry=gpd.points_from_xy(aadf_trim["longitude"], aadf_trim["latitude"]),
        crs=WGS84,
    ).to_crs(BNG)

    roads_bng = openroads.to_crs(BNG).copy()
    roads_centroids = roads_bng[["link_id"]].copy()
    roads_centroids["geometry"] = roads_bng.geometry.centroid
    roads_centroids = gpd.GeoDataFrame(roads_centroids, geometry="geometry", crs=BNG)

    spatial_rows = []
    for year in sorted(aadf_trim["year"].unique()):
        aadf_yr = aadf_gdf[aadf_gdf["year"] == year].copy()
        if aadf_yr.empty:
            continue

        aadf_yr = aadf_yr.drop(columns=["link_id"], errors="ignore")

        joined = gpd.sjoin_nearest(
            roads_centroids,
            aadf_yr,
            how="left",
            distance_col="aadf_snap_distance_m",
        )
        joined = joined[~joined.index.duplicated(keep="first")]

        # Nullify features beyond snap cap — distant match is not meaningful
        feature_cols = [c for c in aadf_keep if c not in ["latitude", "longitude", "year"]]
        beyond_cap = joined["aadf_snap_distance_m"] > aadf_snap_cap_m
        n_beyond = beyond_cap.sum()
        if n_beyond:
            # Cast bool columns to object first to avoid FutureWarning
            for fc in feature_cols:
                if fc in joined.columns and joined[fc].dtype == bool:
                    joined[fc] = joined[fc].astype(object)
            joined.loc[beyond_cap, feature_cols] = np.nan
            logger.info(
                f"    {year}: {n_beyond:,} links beyond {aadf_snap_cap_m}m cap → "
                f"AADF features set to NaN"
            )

        # --- Street name fallback for links still beyond cap ----------------
        # For OpenRoads links with a street_name_clean, try matching to AADF
        # road_name_clean (normalised) — recovers named roads without numbers.
        if "street_name_clean" in openroads.columns and "road_name_clean" in aadf_trim.columns:
            still_beyond = beyond_cap & (
                openroads.set_index("link_id")
                .loc[joined["link_id"].values, "street_name_clean"]
                .values != ""
            ) if "link_id" in joined.columns else beyond_cap

            aadf_named = aadf_yr[
                aadf_yr.get("road_name_clean", pd.Series("", index=aadf_yr.index)) != ""
            ] if "road_name_clean" in aadf_yr.columns else pd.DataFrame()

            if still_beyond.any() and not aadf_named.empty:
                # Normalise AADF road_name for matching
                aadf_name_map = (
                    aadf_yr.assign(
                        aadf_name_norm=aadf_yr.get(
                            "road_name_clean",
                            pd.Series("", index=aadf_yr.index)
                        ).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
                    )
                    .set_index("aadf_name_norm")
                )
                links_beyond = joined[still_beyond]["link_id"]
                or_streets = openroads.set_index("link_id")["street_name_clean"]
                n_name_matched = 0
                for lid in links_beyond:
                    street = or_streets.get(lid, "")
                    if street and street in aadf_name_map.index:
                        row = aadf_name_map.loc[street]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                        for fc in feature_cols:
                            if fc in row.index:
                                joined.loc[joined["link_id"] == lid, fc] = row[fc]
                        joined.loc[joined["link_id"] == lid, "aadf_snap_distance_m"] = 0
                        joined.loc[joined["link_id"] == lid, "aadf_join_method"] = "name_match"
                        n_name_matched += 1
                if n_name_matched:
                    logger.info(
                        f"    {year}: {n_name_matched:,} additional links matched "
                        f"via street name"
                    )

        joined["aadf_join_method"] = joined.get(
            "aadf_join_method", pd.Series("spatial", index=joined.index)
        ).fillna("spatial")

        n_matched = (~beyond_cap).sum()
        logger.info(
            f"    {year}: {n_matched:,} / {len(roads_centroids):,} links matched "
            f"(mean dist: {joined.loc[~beyond_cap, 'aadf_snap_distance_m'].mean():.0f}m)"
        )
        spatial_rows.append(
            joined.drop(columns=["geometry", "index_right"], errors="ignore")
        )

    if not spatial_rows:
        logger.error("No AADF data joined — check aadf_clean.parquet exists and has rows")
        return pd.DataFrame()

    road_features = pd.concat(spatial_rows, ignore_index=True)
    logger.info(
        f"Road features built: {len(road_features):,} link × year rows | "
        f"links: {road_features['link_id'].nunique():,}"
    )
    return road_features


def _attach_webtris_to_aadf(
    aadf: pd.DataFrame,
    webtris: pd.DataFrame,
) -> pd.DataFrame:
    """
    Spatially match WebTRIS sites to AADF count points (nearest neighbour
    per year), then left-join WebTRIS features onto AADF.

    Both datasets have lat/lon. The join is done per year so a 2019 WebTRIS
    reading only attaches to the 2019 AADF row.
    """
    wt_cols = [c for c in WEBTRIS_FEATURE_COLS if c in webtris.columns]
    wt = webtris[wt_cols].copy()

    if "latitude" not in webtris.columns or "longitude" not in webtris.columns:
        logger.warning(
            "WebTRIS data has no lat/lon — cannot spatial-join to AADF. "
            "WebTRIS features will be missing."
        )
        return aadf

    wt_gdf = gpd.GeoDataFrame(
        webtris,
        geometry=gpd.points_from_xy(webtris["longitude"], webtris["latitude"]),
        crs=WGS84,
    ).to_crs(BNG)

    aadf_gdf = gpd.GeoDataFrame(
        aadf,
        geometry=gpd.points_from_xy(aadf["longitude"], aadf["latitude"]),
        crs=WGS84,
    ).to_crs(BNG)

    result_frames = []
    for year in aadf["year"].unique():
        aadf_yr = aadf_gdf[aadf_gdf["year"] == year].copy()
        wt_yr   = wt_gdf[wt_gdf["year"] == year][wt_cols + ["geometry"]] if "year" in wt_gdf.columns \
            else wt_gdf[wt_cols + ["geometry"]]

        if wt_yr.empty:
            result_frames.append(aadf_yr.drop(columns=["geometry"]))
            continue

        joined = gpd.sjoin_nearest(
            aadf_yr,
            wt_yr.drop(columns=["year"] if "year" in wt_yr.columns else []),
            how="left",
            distance_col="webtris_snap_distance_m",
        )
        joined = joined[~joined.index.duplicated(keep="first")]
        result_frames.append(joined.drop(columns=["geometry", "index_right"], errors="ignore"))

    return pd.concat(result_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Build road_link × year final table
# ---------------------------------------------------------------------------

def build_road_link_annual(
    collisions_snapped: gpd.GeoDataFrame,
    road_features: pd.DataFrame,
    openroads: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Aggregate snapped collisions onto OS Open Roads links per year,
    then join road features to produce the final analysis table.

    Output grain: road_link × year
    """
    logger.info("Building road_link × year table")

    col = collisions_snapped.copy()

    if "collision_year" in col.columns:
        col["year"] = col["collision_year"]
    elif "date" in col.columns:
        col["year"] = pd.to_datetime(col["date"], errors="coerce").dt.year

    snapped = col[col["snap_method"].isin(["attribute", "spatial", "weighted"])].copy()
    logger.info(
        f"  Using {len(snapped):,} / {len(col):,} snapped collisions "
        f"({len(snapped)/len(col):.1%})"
    )

    if "vehicle_type" in snapped.columns:
        hgv_codes = {19, 20, 21}
        snapped["involves_hgv"] = snapped["vehicle_type"].isin(hgv_codes)
    else:
        snapped["involves_hgv"] = False

    agg = (
        snapped
        .groupby(["link_id", "year"])
        .agg(
            collision_count            =("collision_index", "count"),
            fatal_count                =("collision_severity", lambda x: (x == 1).sum()),
            serious_count              =("collision_severity", lambda x: (x == 2).sum()),
            slight_count               =("collision_severity", lambda x: (x == 3).sum()),
            casualty_count             =("number_of_casualties", "sum"),
            hgv_collision_count        =("involves_hgv", "sum"),
            mean_vehicles_per_collision=("number_of_vehicles", "mean"),
            pct_attribute_snapped      =("snap_method",
                                         lambda x: (x == "attribute").mean()),
        )
        .reset_index()
    )

    logger.info(
        f"  Collision aggregates: {len(agg):,} link × year rows | "
        f"links: {agg['link_id'].nunique():,} | years: {sorted(agg['year'].unique())}"
    )

    # --- Join road features -------------------------------------------------
    road_feat = road_features.copy()
    road_feat["link_id"] = road_feat["link_id"].astype(agg["link_id"].dtype)

    result = agg.merge(road_feat, on=["link_id", "year"], how="left")

    # Attach OpenRoads road metadata — single road_name, no duplicates
    or_meta = openroads[[
        "link_id", "road_name", "road_name_clean",
        "road_classification", "road_function", "form_of_way",
        "link_length_km", "is_trunk", "is_primary",
    ]].copy()
    # Drop link_length_km from result if already there from AADF to avoid dupe
    if "link_length_km" in result.columns:
        or_meta = or_meta.drop(columns=["link_length_km"])

    result = result.merge(or_meta, on="link_id", how="left")

    # Drop AADF road_name if it exists — OpenRoads road_name is the canonical one
    for col_name in ["road_name_x", "road_name_y"]:
        if col_name in result.columns:
            result = result.drop(columns=[col_name])

    # --- Derived rate -------------------------------------------------------
    if "all_motor_vehicles" in result.columns and "link_length_km" in result.columns:
        vehicle_km = result["all_motor_vehicles"] * result["link_length_km"] * 365
        result["collision_rate_per_mvkm"] = (
            result["collision_count"] / (vehicle_km / 1e6)
        ).replace([np.inf, -np.inf], np.nan)

    # --- COVID flag ---------------------------------------------------------
    if "is_covid" not in result.columns:
        from road_risk.clean import COVID_YEARS
        result["is_covid"] = result["year"].isin(COVID_YEARS)

    logger.info(
        f"Final road_link × year table: {len(result):,} rows × {result.shape[1]} cols"
    )
    if "collision_rate_per_mvkm" in result.columns:
        median_rate = result["collision_rate_per_mvkm"].median()
        logger.info(f"  Collision rate (median): {median_rate:.4f} per M veh-km")

    return result


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_road_link_annual(
    df: pd.DataFrame,
    output_folder: str | Path = None,
) -> None:
    """Save the final road_link × year table to parquet."""
    if output_folder is None:
        output_folder = _ROOT / cfg["paths"]["features"]
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "road_link_annual.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Saved road_link_annual to {path} ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Load all cleaned parquets, run the full join pipeline, and save
    the road_link × year feature table to data/features/.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    processed = _ROOT / "data/processed"

    # --- Load cleaned data --------------------------------------------------
    logger.info("Loading cleaned data ...")

    collisions = pd.read_parquet(processed / "stats19/collision_clean.parquet")
    aadf       = pd.read_parquet(processed / "aadf/aadf_clean.parquet")

    # OS Open Roads — load from processed cache or raw GeoPackage
    or_path = processed / "shapefiles/openroads_yorkshire.parquet"
    if or_path.exists():
        openroads = gpd.read_parquet(or_path)
        logger.info(f"Loaded OS Open Roads from cache ({len(openroads):,} links)")
    else:
        logger.info("OS Open Roads cache not found — loading from GeoPackage ...")
        from road_risk.ingest.ingest_openroads import load_openroads, save_openroads
        openroads = load_openroads()
        save_openroads(openroads, processed / "shapefiles")

    webtris_path = processed / "webtris/webtris_clean.parquet"
    webtris = pd.read_parquet(webtris_path) if webtris_path.exists() else None
    if webtris is None:
        logger.warning("WebTRIS clean parquet not found — proceeding without sensor features")

    # --- Run pipeline -------------------------------------------------------
    # Prefer snap.py weighted output if it exists — it uses multi-criteria
    # scoring (spatial + road class + junction + road number) and is more
    # accurate than the attribute+spatial fallback in snap_collisions_to_roads().
    snapped_w_path = processed / "stats19/snapped_weighted.parquet"
    if snapped_w_path.exists():
        logger.info(
            "Step 1: Loading pre-computed snapped_weighted.parquet "
            "(run snap.py to regenerate)"
        )
        collisions_snapped = pd.read_parquet(snapped_w_path)
    else:
        logger.info("Step 1: Snapping collisions to OS Open Roads links ...")
        collisions_snapped = snap_collisions_to_roads(collisions, openroads)

    logger.info("Step 2: Building road features ...")
    road_features = build_road_features(openroads, aadf, webtris)

    logger.info("Step 3: Building road_link × year table ...")
    result = build_road_link_annual(collisions_snapped, road_features, openroads)

    # --- Summary ------------------------------------------------------------
    print("\n=== road_link_annual ===")
    print(f"  Rows    : {len(result):,}")
    print(f"  Links   : {result['link_id'].nunique():,}")
    print(f"  Years   : {sorted(result['year'].unique())}")
    print(f"  Columns : {result.columns.tolist()}")
    if "collision_rate_per_mvkm" in result.columns:
        print(f"  Collision rate (median): {result['collision_rate_per_mvkm'].median():.4f}")
    if "pct_attribute_snapped" in result.columns:
        print(f"  Mean % attribute-snapped: {result['pct_attribute_snapped'].mean():.1%}")
    if "road_classification" in result.columns:
        print(f"\n  Road classification breakdown:")
        print(result.groupby("road_classification")["collision_count"].sum().to_string())

    save_road_link_annual(result)


if __name__ == "__main__":
    main()