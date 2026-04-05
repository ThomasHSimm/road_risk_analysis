"""
ingest_openroads.py
-------------------
Loader for OS Open Roads GeoPackage.

Source:
  https://osdatahub.os.uk/downloads/open/OpenRoads
  File: oproad_gb.gpkg
  Layers: road_link (LineString), road_node (Point), motorway_junction (Point)

OS Open Roads covers ALL classified roads in GB — motorways, A roads, B roads,
minor roads and unclassified roads. Unlike MRDB (major roads only), this gives
full coverage for snapping STATS19 collisions regardless of road type.

Key columns in road_link layer:
  id                        : unique TOID identifier
  road_classification       : Motorway / A Road / B Road / Minor Road / Local Road
  road_function             : A Road / B Road / Minor Road / Local Street etc.
  form_of_way               : Single Carriageway / Dual Carriageway / Slip Road etc.
  road_classification_number: numeric part of road name (e.g. '62' for M62)
  name_1                    : full road name where available (e.g. 'M62', 'A64')
  length                    : link length in metres (CRS is BNG metres)
  trunk_road                : boolean — National Highways trunk road
  primary_route             : boolean — primary route network
  start_node / end_node     : node TOIDs for network analysis

Coordinate system:
  Raw: EPSG:27700 (British National Grid)
  Output: EPSG:4326 (WGS84) to match STATS19 and AADF

Yorkshire spatial filter:
  Applied at read time via bbox — avoids loading full GB (~4M links) into memory.
"""

import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np

from road_risk.config import _ROOT, cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RAW_FOLDER    = _ROOT / cfg["paths"]["raw"]["shapefiles"]
_DEFAULT_OUTPUT_FOLDER = _ROOT / cfg["paths"]["processed"] / "shapefiles"

TARGET_CRS = "EPSG:4326"
SOURCE_CRS = "EPSG:27700"

# Yorkshire bbox in BNG (metres) — widened 20km beyond Yorkshire boundary
# to capture roads serving collisions near the edge of the police force area.
# Approx WGS84: lon -3.2→0.4, lat 53.1→54.7
YORKSHIRE_BBOX_BNG = (370000, 350000, 590000, 540000)  # minx, miny, maxx, maxy

# Columns to keep
KEEP_COLS = [
    "id",
    "road_classification",
    "road_function",
    "form_of_way",
    "road_classification_number",
    "name_1",
    "length",
    "trunk_road",
    "primary_route",
    "start_node",
    "end_node",
    "geometry",
]

# Map OS road_classification → short prefix for road_name_clean
CLASSIFICATION_PREFIX = {
    "Motorway":    "M",
    "A Road":      "A",
    "B Road":      "B",
    "Minor Road":  "",
    "Local Road":  "",
    "Local Street": "",
    "Unknown":     "",
}

COL_RENAMES = {
    "id":                        "link_id",
    "road_classification":       "road_classification",
    "road_function":             "road_function",
    "form_of_way":               "form_of_way",
    "road_classification_number": "road_number",
    "name_1":                    "road_name",
    "length":                    "link_length_m",
    "trunk_road":                "is_trunk",
    "primary_route":             "is_primary",
    "start_node":                "start_node",
    "end_node":                  "end_node",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_gpkg(folder: Path) -> Path:
    """Find the OS Open Roads GeoPackage in the given folder."""
    # Try known filename first
    for name in ["oproad_gb.gpkg", "oproads_gb.gpkg", "OpenRoads_gpkg.gpkg"]:
        p = folder / name
        if p.exists():
            return p

    # Glob fallback
    matches = sorted(folder.glob("*.gpkg"))
    if matches:
        # Prefer one that isn't MRDB
        non_mrdb = [m for m in matches if "mrdb" not in m.name.lower()]
        if non_mrdb:
            logger.debug(f"Found GeoPackage: {non_mrdb[0].name}")
            return non_mrdb[0]
        return matches[0]

    raise FileNotFoundError(
        f"No OS Open Roads GeoPackage found in {folder}\n"
        f"Download from https://osdatahub.os.uk/downloads/open/OpenRoads "
        f"and place oproad_gb.gpkg in {folder}"
    )


def _build_road_name_clean(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Build road_name_clean for joining to STATS19 road_name_clean.

    Priority:
      1. name_1 / road_name if it looks like a road number (M62, A64, B1234)
      2. Reconstruct from road_classification + road_number
      3. Empty string for unnamed/unclassified roads
    """
    # --- road_name_clean: use road_number directly (already contains prefix) -
    # road_number contains full designation e.g. 'A64', 'M62', 'B1234'
    # road_classification prefix must NOT be prepended — it would double it.
    number = (
        gdf["road_number"]
        .fillna("")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.upper()
    )
    number = number.replace({"0": "", "NAN": "", "NONE": ""})
    gdf["road_name_clean"] = number

    n_numbered = (gdf["road_name_clean"] != "").sum()
    logger.info(
        f"  road_name_clean (M/A/B number): {n_numbered:,} / {len(gdf):,} links "
        f"({n_numbered/len(gdf):.1%})"
    )

    # --- street_name_clean: normalised name_1 (Dale Close → DALECLOSE) ------
    # Used for AADF → OpenRoads name matching where road_name in AADF is a
    # street name rather than a road number.
    if "road_name" in gdf.columns:
        gdf["street_name_clean"] = (
            gdf["road_name"]
            .fillna("")
            .str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
            .str.strip()
        )
    else:
        gdf["street_name_clean"] = ""

    n_street = (gdf["street_name_clean"] != "").sum()
    logger.info(
        f"  street_name_clean (named streets): {n_street:,} / {len(gdf):,} links "
        f"({n_street/len(gdf):.1%})"
    )
    return gdf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_openroads(
    raw_folder: str | Path = _DEFAULT_RAW_FOLDER,
    bbox_bng: tuple = YORKSHIRE_BBOX_BNG,
    target_crs: str = TARGET_CRS,
    layer: str = "road_link",
) -> gpd.GeoDataFrame:
    """
    Load OS Open Roads road_link layer, filter to Yorkshire, reproject to WGS84.

    Parameters
    ----------
    raw_folder : folder containing oproad_gb.gpkg
    bbox_bng   : (minx, miny, maxx, maxy) in BNG metres for spatial filter.
                 Applied at read time so full GB is never loaded into memory.
                 Defaults to generous Yorkshire bounds.
    target_crs : output CRS, defaults to EPSG:4326 (WGS84)
    layer      : GeoPackage layer name, defaults to 'road_link'

    Returns
    -------
    GeoDataFrame with normalised columns and WGS84 geometry.

    Example
    -------
    >>> gdf = load_openroads()
    >>> gdf["road_classification"].value_counts()
    """
    folder = Path(raw_folder)
    gpkg_path = _find_gpkg(folder)
    logger.info(f"Loading OS Open Roads from {gpkg_path.name} (layer='{layer}') ...")

    # bbox filter at read time — avoids loading ~4M GB links
    gdf = gpd.read_file(gpkg_path, layer=layer, bbox=bbox_bng)
    logger.info(f"  Loaded {len(gdf):,} road links within Yorkshire bbox")

    # Set CRS if not already set
    if gdf.crs is None:
        logger.warning(f"No CRS — assuming {SOURCE_CRS}")
        gdf = gdf.set_crs(SOURCE_CRS)

    # Reproject to WGS84
    if str(gdf.crs).upper() != target_crs.upper():
        gdf = gdf.to_crs(target_crs)
        logger.info(f"  Reprojected to {target_crs}")

    # Trim to known columns
    cols_present = [c for c in KEEP_COLS if c in gdf.columns]
    gdf = gdf[cols_present]

    # Rename to project standard names
    gdf = gdf.rename(columns={k: v for k, v in COL_RENAMES.items() if k in gdf.columns})

    # Derive link_length_km
    if "link_length_m" in gdf.columns:
        gdf["link_length_km"] = gdf["link_length_m"] / 1000

    # Normalise string columns
    for col in ["road_name", "road_classification", "road_function", "form_of_way"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].fillna("").str.strip()

    # road_number — strip decimal if it came as float
    if "road_number" in gdf.columns:
        gdf["road_number"] = (
            gdf["road_number"]
            .fillna("")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )

    # Build road_name_clean for Stage 1 snap
    gdf = _build_road_name_clean(gdf)

    # Validate geometry
    null_geom = gdf.geometry.isna()
    if null_geom.any():
        logger.warning(f"  {null_geom.sum()} null geometries — dropping")
        gdf = gdf[~null_geom]

    invalid_geom = ~gdf.geometry.is_valid
    if invalid_geom.any():
        logger.info(f"  Fixing {invalid_geom.sum()} invalid geometries")
        gdf.loc[invalid_geom, "geometry"] = gdf.loc[invalid_geom, "geometry"].buffer(0)

    gdf = gdf.reset_index(drop=True)

    logger.info(
        f"OS Open Roads loaded: {len(gdf):,} links | "
        f"road types:\n{gdf['road_classification'].value_counts().to_string()}"
    )
    return gdf


def save_openroads(
    gdf: gpd.GeoDataFrame,
    output_folder: str | Path = _DEFAULT_OUTPUT_FOLDER,
) -> None:
    """Save OS Open Roads GeoDataFrame to GeoParquet."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = output_folder / "openroads_yorkshire.parquet"
    gdf.to_parquet(out_path, index=False)
    logger.info(f"Saved OS Open Roads to {out_path} ({len(gdf):,} links)")


def main(
    raw_folder: str | Path = None,
    output_folder: str | Path = None,
) -> gpd.GeoDataFrame:
    """Load, filter, and save OS Open Roads for Yorkshire."""
    if raw_folder is None:
        raw_folder = _DEFAULT_RAW_FOLDER
    if output_folder is None:
        output_folder = _DEFAULT_OUTPUT_FOLDER

    gdf = load_openroads(raw_folder)

    print("\n=== OS Open Roads summary ===")
    print(f"  Road links : {len(gdf):,}")
    print(f"  CRS        : {gdf.crs}")
    print(f"  Columns    : {gdf.columns.tolist()}")
    print(f"\n  Road classification:\n{gdf['road_classification'].value_counts().to_string()}")
    if "form_of_way" in gdf.columns:
        print(f"\n  Form of way:\n{gdf['form_of_way'].value_counts().to_string()}")
    n_named = (gdf["road_name_clean"] != "").sum()
    print(f"\n  Named roads: {n_named:,} / {len(gdf):,} ({n_named/len(gdf):.1%})")

    save_openroads(gdf, output_folder)
    return gdf


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    main()