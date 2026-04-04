"""
ingest_mrdb.py
--------------
Loader for the DfT Major Road Network Database (MRDB) shapefile.

Source:
  https://roadtraffic.dft.gov.uk/downloads
  File: "Major Road Network database" — download as zipped shapefile
  (e.g. MRDB_2024.zip). Available for years 2018–2024.

The MRDB is an OS Open Roads-derived dataset of road links on the Major Road
Network. It provides road geometry (LineString) and attributes including road
name, road type, and link identifiers that can be spatially joined to STATS19
collision coordinates and AADF count point locations.

Key attributes in the shapefile:
  CP_Number   : count point ID — links directly to AADF count_point_id
  RoadNumber  : road name (e.g. 'M1', 'A64')
  RoadType    : Motorway, A Road, B Road, etc.
  LenNet      : link length (metres)
  geometry    : LineString in EPSG:27700 (British National Grid)

Coordinate system:
  Raw: EPSG:27700 (British National Grid, metres)
  We reproject to EPSG:4326 (WGS84, lat/lon) to match STATS19 and AADF.

Yorkshire spatial filter:
  Applied using a bounding box in WGS84 before returning.
  Avoids loading the full GB dataset into memory for the pilot.
"""

import logging
import zipfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from road_risk.config import _ROOT, cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RAW_FOLDER = _ROOT / cfg["paths"]["raw"]["shapefiles"]
_DEFAULT_OUTPUT_FOLDER = _ROOT / cfg["paths"]["processed"] / "shapefiles"

# Target CRS — WGS84 to match STATS19 lat/lon and AADF coordinates
TARGET_CRS = "EPSG:4326"

# Yorkshire bounding box in WGS84 [minx, miny, maxx, maxy]
# Generous bounds to capture all Yorkshire roads
YORKSHIRE_BBOX = (-2.20, 53.30, -0.08, 54.60)

# Columns to keep — drop OS-internal fields we don't need
KEEP_COLS = [
    "CP_Number",        # → count_point_id join key for AADF
    "RoadNumber",       # road name
    "RoadType",         # Motorway / A Road / B Road
    "LenNet",           # link length in metres
    "geometry",
]

# Normalised column names after loading
COL_RENAMES = {
    "CP_Number":  "count_point_id",
    "RoadNumber": "road_name",
    "RoadType":   "road_type",
    "LenNet":     "link_length_m",
}

EXPECTED_COLS = ["road_name", "road_type", "geometry"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_shapefile(folder: Path) -> Path:
    """
    Find the MRDB shapefile in the given folder.
    Handles both extracted .shp files and zipped shapefiles.

    Search order:
      1. Any .shp file directly in the folder
      2. Any .shp file one level deep (already extracted zip)
      3. Any .zip file matching MRDB naming — extract and return .shp
    """
    # Direct .shp
    shps = sorted(folder.glob("*.shp"))
    if shps:
        logger.debug(f"Found shapefile: {shps[0].name}")
        return shps[0]

    # One level deep (zip already extracted into subfolder)
    shps = sorted(folder.glob("**/*.shp"))
    if shps:
        logger.debug(f"Found shapefile (subdirectory): {shps[0]}")
        return shps[0]

    # Zip file — extract and return path
    zips = sorted(folder.glob("MRDB*.zip")) or sorted(folder.glob("*.zip"))
    if zips:
        zip_path = zips[0]
        logger.info(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(folder)
        # Find the .shp in the extracted contents
        shps = sorted(folder.glob("**/*.shp"))
        if shps:
            return shps[0]

    raise FileNotFoundError(
        f"No MRDB shapefile found in {folder}\n"
        f"Download 'Major Road Network database' (zipped shapefile) from "
        f"https://roadtraffic.dft.gov.uk/downloads and place in {folder}"
    )


def _bbox_filter(gdf: gpd.GeoDataFrame, bbox: tuple) -> gpd.GeoDataFrame:
    """Filter a GeoDataFrame to rows intersecting a WGS84 bounding box."""
    minx, miny, maxx, maxy = bbox
    bbox_geom = box(minx, miny, maxx, maxy)
    # bbox_geom is in WGS84; gdf must already be reprojected before calling
    mask = gdf.intersects(bbox_geom)
    return gdf[mask].copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mrdb(
    raw_folder: str | Path = _DEFAULT_RAW_FOLDER,
    bbox: tuple | None = YORKSHIRE_BBOX,
    target_crs: str = TARGET_CRS,
) -> gpd.GeoDataFrame:
    """
    Load the MRDB shapefile, reproject to WGS84, and optionally filter
    to a bounding box.

    Parameters
    ----------
    raw_folder : folder containing the MRDB shapefile or zip.
                 Defaults to data/raw/shapefiles/.
    bbox : (minx, miny, maxx, maxy) in WGS84 to spatially filter.
           Defaults to Yorkshire bounding box.
           Pass None to load the full GB dataset (slower).
    target_crs : output CRS. Defaults to EPSG:4326 (WGS84).

    Returns
    -------
    GeoDataFrame with normalised column names and WGS84 geometry.

    Example
    -------
    >>> gdf = load_mrdb()
    >>> gdf.plot()
    """
    folder = Path(raw_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Raw folder not found: {folder}")

    shp_path = _find_shapefile(folder)
    logger.info(f"Loading shapefile: {shp_path.name}")

    gdf = gpd.read_file(shp_path)
    logger.info(f"  Loaded {len(gdf):,} road links (CRS: {gdf.crs})")

    # Reproject to WGS84
    if gdf.crs is None:
        logger.warning("Shapefile has no CRS defined — assuming EPSG:27700")
        gdf = gdf.set_crs("EPSG:27700")
    if str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)
        logger.info(f"  Reprojected to {target_crs}")

    # Trim to known useful columns — keep any we have, skip missing ones
    cols_present = [c for c in KEEP_COLS if c in gdf.columns]
    missing = [c for c in KEEP_COLS if c not in gdf.columns and c != "geometry"]
    if missing:
        logger.warning(f"  Expected columns not found in shapefile: {missing}")
    gdf = gdf[cols_present]

    # Rename to project-standard names
    gdf = gdf.rename(columns={k: v for k, v in COL_RENAMES.items() if k in gdf.columns})

    # Normalise string columns
    for col in ["road_name", "road_type"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].str.strip()

    # Spatial filter
    if bbox is not None:
        before = len(gdf)
        gdf = _bbox_filter(gdf, bbox)
        logger.info(f"  Bbox filter: {before:,} → {len(gdf):,} road links")

    # Validate
    missing_exp = [c for c in EXPECTED_COLS if c not in gdf.columns]
    if missing_exp:
        logger.warning(f"  Expected columns still missing after rename: {missing_exp}")

    logger.info(f"MRDB loaded: {len(gdf):,} links | road types: "
                f"{gdf['road_type'].value_counts().to_dict() if 'road_type' in gdf.columns else 'n/a'}")
    return gdf


def save_mrdb(gdf: gpd.GeoDataFrame, output_folder: str | Path = _DEFAULT_OUTPUT_FOLDER) -> None:
    """
    Save the MRDB GeoDataFrame to GeoParquet (preserves geometry efficiently).

    Parameters
    ----------
    gdf : GeoDataFrame from load_mrdb()
    output_folder : defaults to data/processed/shapefiles/

    Example
    -------
    >>> gdf = load_mrdb()
    >>> save_mrdb(gdf)
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = output_folder / "mrdb_yorkshire.parquet"
    gdf.to_parquet(out_path, index=False)
    logger.info(f"Saved MRDB to {out_path}")


def main(
    raw_folder: str | Path = None,
    output_folder: str | Path = None,
    bbox: tuple | None = YORKSHIRE_BBOX,
) -> gpd.GeoDataFrame:
    """
    Load MRDB shapefile, filter to bbox, and save to GeoParquet.

    Parameters
    ----------
    raw_folder : defaults to data/raw/shapefiles/
    output_folder : defaults to data/processed/shapefiles/
    bbox : spatial filter; defaults to Yorkshire
    """
    if raw_folder is None:
        raw_folder = _DEFAULT_RAW_FOLDER
    if output_folder is None:
        output_folder = _DEFAULT_OUTPUT_FOLDER

    gdf = load_mrdb(raw_folder, bbox=bbox)

    print("\n=== MRDB summary ===")
    print(f"  Road links : {len(gdf):,}")
    print(f"  CRS        : {gdf.crs}")
    print(f"  Columns    : {gdf.columns.tolist()}")
    if "road_type" in gdf.columns:
        print(f"  Road types :\n{gdf['road_type'].value_counts().to_string()}")
    if "count_point_id" in gdf.columns:
        n_with_cp = gdf["count_point_id"].notna().sum()
        print(f"  Links with count_point_id: {n_with_cp:,} / {len(gdf):,}")

    save_mrdb(gdf, output_folder)
    return gdf


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    raw_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(raw_folder=raw_folder)