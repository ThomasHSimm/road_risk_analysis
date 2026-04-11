"""
ingest_test_routes.py
---------------------
Batch processes crowdsourced GPX driving test routes.
Verifies their geographic validity, matches them to an official DVSA 
test centre, and snaps the GPS trace to the OS Open Roads network.

Usage
-----
    python src/road_risk/ingest/ingest_test_routes.py
"""

import logging
from pathlib import Path
import gpxpy
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString

from road_risk.config import _ROOT
from road_risk.snap import _densify_links  # Reuse your existing densification logic

logger = logging.getLogger(__name__)

# --- Constants ---
UK_BBOX = {'lat': (49.5, 60.9), 'lon': (-8.2, 2.0)}
DTC_MAX_DIST_M = 1000  # Start/end point must be within 1km of a test centre
MIN_KM = 5
MAX_KM = 45

# --- Paths ---
RAW_ROUTES_DIR = _ROOT / "data/raw/test_routes"
DTC_SUMMARY_PATH = _ROOT / "data/raw/dvsa/dtc_summary.csv"
OPENROADS_PATH = _ROOT / "data/processed/shapefiles/openroads_yorkshire.parquet"
OUTPUT_PATH = _ROOT / "data/processed/test_routes/snapped_routes.parquet"


def load_dtc_lookup(dtc_path: Path) -> gpd.GeoDataFrame:
    """Loads DVSA Test Centres and projects to BNG for distance matching."""
    logger.info(f"Loading DTC lookup from {dtc_path.name}")
    df = pd.read_csv(dtc_path)
    
    # Filter to UK bounding box
    yorks_df = df[
        df['latitude'].between(*UK_BBOX['lat']) & 
        df['longitude'].between(*UK_BBOX['lon'])
    ].copy()
    
    gdf = gpd.GeoDataFrame(
        yorks_df, 
        geometry=gpd.points_from_xy(yorks_df['longitude'], yorks_df['latitude']),
        crs="EPSG:4326"
    ).to_crs("EPSG:27700")
    
    return gdf


def parse_and_verify_gpx(gpx_path: Path, dtc_gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict]:
    """Reads GPX, verifies bounding box/length, and matches to nearest DTC."""
    try:
        with open(gpx_path, 'r') as f:
            gpx = gpxpy.parse(f)
    except Exception as e:
        return None, {"error": f"Failed to parse GPX: {e}"}

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append((pt.latitude, pt.longitude, pt.time))

    if len(points) < 10:
        return None, {"error": "Too few track points"}

    df = pd.DataFrame(points, columns=['latitude', 'longitude', 'time'])

    # 1. Bounding Box Filter
    if not (df['latitude'].between(*UK_BBOX['lat']).all() and 
            df['longitude'].between(*UK_BBOX['lon']).all()):
        return None, {"error": "Route exits UK study area"}

    # 2. Length Filter
    line = LineString(df[['longitude', 'latitude']].values)
    line_gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326").to_crs("EPSG:27700")
    length_km = line_gdf.geometry.length.iloc[0] / 1000.0
    
    if not (MIN_KM <= length_km <= MAX_KM):
        return None, {"error": f"Invalid length ({length_km:.1f}km)"}

    # 3. DTC Matcher
    start_pt = Point(df.iloc[0]['longitude'], df.iloc[0]['latitude'])
    start_gdf = gpd.GeoDataFrame(geometry=[start_pt], crs="EPSG:4326").to_crs("EPSG:27700")
    
    distances = dtc_gdf.distance(start_gdf.iloc[0].geometry)
    nearest_idx = distances.idxmin()
    min_dist = distances.min()

    if min_dist > DTC_MAX_DIST_M:
        return None, {"error": f"Starts {min_dist:.0f}m from nearest DTC (cap is {DTC_MAX_DIST_M}m)"}

    matched_dtc = dtc_gdf.loc[nearest_idx]

    # 4. End-point check — must return to the same DTC
    end_pt = Point(df.iloc[-1]['longitude'], df.iloc[-1]['latitude'])
    end_gdf = gpd.GeoDataFrame(geometry=[end_pt], crs="EPSG:4326").to_crs("EPSG:27700")
    end_dist = matched_dtc.geometry.distance(end_gdf.iloc[0].geometry)

    if end_dist > DTC_MAX_DIST_M:
        return None, {"error": f"Ends {end_dist:.0f}m from matched DTC '{matched_dtc['name']}' (cap is {DTC_MAX_DIST_M}m)"}

    meta = {
        "file_name": gpx_path.name,
        "dtc_id": matched_dtc['id'],
        "dtc_name": matched_dtc['name'],
        "start_dist_to_dtc_m": min_dist,
        "end_dist_to_dtc_m": end_dist,
        "route_length_km": length_km,
        "n_gps_points": len(df)
    }
    
    return df, meta


def snap_route_sequence(route_df: pd.DataFrame, tree: cKDTree, dense_ids: np.ndarray) -> list:
    """
    Takes high-frequency GPS points, snaps to densified KD-tree, 
    and compresses into a sequence of unique link_ids traversed.
    """
    # Project route points to BNG
    route_gdf = gpd.GeoDataFrame(
        route_df, 
        geometry=gpd.points_from_xy(route_df['longitude'], route_df['latitude']),
        crs="EPSG:4326"
    ).to_crs("EPSG:27700")
    
    route_xy = np.column_stack([route_gdf.geometry.x, route_gdf.geometry.y])
    
    # Query nearest point on road network for each GPS ping
    dists, indices = tree.query(route_xy, k=1, distance_upper_bound=100)
    
    # Filter out points that drifted too far from any road
    valid = dists < np.inf
    snapped_links = dense_ids[indices[valid]]
    
    # Compress sequence (e.g., [A, A, A, B, B, C, C, A] -> [A, B, C, A])
    # This represents the actual links traversed in order.
    route_sequence = []
    if len(snapped_links) > 0:
        route_sequence.append(snapped_links[0])
        for link in snapped_links[1:]:
            if link != route_sequence[-1]:
                route_sequence.append(link)
                
    return route_sequence


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    if not RAW_ROUTES_DIR.exists():
        logger.warning(f"No raw routes directory found at {RAW_ROUTES_DIR}. Please create and add GPX files.")
        return

    # 1. Load lookups and network infrastructure
    dtc_gdf = load_dtc_lookup(DTC_SUMMARY_PATH)
    
    logger.info("Loading OS Open Roads for snapping infrastructure...")
    openroads = gpd.read_parquet(OPENROADS_PATH)
    or_bng = openroads.to_crs("EPSG:27700")
    
    # Reuse your brilliant densification logic to handle long links
    logger.info("Building spatial index (densifying links)...")
    dense_xy, dense_ids = _densify_links(or_bng, interval_m=25.0)
    tree = cKDTree(dense_xy)

    # 2. Batch Runner
    gpx_files = list(RAW_ROUTES_DIR.glob("*.gpx"))
    logger.info(f"Found {len(gpx_files)} GPX files to process.")
    
    successfully_snapped = []
    
    for gpx_file in gpx_files:
        route_df, meta = parse_and_verify_gpx(gpx_file, dtc_gdf)
        
        if route_df is None:
            logger.warning(f"  [REJECTED] {gpx_file.name}: {meta['error']}")
            continue
            
        # Snap the verified route
        link_sequence = snap_route_sequence(route_df, tree, dense_ids)
        
        if not link_sequence:
            logger.warning(f"  [REJECTED] {gpx_file.name}: Failed to snap to road network.")
            continue
            
        meta['link_sequence'] = link_sequence
        meta['n_unique_links'] = len(link_sequence)
        successfully_snapped.append(meta)
        
        logger.info(f"  [SUCCESS] {gpx_file.name} -> {meta['dtc_name']} ({meta['n_unique_links']} links)")

    # 3. Save Output
    if successfully_snapped:
        out_df = pd.DataFrame(successfully_snapped)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(OUTPUT_PATH, index=False)
        logger.info(f"\nProcessing complete! Saved {len(out_df)} valid test routes to {OUTPUT_PATH.name}")
    else:
        logger.warning("\nNo valid routes processed.")

if __name__ == "__main__":
    main()