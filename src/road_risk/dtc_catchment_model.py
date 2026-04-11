"""
dtc_catchment_model.py
----------------------
1. Generates 10-minute drive-time isochrones for Yorkshire Test Centres using osmnx.
2. Joins '1st Attempt' pass rates to isolate environmental difficulty.
3. Spatially aggregates road network features within the isochrone.
4. Trains a Random Forest to explain Pass Rates based on structural road complexity.
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from road_risk.config import _ROOT

logger = logging.getLogger(__name__)

# --- Paths ---
DTC_PATH = _ROOT / "data/raw/dvsa/dtc_summary.csv"
ATTEMPTS_PATH = _ROOT / "data/raw/dvsa/pass_rate_by_attempt.csv"
NET_FEATURES_PATH = _ROOT / "data/features/network_features.parquet"
RISK_SCORES_PATH = _ROOT / "data/models/risk_scores.parquet"
OPENROADS_PATH = _ROOT / "data/processed/shapefiles/openroads_yorkshire.parquet"
OUT_PATH = _ROOT / "data/features/dtc_catchment_profiles.parquet"

TRIP_TIME_MINS = 10
YORKS_BBOX = {'lat': (53.3, 54.7), 'lon': (-2.4, 0.2)}


def get_isochrone_polygon(lat, lon, trip_time_mins=10):
    """Uses osmnx to generate a drive-time polygon around a lat/lon."""
    try:
        # Download graph (approx 8km radius to ensure 10 min drive is captured)
        G = ox.graph_from_point((lat, lon), dist=8000, network_type='drive')
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        
        # Find nearest node to the DTC
        center_node = ox.distance.nearest_nodes(G, lon, lat)
        
        # Calculate subgraph within travel time
        trip_time_sec = trip_time_mins * 60
        subgraph = nx.ego_graph(G, center_node, radius=trip_time_sec, distance='travel_time')
        
        # Extract node points and create a convex hull polygon
        node_points = [Point(data['x'], data['y']) for node, data in subgraph.nodes(data=True)]
        if len(node_points) < 3:
            return Point(lon, lat).buffer(0.05) # Fallback
            
        return gpd.GeoSeries(node_points).unary_union.convex_hull
        
    except Exception as e:
        logger.warning(f"Isochrone failed for {lat},{lon}: {e}. Using fallback buffer.")
        return Point(lon, lat).buffer(0.05) # ~5km fallback in degrees


def build_catchment_profiles():
    logger.info("Loading DTC lookup and generating Drive-Time Isochrones...")
    dtc_df = pd.read_csv(DTC_PATH)
    
    # Filter to Yorkshire to save osmnx download time
    dtc_df = dtc_df[dtc_df['latitude'].between(*YORKS_BBOX['lat']) & 
                    dtc_df['longitude'].between(*YORKS_BBOX['lon'])].copy()

    # Join 1st Attempt Pass Rates
    attempts_df = pd.read_csv(ATTEMPTS_PATH)
    first_attempts = attempts_df[attempts_df['ATTEMPTS_TRUNC'] == 1].copy()
    dtc_df = dtc_df.merge(first_attempts[['TC_NAME', 'pass_rate']], 
                          left_on='name', right_on='TC_NAME', how='left')
    
    # Generate Isochrones
    isochrone_polys = []
    for idx, row in dtc_df.iterrows():
        logger.info(f"  Generating {TRIP_TIME_MINS}-min isochrone for {row['name']}...")
        poly = get_isochrone_polygon(row['latitude'], row['longitude'], TRIP_TIME_MINS)
        isochrone_polys.append(poly)
        
    dtc_gdf = gpd.GeoDataFrame(dtc_df, geometry=isochrone_polys, crs="EPSG:4326").to_crs("EPSG:27700")

    logger.info("Loading network features and risk scores...")
    net_df = pd.read_parquet(NET_FEATURES_PATH)
    risk_df = pd.read_parquet(RISK_SCORES_PATH)
    road_df = net_df.merge(
        risk_df[['link_id', 'estimated_aadt', 'risk_percentile', 'residual_glm']], 
        on='link_id', how='inner'
    )

    logger.info("Projecting OpenRoads to BNG for spatial join...")
    openroads = gpd.read_parquet(OPENROADS_PATH)
    or_centroids = gpd.GeoDataFrame(
        openroads[['link_id']], 
        geometry=openroads.to_crs("EPSG:27700").geometry.centroid, 
        crs="EPSG:27700"
    )
    road_gdf = or_centroids.merge(road_df, on='link_id', how='inner')

    logger.info("Performing Spatial Join (Roads within Isochrones)...")
    joined = gpd.sjoin(road_gdf, dtc_gdf[['id', 'name', 'pass_rate', 'geometry']], how="inner", predicate="intersects")

    logger.info("Aggregating structural features per DTC...")
    catchment_profiles = joined.groupby('id').agg(
        dtc_name=('name', 'first'),
        first_attempt_pass_rate=('pass_rate', 'first'),
        
        # Network Complexity
        junction_density_mean=('degree_mean', 'mean'),
        junction_density_max=('degree_mean', 'max'),
        through_traffic_mean=('betweenness_relative', 'mean'),
        
        # Traffic Exposure
        traffic_volume_median=('estimated_aadt', 'median'),
        
        # Risk & Surprises
        excess_risk_mean=('residual_glm', 'mean'),
        pct_high_risk_links=('risk_percentile', lambda x: (x >= 95).mean() * 100),
        
        # Demographics / Pedestrians
        pop_density_mean=('pop_density_per_km2', 'mean')
    ).reset_index()

    catchment_profiles.to_parquet(OUT_PATH, index=False)
    logger.info(f"Saved {len(catchment_profiles)} DTC Isochrone profiles.")
    
    return catchment_profiles


def train_dtc_model(profiles_df):
    logger.info("Training Environmental Pass Rate Predictor...")
    df = profiles_df.dropna(subset=['first_attempt_pass_rate'])
    
    features = [
        'junction_density_mean', 'junction_density_max', 'through_traffic_mean',
        'traffic_volume_median', 'excess_risk_mean', 'pct_high_risk_links',
        'pop_density_mean'
    ]
    X, y = df[features], df['first_attempt_pass_rate']

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))
    ])

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    logger.info(f"Model CV R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

    model.fit(X, y)
    imp_df = pd.DataFrame({'Feature': features, 'Importance': model.named_steps['rf'].feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=False)
    
    logger.info("\n=== Feature Importance (What drives pass rates?) ===")
    logger.info("\n" + imp_df.to_string(index=False))
    
    return model, imp_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    profiles = build_catchment_profiles()
    model, importances = train_dtc_model(profiles)