"""
network_features.py
-------------------
Computes network topology and population demand features for all OS Open Roads
links. These are the "Big 3" features that replace lat/lon as spatial signal
in the AADT estimator, giving the model genuine causal structure.

Feature 1 — Node degree (junction complexity)
    Mean degree of start/end nodes of each link.
    High = urban junction-heavy road.
    Low = long uninterrupted rural link.

Feature 2 — Betweenness centrality (network importance)
    Approximate betweenness using sampled shortest paths (k=500).
    High = this road is on many shortest paths = likely through-traffic corridor.
    Low = cul-de-sac or genuinely local road.
    Computed on simplified graph (major roads only for speed), then propagated
    to all links via nearest major-road node.

Feature 3 — Distance to nearest major road (node)
    Graph-distance (in km) from each link's nearest node to the nearest
    Motorway or A Road node.
    Small = feeder road, close to trunk network.
    Large = genuinely isolated local road.

Feature 4 — Population density (LSOA join)
    Population per km² of the LSOA whose centroid is nearest to the road link
    centroid, within 2km. Proxy for local demand.
    Requires: data/raw/stats19/lsoa_population.csv
    Download from: https://www.ons.gov.uk/peoplepopulationandcommunity/
                   populationandmigration/populationestimates/datasets/
                   lowersuperoutputareamidyearpopulationestimates

Usage
-----
    python src/road_risk/network_features.py

    Or from model.py:
        from road_risk.network_features import build_network_features
        features_df = build_network_features(openroads)
"""

import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from road_risk.config import _ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED      = _ROOT / "data/processed"
OPENROADS_PATH = PROCESSED / "shapefiles/openroads_yorkshire.parquet"
LSOA_CENT_PATH = _ROOT / "data/raw/stats19/lsoa_centroids.csv"
LSOA_POP_PATH  = _ROOT / "data/raw/stats19/lsoa_population.csv"
OUTPUT_PATH    = _ROOT / "data/features/network_features.parquet"

# Betweenness centrality sample size — higher = more accurate, slower
# 500 gives a good approximation in ~2-3 minutes for Yorkshire
BETWEENNESS_K = 200

# Max distance for population density join (metres)
POP_JOIN_CAP_M = 2000

# Road classifications that count as "major"
MAJOR_CLASSES = {"Motorway", "A Road"}


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def build_graph(openroads: gpd.GeoDataFrame) -> nx.Graph:
    """
    Build an undirected NetworkX graph from OS Open Roads links.

    Nodes = start_node / end_node TOIDs
    Edges = road links with weight = link_length_km

    Parameters
    ----------
    openroads : GeoDataFrame with link_id, start_node, end_node,
                link_length_km, road_classification columns
    """
    logger.info(f"Building graph from {len(openroads):,} road links ...")

    G = nx.Graph()

    # Add edges
    for _, row in openroads.iterrows():
        u = row["start_node"]
        v = row["end_node"]
        if pd.isna(u) or pd.isna(v):
            continue
        G.add_edge(
            u, v,
            link_id=row["link_id"],
            weight=max(row.get("link_length_km", 0.1) or 0.1, 0.001),
            road_classification=row.get("road_classification", "Unknown"),
        )

    logger.info(
        f"  Graph: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges"
    )
    return G


# ---------------------------------------------------------------------------
# Feature 1: Node degree
# ---------------------------------------------------------------------------

def compute_node_degree(
    G: nx.Graph,
    openroads: gpd.GeoDataFrame,
) -> pd.Series:
    """
    Compute mean node degree for each road link.

    Returns
    -------
    Series indexed by link_id with degree_mean values.
    """
    logger.info("Computing node degree ...")
    degree = dict(G.degree())

    results = {}
    for _, row in openroads.iterrows():
        u = row["start_node"]
        v = row["end_node"]
        deg_u = degree.get(u, 1)
        deg_v = degree.get(v, 1)
        results[row["link_id"]] = (deg_u + deg_v) / 2

    s = pd.Series(results, name="degree_mean")
    logger.info(
        f"  degree_mean: median={s.median():.1f}, "
        f"max={s.max():.0f} (high = complex junction area)"
    )
    return s


# ---------------------------------------------------------------------------
# Feature 2: Betweenness centrality
# ---------------------------------------------------------------------------

def compute_betweenness(
    G: nx.Graph,
    openroads: gpd.GeoDataFrame,
    k: int = BETWEENNESS_K,
) -> pd.Series:
    """
    Compute approximate betweenness centrality for each road link.

    Uses NODE betweenness (faster than edge betweenness) with k-sampled
    shortest paths, then aggregates to links as the mean of endpoint values.

    Node betweenness is O(k × V) vs edge betweenness O(k × V × E/V) —
    much faster on large graphs like OS Open Roads (700k+ edges).

    Returns
    -------
    Series indexed by link_id with betweenness values.
    """
    logger.info(
        f"Computing node betweenness centrality (k={k}, ~1-3 mins) ..."
    )

    # Node betweenness — faster than edge betweenness on large graphs
    node_bc = nx.betweenness_centrality(G, k=k, seed=42, weight="weight")

    # Aggregate to links: mean betweenness of start/end nodes
    results = {}
    for _, row in openroads.iterrows():
        u, v = row["start_node"], row["end_node"]
        bc_u = node_bc.get(u, 0.0)
        bc_v = node_bc.get(v, 0.0)
        results[row["link_id"]] = (bc_u + bc_v) / 2

    s = pd.Series(results, name="betweenness").reindex(
        openroads["link_id"], fill_value=0.0
    )

    logger.info(
        f"  betweenness: median={s.median():.6f}, "
        f"p99={s.quantile(0.99):.6f} | "
        f"{(s > 0).sum():,} links with non-zero betweenness"
    )
    return s


# ---------------------------------------------------------------------------
# Feature 3: Distance to nearest major road
# ---------------------------------------------------------------------------

def compute_dist_to_major(
    G: nx.Graph,
    openroads: gpd.GeoDataFrame,
) -> pd.Series:
    """
    Compute graph-distance (km) from each link to the nearest
    Motorway or A Road node.

    Uses multi-source Dijkstra from all major-road nodes simultaneously.

    Returns
    -------
    Series indexed by link_id with dist_to_major_km values.
    """
    logger.info("Computing distance to nearest major road ...")

    # Identify major road nodes
    major_links = openroads[
        openroads["road_classification"].isin(MAJOR_CLASSES)
    ]
    major_nodes = set()
    for _, row in major_links.iterrows():
        if pd.notna(row["start_node"]):
            major_nodes.add(row["start_node"])
        if pd.notna(row["end_node"]):
            major_nodes.add(row["end_node"])

    major_nodes = major_nodes & set(G.nodes())
    logger.info(f"  {len(major_nodes):,} major road nodes as sources")

    if not major_nodes:
        logger.warning("No major road nodes found — returning zeros")
        return pd.Series(0.0, index=openroads["link_id"], name="dist_to_major_km")

    # Multi-source Dijkstra — distance from each node to nearest major node
    node_dist = nx.multi_source_dijkstra_path_length(
        G, sources=major_nodes, weight="weight"
    )

    # Assign to links: use minimum of start/end node distances
    results = {}
    for _, row in openroads.iterrows():
        u, v = row["start_node"], row["end_node"]
        d_u = node_dist.get(u, np.nan)
        d_v = node_dist.get(v, np.nan)
        if pd.notna(d_u) and pd.notna(d_v):
            results[row["link_id"]] = min(d_u, d_v)
        elif pd.notna(d_u):
            results[row["link_id"]] = d_u
        elif pd.notna(d_v):
            results[row["link_id"]] = d_v
        else:
            results[row["link_id"]] = np.nan

    s = pd.Series(results, name="dist_to_major_km")
    logger.info(
        f"  dist_to_major_km: median={s.median():.2f}km, "
        f"p90={s.quantile(0.9):.2f}km"
    )
    return s


# ---------------------------------------------------------------------------
# Feature 4: Population density
# ---------------------------------------------------------------------------

def compute_population_density(
    openroads: gpd.GeoDataFrame,
    lsoa_cent_path: Path = LSOA_CENT_PATH,
    lsoa_pop_path: Path = LSOA_POP_PATH,
    cap_m: float = POP_JOIN_CAP_M,
) -> pd.Series:
    """
    Compute population density (people/km²) for each road link by joining
    to the nearest LSOA centroid within cap_m metres.

    Requires two files:
      lsoa_centroids.csv  — LSOA21CD, x, y (BNG) — already downloaded
      lsoa_population.*   — ONS LSOA mid-year population estimates
                            Excel: "Mid-2024 LSOA 2021" sheet, data from row 4
                            Columns: LAD 2023 Code, LAD 2023 Name,
                                     LSOA 2021 Code, LSOA 2021 Name, Total, ...
                            Save as data/raw/stats19/lsoa_population.xlsx
                            or      data/raw/stats19/lsoa_population.csv

    Returns
    -------
    Series indexed by link_id with pop_density_per_km2 values.
    NaN where no LSOA centroid within cap_m metres.
    """
    if not lsoa_cent_path.exists():
        logger.warning(
            f"LSOA centroids not found at {lsoa_cent_path} — "
            "population density feature will be NaN"
        )
        return pd.Series(np.nan, index=openroads["link_id"], name="pop_density_per_km2")

    # --- Load LSOA centroids (BNG x, y) ------------------------------------
    lsoa_cent = pd.read_csv(
        lsoa_cent_path,
        usecols=["LSOA21CD", "x", "y"],
        encoding="utf-8-sig",
    )

    # --- Load population ----------------------------------------------------
    # Try Excel first, then CSV fallback
    pop_loaded = False
    lsoa_pop = None

    # Check for Excel file
    xl_path = lsoa_pop_path.with_suffix(".xlsx")
    if xl_path.exists():
        logger.info(f"  Reading population from Excel: {xl_path.name}")
        try:
            raw = pd.read_excel(
                xl_path,
                sheet_name="Mid-2024 LSOA 2021",
                header=3,          # row 4 = index 3 is the header
                usecols=[2, 4],    # LSOA 2021 Code, Total
            )
            raw.columns = ["LSOA21CD", "population"]
            raw = raw.dropna(subset=["LSOA21CD", "population"])
            # Remove comma formatting e.g. "1,898" → 1898
            raw["population"] = (
                raw["population"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )
            raw = raw.dropna(subset=["population"])
            lsoa_pop = raw
            pop_loaded = True
            logger.info(f"  Loaded {len(lsoa_pop):,} LSOA population records")
        except Exception as e:
            logger.warning(f"  Failed to read Excel: {e}")

    # CSV fallback
    if not pop_loaded and lsoa_pop_path.with_suffix(".csv").exists():
        csv_path = lsoa_pop_path.with_suffix(".csv")
        logger.info(f"  Reading population from CSV: {csv_path.name}")
        raw = pd.read_csv(csv_path, encoding="utf-8-sig")
        raw.columns = raw.columns.str.strip()
        # Find LSOA code and total columns
        code_col = next(
            (c for c in raw.columns if "lsoa" in c.lower() and "code" in c.lower()),
            None
        )
        tot_col = next(
            (c for c in raw.columns if c.lower() in ("total", "population", "all ages")),
            None
        )
        if code_col and tot_col:
            raw = raw.rename(columns={code_col: "LSOA21CD", tot_col: "population"})
            raw["population"] = (
                raw["population"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )
            lsoa_pop = raw[["LSOA21CD", "population"]].dropna()
            pop_loaded = True
            logger.info(f"  Loaded {len(lsoa_pop):,} LSOA population records from CSV")

    # --- Merge population onto centroids ------------------------------------
    if pop_loaded and lsoa_pop is not None:
        lsoa_cent = lsoa_cent.merge(lsoa_pop, on="LSOA21CD", how="left")
        n_matched = lsoa_cent["population"].notna().sum()
        logger.info(
            f"  Population matched for {n_matched:,} / {len(lsoa_cent):,} LSOAs"
        )

        # Approximate area: use mean LSOA area for Yorkshire (~2.9 km²)
        # For better accuracy download LSOA boundaries and compute area
        MEAN_LSOA_AREA_KM2 = 2.9
        lsoa_cent["pop_density"] = (
            lsoa_cent["population"] / MEAN_LSOA_AREA_KM2
        )
        use_density = True
    else:
        logger.warning(
            f"Population file not found. Looked for:\n"
            f"  {xl_path}\n"
            f"  {lsoa_pop_path.with_suffix('.csv')}\n"
            f"Falling back to LSOA count proxy.\n"
            f"Download ONS LSOA mid-year estimates Excel and save as:\n"
            f"  {xl_path}"
        )
        use_density = False

    # --- Spatial join -------------------------------------------------------
    lsoa_xy  = lsoa_cent[["x", "y"]].values
    or_bng   = openroads.to_crs("EPSG:27700").copy()
    road_xy  = np.column_stack([
        or_bng.geometry.centroid.x,
        or_bng.geometry.centroid.y,
    ])

    tree = cKDTree(lsoa_xy)
    dists, indices = tree.query(road_xy, k=1, distance_upper_bound=cap_m)
    valid = dists < cap_m

    if use_density:
        pop_values = lsoa_cent["pop_density"].values
        result = np.full(len(road_xy), np.nan)
        result[valid] = pop_values[indices[valid]]
        s = pd.Series(result, index=openroads["link_id"], name="pop_density_per_km2")
    else:
        # Fallback: count LSOAs within 1km as urbanisation proxy
        dists_5, _ = tree.query(road_xy, k=5, distance_upper_bound=1000)
        lsoa_count = (dists_5 < 1000).sum(axis=1).astype(float)
        lsoa_count[~valid] = np.nan
        s = pd.Series(
            lsoa_count, index=openroads["link_id"],
            name="pop_density_per_km2"
        )

    n_valid = s.notna().sum()
    logger.info(
        f"  pop_density_per_km2: {n_valid:,} / {len(openroads):,} links matched "
        f"(median={s.median():.0f})"
    )
    return s


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_network_features(
    openroads: gpd.GeoDataFrame = None,
    openroads_path: Path = OPENROADS_PATH,
    output_path: Path = OUTPUT_PATH,
    betweenness_k: int = BETWEENNESS_K,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Compute all network and population features for OS Open Roads links.

    Caches result to data/features/network_features.parquet so subsequent
    runs are instant. Use force_recompute=True to regenerate.

    Parameters
    ----------
    openroads       : pre-loaded GeoDataFrame (optional — loads from path if None)
    openroads_path  : path to openroads_yorkshire.parquet
    output_path     : where to cache the feature table
    betweenness_k   : sample size for betweenness (higher = slower + more accurate)
    force_recompute : if True, ignores cache

    Returns
    -------
    DataFrame indexed by link_id with columns:
        degree_mean, betweenness, dist_to_major_km, pop_density_per_km2
    """
    if output_path.exists() and not force_recompute:
        logger.info(f"Loading cached network features from {output_path}")
        return pd.read_parquet(output_path)

    if openroads is None:
        logger.info(f"Loading OS Open Roads from {openroads_path}")
        openroads = gpd.read_parquet(openroads_path)
        logger.info(f"  {len(openroads):,} links loaded")

    # Build graph
    G = build_graph(openroads)

    # Compute features
    degree      = compute_node_degree(G, openroads)
    betweenness = compute_betweenness(G, openroads, k=betweenness_k)
    dist_major  = compute_dist_to_major(G, openroads)
    pop_density = compute_population_density(openroads)

    # Combine
    features = pd.DataFrame({
        "link_id":             openroads["link_id"].values,
        "degree_mean":         degree.values,
        "betweenness":         betweenness.values,
        "dist_to_major_km":    dist_major.values,
        "pop_density_per_km2": pop_density.values,
    })

    # Log feature summaries
    logger.info("\n=== Network feature summary ===")
    for col in ["degree_mean", "betweenness", "dist_to_major_km", "pop_density_per_km2"]:
        vals = features[col].dropna()
        logger.info(
            f"  {col:25s}: median={vals.median():.4f}, "
            f"p25={vals.quantile(0.25):.4f}, "
            f"p75={vals.quantile(0.75):.4f}, "
            f"nulls={features[col].isna().sum():,}"
        )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    logger.info(f"Saved network features to {output_path} ({len(features):,} rows)")

    return features


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if cache exists"
    )
    parser.add_argument(
        "--k", type=int, default=BETWEENNESS_K,
        help=f"Betweenness sample size (default: {BETWEENNESS_K})"
    )
    args = parser.parse_args()

    features = build_network_features(
        betweenness_k=args.k,
        force_recompute=args.force,
    )

    print("\n=== Network features ===")
    print(features.describe().round(4).to_string())
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()