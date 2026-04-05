"""
snap.py
-------
Two methods for snapping STATS19 collisions to OS Open Roads links.

snap_weighted(collisions, openroads)
    Sophisticated multi-criteria scoring snap.
    For each collision, finds the top K=20 nearest candidate links within
    a search radius, scores each on four dimensions, returns the best match.

    Scoring dimensions:
      1. Spatial distance     (40%) — exponential decay, 100m half-life
      2. Road classification  (25%) — does link road type match collision road class?
      3. Form of way/junction (25%) — slip road collision → slip road link
      4. Road number          (10%) — low weight, known quality issues

    Returns collision DataFrame with link_id, snap_score, snap_distance_m,
    and per-dimension scores for inspection.

snap_quick(collisions, openroads)
    Simple nearest-neighbour spatial snap within a distance cap.
    Fast — one sjoin_nearest call. Useful as a baseline to compare against
    snap_weighted, and for rapid iteration on other pipeline stages.

Both functions return the same output schema so they are interchangeable
downstream in join.py.

Usage
-----
    from road_risk.snap import snap_weighted, snap_quick, compare_snaps

    snapped_w = snap_weighted(collisions, openroads)
    snapped_q = snap_quick(collisions, openroads, cap_m=500)
    comparison = compare_snaps(snapped_w, snapped_q)
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from road_risk.config import _ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WGS84 = "EPSG:4326"
BNG   = "EPSG:27700"

# Weighted snap parameters
K_CANDIDATES   = 20      # nearest links to consider per collision
SEARCH_RADIUS  = 500     # metres — candidate search radius
HALF_LIFE_M    = 100     # metres — spatial score half-life

# Dimension weights — must sum to 1.0
W_SPATIAL  = 0.40
W_CLASS    = 0.25
W_JUNCTION = 0.25
W_NUMBER   = 0.10

# Quick snap default cap
QUICK_CAP_M = 500

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# STATS19 first_road_class → set of preferred OS Open Roads road_classification
# values. Score = 1.0 if in preferred, 0.5 if in partial, 0.0 if in penalty.
ROAD_CLASS_SCORES = {
    1: {  # Motorway
        "preferred": {"Motorway"},
        "partial":   {"A Road"},
        "penalty":   {"B Road", "Classified Unnumbered", "Unclassified",
                      "Not Classified", "Unknown"},
    },
    2: {  # A(M) — motorway-standard A road
        "preferred": {"Motorway", "A Road"},
        "partial":   {"Classified Unnumbered"},
        "penalty":   {"B Road", "Unclassified"},
    },
    3: {  # A Road
        "preferred": {"A Road"},
        "partial":   {"Classified Unnumbered"},
        "penalty":   {"Motorway", "B Road", "Unclassified"},
    },
    4: {  # B Road
        "preferred": {"B Road"},
        "partial":   {"Classified Unnumbered", "A Road"},
        "penalty":   {"Motorway", "Unclassified"},
    },
    5: {  # C Road
        "preferred": {"Classified Unnumbered"},
        "partial":   {"B Road", "Not Classified"},
        "penalty":   {"Motorway"},
    },
    6: {  # Unclassified
        "preferred": {"Unclassified", "Not Classified", "Unknown",
                      "Classified Unnumbered"},
        "partial":   {"B Road"},
        "penalty":   {"Motorway"},
    },
}

# STATS19 junction_detail → form_of_way score rules
# Each entry: (preferred set, neutral set, penalised set)
# Anything not listed gets score 0.5 (neutral)
JUNCTION_SCORES = {
    0: {  # Not at junction — should NOT be on slip road or roundabout
        "preferred": {"Single Carriageway", "Dual Carriageway",
                      "Collapsed Dual Carriageway"},
        "partial":   {"Shared Use Carriageway"},
        "penalty":   {"Slip Road", "Roundabout"},
    },
    13: {  # T or staggered junction
        "preferred": {"Single Carriageway", "Dual Carriageway"},
        "partial":   {"Collapsed Dual Carriageway", "Roundabout"},
        "penalty":   {"Slip Road"},
    },
    16: {  # Crossroads
        "preferred": {"Single Carriageway", "Dual Carriageway"},
        "partial":   {"Collapsed Dual Carriageway"},
        "penalty":   {"Slip Road"},
    },
    17: {  # Multiple junction
        "preferred": {"Single Carriageway", "Roundabout"},
        "partial":   {"Dual Carriageway", "Collapsed Dual Carriageway"},
        "penalty":   {"Slip Road"},
    },
    18: {  # Private drive / entrance
        "preferred": {"Single Carriageway"},
        "partial":   {"Classified Unnumbered", "Not Classified"},
        "penalty":   {"Motorway", "Slip Road"},
    },
    19: {  # Other junction
        "preferred": {"Single Carriageway", "Roundabout", "Slip Road"},
        "partial":   {"Dual Carriageway", "Collapsed Dual Carriageway"},
        "penalty":   set(),
    },
    -1: {  # Unknown — no constraint
        "preferred": set(),
        "partial":   set(),
        "penalty":   set(),
    },
    99: {  # Unknown
        "preferred": set(),
        "partial":   set(),
        "penalty":   set(),
    },
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _densify_links(
    links_bng: gpd.GeoDataFrame,
    interval_m: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate points along each road link at fixed intervals.

    Instead of one centroid per link, generates a point every interval_m
    metres along the actual LineString geometry. This ensures that even
    a 2km motorway link is represented by ~80 points, so a collision
    60m from the road will always find a nearby candidate.

    Parameters
    ----------
    links_bng  : GeoDataFrame in BNG (metres), with link_id column
    interval_m : spacing between interpolated points in metres

    Returns
    -------
    xy       : (N, 2) array of point coordinates
    link_ids : (N,) array of link_id for each point (same index as xy)
    """
    xs, ys, ids = [], [], []

    for _, row in links_bng.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        length = geom.length
        link_id = row["link_id"]

        # Always include start, end, and one point every interval_m
        n_points = max(2, int(length / interval_m) + 1)
        distances = np.linspace(0, length, n_points)

        for d in distances:
            pt = geom.interpolate(d)
            xs.append(pt.x)
            ys.append(pt.y)
            ids.append(link_id)

    xy       = np.column_stack([xs, ys])
    link_ids = np.array(ids)
    logger.info(
        f"  Densified {len(links_bng):,} links → "
        f"{len(xy):,} points at {interval_m}m interval"
    )
    return xy, link_ids


def _spatial_score(distance_m: np.ndarray, half_life: float = HALF_LIFE_M) -> np.ndarray:
    """
    Exponential decay spatial score.
    Score = 1.0 at distance 0, 0.5 at half_life metres, ~0 at 5× half_life.
    """
    return np.exp(-distance_m * np.log(2) / half_life)


def _road_class_score(
    stats19_class: int,
    or_classification: pd.Series,
) -> np.ndarray:
    """
    Score how well OS Open Roads road_classification matches
    STATS19 first_road_class for a set of candidate links.
    """
    rules = ROAD_CLASS_SCORES.get(stats19_class)
    if rules is None:
        return np.full(len(or_classification), 0.5)

    scores = np.full(len(or_classification), 0.5)  # default neutral
    scores[or_classification.isin(rules["preferred"]).values] = 1.0
    scores[or_classification.isin(rules["penalty"]).values]   = 0.0
    return scores


def _junction_score(
    junction_detail: int,
    or_form_of_way: pd.Series,
) -> np.ndarray:
    """
    Score how well OS Open Roads form_of_way matches
    STATS19 junction_detail for a set of candidate links.
    """
    rules = JUNCTION_SCORES.get(junction_detail, JUNCTION_SCORES[-1])

    scores = np.full(len(or_form_of_way), 0.5)  # default neutral
    if rules["preferred"]:
        scores[or_form_of_way.isin(rules["preferred"]).values] = 1.0
    if rules["penalty"]:
        scores[or_form_of_way.isin(rules["penalty"]).values]   = 0.0
    if rules["partial"]:
        # partial only if not already set to preferred/penalty
        is_partial = or_form_of_way.isin(rules["partial"]).values
        unset = (scores == 0.5)
        scores[is_partial & unset] = 0.3
    return scores


def _road_number_score(
    stats19_road_name: str,
    or_road_name_clean: pd.Series,
) -> np.ndarray:
    """
    Score based on road number match.
    1.0 = exact match, 0.5 = no road number in STATS19 (unknown), 0.1 = mismatch.
    Low weight (10%) because of known STATS19 road number quality issues.
    """
    if not stats19_road_name:
        return np.full(len(or_road_name_clean), 0.5)  # unknown — neutral

    scores = np.where(
        or_road_name_clean.values == stats19_road_name,
        1.0,
        0.1,
    )
    # Upgrade links with no road name to neutral — they can't contradict
    scores[or_road_name_clean.values == ""] = 0.5
    return scores


# ---------------------------------------------------------------------------
# Weighted snap
# ---------------------------------------------------------------------------

def snap_weighted(
    collisions: pd.DataFrame,
    openroads: gpd.GeoDataFrame,
    k: int = K_CANDIDATES,
    search_radius_m: float = SEARCH_RADIUS,
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Snap collisions to OS Open Roads links using multi-criteria weighted scoring.

    For each collision with valid coordinates, finds the K nearest candidate
    links within search_radius_m, scores each on four dimensions, and returns
    the link with the highest composite score.

    Parameters
    ----------
    collisions     : cleaned collision DataFrame — needs latitude, longitude,
                     first_road_class, junction_detail, road_name_clean,
                     coords_valid
    openroads      : OS Open Roads GeoDataFrame — needs link_id, geometry,
                     road_classification, form_of_way, road_name_clean
    k              : number of nearest candidates to score per collision
    search_radius_m: maximum search radius in metres
    weights        : override default dimension weights
                     e.g. {'spatial': 0.5, 'class': 0.2, 'junction': 0.2, 'number': 0.1}

    Returns
    -------
    DataFrame with all original collision columns plus:
      link_id           : best matching OS Open Roads link
      snap_distance_m   : distance to matched link (metres)
      snap_score        : composite score 0–1
      score_spatial     : spatial dimension score
      score_class       : road classification score
      score_junction    : junction/form-of-way score
      score_number      : road number score
      snap_method       : 'weighted', 'weighted_fallback', or 'unmatched'
    """
    w = weights or {
        "spatial":  W_SPATIAL,
        "class":    W_CLASS,
        "junction": W_JUNCTION,
        "number":   W_NUMBER,
    }
    assert abs(sum(w.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    logger.info(
        f"snap_weighted: {len(collisions):,} collisions × "
        f"{len(openroads):,} links | k={k}, radius={search_radius_m}m"
    )

    # --- Project to BNG for metre-based distance calculations ---------------
    valid_mask = collisions["coords_valid"].fillna(False) \
        if "coords_valid" in collisions.columns \
        else pd.Series(True, index=collisions.index)

    valid_col = collisions[valid_mask].copy()
    invalid_col = collisions[~valid_mask].copy()

    coll_gdf = gpd.GeoDataFrame(
        valid_col,
        geometry=gpd.points_from_xy(valid_col["longitude"], valid_col["latitude"]),
        crs=WGS84,
    ).to_crs(BNG)

    or_bng = openroads.to_crs(BNG).copy()

    # Build collision coordinate array
    coll_xy = np.column_stack([coll_gdf.geometry.x, coll_gdf.geometry.y])

    # Densify road links — interpolate points every 25m along each LineString.
    # This replaces centroid lookup (which fails for long links) with geometry-
    # aware distance so a collision 60m from a 2km motorway link is found correctly.
    logger.info("  Densifying road links ...")
    dense_xy, dense_ids = _densify_links(or_bng, interval_m=25.0)

    # Build KD-tree on densified points
    logger.info("  Building KD-tree ...")
    tree = cKDTree(dense_xy)

    # For each collision, find K nearest densified points within search radius
    # Each point carries its parent link_id — deduplicate to get unique candidate links
    distances, indices = tree.query(
        coll_xy,
        k=min(k, len(dense_xy)),
        distance_upper_bound=search_radius_m,
        workers=-1,
    )

    logger.info("  Scoring candidates ...")

    # Build a lookup from link_id → row in or_bng for fast candidate retrieval
    or_reset = or_bng.reset_index(drop=True)
    link_id_to_idx = {lid: i for i, lid in enumerate(or_reset["link_id"])}

    results = []

    for i, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        # Filter out inf (beyond search radius)
        valid_k = row_distances < np.inf
        if not valid_k.any():
            results.append({
                "link_id": pd.NA,
                "snap_distance_m": np.nan,
                "snap_score": np.nan,
                "score_spatial":  np.nan,
                "score_class":    np.nan,
                "score_junction": np.nan,
                "score_number":   np.nan,
                "snap_method": "unmatched",
            })
            continue

        dists_raw   = row_distances[valid_k]
        point_ids   = dense_ids[row_indices[valid_k]]

        # Deduplicate: for each unique link keep the closest dense point distance
        seen = {}
        for dist, lid in zip(dists_raw, point_ids):
            if lid not in seen or dist < seen[lid]:
                seen[lid] = dist

        cand_link_ids = list(seen.keys())
        cand_dists    = np.array([seen[lid] for lid in cand_link_ids])

        # Retrieve candidate rows from openroads
        cand_indices  = [link_id_to_idx[lid] for lid in cand_link_ids
                         if lid in link_id_to_idx]
        if not cand_indices:
            results.append({
                "link_id": pd.NA,
                "snap_distance_m": np.nan,
                "snap_score": np.nan,
                "score_spatial":  np.nan,
                "score_class":    np.nan,
                "score_junction": np.nan,
                "score_number":   np.nan,
                "snap_method": "unmatched",
            })
            continue

        candidates = or_reset.iloc[cand_indices]
        dists      = cand_dists[:len(cand_indices)]

        collision_row = valid_col.iloc[i]

        # Score each dimension
        s_spatial  = _spatial_score(dists, HALF_LIFE_M)

        road_class  = int(collision_row.get("first_road_class", 0) or 0)
        s_class     = _road_class_score(road_class, candidates["road_classification"])

        junc_detail = int(collision_row.get("junction_detail", -1) or -1)
        s_junction  = _junction_score(junc_detail, candidates["form_of_way"])

        road_name   = str(collision_row.get("road_name_clean", "") or "")
        s_number    = _road_number_score(road_name, candidates["road_name_clean"])

        composite = (
            w["spatial"]  * s_spatial  +
            w["class"]    * s_class    +
            w["junction"] * s_junction +
            w["number"]   * s_number
        )

        best_idx  = np.argmax(composite)
        best_link = candidates.iloc[best_idx]

        results.append({
            "link_id":         best_link["link_id"],
            "snap_distance_m": float(dists[best_idx]),
            "snap_score":      float(composite[best_idx]),
            "score_spatial":   float(s_spatial[best_idx]),
            "score_class":     float(s_class[best_idx]),
            "score_junction":  float(s_junction[best_idx]),
            "score_number":    float(s_number[best_idx]),
            "snap_method":     "weighted",
        })

        if i % 10000 == 0 and i > 0:
            logger.info(f"  ... {i:,} / {len(valid_col):,} processed")

    result_df = pd.DataFrame(results, index=valid_col.index)
    out = valid_col.copy()
    for col in result_df.columns:
        out[col] = result_df[col]

    # Append invalid coord rows
    if not invalid_col.empty:
        for col in ["link_id", "snap_distance_m", "snap_score",
                    "score_spatial", "score_class", "score_junction",
                    "score_number"]:
            invalid_col[col] = np.nan
        invalid_col["snap_method"] = "invalid_coords"
        out = pd.concat([out, invalid_col], ignore_index=True)

    # Summary
    method_counts = out["snap_method"].value_counts()
    n_matched = (out["snap_method"] == "weighted").sum()
    logger.info(
        f"snap_weighted complete:\n{method_counts.to_string()}\n"
        f"  Matched: {n_matched:,} / {len(collisions):,} "
        f"({n_matched/len(collisions):.1%})\n"
        f"  Mean score (matched): "
        f"{out.loc[out['snap_method']=='weighted', 'snap_score'].mean():.3f}\n"
        f"  Mean distance (matched): "
        f"{out.loc[out['snap_method']=='weighted', 'snap_distance_m'].mean():.1f}m"
    )
    return out


# ---------------------------------------------------------------------------
# Quick snap
# ---------------------------------------------------------------------------

def snap_quick(
    collisions: pd.DataFrame,
    openroads: gpd.GeoDataFrame,
    cap_m: float = QUICK_CAP_M,
) -> pd.DataFrame:
    """
    Simple nearest-neighbour spatial snap within a distance cap.

    Fast — single geopandas sjoin_nearest call using STRtree index.
    No scoring — just closest link within cap_m metres.

    Use for:
      - Baseline comparison against snap_weighted
      - Rapid iteration when pipeline code is changing
      - Links that will be reviewed manually anyway

    Parameters
    ----------
    collisions : cleaned collision DataFrame
    openroads  : OS Open Roads GeoDataFrame
    cap_m      : maximum snap distance in metres (default 500m)

    Returns
    -------
    DataFrame with same schema as snap_weighted output.
    score columns are NaN (not computed).
    snap_method = 'spatial' (within cap) or 'unmatched' (beyond cap)
    """
    logger.info(
        f"snap_quick: {len(collisions):,} collisions | cap={cap_m}m"
    )

    valid_mask = collisions["coords_valid"].fillna(False) \
        if "coords_valid" in collisions.columns \
        else pd.Series(True, index=collisions.index)

    valid_col   = collisions[valid_mask].copy()
    invalid_col = collisions[~valid_mask].copy()

    coll_gdf = gpd.GeoDataFrame(
        valid_col,
        geometry=gpd.points_from_xy(valid_col["longitude"], valid_col["latitude"]),
        crs=WGS84,
    ).to_crs(BNG)

    or_bng = openroads[["link_id", "geometry"]].to_crs(BNG)

    joined = gpd.sjoin_nearest(
        coll_gdf[["geometry"]],
        or_bng,
        how="left",
        distance_col="snap_distance_m",
    )
    joined = joined[~joined.index.duplicated(keep="first")]

    valid_col["link_id"]         = joined["link_id"].values
    valid_col["snap_distance_m"] = joined["snap_distance_m"].values
    valid_col["snap_score"]      = np.nan
    valid_col["score_spatial"]   = np.nan
    valid_col["score_class"]     = np.nan
    valid_col["score_junction"]  = np.nan
    valid_col["score_number"]    = np.nan

    within_cap = valid_col["snap_distance_m"] <= cap_m
    valid_col["snap_method"] = np.where(within_cap, "spatial", "unmatched")

    # Null out link_id for beyond-cap matches
    valid_col.loc[~within_cap, "link_id"] = pd.NA

    if not invalid_col.empty:
        for c in ["link_id", "snap_distance_m", "snap_score",
                  "score_spatial", "score_class", "score_junction", "score_number"]:
            invalid_col[c] = np.nan
        invalid_col["snap_method"] = "invalid_coords"
        valid_col = pd.concat([valid_col, invalid_col], ignore_index=True)

    method_counts = valid_col["snap_method"].value_counts()
    n_matched = (valid_col["snap_method"] == "spatial").sum()
    logger.info(
        f"snap_quick complete:\n{method_counts.to_string()}\n"
        f"  Matched: {n_matched:,} / {len(collisions):,} "
        f"({n_matched/len(collisions):.1%})"
    )
    return valid_col


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_snaps(
    weighted: pd.DataFrame,
    quick: pd.DataFrame,
    id_col: str = "collision_index",
) -> pd.DataFrame:
    """
    Compare snap_weighted and snap_quick results side by side.

    Returns a DataFrame with one row per collision showing:
      - Whether both methods agree on the link
      - Where they disagree: the two link IDs and their scores/distances
      - The scoring breakdown for the weighted result

    Useful for assessing how much the weighted approach improves on quick.

    Parameters
    ----------
    weighted : output of snap_weighted()
    quick    : output of snap_quick()
    id_col   : collision identifier column

    Returns
    -------
    DataFrame with columns:
      collision_index, w_link_id, q_link_id, agree,
      w_distance_m, q_distance_m, w_score,
      score_spatial, score_class, score_junction, score_number,
      w_method, q_method
    """
    w = weighted[[id_col, "link_id", "snap_distance_m", "snap_score",
                  "score_spatial", "score_class", "score_junction",
                  "score_number", "snap_method"]].copy()
    w.columns = [id_col, "w_link_id", "w_distance_m", "w_score",
                 "score_spatial", "score_class", "score_junction",
                 "score_number", "w_method"]

    q = quick[[id_col, "link_id", "snap_distance_m", "snap_method"]].copy()
    q.columns = [id_col, "q_link_id", "q_distance_m", "q_method"]

    comp = w.merge(q, on=id_col, how="outer")
    comp["agree"] = comp["w_link_id"] == comp["q_link_id"]

    # Summary stats
    both_matched = (
        comp["w_method"].isin(["weighted"]) &
        comp["q_method"].isin(["spatial"])
    )
    n_agree    = comp.loc[both_matched, "agree"].sum()
    n_disagree = both_matched.sum() - n_agree
    pct_agree  = n_agree / both_matched.sum() if both_matched.sum() > 0 else 0

    logger.info(
        f"Snap comparison:\n"
        f"  Both matched    : {both_matched.sum():,}\n"
        f"  Agree on link   : {n_agree:,} ({pct_agree:.1%})\n"
        f"  Disagree on link: {n_disagree:,} ({1-pct_agree:.1%})\n"
        f"  Weighted-only matched : "
        f"{(comp['w_method']=='weighted') .sum() - both_matched.sum():,}\n"
        f"  Quick-only matched    : "
        f"{(comp['q_method']=='spatial').sum() - both_matched.sum():,}"
    )
    return comp


# ---------------------------------------------------------------------------
# Main — run both methods and compare
# ---------------------------------------------------------------------------

def main() -> None:
    import geopandas as gpd
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    processed = _ROOT / "data/processed"

    collisions = pd.read_parquet(processed / "stats19/collision_clean.parquet")

    # Load OS Open Roads — regenerate from GeoPackage if cache missing
    or_cache = processed / "shapefiles/openroads_yorkshire.parquet"
    if or_cache.exists():
        openroads = gpd.read_parquet(or_cache)
        logger.info(f"Loaded OS Open Roads from cache ({len(openroads):,} links)")
    else:
        logger.info("Cache not found — loading from GeoPackage ...")
        from road_risk.ingest.ingest_openroads import load_openroads, save_openroads
        openroads = load_openroads()
        save_openroads(openroads, processed / "shapefiles")

    logger.info("Running snap_weighted ...")
    snapped_w = snap_weighted(collisions, openroads)
    snapped_w.to_parquet(processed / "stats19/snapped_weighted.parquet", index=False)
    logger.info("Saved snapped_weighted.parquet")

    logger.info("Running snap_quick ...")
    snapped_q = snap_quick(collisions, openroads, cap_m=500)
    snapped_q.to_parquet(processed / "stats19/snapped_quick.parquet", index=False)
    logger.info("Saved snapped_quick.parquet")

    logger.info("Comparing ...")
    comp = compare_snaps(snapped_w, snapped_q)

    print("\n=== Snap comparison ===")
    print(f"  Weighted matched : {(snapped_w['snap_method']=='weighted').sum():,} "
          f"({(snapped_w['snap_method']=='weighted').mean():.1%})")
    print(f"  Quick matched    : {(snapped_q['snap_method']=='spatial').sum():,} "
          f"({(snapped_q['snap_method']=='spatial').mean():.1%})")
    print(f"\n  Agreement rate   : {comp['agree'].mean():.1%}")
    print(f"\n  Weighted score distribution:")
    print(snapped_w["snap_score"].describe().round(3))
    print(f"\n  Score by road classification:")
    joined = snapped_w.merge(
        openroads[["link_id", "road_classification", "form_of_way"]],
        on="link_id", how="left"
    )
    print(joined.groupby("road_classification")["snap_score"].mean().round(3))


if __name__ == "__main__":
    main()