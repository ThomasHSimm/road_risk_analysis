"""
model.py
--------
Two-stage traffic and collision risk modelling.

Stage 1a — Static AADT estimator
    Predicts annual average daily traffic flow for all OS Open Roads links.
    Trained on AADF count points (2,240 locations × 3 years).
    Features: road classification, form of way, is_trunk, is_primary,
              link_length_km, lat/lon, year.
    Applied to all 705,672 OS Open Roads links to fill coverage gaps.

Stage 1b — Temporal traffic profile
    Models monthly variation in traffic volume and large vehicle percentage
    using WebTRIS sensor data (dense motorway/trunk road coverage).
    Output: per-road-type seasonal multipliers and weekday/weekend ratios.
    Used to enrich collision records with time-of-day context.

Stage 2 — Collision count model (TODO — depends on Stage 1 output)
    Predicts collision_count per road link per year.
    Uses estimated_aadt from Stage 1a as exposure offset.

Usage
-----
    python src/road_risk/model.py --stage traffic
    python src/road_risk/model.py --stage temporal
    python src/road_risk/model.py --stage all
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from road_risk.config import _ROOT, cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED = _ROOT / "data/processed"
FEATURES = _ROOT / "data/features"
MODELS = _ROOT / "data/models"

AADF_PATH = PROCESSED / "aadf/aadf_clean.parquet"
WEBTRIS_RAW = _ROOT / "data/raw/webtris"
WEBTRIS_PATH = PROCESSED / "webtris/webtris_clean.parquet"
OPENROADS_PATH = PROCESSED / "shapefiles/openroads_yorkshire.parquet"
SITES_PATH = _ROOT / "data/raw/webtris/sites.parquet"
RLA_PATH = FEATURES / "road_link_annual.parquet"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROAD_CLASS_ORDER = [
    "Motorway",
    "A Road",
    "B Road",
    "Classified Unnumbered",
    "Not Classified",
    "Unclassified",
    "Unknown",
]

FORM_OF_WAY_ORDER = [
    "Dual Carriageway",
    "Collapsed Dual Carriageway",
    "Motorway",
    "Slip Road",
    "Roundabout",
    "Single Carriageway",
    "Shared Use Carriageway",
    "Guided Busway",
]

MONTH_ORDER = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Stage 1a — Static AADT estimator
# ---------------------------------------------------------------------------


def build_aadt_features(aadf: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix and target for AADT estimation.

    Features
    --------
    road_class_ord   : ordinal encoding of road_type (Major=1, Minor=0)
    is_trunk         : 1 if trunk road (from road_name prefix M/A)
    latitude         : lat of count point
    longitude        : lon of count point
    link_length_km   : link length (NaN for many minor roads)
    hgv_proportion   : HGV fraction (useful cross-feature)
    year_norm        : normalised year (0=2019, 0.5=2021, 1=2023)
    is_covid         : COVID year flag

    Network features (joined from network_features.parquet if available):
    degree_mean      : mean node degree at link endpoints (junction complexity)
    betweenness      : edge betweenness centrality (through-traffic proxy)
    dist_to_major_km : graph distance to nearest major road node
    pop_density_per_km2 : population density of nearest LSOA

    Target
    ------
    log_aadt : log(all_motor_vehicles + 1) — log-transform stabilises variance
    """
    df = aadf.copy()

    # Road type ordinal — Major=1, Minor=0
    df["road_class_ord"] = (df["road_type"] == "Major").astype(int)

    # Trunk road proxy from road name
    df["is_trunk"] = df["road_name"].str.match(r"^[MA]\d", na=False).astype(int)

    # Year normalisation
    year_min, year_max = df["year"].min(), df["year"].max()
    df["year_norm"] = (df["year"] - year_min) / max(year_max - year_min, 1)

    # Log target
    target = np.log1p(df["all_motor_vehicles"])

    feature_cols = [
        "road_class_ord",
        "is_trunk",
        "latitude",
        "longitude",
        "year_norm",
        "is_covid",
        "hgv_proportion",
    ]

    # link_length_km has ~21% missingness — impute with median per road type
    if "link_length_km" in df.columns:
        df["link_length_km"] = df.groupby("road_type")["link_length_km"].transform(
            lambda x: x.fillna(x.median() if x.notna().any() else 0)
        )
        feature_cols.append("link_length_km")

    # Join network features — snap AADF count points to nearest OS Open Roads
    # link via KD-tree, then look up network features for that link.
    # Cannot use road_link_annual as bridge — it only contains collision links
    # (22k) not AADF count point links (2,240), so almost no overlap.
    net_path = _ROOT / "data/features/network_features.parquet"
    or_path = _ROOT / "data/processed/shapefiles/openroads_yorkshire.parquet"

    if net_path.exists() and or_path.exists():
        import geopandas as gpd
        from scipy.spatial import cKDTree

        net = pd.read_parquet(net_path)
        # Use all columns except link_id — picks up new features automatically
        net_cols = [c for c in net.columns if c != "link_id"]

        # Build KD-tree from OS Open Roads link centroids (BNG)
        or_gdf = gpd.read_parquet(or_path)
        or_bng = or_gdf.to_crs("EPSG:27700")
        link_xy = np.column_stack(
            [
                or_bng.geometry.centroid.x,
                or_bng.geometry.centroid.y,
            ]
        )
        link_ids = or_gdf["link_id"].values
        tree = cKDTree(link_xy)

        # Convert AADF lat/lon to BNG for matching
        import pyproj

        wgs_to_bng = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:27700", always_xy=True
        )
        aadf_e, aadf_n = wgs_to_bng.transform(
            df["longitude"].values, df["latitude"].values
        )
        aadf_xy = np.column_stack([aadf_e, aadf_n])

        # Snap each AADF count point to nearest OS Open Roads link (2km cap)
        dists, idx = tree.query(aadf_xy, k=1, distance_upper_bound=2000)
        valid = dists < 2000

        # idx may equal len(link_ids) for unmatched points — safe index with valid mask
        df["_snapped_link_id"] = None
        df.loc[valid, "_snapped_link_id"] = link_ids[idx[valid]]

        df = df.merge(
            net[["link_id"] + net_cols].rename(columns={"link_id": "_snapped_link_id"}),
            on="_snapped_link_id",
            how="left",
        )
        df = df.drop(columns=["_snapped_link_id"])

        n_joined = df["degree_mean"].notna().sum()
        logger.info(
            f"  Network features joined for {n_joined:,} / {len(df):,} "
            f"AADF count points ({n_joined/len(df):.1%}) "
            f"via direct KD-tree snap"
        )
        for col in net_cols:
            if col in df.columns:
                feature_cols.append(col)
    else:
        logger.info(
            f"  Network features or OpenRoads not found — "
            f"training without network features (CV R² will be lower)"
        )

    X = df[feature_cols].copy()
    return X, target, df


def train_aadt_estimator(aadf: pd.DataFrame) -> tuple:
    """
    Train a gradient boosting model to estimate AADT from road attributes.

    Uses GroupKFold cross-validation grouped by count_point_id to prevent
    data leakage — the same count point appears in multiple years, so it
    must not appear in both train and test folds.

    Returns
    -------
    model    : fitted GradientBoostingRegressor
    metrics  : dict of cross-validated MAE, RMSE, R²
    features : list of feature column names
    """
    logger.info("Training AADT estimator ...")

    X, y, df = build_aadt_features(aadf)
    groups = df["count_point_id"].values

    model = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbose=0,
    )

    # GroupKFold — same count point stays in same fold
    cv = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )
    cv_mae = cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    # Fit on full dataset
    model.fit(X, y)

    # Permutation importance (HistGBR doesn't expose feature_importances_ directly)
    perm = permutation_importance(model, X, y, n_repeats=5, random_state=RANDOM_STATE)
    importance = pd.Series(perm.importances_mean, index=X.columns).sort_values(
        ascending=False
    )

    metrics = {
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "cv_mae_mean": float(-cv_mae.mean()),  # back to positive
        "cv_mae_std": float(cv_mae.std()),
        "n_train": len(X),
        "n_features": len(X.columns),
    }

    logger.info(
        f"  AADT estimator CV R²: {metrics['cv_r2_mean']:.3f} "
        f"(±{metrics['cv_r2_std']:.3f}) | "
        f"MAE: {metrics['cv_mae_mean']:.3f} log-units"
    )
    logger.info(f"  Feature importance:\n{importance.to_string()}")

    return model, metrics, X.columns.tolist()


def apply_aadt_estimator(
    model,
    feature_cols: list,
    openroads,
    aadf: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the trained AADT estimator to all OS Open Roads links.

    Returns
    -------
    DataFrame with link_id and estimated_aadt for all links × AADF years.
    """
    import geopandas as gpd

    logger.info(
        f"Applying AADT estimator to {len(openroads):,} OS Open Roads links ..."
    )

    # Build prediction features matching training schema
    or_df = openroads.copy()
    if isinstance(or_df, gpd.GeoDataFrame):
        bng = or_df.to_crs("EPSG:27700")
        centroids = bng.geometry.centroid.to_crs("EPSG:4326")
        or_df["latitude"] = centroids.y
        or_df["longitude"] = centroids.x

    or_df["road_class_ord"] = (
        or_df["road_classification"].isin(["Motorway", "A Road"]).astype(int)
    )
    or_df["is_trunk"] = or_df.get(
        "is_trunk", pd.Series(False, index=or_df.index)
    ).astype(int)

    # Join network features if available
    net_path = _ROOT / "data/features/network_features.parquet"
    if net_path.exists():
        net = pd.read_parquet(net_path)
        net_cols = [c for c in net.columns if c != "link_id"]
        or_df = or_df.merge(net[["link_id"] + net_cols], on="link_id", how="left")
        n_joined = or_df["degree_mean"].notna().sum()
        logger.info(
            f"  Network features joined for {n_joined:,} / {len(or_df):,} links"
        )

    # Predict for each AADF year
    years = sorted(aadf["year"].unique())
    year_min, year_max = years[0], years[-1]
    aadf_medians = aadf.groupby("road_type")[
        ["hgv_proportion", "link_length_km"]
    ].median()
    road_type_map = {"Motorway": "Major", "A Road": "Major"}

    frames = []
    for year in years:
        pred_df = or_df.copy()
        pred_df["year"] = year
        pred_df["year_norm"] = (year - year_min) / max(year_max - year_min, 1)
        pred_df["is_covid"] = year in {2020, 2021}
        pred_df["hgv_proportion"] = np.nan

        pred_df["_road_type"] = (
            pred_df["road_classification"].map(road_type_map).fillna("Minor")
        )

        for col in ["hgv_proportion", "link_length_km"]:
            if col in feature_cols:
                for rtype in ["Major", "Minor"]:
                    mask = pred_df["_road_type"] == rtype
                    if rtype in aadf_medians.index:
                        pred_df.loc[mask & pred_df[col].isna(), col] = aadf_medians.loc[
                            rtype, col
                        ]

        # Only use feature columns that exist (network features may be absent)
        available_cols = [c for c in feature_cols if c in pred_df.columns]
        X_pred = pred_df[available_cols].fillna(0)
        log_pred = model.predict(X_pred)
        pred_df["estimated_aadt"] = np.expm1(log_pred).round().astype(int)
        pred_df["estimated_aadt"] = pred_df["estimated_aadt"].clip(lower=1)
        frames.append(pred_df[["link_id", "year", "estimated_aadt"]])

    result = pd.concat(frames, ignore_index=True)
    logger.info(
        f"  Estimated AADT: median={result['estimated_aadt'].median():,.0f} "
        f"vehicles/day | range {result['estimated_aadt'].min():,}–"
        f"{result['estimated_aadt'].max():,}"
    )
    return result


# ---------------------------------------------------------------------------
# Stage 1b — Temporal traffic profile (WebTRIS)
# ---------------------------------------------------------------------------


def build_temporal_profiles(
    raw_folder: Path = WEBTRIS_RAW,
    sites_path: Path = SITES_PATH,
) -> pd.DataFrame:
    """
    Build monthly traffic profiles from WebTRIS raw chunk parquets.

    For each road corridor (M62, M1, A1M etc), computes:
      - Monthly index: traffic in month M relative to annual mean (1.0 = average)
      - Weekday/weekend ratio: awt24hour / adt24hour per month
      - Large vehicle seasonal pattern: adt24largevehiclepercentage by month

    Uses raw chunks (not webtris_clean) to get monthly grain before aggregation.

    Returns
    -------
    DataFrame at road_prefix × month grain with seasonal multipliers.
    """
    logger.info("Building temporal traffic profiles from WebTRIS ...")

    sites = pd.read_parquet(sites_path)[
        ["site_id", "description", "latitude", "longitude"]
    ]
    sites["road_prefix"] = sites["description"].str[:4].str.strip()

    # Load all raw chunks for Yorkshire sites
    chunks = sorted(raw_folder.glob("site_*_*.parquet"))
    if not chunks:
        raise FileNotFoundError(f"No WebTRIS chunks found in {raw_folder}")

    yorkshire_sites = set(
        sites[
            sites["latitude"].between(53.3, 54.5)
            & sites["longitude"].between(-2.8, 0.0)
        ]["site_id"]
    )

    frames = []
    for chunk in chunks:
        site_id = int(chunk.stem.split("_")[1])
        if site_id not in yorkshire_sites:
            continue
        df = pd.read_parquet(chunk)
        df["site_id"] = site_id
        frames.append(df)

    if not frames:
        raise ValueError("No Yorkshire WebTRIS chunks found")

    raw = pd.concat(frames, ignore_index=True)
    logger.info(
        f"  Loaded {len(raw):,} monthly rows from {len(yorkshire_sites):,} sites"
    )

    # Coerce numeric
    for col in ["adt24hour", "awt24hour", "adt24largevehiclepercentage"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Join road prefix
    raw = raw.merge(sites[["site_id", "road_prefix"]], on="site_id", how="left")

    # Standardise month order
    raw["month_num"] = (
        pd.Categorical(raw["monthname"], categories=MONTH_ORDER, ordered=True).codes + 1
    )
    raw["monthname"] = pd.Categorical(
        raw["monthname"], categories=MONTH_ORDER, ordered=True
    )

    # Aggregate by road_prefix × month across all years and sites
    profile = (
        raw.groupby(["road_prefix", "monthname", "month_num"])
        .agg(
            mean_adt24=("adt24hour", "mean"),
            mean_awt24=("awt24hour", "mean"),
            mean_large_pct=("adt24largevehiclepercentage", "mean"),
            n_site_months=("site_id", "count"),
        )
        .reset_index()
        .sort_values(["road_prefix", "month_num"])
    )

    # Seasonal index: month flow / mean annual flow per road prefix
    annual_mean = profile.groupby("road_prefix")["mean_adt24"].transform("mean")
    profile["seasonal_index"] = profile["mean_adt24"] / annual_mean.replace(0, np.nan)

    # Weekday/weekend ratio (>1 = weekend busier, <1 = weekday busier)
    profile["weekday_weekend_ratio"] = profile["mean_awt24"] / profile[
        "mean_adt24"
    ].replace(0, np.nan)

    logger.info(
        f"  Profiles built for {profile['road_prefix'].nunique()} road types × "
        f"12 months"
    )
    logger.info(
        f"  Seasonal index range: "
        f"{profile['seasonal_index'].min():.3f} – {profile['seasonal_index'].max():.3f}"
    )
    return profile


def plot_temporal_profiles(profiles: pd.DataFrame) -> None:
    """
    Plot seasonal traffic profiles for the main road types.
    Saves to data/models/temporal_profiles.png
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return

    top_roads = (
        profiles.groupby("road_prefix")["n_site_months"]
        .sum()
        .nlargest(6)
        .index.tolist()
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, road in zip(axes, top_roads):
        data = profiles[profiles["road_prefix"] == road].sort_values("month_num")
        ax.bar(data["monthname"], data["seasonal_index"], color="steelblue", alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(
            data["monthname"],
            data["mean_large_pct"],
            color="crimson",
            marker="o",
            linewidth=2,
            markersize=4,
        )
        ax2.set_ylabel("Large vehicle %", color="crimson")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{road} (n={data['n_site_months'].sum():,})")
        ax.set_ylabel("Seasonal index (1.0 = annual mean)")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle(
        "WebTRIS seasonal traffic profiles — Yorkshire motorways/trunk roads",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    out = MODELS / "temporal_profiles.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved temporal profiles plot to {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Stage 2 — Collision count model (Poisson GLM + XGBoost)
# ---------------------------------------------------------------------------

ROAD_CLASS_ORDINAL = {
    "Motorway": 6,
    "A Road": 5,
    "B Road": 4,
    "Classified Unnumbered": 3,
    "Not Classified": 2,
    "Unclassified": 1,
    "Unknown": 0,
}

FORM_OF_WAY_ORDINAL = {
    "Dual Carriageway": 4,
    "Collapsed Dual Carriageway": 3,
    "Slip Road": 2,
    "Roundabout": 2,
    "Single Carriageway": 1,
    "Shared Use Carriageway": 1,
    "Guided Busway": 0,
}


def build_collision_dataset(
    openroads,
    aadt_estimates: pd.DataFrame,
    rla: pd.DataFrame,
    net_features: pd.DataFrame | None = None,
    years: list | None = None,
) -> pd.DataFrame:
    """
    Build full collision dataset for Poisson modelling.

    Critical: includes links with ZERO collisions, not just links where
    collisions happened. Without zeros the model learns "how many given ≥1"
    not "collision risk across all roads."

    Steps:
    1. All OS Open Roads links × years (705k × n_years)
    2. Left-join collision counts from road_link_annual (NaN → 0)
    3. Left-join AADT estimates from Stage 1a
    4. Filter to rows where AADT is available (exposure denominator required)
    5. Compute offset = log(AADT × link_length_km × 365 / 1e6)

    Parameters
    ----------
    openroads      : GeoDataFrame with link geometry and attributes
    aadt_estimates : DataFrame with link_id, year, estimated_aadt
    rla            : road_link_annual.parquet — collision counts per link × year
    net_features   : optional network features DataFrame
    years          : years to include (defaults to all in aadt_estimates)

    Returns
    -------
    DataFrame at link_id × year grain with:
      collision_count, log_offset, feature columns, confidence flags
    """
    import geopandas as gpd

    if years is None:
        years = sorted(aadt_estimates["year"].unique())

    logger.info(
        f"Building collision dataset: {len(openroads):,} links × "
        f"{len(years)} years ..."
    )

    # --- Base: all links × years -------------------------------------------
    links = openroads[
        [
            "link_id",
            "road_classification",
            "form_of_way",
            "link_length_km",
            "is_trunk",
            "is_primary",
        ]
    ].copy()

    base = pd.DataFrame(
        {
            "link_id": np.repeat(links["link_id"].values, len(years)),
            "year": np.tile(years, len(links)),
        }
    )
    base = base.merge(links, on="link_id", how="left")

    logger.info(f"  Base table: {len(base):,} rows")

    # --- Join collision counts (0 for links with no collisions) -------------
    rla_cols = [
        "link_id",
        "year",
        "collision_count",
        "fatal_count",
        "serious_count",
        "slight_count",
        "casualty_count",
        # STATS19 contextual aggregates
        "pct_urban",
        "pct_dark",
        "pct_junction",
        "pct_near_crossing",
        "mean_speed_limit",
    ]
    rla_trim = rla[[c for c in rla_cols if c in rla.columns]].copy()

    base = base.merge(rla_trim, on=["link_id", "year"], how="left")
    base["collision_count"] = base["collision_count"].fillna(0).astype(int)
    base["fatal_count"] = base["fatal_count"].fillna(0).astype(int)
    base["serious_count"] = base["serious_count"].fillna(0).astype(int)

    n_with_collisions = (base["collision_count"] > 0).sum()
    logger.info(
        f"  Collisions joined: {n_with_collisions:,} link-years with ≥1 collision "
        f"({n_with_collisions/len(base):.2%} of all link-years)"
    )

    # --- Join AADT estimates ------------------------------------------------
    base = base.merge(aadt_estimates, on=["link_id", "year"], how="left")

    # Filter to rows where we have AADT (required for exposure offset)
    n_before = len(base)
    base = base[base["estimated_aadt"].notna()].copy()
    logger.info(
        f"  After AADT filter: {len(base):,} / {n_before:,} rows "
        f"({len(base)/n_before:.1%})"
    )

    # --- Compute log offset -------------------------------------------------
    # offset = log(AADT × link_length_km × 365 / 1e6)
    # = log(vehicle-km in millions per year)
    # NaN link_length_km → use median per road classification
    median_len = base.groupby("road_classification")["link_length_km"].transform(
        lambda x: x.fillna(x.median() if x.notna().any() else 0.5)
    )
    base["link_length_km"] = base["link_length_km"].fillna(median_len)

    vehicle_km_M = base["estimated_aadt"] * base["link_length_km"] * 365 / 1e6
    base["log_offset"] = np.log(vehicle_km_M.clip(lower=1e-6))

    # --- Encode road features -----------------------------------------------
    base["road_class_ord"] = (
        base["road_classification"].map(ROAD_CLASS_ORDINAL).fillna(0).astype(int)
    )
    base["form_of_way_ord"] = (
        base["form_of_way"].map(FORM_OF_WAY_ORDINAL).fillna(1).astype(int)
    )
    base["is_motorway"] = (base["road_classification"] == "Motorway").astype(int)
    base["is_a_road"] = (base["road_classification"] == "A Road").astype(int)
    base["is_slip_road"] = (base["form_of_way"] == "Slip Road").astype(int)
    base["is_roundabout"] = (base["form_of_way"] == "Roundabout").astype(int)
    base["is_dual"] = (
        base["form_of_way"]
        .isin(["Dual Carriageway", "Collapsed Dual Carriageway"])
        .astype(int)
    )
    base["is_trunk"] = base["is_trunk"].fillna(False).astype(int)
    base["is_primary"] = base["is_primary"].fillna(False).astype(int)

    # Temporal
    base["is_covid"] = base["year"].isin({2020, 2021}).astype(int)
    base["year_norm"] = (base["year"] - base["year"].min()) / max(
        base["year"].max() - base["year"].min(), 1
    )
    base["log_link_length"] = np.log(base["link_length_km"].clip(lower=0.001))

    # --- Join network features if available ---------------------------------
    if net_features is not None:
        base = base.merge(net_features, on="link_id", how="left")
        n_net = base["degree_mean"].notna().sum()
        logger.info(
            f"  Network features joined: {n_net:,} / {len(base):,} rows "
            f"({n_net/len(base):.1%})"
        )

    logger.info(
        f"  Collision dataset: {len(base):,} rows | "
        f"zeros={( base['collision_count']==0).sum():,} "
        f"({(base['collision_count']==0).mean():.1%})"
    )
    return base


def train_collision_glm(df: pd.DataFrame) -> tuple:
    """
    Fit a Poisson GLM for collision counts with AADT exposure offset.

    Model:
        log(E[collisions]) = β×X + log(AADT × length_km × 365 / 1e6)

    The log_offset term anchors predictions to traffic exposure so the
    model learns risk factors *above and beyond* what traffic volume explains.

    Uses statsmodels for interpretable coefficients and confidence intervals.

    Returns
    -------
    result   : statsmodels GLM result object
    features : list of feature column names used
    summary  : dict of key metrics
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError(
            "statsmodels required for Poisson GLM. "
            "Install with: pip install statsmodels"
        )

    logger.info("Fitting Poisson GLM (statsmodels) ...")

    # Core features — always required, well populated
    core_cols = [
        "road_class_ord",
        "form_of_way_ord",
        "is_motorway",
        "is_a_road",
        "is_slip_road",
        "is_roundabout",
        "is_dual",
        "is_trunk",
        "is_primary",
        "log_link_length",
        "is_covid",
        "year_norm",
    ]

    # Network features — include where coverage >50% of rows, else impute median
    # This prevents sparse features (speed 43%, lanes 5%) from dropping 95% of rows
    network_feat_candidates = [
        "degree_mean",
        "betweenness",
        "betweenness_relative",
        "dist_to_major_km",
        "pop_density_per_km2",
        "speed_limit_mph",
        "lanes",
        "is_unpaved",
        # NOTE: pct_urban, pct_dark, pct_junction excluded — data leakage
        # (derived from collision data, only non-null on collision links)
        # Use as lagged features or post-hoc analysis only
    ]

    feature_cols = list(core_cols)
    for col in network_feat_candidates:
        if col not in df.columns:
            continue
        coverage = df[col].notna().mean()
        if coverage > 0.5:
            # Good coverage — include as-is, rows with NaN will be dropped
            feature_cols.append(col)
        elif coverage > 0.05:
            # Sparse — impute with median so we don't lose rows
            median_val = df[col].median()
            df[f"{col}_imputed"] = df[col].fillna(median_val)
            feature_cols.append(f"{col}_imputed")
            logger.info(
                f"  {col}: {coverage:.1%} coverage — imputing median "
                f"({median_val:.2f}) to retain rows"
            )

    # Drop rows missing core features or offset only
    model_df = df[feature_cols + ["collision_count", "log_offset"]].dropna()
    logger.info(
        f"  GLM training rows: {len(model_df):,} "
        f"(dropped {len(df)-len(model_df):,} with missing features)"
    )

    X = sm.add_constant(model_df[feature_cols].astype(float))
    y = model_df["collision_count"].astype(int)
    offset = model_df["log_offset"].astype(float)

    glm = sm.GLM(
        y,
        X,
        family=sm.families.Poisson(),
        offset=offset,
    )
    result = glm.fit(maxiter=100)

    # Key metrics
    summary = {
        "n_obs": len(model_df),
        "deviance": float(result.deviance),
        "null_deviance": float(result.null_deviance),
        "pseudo_r2": float(1 - result.deviance / result.null_deviance),
        "aic": float(result.aic),
        "converged": result.converged,
        "features": feature_cols,
    }

    logger.info(
        f"  Poisson GLM: pseudo-R²={summary['pseudo_r2']:.3f} | "
        f"deviance={summary['deviance']:,.0f} | "
        f"AIC={summary['aic']:,.0f} | "
        f"converged={summary['converged']}"
    )

    # Log significant coefficients
    coef_df = pd.DataFrame(
        {
            "coef": result.params,
            "pvalue": result.pvalues,
            "ci_low": result.conf_int()[0],
            "ci_high": result.conf_int()[1],
        }
    ).round(4)
    sig = coef_df[coef_df["pvalue"] < 0.05].sort_values("coef", ascending=False)
    logger.info(f"  Significant coefficients (p<0.05):\n{sig.to_string()}")

    return result, feature_cols, summary


def train_collision_xgb(df: pd.DataFrame) -> tuple:
    """
    Fit an XGBoost Poisson regression for collision counts.

    Complements the GLM — captures non-linear interactions that the
    linear GLM misses. Uses log_offset as an exposure weight.

    Returns
    -------
    model    : fitted XGBRegressor
    features : list of feature column names
    metrics  : dict of evaluation metrics
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost required. Install with: pip install xgboost")

    logger.info("Fitting XGBoost Poisson model ...")

    feature_cols = [
        "road_class_ord",
        "form_of_way_ord",
        "is_motorway",
        "is_a_road",
        "is_slip_road",
        "is_roundabout",
        "is_dual",
        "is_trunk",
        "is_primary",
        "log_link_length",
        "is_covid",
        "year_norm",
        "estimated_aadt",
    ]
    network_feat_candidates = [
        "degree_mean",
        "betweenness",
        "betweenness_relative",
        "dist_to_major_km",
        "pop_density_per_km2",
        "speed_limit_mph",
        "lanes",
        "is_unpaved",
        # NOTE: pct_urban, pct_dark, pct_junction excluded — data leakage
    ]
    for col in network_feat_candidates:
        if col in df.columns and df[col].notna().sum() > 100:
            feature_cols.append(col)

    # XGBoost handles NaN natively — only drop rows missing core cols or target
    core_required = [
        "collision_count",
        "log_offset",
        "road_class_ord",
        "log_link_length",
        "year_norm",
    ]
    model_df = df[feature_cols + ["collision_count", "log_offset"]].copy()
    model_df = model_df.dropna(
        subset=[c for c in core_required if c in model_df.columns]
    )
    logger.info(f"  XGBoost training rows: {len(model_df):,}")

    X = model_df[feature_cols].astype(float)
    y = model_df["collision_count"].astype(float)

    # GroupKFold by link_id — same link shouldn't appear in train and test
    from sklearn.model_selection import GroupKFold

    groups = model_df.index  # each row is a link-year; use link_id if available

    model = XGBRegressor(
        objective="count:poisson",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    # Simple train/test split (80/20) for evaluation
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    offset_train = model_df.loc[X_train.index, "log_offset"].values
    offset_test = model_df.loc[X_test.index, "log_offset"].values

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)

    # Poisson deviance on test set
    eps = 1e-6
    deviance = 2 * np.sum(
        np.where(y_test > 0, y_test * np.log((y_test + eps) / (y_pred + eps)), 0)
        - (y_test - y_pred)
    )
    null_pred = np.full_like(y_pred, y_test.mean())
    null_dev = 2 * np.sum(
        np.where(y_test > 0, y_test * np.log((y_test + eps) / (null_pred + eps)), 0)
        - (y_test - null_pred)
    )
    pseudo_r2 = 1 - deviance / null_dev if null_dev > 0 else np.nan

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )

    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "pseudo_r2": float(pseudo_r2),
        "deviance": float(deviance),
        "features": feature_cols,
    }

    logger.info(
        f"  XGBoost Poisson: pseudo-R²={pseudo_r2:.3f} | "
        f"test deviance={deviance:,.0f}"
    )
    logger.info(
        f"  Feature importance (top 10):\n" f"{importance.head(10).to_string()}"
    )

    return model, feature_cols, metrics


def apply_collision_model(
    glm_result,
    xgb_model,
    glm_features: list,
    xgb_features: list,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply both collision models to produce risk scores for all link-years.

    Outputs
    -------
    predicted_glm      : Poisson GLM predicted collision rate
    predicted_xgb      : XGBoost predicted collision rate
    residual_glm       : observed - predicted (GLM) — positive = more dangerous
                         than model expects given traffic
    risk_score         : standardised risk percentile (0–100)
    confidence         : 'high' where AADT is measured, 'modelled' elsewhere
    """
    import statsmodels.api as sm

    results = df.copy()

    # GLM predictions
    X_glm = sm.add_constant(
        df[glm_features].fillna(0).astype(float),
        has_constant="add",
    )
    # Ensure constant column exists even if model was fit with it
    if "const" not in X_glm.columns:
        X_glm.insert(0, "const", 1.0)

    results["predicted_glm"] = glm_result.predict(
        X_glm, offset=df["log_offset"].fillna(0)
    )

    # XGBoost predictions
    X_xgb = df[xgb_features].fillna(0).astype(float)
    results["predicted_xgb"] = xgb_model.predict(X_xgb)

    # Residual (observed - expected) — key risk signal
    results["residual_glm"] = results["collision_count"] - results["predicted_glm"]

    # Normalised risk score: predicted_glm as percentile across all links
    results["risk_percentile"] = results["predicted_glm"].rank(pct=True) * 100

    # For links with ≥1 collision: residual percentile (unexplained risk)
    has_collision = results["collision_count"] > 0
    results.loc[has_collision, "excess_risk_percentile"] = (
        results.loc[has_collision, "residual_glm"].rank(pct=True) * 100
    )

    logger.info(
        f"  Risk scores applied to {len(results):,} link-years\n"
        f"  Mean predicted collisions/year: {results['predicted_glm'].mean():.4f}\n"
        f"  Links in top 1% risk: "
        f"{(results['risk_percentile'] >= 99).sum():,}"
    )
    return results


def save_collision_outputs(
    glm_result,
    xgb_model,
    glm_features: list,
    xgb_features: list,
    glm_summary: dict,
    xgb_metrics: dict,
    risk_scores: pd.DataFrame,
) -> None:
    """Save Stage 2 model outputs."""
    import pickle

    MODELS.mkdir(parents=True, exist_ok=True)

    # GLM result
    glm_result.save(str(MODELS / "collision_glm.pkl"))
    logger.info(f"  Saved Poisson GLM to {MODELS / 'collision_glm.pkl'}")

    # XGBoost model
    xgb_model.save_model(str(MODELS / "collision_xgb.json"))
    logger.info(f"  Saved XGBoost model to {MODELS / 'collision_xgb.json'}")

    # Risk scores
    risk_out = MODELS / "risk_scores.parquet"
    risk_scores[
        [
            "link_id",
            "year",
            "collision_count",
            "predicted_glm",
            "predicted_xgb",
            "residual_glm",
            "risk_percentile",
        ]
    ].to_parquet(risk_out, index=False)
    logger.info(f"  Saved risk scores to {risk_out} ({len(risk_scores):,} rows)")

    # Metrics summary
    metrics_path = MODELS / "collision_metrics.json"
    import json

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "glm": glm_summary,
                "xgb": xgb_metrics,
            },
            f,
            indent=2,
        )
    logger.info(f"  Saved metrics to {metrics_path}")


def save_models(
    aadt_model,
    aadt_metrics: dict,
    aadt_features: list,
    aadt_estimates: pd.DataFrame,
    temporal_profiles: pd.DataFrame,
) -> None:
    """Save all model outputs to data/models/."""
    import pickle

    MODELS.mkdir(parents=True, exist_ok=True)

    # AADT model
    with open(MODELS / "aadt_estimator.pkl", "wb") as f:
        pickle.dump(
            {
                "model": aadt_model,
                "features": aadt_features,
                "metrics": aadt_metrics,
            },
            f,
        )
    logger.info(f"  Saved AADT estimator to {MODELS / 'aadt_estimator.pkl'}")

    # AADT estimates for all links
    aadt_estimates.to_parquet(MODELS / "aadt_estimates.parquet", index=False)
    logger.info(f"  Saved AADT estimates ({len(aadt_estimates):,} rows)")

    # Temporal profiles
    temporal_profiles.to_parquet(MODELS / "temporal_profiles.parquet", index=False)
    logger.info(f"  Saved temporal profiles ({len(temporal_profiles):,} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(stage: str = "all") -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    import geopandas as gpd

    if stage in ("traffic", "all"):
        logger.info("=== Stage 1a: AADT estimator ===")

        aadf = pd.read_parquet(AADF_PATH)
        openroads = gpd.read_parquet(OPENROADS_PATH)

        model, metrics, features = train_aadt_estimator(aadf)
        estimates = apply_aadt_estimator(model, features, openroads, aadf)

        print("\n=== AADT estimator results ===")
        print(f"  CV R²  : {metrics['cv_r2_mean']:.3f} (±{metrics['cv_r2_std']:.3f})")
        print(
            f"  CV MAE : {metrics['cv_mae_mean']:.3f} log-units "
            f"≈ ×{np.expm1(metrics['cv_mae_mean']):.2f} multiplicative error"
        )
        print(f"  Features: {features}")
        print(f"\n  Estimated AADT distribution:")
        print(estimates["estimated_aadt"].describe().round(0).to_string())

        X_train, y_train, _ = build_aadt_features(aadf)
        y_pred = model.predict(X_train)
        print(f"\n  Training residuals (log scale):")
        print(f"    MAE  : {mean_absolute_error(y_train, y_pred):.4f}")
        print(f"    RMSE : {mean_squared_error(y_train, y_pred)**0.5:.4f}")
        print(f"    R²   : {r2_score(y_train, y_pred):.4f}")

    if stage in ("temporal", "all"):
        logger.info("=== Stage 1b: Temporal profiles ===")

        profiles = build_temporal_profiles()

        print("\n=== Temporal profiles ===")
        print(f"  Road types: {sorted(profiles['road_prefix'].unique())}")
        print(f"\n  Seasonal index by road type (Jan vs Jul):")
        for road in profiles["road_prefix"].unique():
            rd = profiles[profiles["road_prefix"] == road]
            jan = rd[rd["monthname"] == "Jan"]["seasonal_index"].values
            jul = rd[rd["monthname"] == "Jul"]["seasonal_index"].values
            if len(jan) and len(jul):
                print(
                    f"    {road:6s}: Jan={jan[0]:.3f}, Jul={jul[0]:.3f} "
                    f"({'summer-heavy' if jul[0]>jan[0] else 'winter-heavy'})"
                )

        print(f"\n  Large vehicle % by month (M62):")
        m62 = profiles[profiles["road_prefix"].str.startswith("M62")]
        if not m62.empty:
            print(
                m62[
                    [
                        "monthname",
                        "mean_large_pct",
                        "seasonal_index",
                        "weekday_weekend_ratio",
                    ]
                ].to_string(index=False)
            )

        plot_temporal_profiles(profiles)

    if stage in ("collision", "all"):
        logger.info("=== Stage 2: Collision model ===")

        # Load data
        openroads = gpd.read_parquet(OPENROADS_PATH)
        rla = pd.read_parquet(RLA_PATH)

        # Load AADT estimates (run --stage traffic first)
        aadt_est_path = MODELS / "aadt_estimates.parquet"
        if not aadt_est_path.exists():
            # Fall back to running Stage 1 inline
            logger.warning("aadt_estimates.parquet not found — running Stage 1a first")
            aadf = pd.read_parquet(AADF_PATH)
            m, mets, feats = train_aadt_estimator(aadf)
            estimates = apply_aadt_estimator(m, feats, openroads, aadf)
            MODELS.mkdir(parents=True, exist_ok=True)
            estimates.to_parquet(aadt_est_path, index=False)
        else:
            estimates = pd.read_parquet(aadt_est_path)

        # Load network features if available
        net_path = _ROOT / "data/features/network_features.parquet"
        net_features = pd.read_parquet(net_path) if net_path.exists() else None
        if net_features is None:
            logger.warning("Network features not found — run network_features.py first")

        # Build full dataset (all links × years, including zeros)
        df = build_collision_dataset(
            openroads,
            estimates,
            rla,
            net_features=net_features,
        )

        # Fit GLM
        glm_result, glm_features, glm_summary = train_collision_glm(df)

        # Fit XGBoost
        try:
            xgb_model, xgb_features, xgb_metrics = train_collision_xgb(df)
            has_xgb = True
        except ImportError:
            logger.warning("XGBoost not installed — skipping. pip install xgboost")
            has_xgb = False

        # Apply models and compute risk scores
        if has_xgb:
            risk_scores = apply_collision_model(
                glm_result, xgb_model, glm_features, xgb_features, df
            )
        else:
            # GLM only
            import statsmodels.api as sm

            risk_scores = df.copy()
            X_glm = sm.add_constant(df[glm_features].fillna(0).astype(float))
            risk_scores["predicted_glm"] = glm_result.predict(
                X_glm, offset=df["log_offset"].fillna(0)
            )
            risk_scores["residual_glm"] = (
                risk_scores["collision_count"] - risk_scores["predicted_glm"]
            )
            risk_scores["risk_percentile"] = (
                risk_scores["predicted_glm"].rank(pct=True) * 100
            )

        print("\n=== Collision model results ===")
        print(f"  Poisson GLM pseudo-R²: {glm_summary['pseudo_r2']:.3f}")
        print(f"  Training rows: {glm_summary['n_obs']:,}")
        if has_xgb:
            print(f"  XGBoost pseudo-R²: {xgb_metrics['pseudo_r2']:.3f}")

        print(f"\n  Top 1% highest-risk links:")
        top = risk_scores[risk_scores["risk_percentile"] >= 99].sort_values(
            "predicted_glm", ascending=False
        )
        print(f"    {len(top):,} link-years in top 1%")
        if "road_classification" in top.columns:
            print(f"    Road type breakdown:")
            print(top["road_classification"].value_counts().head(5).to_string())

        print(f"\n  GLM coefficients (significant):")
        coef_df = pd.DataFrame(
            {
                "coef": glm_result.params,
                "pvalue": glm_result.pvalues,
            }
        ).round(4)
        print(
            coef_df[coef_df["pvalue"] < 0.05]
            .sort_values("coef", ascending=False)
            .to_string()
        )

        # Save
        if has_xgb:
            save_collision_outputs(
                glm_result,
                xgb_model,
                glm_features,
                xgb_features,
                glm_summary,
                xgb_metrics,
                risk_scores,
            )
        else:
            MODELS.mkdir(parents=True, exist_ok=True)
            glm_result.save(str(MODELS / "collision_glm.pkl"))
            risk_scores[
                [
                    "link_id",
                    "year",
                    "collision_count",
                    "predicted_glm",
                    "residual_glm",
                    "risk_percentile",
                ]
            ].to_parquet(MODELS / "risk_scores.parquet", index=False)
            logger.info("Saved GLM outputs")

    if stage == "all":
        logger.info("=== All stages complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road risk traffic models")
    parser.add_argument(
        "--stage",
        choices=["traffic", "temporal", "collision", "all"],
        default="all",
        help="Which model stage to run (default: all)",
    )
    args = parser.parse_args()
    main(stage=args.stage)
