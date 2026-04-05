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

PROCESSED   = _ROOT / "data/processed"
FEATURES    = _ROOT / "data/features"
MODELS      = _ROOT / "data/models"

AADF_PATH       = PROCESSED / "aadf/aadf_clean.parquet"
WEBTRIS_RAW     = _ROOT / "data/raw/webtris"
WEBTRIS_PATH    = PROCESSED / "webtris/webtris_clean.parquet"
OPENROADS_PATH  = PROCESSED / "shapefiles/openroads_yorkshire.parquet"
SITES_PATH      = _ROOT / "data/raw/webtris/sites.parquet"
RLA_PATH        = FEATURES / "road_link_annual.parquet"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROAD_CLASS_ORDER = [
    "Motorway", "A Road", "B Road",
    "Classified Unnumbered", "Not Classified", "Unclassified", "Unknown",
]

FORM_OF_WAY_ORDER = [
    "Dual Carriageway", "Collapsed Dual Carriageway", "Motorway",
    "Slip Road", "Roundabout", "Single Carriageway",
    "Shared Use Carriageway", "Guided Busway",
]

MONTH_ORDER = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
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
        "road_class_ord", "is_trunk",
        "latitude", "longitude",
        "year_norm", "is_covid",
        "hgv_proportion",
    ]

    # link_length_km has ~21% missingness — impute with median per road type
    if "link_length_km" in df.columns:
        df["link_length_km"] = df.groupby("road_type")["link_length_km"].transform(
            lambda x: x.fillna(x.median() if x.notna().any() else 0)
        )
        feature_cols.append("link_length_km")

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
        model, X, y, groups=groups, cv=cv,
        scoring="r2", n_jobs=-1,
    )
    cv_mae = cross_val_score(
        model, X, y, groups=groups, cv=cv,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )

    # Fit on full dataset
    model.fit(X, y)

    # Permutation importance (HistGBR doesn't expose feature_importances_ directly)
    perm = permutation_importance(model, X, y, n_repeats=5, random_state=RANDOM_STATE)
    importance = pd.Series(
        perm.importances_mean, index=X.columns
    ).sort_values(ascending=False)

    metrics = {
        "cv_r2_mean":  float(cv_scores.mean()),
        "cv_r2_std":   float(cv_scores.std()),
        "cv_mae_mean": float(-cv_mae.mean()),   # back to positive
        "cv_mae_std":  float(cv_mae.std()),
        "n_train":     len(X),
        "n_features":  len(X.columns),
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

    logger.info(f"Applying AADT estimator to {len(openroads):,} OS Open Roads links ...")

    # Build prediction features matching training schema
    or_df = openroads.copy()
    if isinstance(or_df, gpd.GeoDataFrame):
        # Get centroid lat/lon for prediction
        bng = or_df.to_crs("EPSG:27700")
        centroids = bng.geometry.centroid.to_crs("EPSG:4326")
        or_df["latitude"]  = centroids.y
        or_df["longitude"] = centroids.x

    or_df["road_class_ord"] = or_df["road_classification"].isin(
        ["Motorway", "A Road"]
    ).astype(int)
    or_df["is_trunk"] = or_df.get("is_trunk", pd.Series(False, index=or_df.index)).astype(int)

    # Predict for each AADF year
    years = sorted(aadf["year"].unique())
    year_min, year_max = years[0], years[-1]
    frames = []

    for year in years:
        pred_df = or_df.copy()
        pred_df["year"] = year
        pred_df["year_norm"]    = (year - year_min) / max(year_max - year_min, 1)
        pred_df["is_covid"]     = year in {2020, 2021}
        pred_df["hgv_proportion"] = np.nan  # unknown for most links

        # Impute missing features with AADF medians
        aadf_medians = aadf.groupby("road_type")[["hgv_proportion", "link_length_km"]].median()
        road_type_map = {"Motorway": "Major", "A Road": "Major"}
        pred_df["_road_type"] = pred_df["road_classification"].map(road_type_map).fillna("Minor")

        for col in ["hgv_proportion", "link_length_km"]:
            if col in feature_cols:
                for rtype in ["Major", "Minor"]:
                    mask = pred_df["_road_type"] == rtype
                    if rtype in aadf_medians.index:
                        pred_df.loc[mask & pred_df[col].isna(), col] = \
                            aadf_medians.loc[rtype, col]

        X_pred = pred_df[feature_cols].fillna(0)
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

    sites = pd.read_parquet(sites_path)[["site_id", "description", "latitude", "longitude"]]
    sites["road_prefix"] = sites["description"].str[:4].str.strip()

    # Load all raw chunks for Yorkshire sites
    chunks = sorted(raw_folder.glob("site_*_*.parquet"))
    if not chunks:
        raise FileNotFoundError(f"No WebTRIS chunks found in {raw_folder}")

    yorkshire_sites = set(sites[
        sites["latitude"].between(53.3, 54.5) &
        sites["longitude"].between(-2.8, 0.0)
    ]["site_id"])

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
    logger.info(f"  Loaded {len(raw):,} monthly rows from {len(yorkshire_sites):,} sites")

    # Coerce numeric
    for col in ["adt24hour", "awt24hour", "adt24largevehiclepercentage"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Join road prefix
    raw = raw.merge(sites[["site_id", "road_prefix"]], on="site_id", how="left")

    # Standardise month order
    raw["month_num"] = pd.Categorical(
        raw["monthname"], categories=MONTH_ORDER, ordered=True
    ).codes + 1
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
    profile["weekday_weekend_ratio"] = (
        profile["mean_awt24"] / profile["mean_adt24"].replace(0, np.nan)
    )

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
        profiles.groupby("road_prefix")["n_site_months"].sum()
        .nlargest(6).index.tolist()
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, road in zip(axes, top_roads):
        data = profiles[profiles["road_prefix"] == road].sort_values("month_num")
        ax.bar(data["monthname"], data["seasonal_index"],
               color="steelblue", alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(data["monthname"], data["mean_large_pct"],
                 color="crimson", marker="o", linewidth=2, markersize=4)
        ax2.set_ylabel("Large vehicle %", color="crimson")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{road} (n={data['n_site_months'].sum():,})")
        ax.set_ylabel("Seasonal index (1.0 = annual mean)")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle(
        "WebTRIS seasonal traffic profiles — Yorkshire motorways/trunk roads",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    out = MODELS / "temporal_profiles.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved temporal profiles plot to {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

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
        pickle.dump({
            "model": aadt_model,
            "features": aadt_features,
            "metrics": aadt_metrics,
        }, f)
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

        aadf      = pd.read_parquet(AADF_PATH)
        openroads = gpd.read_parquet(OPENROADS_PATH)

        # Train
        model, metrics, features = train_aadt_estimator(aadf)

        # Apply to all links
        estimates = apply_aadt_estimator(model, features, openroads, aadf)

        print("\n=== AADT estimator results ===")
        print(f"  CV R²  : {metrics['cv_r2_mean']:.3f} (±{metrics['cv_r2_std']:.3f})")
        print(f"  CV MAE : {metrics['cv_mae_mean']:.3f} log-units "
              f"≈ {np.expm1(metrics['cv_mae_mean']):.0f} vehicles/day")
        print(f"  Features: {features}")
        print(f"\n  Estimated AADT distribution:")
        print(estimates["estimated_aadt"].describe().round(0).to_string())

        # Sense check: compare estimates vs actual for training data
        X_train, y_train, _ = build_aadt_features(aadf)
        y_pred = model.predict(X_train)
        residuals = y_train - y_pred
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
                print(f"    {road:6s}: Jan={jan[0]:.3f}, Jul={jul[0]:.3f} "
                      f"({'summer-heavy' if jul[0]>jan[0] else 'winter-heavy'})")

        print(f"\n  Large vehicle % by month (M62):")
        m62 = profiles[profiles["road_prefix"].str.startswith("M62")]
        if not m62.empty:
            print(m62[["monthname", "mean_large_pct", "seasonal_index",
                        "weekday_weekend_ratio"]].to_string(index=False))

        plot_temporal_profiles(profiles)

    if stage == "all":
        logger.info("=== Saving all model outputs ===")
        save_models(model, metrics, features, estimates, profiles)
        logger.info("=== Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road risk traffic models")
    parser.add_argument(
        "--stage",
        choices=["traffic", "temporal", "all"],
        default="all",
        help="Which model stage to run (default: all)",
    )
    args = parser.parse_args()
    main(stage=args.stage)
