"""
features.py
-----------
Feature engineering for the road risk model.

Transforms road_link_annual.parquet (output of join.py) into a model-ready
feature table at road_link × year grain.

Input columns (from road_link_annual.parquet):
  Identifiers : link_id, year
  Collisions  : collision_count, fatal_count, serious_count, slight_count,
                casualty_count, hgv_collision_count, mean_vehicles_per_collision
  AADF traffic: all_motor_vehicles, all_hgvs, hgv_proportion, lgv_proportion,
                cars_proportion, heavy_vehicle_prop, link_length_km, road_type,
                estimation_method, latitude, longitude
  WebTRIS     : mean_daily_flow, large_vehicle_pct, mean_weekday_flow,
                large_vehicle_weekday_pct, site_id
  Road attrs  : road_classification, road_function, form_of_way,
                road_name_clean, is_trunk, is_primary
  Flags       : is_covid, aadf_snap_distance_m, aadf_join_method
  Target      : collision_rate_per_mvkm

Feature engineering steps:
  1. Target variable — collision_rate_per_mvkm + severity breakdown
  2. Traffic features — volume, HGV%, weekday/weekend ratio
  3. Road type encoding — road_classification, form_of_way ordinal + binary flags
  4. Temporal features — year, COVID flag, pre/post-COVID period
  5. Lag features — previous year collision rate per link
  6. Confidence flags — AADF available, WebTRIS available, snap quality

Output: data/features/model_features.parquet
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from road_risk.config import _ROOT, cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_INPUT  = _ROOT / cfg["paths"]["features"] / "road_link_annual.parquet"
_DEFAULT_OUTPUT = _ROOT / cfg["paths"]["features"] / "model_features.parquet"

COVID_YEARS    = {2020, 2021}
PRE_COVID      = {2015, 2016, 2017, 2018, 2019}
POST_COVID     = {2022, 2023, 2024}

# Minimum score threshold for including snapped collisions in rate calculation
# Collisions below this were likely mis-snapped — exclude from rate, keep count
MIN_SNAP_SCORE = 0.0  # set to e.g. 0.7 to be strict; 0.0 keeps all

# Ordinal encoding for road_classification — higher = more major
ROAD_CLASS_ORDINAL = {
    "Motorway":             6,
    "A Road":               5,
    "B Road":               4,
    "Classified Unnumbered":3,
    "Not Classified":       2,
    "Unclassified":         1,
    "Unknown":              0,
}

# Ordinal encoding for form_of_way — higher = more complex/separated
FORM_OF_WAY_ORDINAL = {
    "Motorway":                  5,  # shouldn't appear but handle gracefully
    "Dual Carriageway":          4,
    "Collapsed Dual Carriageway":3,
    "Slip Road":                 2,
    "Roundabout":                2,
    "Single Carriageway":        1,
    "Shared Use Carriageway":    1,
    "Guided Busway":             0,
}


# ---------------------------------------------------------------------------
# Feature engineering functions
# ---------------------------------------------------------------------------

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build and validate target variables.

    Primary target:
        collision_rate_per_mvkm : collisions per million vehicle-km
            = collision_count / (all_motor_vehicles × link_length_km × 365 / 1e6)
            NaN where AADF data is unavailable (no traffic denominator)

    Secondary targets (for multi-output or severity-weighted models):
        fatal_rate_per_mvkm
        serious_rate_per_mvkm
        ksi_rate_per_mvkm     : killed + seriously injured combined

    Notes:
        - collision_rate_per_mvkm is already computed in join.py but may be
          stale or missing; recompute here for consistency
        - Rates > 99th percentile are capped and flagged as outliers
          (likely very short links with 1 collision / tiny denominator)
    """
    df = df.copy()

    # Recompute collision rate from first principles
    has_aadf = df["all_motor_vehicles"].notna() & df["link_length_km"].notna()
    vehicle_km = df["all_motor_vehicles"] * df["link_length_km"] * 365

    df["collision_rate_per_mvkm"] = np.where(
        has_aadf & (vehicle_km > 0),
        df["collision_count"] / (vehicle_km / 1e6),
        np.nan,
    )
    df["fatal_rate_per_mvkm"] = np.where(
        has_aadf & (vehicle_km > 0),
        df["fatal_count"] / (vehicle_km / 1e6),
        np.nan,
    )
    df["serious_rate_per_mvkm"] = np.where(
        has_aadf & (vehicle_km > 0),
        df["serious_count"] / (vehicle_km / 1e6),
        np.nan,
    )
    df["ksi_rate_per_mvkm"] = np.where(
        has_aadf & (vehicle_km > 0),
        (df["fatal_count"] + df["serious_count"]) / (vehicle_km / 1e6),
        np.nan,
    )

    # Cap extreme outliers (very short links inflate rate artificially)
    for col in ["collision_rate_per_mvkm", "fatal_rate_per_mvkm",
                "serious_rate_per_mvkm", "ksi_rate_per_mvkm"]:
        cap = df[col].quantile(0.99)
        n_capped = (df[col] > cap).sum()
        if n_capped:
            df[f"{col}_capped"] = df[col].clip(upper=cap)
            logger.info(f"  {col}: {n_capped} outliers capped at {cap:.4f}")

    # Severity mix (proportion of collisions by severity)
    total = df["collision_count"].replace(0, np.nan)
    df["pct_fatal"]   = df["fatal_count"]   / total
    df["pct_serious"] = df["serious_count"] / total
    df["pct_slight"]  = df["slight_count"]  / total
    df["pct_ksi"]     = (df["fatal_count"] + df["serious_count"]) / total

    n_with_rate = df["collision_rate_per_mvkm"].notna().sum()
    logger.info(
        f"  Target built: {n_with_rate:,} / {len(df):,} rows have collision rate "
        f"({n_with_rate/len(df):.1%} — requires AADF data)"
    )
    return df


def build_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer traffic volume and vehicle mix features.

    Sources:
        AADF — annual average daily flow, HGV/LGV proportions
        WebTRIS — sensor-based mean daily flow, large vehicle %

    Features added:
        log_motor_vehicles       : log(all_motor_vehicles + 1) — normalises skew
        log_daily_flow           : log(mean_daily_flow + 1) — WebTRIS equivalent
        hgv_pct_aadf             : all_hgvs / all_motor_vehicles × 100
        large_vehicle_pct        : WebTRIS large vehicle % (already in source)
        hgv_source               : 'webtris', 'aadf', or 'none' — which HGV source used
        weekday_weekend_ratio    : mean_daily_flow / mean_weekday_flow (>1 = weekend-heavy)
        vehicles_per_km          : all_motor_vehicles / link_length_km — traffic intensity
        aadf_available           : bool — AADF data joined
        webtris_available        : bool — WebTRIS sensor data available

    NaN handling:
        AADF NaN = no count point within 2km (not zero traffic)
        WebTRIS NaN = non-trunk road (not covered by National Highways sensors)
        Both are left as NaN — the model must handle missing traffic features
        (tree-based models handle NaN natively; linear models need imputation)
    """
    df = df.copy()

    # Availability flags
    df["aadf_available"]    = df["all_motor_vehicles"].notna()
    df["webtris_available"] = df["mean_daily_flow"].notna()

    # Log-transform volume (handles skewness, preserves NaN)
    df["log_motor_vehicles"] = np.log1p(df["all_motor_vehicles"])
    df["log_daily_flow"]     = np.log1p(df["mean_daily_flow"])

    # HGV % from AADF (already as proportion 0–1, convert to %)
    df["hgv_pct_aadf"] = df["hgv_proportion"] * 100

    # Preferred HGV source: WebTRIS where available (sensor-based), else AADF
    df["hgv_pct_best"] = np.where(
        df["webtris_available"],
        df["large_vehicle_pct"],    # WebTRIS large vehicle %
        df["hgv_pct_aadf"],         # AADF HGV proportion %
    )
    df["hgv_source"] = np.where(
        df["webtris_available"], "webtris",
        np.where(df["aadf_available"], "aadf", "none")
    )

    # Weekday/weekend ratio (>1 = busier weekends, <1 = busier weekdays)
    df["weekday_weekend_ratio"] = np.where(
        df["mean_weekday_flow"].notna() & (df["mean_weekday_flow"] > 0),
        df["mean_daily_flow"] / df["mean_weekday_flow"],
        np.nan,
    )

    # Traffic intensity per km of road
    df["vehicles_per_km"] = np.where(
        df["aadf_available"] & (df["link_length_km"] > 0),
        df["all_motor_vehicles"] / df["link_length_km"],
        np.nan,
    )
    df["log_vehicles_per_km"] = np.log1p(df["vehicles_per_km"])

    # HGV absolute count (useful alongside proportion)
    df["hgv_daily"] = df["all_hgvs"]
    df["log_hgv_daily"] = np.log1p(df["all_hgvs"])

    n_aadf = df["aadf_available"].sum()
    n_wt   = df["webtris_available"].sum()
    logger.info(
        f"  Traffic features: AADF={n_aadf:,} ({n_aadf/len(df):.1%}), "
        f"WebTRIS={n_wt:,} ({n_wt/len(df):.1%})"
    )
    return df


def build_road_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode road type and geometry features.

    Features added:
        road_class_ordinal   : 0–6 ordinal encoding of road_classification
        form_of_way_ordinal  : 0–5 ordinal encoding of form_of_way
        is_motorway          : bool
        is_a_road            : bool
        is_b_road            : bool
        is_unclassified      : bool
        is_slip_road         : bool
        is_roundabout        : bool
        is_dual_carriageway  : bool (Dual + Collapsed Dual)
        log_link_length_km   : log(link_length_km + 0.001)
    """
    df = df.copy()

    # Ordinal encodings
    df["road_class_ordinal"] = (
        df["road_classification"]
        .map(ROAD_CLASS_ORDINAL)
        .fillna(0)
        .astype(int)
    )
    df["form_of_way_ordinal"] = (
        df["form_of_way"]
        .map(FORM_OF_WAY_ORDINAL)
        .fillna(1)
        .astype(int)
    )

    # Binary road type flags
    df["is_motorway"]        = df["road_classification"] == "Motorway"
    df["is_a_road"]          = df["road_classification"] == "A Road"
    df["is_b_road"]          = df["road_classification"] == "B Road"
    df["is_unclassified"]    = df["road_classification"].isin(
        ["Unclassified", "Not Classified", "Unknown", "Classified Unnumbered"]
    )
    df["is_slip_road"]       = df["form_of_way"] == "Slip Road"
    df["is_roundabout"]      = df["form_of_way"] == "Roundabout"
    df["is_dual_carriageway"] = df["form_of_way"].isin(
        ["Dual Carriageway", "Collapsed Dual Carriageway"]
    )

    # Link length
    df["log_link_length_km"] = np.log(df["link_length_km"].clip(lower=0.001))

    logger.info(
        f"  Road features encoded | "
        f"motorway={df['is_motorway'].sum():,}, "
        f"a_road={df['is_a_road'].sum():,}, "
        f"b_road={df['is_b_road'].sum():,}, "
        f"unclassified={df['is_unclassified'].sum():,}"
    )
    return df


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build temporal features.

    Features added:
        year_norm       : year normalised to 0–1 over the study period
        period          : 'pre_covid', 'covid', 'post_covid'
        is_covid        : already present — kept as-is
        years_from_2019 : signed distance from last pre-COVID year
    """
    df = df.copy()

    year_min = df["year"].min()
    year_max = df["year"].max()
    df["year_norm"] = (df["year"] - year_min) / max(year_max - year_min, 1)

    df["period"] = np.select(
        [df["year"].isin(PRE_COVID), df["year"].isin(COVID_YEARS)],
        ["pre_covid", "covid"],
        default="post_covid",
    )

    df["years_from_2019"] = df["year"] - 2019

    logger.info(
        f"  Temporal features | periods: {df['period'].value_counts().to_dict()}"
    )
    return df


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lag features — previous year's collision rate per road link.

    For a road_link × year model, the previous year's rate is a strong
    predictor (persistence effect). Lagged features require at least 2
    years of data per link.

    Features added:
        collision_rate_lag1      : collision_rate_per_mvkm from previous year
        collision_count_lag1     : collision_count from previous year
        rate_change_1yr          : collision_rate_per_mvkm - lag1 (trend)

    NaN for:
        - First year a link appears (no previous year)
        - Links with only 1 year of data
        - Links where AADF data is missing in either year
    """
    df = df.copy()
    df = df.sort_values(["link_id", "year"])

    # Shift within each link group
    lag = (
        df.groupby("link_id")[["collision_rate_per_mvkm", "collision_count"]]
        .shift(1)
    )
    df["collision_rate_lag1"]  = lag["collision_rate_per_mvkm"]
    df["collision_count_lag1"] = lag["collision_count"]

    # Only valid if previous year is actually year - 1 (not a gap)
    prev_year = df.groupby("link_id")["year"].shift(1)
    not_consecutive = (df["year"] - prev_year) != 1
    df.loc[not_consecutive, "collision_rate_lag1"]  = np.nan
    df.loc[not_consecutive, "collision_count_lag1"] = np.nan

    # Year-on-year rate change
    df["rate_change_1yr"] = df["collision_rate_per_mvkm"] - df["collision_rate_lag1"]

    n_with_lag = df["collision_rate_lag1"].notna().sum()
    logger.info(
        f"  Lag features: {n_with_lag:,} / {len(df):,} rows have lag1 rate "
        f"({n_with_lag/len(df):.1%})"
    )
    return df


def build_confidence_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add data confidence flags useful for filtering in modelling.

    Flags added:
        has_aadf            : AADF traffic data available
        has_webtris         : WebTRIS sensor data available
        has_rate            : collision_rate_per_mvkm is computable
        is_major_road       : motorway or A road
        snap_quality        : 'high' (score>=0.8), 'medium' (>=0.6), 'low' (<0.6)
                              NaN if pct_attribute_snapped is stale (all 0)

    Use case: model on `has_rate == True` rows only; use `is_major_road`
    as a stratification variable.
    """
    df = df.copy()

    df["has_aadf"]    = df["all_motor_vehicles"].notna()
    df["has_webtris"] = df["mean_daily_flow"].notna()
    df["has_rate"]    = df["collision_rate_per_mvkm"].notna()
    df["is_major_road"] = df["road_classification"].isin(["Motorway", "A Road"])

    logger.info(
        f"  Confidence flags: "
        f"has_rate={df['has_rate'].sum():,} ({df['has_rate'].mean():.1%}), "
        f"is_major_road={df['is_major_road'].sum():,} ({df['is_major_road'].mean():.1%})"
    )
    return df


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and order the final feature columns for the model.

    Drops raw source columns that have been superseded by engineered features.
    Keeps identifiers and targets separate from model features.
    """
    # Identifiers — not model inputs
    id_cols = ["link_id", "year", "road_name_clean"]

    # Target variables
    target_cols = [
        "collision_count",
        "collision_rate_per_mvkm",
        "collision_rate_per_mvkm_capped",
        "fatal_rate_per_mvkm",
        "serious_rate_per_mvkm",
        "ksi_rate_per_mvkm",
        "pct_fatal", "pct_serious", "pct_slight", "pct_ksi",
    ]

    # Model features
    feature_cols = [
        # Traffic
        "log_motor_vehicles",
        "log_daily_flow",
        "log_vehicles_per_km",
        "log_hgv_daily",
        "hgv_pct_best",
        "hgv_pct_aadf",
        "large_vehicle_pct",
        "weekday_weekend_ratio",
        # Road type
        "road_class_ordinal",
        "form_of_way_ordinal",
        "is_motorway",
        "is_a_road",
        "is_b_road",
        "is_unclassified",
        "is_slip_road",
        "is_roundabout",
        "is_dual_carriageway",
        "is_trunk",
        "is_primary",
        "log_link_length_km",
        # Temporal
        "year",
        "year_norm",
        "years_from_2019",
        "is_covid",
        # Lag
        "collision_rate_lag1",
        "collision_count_lag1",
        "rate_change_1yr",
        # Confidence / metadata
        "has_aadf",
        "has_webtris",
        "has_rate",
        "is_major_road",
        "aadf_available",
        "webtris_available",
        "hgv_source",
        "period",
        "aadf_snap_distance_m",
    ]

    # Keep only columns that exist
    id_cols      = [c for c in id_cols      if c in df.columns]
    target_cols  = [c for c in target_cols  if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    out = df[id_cols + target_cols + feature_cols].copy()
    logger.info(
        f"  Final feature table: {len(out):,} rows × {out.shape[1]} cols "
        f"({len(feature_cols)} features, {len(target_cols)} targets)"
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_features(
    input_path: str | Path = _DEFAULT_INPUT,
    output_path: str | Path = _DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Loads road_link_annual.parquet, applies all feature engineering steps,
    and saves model_features.parquet.

    Parameters
    ----------
    input_path  : path to road_link_annual.parquet
    output_path : path to save model_features.parquet

    Returns
    -------
    Feature DataFrame
    """
    logger.info(f"Loading road_link_annual from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"  Loaded: {len(df):,} rows × {df.shape[1]} cols")

    logger.info("Building target variables ...")
    df = build_target(df)

    logger.info("Building traffic features ...")
    df = build_traffic_features(df)

    logger.info("Building road type features ...")
    df = build_road_features(df)

    logger.info("Building temporal features ...")
    df = build_temporal_features(df)

    logger.info("Building lag features ...")
    df = build_lag_features(df)

    logger.info("Building confidence flags ...")
    df = build_confidence_flags(df)

    logger.info("Selecting model columns ...")
    df = select_model_columns(df)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info(f"Saved model_features to {out} ({len(df):,} rows)")

    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    df = build_features()

    print("\n=== Feature table summary ===")
    print(f"  Rows     : {len(df):,}")
    print(f"  Columns  : {df.shape[1]}")
    print(f"\n  Target coverage:")
    print(f"    has_rate       : {df['has_rate'].sum():,} ({df['has_rate'].mean():.1%})")
    print(f"    has_aadf       : {df['has_aadf'].sum():,} ({df['has_aadf'].mean():.1%})")
    print(f"    has_webtris    : {df['has_webtris'].sum():,} ({df['has_webtris'].mean():.1%})")
    print(f"    has_lag1       : {df['collision_rate_lag1'].notna().sum():,} "
          f"({df['collision_rate_lag1'].notna().mean():.1%})")
    print(f"\n  Collision rate (where available):")
    rate = df.loc[df['has_rate'], 'collision_rate_per_mvkm']
    print(f"    Median : {rate.median():.4f} per M veh-km")
    print(f"    Mean   : {rate.mean():.4f}")
    print(f"    Std    : {rate.std():.4f}")
    print(f"    p99    : {rate.quantile(0.99):.4f}")
    print(f"\n  Feature missingness (>5% missing):")
    miss = df.isna().mean()
    high_miss = miss[miss > 0.05].sort_values(ascending=False)
    if len(high_miss):
        print(high_miss.map("{:.1%}".format).to_string())
    else:
        print("    None")
    print(f"\n  Period breakdown:")
    print(df['period'].value_counts().to_string())
    print(f"\n  Road classification breakdown:")
    # road_class_ordinal back to name for readability
    ordinal_to_name = {v: k for k, v in ROAD_CLASS_ORDINAL.items()}
    print(df['road_class_ordinal'].map(ordinal_to_name).value_counts().to_string())


if __name__ == "__main__":
    main()