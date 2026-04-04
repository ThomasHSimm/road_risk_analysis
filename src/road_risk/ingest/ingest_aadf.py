"""
ingest_aadf.py
--------------
Loader for DfT Annual Average Daily Flow (AADF) by direction data.

Source:
  https://roadtraffic.dft.gov.uk/downloads
  File: dft_traffic_counts_aadf_by_direction.zip

Unlike STATS19, AADF is published as a SINGLE file covering all years (2000–2024).
The zip contains one large CSV — we read it once, filter, and cache to parquet.

Schema (from DfT metadata):
  count_point_id, year, region_id, region_name,
  local_authority_id, local_authority_name,
  road_name, road_type,
  start_junction_road_name, end_junction_road_name,
  easting, northing, latitude, longitude,
  link_length_km, link_length_miles,
  estimation_method, estimation_method_detailed,
  direction_of_travel,
  pedal_cycles, two_wheeled_motor_vehicles, cars_and_taxis,
  buses_and_coaches, lgvs,
  hgvs_2_rigid_axle, hgvs_3_rigid_axle, hgvs_4_or_more_rigid_axle,
  hgvs_3_or_4_articulated_axle, hgvs_5_articulated_axle, hgvs_6_articulated_axle,
  all_hgvs, all_motor_vehicles

Key notes:
  - direction_of_travel: N/S/E/W = directional counts; C = combined (no split available)
  - Different directions for the SAME count_point_id CAN be summed to get total flow
  - Different count_point_ids must NOT be summed (different road links)
  - all_hgvs = sum of all 6 HGV sub-type columns
"""

import logging
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from road_risk.config import _ROOT, cfg

logger = logging.getLogger(__name__)

_DEFAULT_YEARS       = cfg["years"]["full_range"]
_DEFAULT_REGION      = "Yorkshire and the Humber"
_DEFAULT_RAW_FOLDER  = _ROOT / cfg["paths"]["raw"]["aadf"]
_DEFAULT_OUTPUT_FOLDER = _ROOT / cfg["paths"]["processed"] / "aadf"

# All HGV sub-type columns (for validation / derived features)
HGV_SUBTYPES = [
    "hgvs_2_rigid_axle",
    "hgvs_3_rigid_axle",
    "hgvs_4_or_more_rigid_axle",
    "hgvs_3_or_4_articulated_axle",
    "hgvs_5_articulated_axle",
    "hgvs_6_articulated_axle",
]

# All motor vehicle breakdown columns (excludes pedal_cycles — not motor vehicles)
VEHICLE_COLS = [
    "two_wheeled_motor_vehicles",
    "cars_and_taxis",
    "buses_and_coaches",
    "lgvs",
    "all_hgvs",
]

# Columns to keep after loading — trim memory before any processing
KEEP_COLS = [
    "count_point_id",
    "year",
    "region_id",
    "region_name",
    "local_authority_id",
    "local_authority_name",
    "road_name",
    "road_type",
    "start_junction_road_name",
    "end_junction_road_name",
    "easting",
    "northing",
    "latitude",
    "longitude",
    "link_length_km",
    "link_length_miles",
    "estimation_method",
    "direction_of_travel",
    "pedal_cycles",
    "two_wheeled_motor_vehicles",
    "cars_and_taxis",
    "buses_and_coaches",
    "lgvs",
    *HGV_SUBTYPES,
    "all_hgvs",
    "all_motor_vehicles",
]

EXPECTED_COLS = [
    "count_point_id", "year", "region_name", "road_name", "road_type",
    "direction_of_travel", "all_hgvs", "all_motor_vehicles",
    "latitude", "longitude",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_csv_in_zip(zip_path: Path) -> str:
    """Return the name of the first CSV inside the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(f"No CSV found inside {zip_path.name}")
    if len(csvs) > 1:
        logger.warning(f"Multiple CSVs in zip — using first: {csvs[0]}")
    return csvs[0]


def _read_zip(zip_path: Path) -> pd.DataFrame:
    """Read the CSV from inside the zip directly, without extracting to disk."""
    csv_name = _find_csv_in_zip(zip_path)
    logger.info(f"Reading '{csv_name}' from {zip_path.name} ...")

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(csv_name) as f:
            raw = f.read()

    df = pd.read_csv(
        BytesIO(raw),
        low_memory=False,
        dtype={"count_point_id": str},   # keep as string for joining
    )
    df.columns = df.columns.str.lower().str.strip()
    logger.info(f"  Loaded {len(df):,} rows × {df.shape[1]} cols (pre-filter)")
    return df


def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"AADF file {path.name} is missing expected columns: {missing}\n"
            f"Actual columns: {df.columns.tolist()}"
        )


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add useful derived columns for feature engineering later:
      - hgv_proportion     : all_hgvs / all_motor_vehicles
      - lgv_proportion     : lgvs / all_motor_vehicles
      - cars_proportion    : cars_and_taxis / all_motor_vehicles
      - heavy_vehicle_prop : (all_hgvs + lgvs) / all_motor_vehicles
    """
    total = df["all_motor_vehicles"].replace(0, pd.NA)  # avoid div/0

    df["hgv_proportion"]      = df["all_hgvs"] / total
    df["lgv_proportion"]      = df["lgvs"] / total
    df["cars_proportion"]     = df["cars_and_taxis"] / total
    df["heavy_vehicle_prop"]  = (df["all_hgvs"] + df["lgvs"]) / total

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_aadf(
    raw_folder: str | Path = _DEFAULT_RAW_FOLDER,
    years: list[int] | None = None,
    region_name: str | None = _DEFAULT_REGION,
    road_types: list[str] | None = None,
    directions: list[str] | None = None,
    cache_parquet: bool = True,
) -> pd.DataFrame:
    """
    Load the DfT AADF by-direction dataset.

    Parameters
    ----------
    raw_folder : path to data/raw/aadf/
                 Defaults to the path defined in config/settings.yaml.
    years : years to keep.
            Defaults to full_range from config/settings.yaml.
    region_name : DfT region to filter on.
                  Defaults to 'Yorkshire and The Humber'.
                  Pass None to load all regions (large — ~4M rows GB-wide).
    road_types : list of road_type values to keep, e.g. ['Major'].
                 Defaults to all road types.
    directions : direction_of_travel values to keep.
                 Options: N, S, E, W (directional) or C (combined).
                 Defaults to all.
    cache_parquet : if True, save filtered result to
                    data/raw/aadf/aadf_filtered.parquet so subsequent
                    loads are fast (avoids re-reading the large zip).

    Returns
    -------
    DataFrame with one row per count_point_id × year × direction,
    plus derived proportion columns.

    Example
    -------
    >>> df = load_aadf("data/raw/aadf", years=list(range(2015, 2025)))
    >>> df.groupby("year")["all_motor_vehicles"].sum()
    """
    folder = Path(raw_folder)
    zip_path = folder / "dft_traffic_counts_aadf_by_direction.zip"
    parquet_path = folder / "aadf_filtered.parquet"

    if years is None:
        years = _DEFAULT_YEARS
    # region_name already defaults to _DEFAULT_REGION via signature

    # --- Fast path: parquet cache -------------------------------------------
    if cache_parquet and parquet_path.exists():
        logger.info(f"Loading from parquet cache: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        # Still apply year / direction filters in case cache is broader
        df = df[df["year"].isin(years)]
        if directions:
            df = df[df["direction_of_travel"].isin(directions)]
        logger.info(f"  {len(df):,} rows after cache filters")
        return df

    # --- Full load from zip --------------------------------------------------
    if not zip_path.exists():
        raise FileNotFoundError(
            f"AADF zip not found: {zip_path}\n"
            f"Download from https://roadtraffic.dft.gov.uk/downloads"
        )

    df = _read_zip(zip_path)
    _validate_columns(df, zip_path)

    # Trim to known columns (future-proof against DfT adding new cols)
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available]

    # --- Filters -------------------------------------------------------------
    before = len(df)

    df = df[df["year"].isin(years)]
    logger.info(f"  Year filter ({years[0]}–{years[-1]}): {before:,} → {len(df):,} rows")

    if region_name:
        before = len(df)
        df = df[df["region_name"] == region_name]
        logger.info(f"  Region filter ('{region_name}'): {before:,} → {len(df):,} rows")

    if road_types:
        before = len(df)
        df = df[df["road_type"].isin(road_types)]
        logger.info(f"  Road type filter {road_types}: {before:,} → {len(df):,} rows")

    if directions:
        before = len(df)
        df = df[df["direction_of_travel"].isin(directions)]
        logger.info(f"  Direction filter {directions}: {before:,} → {len(df):,} rows")

    # --- Derived columns -----------------------------------------------------
    df = _add_derived_columns(df)
    logger.info(f"AADF loaded: {len(df):,} rows × {df.shape[1]} cols")

    # --- Cache ---------------------------------------------------------------
    if cache_parquet:
        df.to_parquet(parquet_path, index=False)
        logger.info(f"  Cached to {parquet_path}")

    return df


def aggregate_bidirectional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse directional rows (N+S or E+W) into a single combined flow
    per count_point_id × year.

    Only rows with direction_of_travel in (N, S, E, W) are summed.
    Rows already marked 'C' (combined) are passed through unchanged.
    This gives a single-row-per-link view for joining to STATS19/MRDB.

    Non-flow columns (road metadata, lat/lon) are taken from the first
    row in each group — they are constant within a count point.

    Returns
    -------
    DataFrame at count_point_id × year grain.
    """
    meta_cols = [
        "count_point_id", "year", "region_id", "region_name",
        "local_authority_id", "local_authority_name",
        "road_name", "road_type",
        "start_junction_road_name", "end_junction_road_name",
        "easting", "northing", "latitude", "longitude",
        "link_length_km", "link_length_miles", "estimation_method",
    ]
    flow_cols = [
        "pedal_cycles", "two_wheeled_motor_vehicles", "cars_and_taxis",
        "buses_and_coaches", "lgvs",
        *HGV_SUBTYPES,
        "all_hgvs", "all_motor_vehicles",
    ]

    directional = df[df["direction_of_travel"].isin(["N", "S", "E", "W"])]
    combined    = df[df["direction_of_travel"] == "C"]

    if len(directional) == 0:
        logger.warning("No directional rows found — returning combined rows only")
        return combined.copy()

    # Take metadata from first row per group (stable within a count point)
    meta = (
        directional
        .sort_values("direction_of_travel")
        .groupby(["count_point_id", "year"])[meta_cols[2:]]
        .first()
        .reset_index()
    )

    # Sum flows across directions
    flows = (
        directional
        .groupby(["count_point_id", "year"])[flow_cols]
        .sum()
        .reset_index()
    )

    result = meta.merge(flows, on=["count_point_id", "year"])
    result["direction_of_travel"] = "combined"
    result = _add_derived_columns(result)

    # Append any 'C' rows for count points that never had directional data
    cp_in_directional = set(result["count_point_id"].unique())
    combined_only = combined[~combined["count_point_id"].isin(cp_in_directional)]

    final = pd.concat([result, combined_only], ignore_index=True)
    logger.info(
        f"aggregate_bidirectional: {len(df):,} directional/combined rows "
        f"→ {len(final):,} count-point × year rows"
    )
    return final


def save_aadf(df: pd.DataFrame, output_folder: str | Path) -> None:
    """
    Save AADF data to parquet file.

    Parameters
    ----------
    df : DataFrame from load_aadf()
    output_folder : path to save parquet file (typically data/processed/aadf)

    Example
    -------
    >>> df = load_aadf("data/raw/aadf")
    >>> save_aadf(df, "data/processed/aadf")
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "aadf_raw.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved AADF to {output_path}")


def save_aadf_bidirectional(df: pd.DataFrame, output_folder: str | Path) -> None:
    """
    Save aggregated bidirectional AADF data to parquet file.

    Parameters
    ----------
    df : DataFrame from aggregate_bidirectional()
    output_folder : path to save parquet file (typically data/processed/aadf)

    Example
    -------
    >>> df = load_aadf("data/raw/aadf")
    >>> agg = aggregate_bidirectional(df)
    >>> save_aadf_bidirectional(agg, "data/processed/aadf")
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "aadf_bidirectional.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved aggregated AADF to {output_path}")


def main(
    raw_folder: str | Path = None,
    output_folder: str | Path = None,
    years: list[int] | None = None,
    region_name: str | None = _DEFAULT_REGION,
) -> None:
    """
    Load AADF, aggregate to bidirectional, and save to parquet.

    Parameters
    ----------
    raw_folder : path to data/raw/aadf; defaults to config
    output_folder : path to save processed data; defaults to data/processed/aadf
    years : years to filter by; defaults to config years
    region_name : region to filter; defaults to 'Yorkshire and the Humber'
    """
    if raw_folder is None:
        raw_folder = _DEFAULT_RAW_FOLDER
    if output_folder is None:
        output_folder = _DEFAULT_OUTPUT_FOLDER
    if years is None:
        years = _DEFAULT_YEARS

    logger.info(f"Loading AADF from: {raw_folder}")
    df = load_aadf(raw_folder, years=years, region_name=region_name)

    print("\n=== AADF summary ===")
    print(f"  Rows:        {len(df):,}")
    print(f"  Count points:{df['count_point_id'].nunique():,}")
    print(f"  Years:       {sorted(df['year'].unique())}")
    print(f"  Road types:  {df['road_type'].value_counts().to_dict()}")
    print(f"  Directions:  {df['direction_of_travel'].value_counts().to_dict()}")
    print(f"\n  HGV proportion (mean): {df['hgv_proportion'].mean():.3f}")

    # Save raw directional data
    logger.info(f"Saving to: {output_folder}")
    save_aadf(df, output_folder)

    # Aggregate and save bidirectional
    agg = aggregate_bidirectional(df)
    print(f"\n=== Bidirectional aggregate ===")
    print(f"  Rows: {len(agg):,}  (count_point_id × year)")
    save_aadf_bidirectional(agg, output_folder)


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