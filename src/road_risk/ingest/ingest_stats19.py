"""
ingest_stats19.py
-----------------
Loader for DfT STATS19 road casualty statistics.

Handles:
  - Large combined CSV files (spanning 1979–latest year, not split by year)
  - All three tables: collisions, vehicles, casualties
  - Filtering by police_force code (default: Yorkshire region)
  - Year-range filtering via 'year' column in data

Source:
  https://www.gov.uk/government/statistical-data-sets/road-safety-open-data

Expected raw folder layout:
  data/raw/stats19/
      dft-road-casualty-statistics-collision-1979-latest-published-year.csv
      dft-road-casualty-statistics-vehicle-1979-latest-published-year.csv
      dft-road-casualty-statistics-casualty-1979-latest-published-year.csv
"""

import logging
from pathlib import Path

import pandas as pd

from road_risk.config import _ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Yorkshire police force codes in STATS19
YORKSHIRE_FORCE_CODES = {
    4: "West Yorkshire",
    5: "South Yorkshire",
    6: "North Yorkshire",
    7: "Humberside",
}

# Both naming conventions, tried in order
TABLE_NAME_VARIANTS = {
    "collision": ["collision", "accident"],  # new name first
    "vehicle": ["vehicle"],
    "casualty": ["casualty"],
}

# Columns to parse as dates
DATE_COLS = {
    "collision": ["date"],
    "vehicle": [],
    "casualty": [],
}

# Minimal expected columns per table — used for validation
EXPECTED_COLS = {
    "collision": [
        "collision_index",
        "collision_severity",
        "collision_date",
        "police_force",
        "road_type",
        "speed_limit",
        "latitude",
        "longitude",
    ],
    "vehicle": [
        "collision_index",
        "vehicle_type",
        "vehicle_manoeuvre",
        "age_of_driver",
        "age_of_vehicle",
    ],
    "casualty": [
        "collision_index",
        "casualty_type",
        "casualty_severity",
        "age_of_casualty",
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_file(folder: Path, table: str) -> Path | None:
    """
    Find the combined STATS19 file for a given table.
    Looks for any file matching the table name (e.g., '*collision*.csv').
    Returns the Path if found, None otherwise.
    """
    # Try known naming variants
    for variant in TABLE_NAME_VARIANTS[table]:
        patterns = [
            f"*{variant}*.csv",
        ]
        for pattern in patterns:
            matches = sorted(folder.glob(pattern))
            if matches:
                logger.debug(f"  Found '{table}' file: {matches[0].name}")
                return matches[0]

    # Nothing found — log what IS in the folder to help diagnose
    all_csvs = sorted(folder.glob("*.csv"))
    if all_csvs:
        logger.debug(
            f"  No match for table='{table}'. "
            f"Files in folder: {[f.name for f in all_csvs]}"
        )
    else:
        logger.debug(f"  Folder appears empty or has no CSVs: {folder}")
    return None


def _load_single(
    path: Path,
    table: str,
    force_codes: set[int],
    years: list[int] | None = None,
    valid_indices: set | None = None,
) -> pd.DataFrame:
    """
    Load one combined CSV file, apply filters, parse dates, validate columns.

    Parameters
    ----------
    valid_indices : set of collision_index values to keep for vehicle/casualty tables.
                    When provided, filters the table immediately after loading —
                    avoids holding the full GB dataset in memory.
    """
    logger.info(f"  Reading {path.name}")

    df = pd.read_csv(
        path,
        low_memory=False,
        parse_dates=DATE_COLS[table] or False,
        dayfirst=True,  # DfT uses DD/MM/YYYY
    )

    # Normalise column names: lowercase, strip whitespace
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # Filter by year if specified — handle both 'year' and 'collision_year' column names
    year_col = next((c for c in ["year", "collision_year"] if c in df.columns), None)
    if years is not None and year_col:
        before = len(df)
        df = df[df[year_col].isin(years)]
        logger.info(f"    Filtered {before:,} → {len(df):,} rows (year in {years})")

    # Pre-filter vehicle/casualty to Yorkshire collision indices
    # This avoids holding the full GB dataset in memory
    if table in ("vehicle", "casualty") and valid_indices is not None:
        before = len(df)
        df = df[df["collision_index"].isin(valid_indices)]
        logger.info(
            f"    Filtered {before:,} → {len(df):,} rows (collision_index in Yorkshire set)"
        )

    # Filter to region of interest (collision table has police_force)
    if table == "collision" and force_codes:
        before = len(df)
        df = df[df["police_force"].isin(force_codes)]
        logger.info(
            f"    Filtered {before:,} → {len(df):,} rows "
            f"(police_force in {sorted(force_codes)})"
        )

    # Validate expected columns are present
    missing = [c for c in EXPECTED_COLS[table] if c not in df.columns]
    if missing:
        logger.warning(f"    Expected columns missing in {path.name}: {missing}")

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_stats19(
    raw_folder: str | Path,
    years: list[int] | None = None,
    police_force_codes: list[int] | None = None,
    tables: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load STATS19 data from combined CSV files, optionally filtered by
    year and police force codes.

    Parameters
    ----------
    raw_folder : path to data/raw/stats19/
    years : list of years to filter by (e.g. list(range(2015, 2025)))
            Defaults to None (all years in the data). Only applied if 'year' column exists.
    police_force_codes : STATS19 police_force integers to keep.
                         Defaults to Yorkshire region (4, 5, 6, 7).
                         Pass an empty list [] to skip filtering.
    tables : which tables to load — any subset of
             ['collision', 'vehicle', 'casualty'].
             Defaults to all three.

    Returns
    -------
    dict with keys 'collision', 'vehicle', 'casualty' (whichever were requested),
    each containing a DataFrame with filtered data.

    Example
    -------
    >>> data = load_stats19("data/raw/stats19", years=range(2015, 2025))
    >>> collisions = data["collision"]
    >>> vehicles   = data["vehicle"]
    >>> casualties = data["casualty"]
    """
    folder = Path(raw_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Raw folder not found: {folder}")

    if police_force_codes is None:
        police_force_codes = list(YORKSHIRE_FORCE_CODES.keys())
    force_set = set(police_force_codes)

    if tables is None:
        tables = ["collision", "vehicle", "casualty"]

    results: dict[str, pd.DataFrame] = {}

    # Always load collision first so we can pre-filter vehicle/casualty
    # to Yorkshire collision indices — avoids loading full GB data into memory
    load_order = ["collision"] + [t for t in tables if t != "collision"]
    valid_indices: set | None = None

    for table in load_order:
        if table not in tables:
            continue
        logger.info(f"Loading table '{table}'")
        path = _find_file(folder, table)
        if path is None:
            logger.warning(
                f"  No file found for table='{table}' in {folder} — skipping"
            )
            continue
        df = _load_single(
            path, table, force_set, years=years, valid_indices=valid_indices
        )
        results[table] = df
        logger.info(f"'{table}' total rows loaded: {len(df):,}")

        # After loading collision, capture the Yorkshire index set
        if table == "collision" and "collision_index" in df.columns:
            valid_indices = set(df["collision_index"])
            logger.info(
                f"  Captured {len(valid_indices):,} Yorkshire collision indices for pre-filtering"
            )

    return results


def join_stats19(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join all three STATS19 tables on collision_index.

    The result is casualty-level (one row per casualty), with vehicle
    and collision attributes denormalised in.

    Note: vehicle and casualty tables are NOT pre-filtered by police_force;
    they are joined against the already-filtered collision table, so only
    Yorkshire incidents are retained.

    Parameters
    ----------
    data : dict returned by load_stats19()

    Returns
    -------
    DataFrame at casualty granularity with collision and vehicle fields.
    """
    collision = data.get("collision")
    vehicle = data.get("vehicle")
    casualty = data.get("casualty")

    if collision is None:
        raise ValueError("collision table is required for joining")

    # Keep only collision_index values present in the filtered collisions
    valid_indices = set(collision["collision_index"])

    joined = collision.copy()

    if vehicle is not None:
        veh_filtered = vehicle[vehicle["collision_index"].isin(valid_indices)]
        joined = joined.merge(
            veh_filtered,
            on=["collision_index"],
            how="left",
            suffixes=("", "_veh"),
        )
        logger.info(f"After vehicle join: {len(joined):,} rows")

    if casualty is not None:
        cas_filtered = casualty[casualty["collision_index"].isin(valid_indices)]
        joined = joined.merge(
            cas_filtered,
            on=["collision_index"],
            how="left",
            suffixes=("", "_cas"),
        )
        logger.info(f"After casualty join: {len(joined):,} rows")

    return joined


def save_stats19(data: dict[str, pd.DataFrame], output_folder: str | Path) -> None:
    """
    Save loaded STATS19 tables to parquet files in the given folder.

    Parameters
    ----------
    data : dict returned by load_stats19()
    output_folder : path to save parquet files (typically data/processed/stats19)

    Example
    -------
    >>> data = load_stats19("data/raw/stats19")
    >>> save_stats19(data, "data/processed/stats19")
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for table_name, df in data.items():
        output_path = output_folder / f"{table_name}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved '{table_name}' to {output_path}")


# ---------------------------------------------------------------------------
# Quick smoke test — run directly to validate against real files
# ---------------------------------------------------------------------------


def main(
    raw_folder: str | Path = None,
    output_folder: str | Path = None,
    years: list[int] | None = None,
    police_force_codes: list[int] | None = None,
) -> None:
    """
    Load STATS19 data, optionally join tables, and save to parquet.

    Parameters
    ----------
    raw_folder : path to data/raw/stats19; defaults to _ROOT / "data/raw/stats19"
    output_folder : path to save processed data; defaults to _ROOT / "data/processed/stats19"
    years : years to filter by; defaults to 2015–2024
    police_force_codes : police force codes to filter; defaults to Yorkshire
    """
    if raw_folder is None:
        raw_folder = _ROOT / "data/raw/stats19"
    if output_folder is None:
        output_folder = _ROOT / "data/processed/stats19"
    if years is None:
        years = list(range(2015, 2025))

    logger.info(f"Loading from: {raw_folder}")
    data = load_stats19(raw_folder, years=years, police_force_codes=police_force_codes)

    print("\n=== Loaded tables ===")
    for name, df in data.items():
        print(f"  {name:12s}  {len(df):>8,} rows  |  {df.columns.tolist()[:6]} ...")

    if len(data) == 3:
        logger.info("Joining tables on collision_index...")
        joined = join_stats19(data)
        print(
            f"\n=== Joined table ===\n  {len(joined):,} rows  x  {joined.shape[1]} cols"
        )

        # Save individual tables
        logger.info(f"Saving to: {output_folder}")
        save_stats19(data, output_folder)

        # Also save joined table
        joined_path = Path(output_folder) / "joined.parquet"
        joined.to_parquet(joined_path, index=False)
        logger.info(f"Saved joined table to {joined_path}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    raw_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(raw_folder=raw_folder)