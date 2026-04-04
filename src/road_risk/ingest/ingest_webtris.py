"""
ingest_webtris.py
-----------------
Loader for National Highways WebTRIS traffic sensor data via pytris.

Install:
  pip install pyTRIS

Usage:
  from road_risk.ingest.ingest_webtris import get_yorkshire_sites, pull_yorkshire

The pytris API object is the entry point for all requests:
  from pytris import API
  api = API()
  sites = api.sites()          # Sites resource
  reports = api.daily_reports()  # Reports resource
  result = reports.get(sites=site_id, start_date='01012023', end_date='31122023')
  df = result.to_frame()

Coverage note:
  WebTRIS covers the National Highways network only — motorways and major
  trunk roads. In Yorkshire this is mainly M1, M62, M18, M621, A1(M), A64(M).
  Non-National Highways roads (most of rural Yorkshire) are not covered.

Vehicle length bands returned by WebTRIS daily reports:
  0 - 520 cm    motorcycles / very short
  521 - 660 cm  cars
  661 - 900 cm  large cars / small vans
  901 - 1160 cm vans / minibuses
  1160 - 1260 cm 2-axle rigid HGVs
  > 1260 cm     3+ axle / articulated HGVs
"""

import logging
import time
from pathlib import Path

import pandas as pd
from pytris import API

from road_risk.config import _ROOT, cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RAW_FOLDER = _ROOT / cfg["paths"]["raw"]["webtris"]
_DEFAULT_OUTPUT_FOLDER = _ROOT / cfg["paths"]["processed"] / "webtris"
_DEFAULT_YEARS = cfg["years"]["full_range"]

# Yorkshire bounding box WGS84
YORKSHIRE_BBOX = {
    "min_lat": 53.30,
    "max_lat": 54.60,
    "min_lon": -2.20,
    "max_lon": -0.08,
}

# Seconds to wait between API calls
_API_DELAY = 0.5

LENGTH_BAND_COLS = [
    "len_0_520_cm",
    "len_521_660_cm",
    "len_661_900_cm",
    "len_901_1160_cm",
    "len_1160_1260_cm",
    "len_gt_1260_cm",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _in_yorkshire_bbox(lat: float, lon: float) -> bool:
    return (
        YORKSHIRE_BBOX["min_lat"] <= lat <= YORKSHIRE_BBOX["max_lat"]
        and YORKSHIRE_BBOX["min_lon"] <= lon <= YORKSHIRE_BBOX["max_lon"]
    )


def _year_date_range(year: int) -> tuple[str, str]:
    """Return (start_date, end_date) strings in DDMMYYYY format for a full year."""
    return f"0101{year}", f"3112{year}"


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and map length band columns to project standard."""
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )

    # pytris returns length bands with names like '0__520_cm', '521__660_cm' etc.
    # Map to our standard names — try multiple possible variants.
    band_map = {
        "0__520_cm": "len_0_520_cm",
        "521__660_cm": "len_521_660_cm",
        "661__900_cm": "len_661_900_cm",
        "901__1160_cm": "len_901_1160_cm",
        "1160__1260_cm": "len_1160_1260_cm",
        "1160_cm": "len_gt_1260_cm",
        "0_520_cm": "len_0_520_cm",
        "521_660_cm": "len_521_660_cm",
        "661_900_cm": "len_661_900_cm",
        "901_1160_cm": "len_901_1160_cm",
        "1160_1260_cm": "len_1160_1260_cm",
        "gt_1260_cm": "len_gt_1260_cm",
    }
    return df.rename(columns={k: v for k, v in band_map.items() if k in df.columns})


def _add_length_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived vehicle length proportion columns:
      hgv_length_proxy  : proportion > 1160 cm (rigid + artic HGVs)
      long_vehicle_prop : proportion > 901 cm  (vans + all HGVs)
    """
    available = [c for c in LENGTH_BAND_COLS if c in df.columns]
    if not available:
        return df

    total = df[available].sum(axis=1).replace(0, pd.NA)
    hgv_bands = [c for c in ["len_1160_1260_cm", "len_gt_1260_cm"] if c in df.columns]
    long_bands = [
        c
        for c in ["len_901_1160_cm", "len_1160_1260_cm", "len_gt_1260_cm"]
        if c in df.columns
    ]

    if hgv_bands:
        df["hgv_length_proxy"] = df[hgv_bands].sum(axis=1) / total
    if long_bands:
        df["long_vehicle_prop"] = df[long_bands].sum(axis=1) / total

    return df


# ---------------------------------------------------------------------------
# Public API — site discovery
# ---------------------------------------------------------------------------


def get_all_sites(cache_folder: Path | None = None) -> pd.DataFrame:
    """
    Fetch all WebTRIS sites and return as a DataFrame.
    Caches to sites.parquet in cache_folder to avoid re-fetching.
    """
    if cache_folder is not None:
        cache_path = Path(cache_folder) / "sites.parquet"
        if cache_path.exists():
            logger.info(f"Loading sites from cache: {cache_path}")
            return pd.read_parquet(cache_path)

    logger.info("Fetching all WebTRIS sites ...")
    api = API()
    sites_resource = api.sites()

    rows = []
    for site in sites_resource.all():
        rows.append(
            {
                "site_id": site.id,
                "name": getattr(site, "name", None),
                "description": getattr(site, "description", None),
                "latitude": getattr(site, "latitude", None),
                "longitude": getattr(site, "longitude", None),
                "status": getattr(site, "status", None),
            }
        )

    df = pd.DataFrame(rows)
    logger.info(f"  Found {len(df):,} total sites")

    if cache_folder is not None:
        Path(cache_folder).mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"  Cached to {cache_path}")

    return df


def get_yorkshire_sites(cache_folder: Path | None = None) -> pd.DataFrame:
    """Return WebTRIS sites within the Yorkshire bounding box."""
    all_sites = get_all_sites(cache_folder=cache_folder)

    has_coords = all_sites["latitude"].notna() & all_sites["longitude"].notna()
    in_bbox = all_sites[has_coords].apply(
        lambda r: _in_yorkshire_bbox(float(r["latitude"]), float(r["longitude"])),
        axis=1,
    )
    yorkshire = all_sites[has_coords][in_bbox].copy()
    logger.info(f"Yorkshire sites: {len(yorkshire)} / {len(all_sites)} total")
    return yorkshire


# ---------------------------------------------------------------------------
# Public API — data pull
# ---------------------------------------------------------------------------


def pull_site_year_annual(
    site_id: int | str,
    year: int,
    api: API | None = None,
) -> pd.DataFrame:
    """
    Pull ANNUAL report for one site and one year.

    One API call per site-year — use this for the main dataset pull.
    Returns one row per site per year with total flows and length bands.

    Returns normalised DataFrame, or empty DataFrame if no data / error.
    """
    if api is None:
        api = API()

    start, end = _year_date_range(year)

    try:
        reports = api.annual_reports()
        result = reports.get(sites=int(site_id), start_date=start, end_date=end)
        df = result.to_frame()
    except Exception as exc:
        logger.warning(f"  Site {site_id} / {year}: failed ({exc})")
        return pd.DataFrame()

    if df is None or df.empty:
        logger.debug(f"  Site {site_id} / {year}: no data returned")
        return pd.DataFrame()

    df = _normalise_columns(df)
    df["site_id"] = site_id
    df["_pull_year"] = year
    return df


def pull_site_year_daily(
    site_id: int | str,
    year: int,
    api: API | None = None,
) -> pd.DataFrame:
    """
    Pull 15-minute interval data aggregated to DAILY totals for one site / year.

    ⚠ SLOW — pytris paginates at 200 rows, so one site-year = ~176 HTTP requests.
    Only use this for a small sample of sites for temporal fingerprint analysis.
    For the main dataset use pull_site_year_annual() instead.

    Aggregates 96 × 15-min intervals to daily totals before returning,
    reducing rows from ~35,000 to ~365 per site-year.

    Returns normalised DataFrame, or empty DataFrame if no data / error.
    """
    if api is None:
        api = API()

    start, end = _year_date_range(year)

    try:
        reports = api.daily_reports()
        result = reports.get(sites=int(site_id), start_date=start, end_date=end)
        df = result.to_frame()
    except Exception as exc:
        logger.warning(f"  Site {site_id} / {year}: failed ({exc})")
        return pd.DataFrame()

    if df is None or df.empty:
        logger.debug(f"  Site {site_id} / {year}: no data returned")
        return pd.DataFrame()

    df = _normalise_columns(df)

    # Aggregate 15-min intervals to daily totals
    # Flow columns are summed; keep date as the group key
    date_col = next((c for c in ["report_date", "date"] if c in df.columns), None)
    if date_col:
        flow_cols = [
            c for c in df.columns if c not in [date_col, "site_id", "_pull_year"]
        ]
        numeric = df[flow_cols].apply(pd.to_numeric, errors="coerce")
        df = numeric.groupby(df[date_col]).sum().reset_index()
        df.rename(columns={date_col: "report_date"}, inplace=True)

    df["site_id"] = site_id
    df["_pull_year"] = year
    return df


def pull_temporal_sample(
    site_ids: list[int | str],
    years: list[int],
    raw_folder: str | Path = _DEFAULT_RAW_FOLDER,
) -> pd.DataFrame:
    """
    Pull daily interval data for a SMALL SAMPLE of sites for temporal
    fingerprint analysis (HGV vs car daily/hourly patterns).

    This is intentionally separate from the main pull — only run on
    a handful of representative sites, not the full Yorkshire network.

    Parameters
    ----------
    site_ids : small list of site IDs to pull (suggest 5-10 max)
    years : years to pull
    raw_folder : where to cache chunk parquets

    Returns
    -------
    Daily-aggregated DataFrame for the sample sites.
    """
    raw_folder = Path(raw_folder)
    api = API()
    frames: list[pd.DataFrame] = []

    logger.info(f"Pulling temporal sample: {len(site_ids)} sites × {len(years)} years")
    logger.warning("This is slow (~88s per site-year) — keep site_ids small")

    for site_id in site_ids:
        for year in years:
            chunk_path = raw_folder / f"daily_{site_id}_{year}.parquet"
            if chunk_path.exists():
                frames.append(pd.read_parquet(chunk_path))
                continue
            df = pull_site_year_daily(site_id, year, api=api)
            if not df.empty:
                df.to_parquet(chunk_path, index=False)
                frames.append(df)
            time.sleep(_API_DELAY)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = _add_length_proportions(combined)
    return combined


def pull_yorkshire(
    years: list[int] | None = None,
    raw_folder: str | Path = _DEFAULT_RAW_FOLDER,
    output_folder: str | Path = _DEFAULT_OUTPUT_FOLDER,
    cache_per_site: bool = True,
) -> pd.DataFrame:
    """
    Pull WebTRIS daily report data for all Yorkshire sites across multiple years.

    Caches per-site parquet files as it goes — safe to interrupt and resume.

    Parameters
    ----------
    years : list of years to pull. Defaults to full_range from config.
    raw_folder : where to cache per-site parquet and sites list.
    output_folder : where to save the combined output parquet.
    cache_per_site : save a parquet per site as each completes.

    Returns
    -------
    Combined DataFrame for all Yorkshire sites × years,
    with vehicle length proportion columns added.
    """
    if years is None:
        years = _DEFAULT_YEARS

    raw_folder = Path(raw_folder)
    output_folder = Path(output_folder)
    raw_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    sites = get_yorkshire_sites(cache_folder=raw_folder)
    if sites.empty:
        raise RuntimeError("No Yorkshire sites found — check API connectivity")

    logger.info(f"Pulling {len(sites)} Yorkshire sites × {len(years)} years")

    api = API()
    all_frames: list[pd.DataFrame] = []

    for i, row in enumerate(sites.itertuples(), 1):
        site_id = row.site_id
        site_cache = raw_folder / f"site_{site_id}.parquet"

        if site_cache.exists():
            logger.info(
                f"[{i}/{len(sites)}] Site {site_id} ({row.description}): from cache"
            )
            all_frames.append(pd.read_parquet(site_cache))
            continue

        logger.info(
            f"[{i}/{len(sites)}] Site {site_id} ({row.description}): pulling ..."
        )
        site_frames: list[pd.DataFrame] = []

        for year in years:
            # Check if this site-year chunk already saved
            chunk_path = raw_folder / f"site_{site_id}_{year}.parquet"
            if chunk_path.exists():
                logger.debug(f"  Site {site_id} / {year}: from chunk cache")
                site_frames.append(pd.read_parquet(chunk_path))
                continue

            df_year = pull_site_year_annual(site_id, year, api=api)
            if not df_year.empty:
                # Save immediately after each API call
                df_year.to_parquet(chunk_path, index=False)
                logger.debug(
                    f"  Site {site_id} / {year}: saved {len(df_year):,} rows → {chunk_path.name}"
                )
                site_frames.append(df_year)
            time.sleep(_API_DELAY)

        if not site_frames:
            logger.warning(f"  No data for site {site_id} — skipping")
            continue

        site_df = pd.concat(site_frames, ignore_index=True)
        site_df = _add_length_proportions(site_df)

        if cache_per_site:
            site_df.to_parquet(site_cache, index=False)

        all_frames.append(site_df)

    if not all_frames:
        raise RuntimeError("No data retrieved for any Yorkshire site")

    combined = pd.concat(all_frames, ignore_index=True)
    return combined


def combine_raw(
    raw_folder: str | Path = _DEFAULT_RAW_FOLDER,
) -> pd.DataFrame:
    """
    Reconstruct the combined DataFrame from per-site-year chunk parquets.

    Useful if the full pull was interrupted — call this to get everything
    collected so far without re-pulling completed chunks.

    Parameters
    ----------
    raw_folder : folder containing site_{id}_{year}.parquet chunk files

    Returns
    -------
    Combined DataFrame with length proportions added.
    """
    raw_folder = Path(raw_folder)
    chunks = sorted(raw_folder.glob("site_*_*.parquet"))

    if not chunks:
        raise FileNotFoundError(f"No chunk parquets found in {raw_folder}")

    logger.info(f"Combining {len(chunks)} chunk files from {raw_folder}")
    frames = [pd.read_parquet(c) for c in chunks]
    combined = pd.concat(frames, ignore_index=True)
    combined = _add_length_proportions(combined)
    logger.info(
        f"  Combined: {len(combined):,} rows, "
        f"{combined['site_id'].nunique()} sites, "
        f"{sorted(combined['_pull_year'].unique())} years"
    )
    return combined


def save_webtris(
    df: pd.DataFrame,
    output_folder: str | Path = _DEFAULT_OUTPUT_FOLDER,
    years: list[int] | None = None,
) -> None:
    """
    Save WebTRIS DataFrame to parquet.

    Parameters
    ----------
    df : DataFrame from pull_yorkshire()
    output_folder : defaults to data/processed/webtris/
    years : used to name the file; inferred from df if not provided

    Example
    -------
    >>> df = pull_yorkshire()
    >>> save_webtris(df)
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if years is None:
        years = sorted(df["_pull_year"].unique()) if "_pull_year" in df.columns else []
    suffix = f"_{min(years)}_{max(years)}" if years else ""
    out_path = output_folder / f"webtris_yorkshire{suffix}.parquet"

    df.to_parquet(out_path, index=False)
    logger.info(f"Saved WebTRIS data to {out_path}")


# ---------------------------------------------------------------------------
# Main / smoke test
# ---------------------------------------------------------------------------


def main(
    years: list[int] | None = None,
    raw_folder: str | Path = None,
    output_folder: str | Path = None,
    sites_only: bool = False,
) -> None:
    if raw_folder is None:
        raw_folder = _DEFAULT_RAW_FOLDER
    if output_folder is None:
        output_folder = _DEFAULT_OUTPUT_FOLDER

    if sites_only:
        sites = get_yorkshire_sites(cache_folder=Path(raw_folder))
        print(f"\n=== Yorkshire WebTRIS sites ({len(sites)}) ===")
        print(
            sites[["site_id", "description", "latitude", "longitude"]].to_string(
                index=False
            )
        )
        return

    df = pull_yorkshire(years=years, raw_folder=raw_folder, output_folder=output_folder)

    print(f"\n=== WebTRIS pull complete ===")
    print(f"  Rows  : {len(df):,}")
    print(f"  Sites : {df['site_id'].nunique()}")
    print(f"  Cols  : {df.columns.tolist()}")
    if "hgv_length_proxy" in df.columns:
        print(f"  Mean HGV proxy : {df['hgv_length_proxy'].mean():.3f}")

    save_webtris(df, output_folder=output_folder, years=years)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    main(sites_only="--sites" in sys.argv)
