"""
road_risk/config.py
-------------------
Loads config/settings.yaml and exposes typed accessors.
Import this everywhere instead of hardcoding paths or constants.

Usage:
    from road_risk.config import cfg, PATHS, YEARS, FORCE_CODES
"""

from pathlib import Path

import yaml

# config.py lives at src/road_risk/config.py
# .parent      → src/road_risk/
# .parent.parent → src/
# .parent.parent.parent → project root (where config/ and data/ live)
_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _ROOT / "config" / "settings.yaml"


def _load() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


cfg: dict = _load()

# --- Convenience aliases ---------------------------------------------------

YEARS       = cfg["years"]
FORCE_CODES = cfg["geography"]["police_force_codes"]
WEBTRIS_URL = cfg["webtris"]["base_url"]

# Resolve paths relative to project root
PATHS = {
    section: {k: _ROOT / v for k, v in vals.items()}
    if isinstance(vals, dict)
    else _ROOT / vals
    for section, vals in cfg["paths"].items()
}


def get_raw_path(source: str) -> Path:
    """Return the raw data folder for a given source name."""
    return PATHS["raw"][source]


def get_processed_path(filename: str) -> Path:
    """Return a path inside data/processed/."""
    return PATHS["processed"] / filename


def get_features_path(filename: str) -> Path:
    """Return a path inside data/features/."""
    return PATHS["features"] / filename