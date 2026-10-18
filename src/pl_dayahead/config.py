"""Project-wide settings and paths."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "pl_power.duckdb"

BIDDING_ZONE = "PL"
START_DATE = date(2023, 1, 1)  # first delivery day ingested
SDAC_15MIN_START = date(2025, 10, 1)  # first delivery day with 15-minute day-ahead products


def entsoe_token() -> str:
    """ENTSO-E API token from .env in the project root (never committed)."""
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.environ.get("ENTSOE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("ENTSOE_API_TOKEN is not set: add it to .env in the project root")
    return token