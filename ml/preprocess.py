"""
Clean raw Games data and produce a game-level DataFrame ready for feature engineering.

Mirrors the preprocessing logic from Chapter 4 of the report:
  - Keep relevant columns
  - Drop rows with missing critical fields
  - Derive binary target ``home_win``
  - Parse and normalise datetime
  - Persist cleaned CSV to ``artifacts/games_clean.csv``
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import load_games

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

REQUIRED_COLS = [
    "gameDateTimeEst",
    "hometeamId",
    "awayteamId",
    "homeScore",
    "awayScore",
    "winner",
    "gameType",
]


def preprocess_games(
    raw_path: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Return a cleaned DataFrame and optionally persist to *out_path*."""
    df = load_games(raw_path)

    available = [c for c in REQUIRED_COLS if c in df.columns]
    df = df[available].copy()

    critical = ["homeScore", "awayScore", "winner", "hometeamId", "awayteamId"]
    df.dropna(subset=[c for c in critical if c in df.columns], inplace=True)

    df["home_win"] = (df["winner"] == df["hometeamId"]).astype(int)

    if "gameDateTimeEst" in df.columns:
        df["gameDateTimeEst"] = pd.to_datetime(
            df["gameDateTimeEst"], errors="coerce", utc=True
        )
        df.dropna(subset=["gameDateTimeEst"], inplace=True)
        df.sort_values("gameDateTimeEst", inplace=True)

    df.reset_index(drop=True, inplace=True)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df


def run_default() -> pd.DataFrame:
    out = _ARTIFACTS_DIR / "games_clean.csv"
    df = preprocess_games(out_path=out)
    print(f"Saved {len(df)} rows → {out}")
    return df


if __name__ == "__main__":
    run_default()
