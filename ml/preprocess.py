"""
Clean raw Games data and produce a game-level DataFrame ready for
feature engineering.

Follows the preprocessing logic from Chapter 4 of the report:
  - Select the required columns (including gameId for joins)
  - Filter to Regular Season games only
  - Drop rows with missing critical fields
  - Derive binary target ``home_win``
  - Parse and normalise datetime
  - Persist cleaned CSV to ``artifacts/games_clean.csv``
"""
import pandas as pd
from pathlib import Path

from .data_loader import load_games

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "Games.csv"
OUT_FILE = BASE_DIR / "artifacts" / "games_clean.csv"


def main(raw_path=None):
    df = load_games(raw_path or RAW_FILE)

    df = df[
        [
            "gameId",
            "gameDateTimeEst",
            "hometeamId",
            "awayteamId",
            "homeScore",
            "awayScore",
            "winner",
            "gameType",
        ]
    ]

    # Filter to Regular Season only (exclude preseason, playoffs, all-star)
    df = df[df["gameType"] == "Regular Season"].copy()

    df = df.dropna(
        subset=[
            "homeScore",
            "awayScore",
            "winner",
            "hometeamId",
            "awayteamId",
            "gameDateTimeEst",
        ]
    )

    # Target: 1 if home team won, else 0
    df["home_win"] = (df["winner"] == df["hometeamId"]).astype(int)

    # Parse datetime safely and unify timezone
    df["gameDateTimeEst"] = pd.to_datetime(
        df["gameDateTimeEst"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["gameDateTimeEst"])
    df = df.sort_values("gameDateTimeEst").reset_index(drop=True)

    OUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print("Saved cleaned dataset to:", OUT_FILE)
    print("Rows:", len(df))
    if len(df) > 0:
        print("Date range:", df["gameDateTimeEst"].iloc[0], "→", df["gameDateTimeEst"].iloc[-1])

    return df


if __name__ == "__main__":
    main()
