"""
Load raw NBA CSV files into DataFrames.

Expected files under DATA_DIR (configured in Django settings or passed explicitly):
  - Games.csv         : game-level rows with scores, teams, dates
  - PlayerGameStats.csv : per-player-per-game box-score rows
  - Lineups.csv       : stint/lineup rows with minutes and plus-minus
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _data_dir() -> Path:
    try:
        from django.conf import settings

        return Path(settings.DATA_DIR)
    except Exception:
        return Path(os.environ.get("NBA_DATA_DIR", _DEFAULT_DATA_DIR))


def load_games(path: Path | None = None) -> pd.DataFrame:
    path = path or _data_dir() / "Games.csv"
    return pd.read_csv(path, low_memory=False)


def load_player_game_stats(path: Path | None = None) -> pd.DataFrame:
    path = path or _data_dir() / "PlayerGameStats.csv"
    return pd.read_csv(path, low_memory=False)


def load_lineups(path: Path | None = None) -> pd.DataFrame:
    path = path or _data_dir() / "Lineups.csv"
    return pd.read_csv(path, low_memory=False)
