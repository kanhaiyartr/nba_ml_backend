"""
Load raw NBA CSV files into DataFrames.

Default paths point at ``artifacts/assets/`` where the real NBA data lives.
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "artifacts" / "assets"


def load_games(path=None):
    path = path or ASSETS_DIR / "Games.csv"
    return pd.read_csv(path, low_memory=False)


def load_team_statistics(path=None):
    path = path or ASSETS_DIR / "TeamStatistics.csv"
    return pd.read_csv(path, low_memory=False)


def load_player_statistics(path=None):
    path = path or ASSETS_DIR / "PlayerStatistics.csv"
    return pd.read_csv(path, low_memory=False)


def load_players(path=None):
    path = path or ASSETS_DIR / "Players.csv"
    return pd.read_csv(path, low_memory=False)


def load_team_histories(path=None):
    path = path or ASSETS_DIR / "TeamHistories.csv"
    return pd.read_csv(path, low_memory=False)
