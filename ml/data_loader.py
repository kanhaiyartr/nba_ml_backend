"""
Load raw NBA CSV files into DataFrames.

Default paths point at ``data/`` (settings.DATA_DIR) where the raw
NBA CSV files are stored.
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_games(path=None):
    path = path or DATA_DIR / "Games.csv"
    return pd.read_csv(path, low_memory=False)


def load_team_statistics(path=None):
    path = path or DATA_DIR / "TeamStatistics.csv"
    return pd.read_csv(path, low_memory=False)


def load_player_statistics(path=None):
    path = path or DATA_DIR / "PlayerStatistics.csv"
    return pd.read_csv(path, low_memory=False)


def load_players(path=None):
    path = path or DATA_DIR / "Players.csv"
    return pd.read_csv(path, low_memory=False)


def load_team_histories(path=None):
    path = path or DATA_DIR / "TeamHistories.csv"
    return pd.read_csv(path, low_memory=False)
