"""
Load raw NBA CSV files into DataFrames.

Structure mirrors the report's Chapter 4 data_loader.py, extended with
loaders for player stats and lineup data needed for chemistry features.
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def load_games(path=None):
    """Load Games.csv — matches the report's data_loader.load_games()."""
    path = path or BASE_DIR / "data" / "Games.csv"
    return pd.read_csv(path, low_memory=False)


def load_player_game_stats(path=None):
    path = path or BASE_DIR / "data" / "PlayerGameStats.csv"
    return pd.read_csv(path, low_memory=False)


def load_lineups(path=None):
    path = path or BASE_DIR / "data" / "Lineups.csv"
    return pd.read_csv(path, low_memory=False)
