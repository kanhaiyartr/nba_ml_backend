"""
Elo rating system for NBA teams.

Computes a running Elo score for every team based on game results.
The Elo difference between two teams is one of the strongest predictors
of game outcomes in competitive sports.

Parameters
----------
K : update factor (higher = more reactive to recent results)
HOME_ADV : bonus Elo points for the home team in expected-score calc
MEAN_REVERT : fraction of reversion to mean at season boundaries
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INITIAL_ELO = 1500.0
K = 25.0
HOME_ADV = 70.0
MEAN_REVERT = 0.6


def _expected_score(elo_a: float, elo_b: float, home_adv: float = 0.0) -> float:
    """Probability that team A wins given Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b + home_adv) / 400.0))


def compute_elo_ratings(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through games chronologically and compute Elo for each team.

    Adds columns ``home_elo`` and ``away_elo`` (the ratings GOING INTO
    each game, i.e. before the result is applied) plus ``elo_diff``
    (home_elo - away_elo).

    Applies mean-reversion between seasons to handle roster turnover.
    """
    df = games_df.copy()
    df.sort_values("gameDateTimeEst", inplace=True)
    df.reset_index(drop=True, inplace=True)

    elos: dict[int, float] = {}
    home_elo_col = np.zeros(len(df))
    away_elo_col = np.zeros(len(df))

    prev_season = None

    for i, row in df.iterrows():
        home_id = int(row["hometeamId"])
        away_id = int(row["awayteamId"])

        dt = row["gameDateTimeEst"]
        if hasattr(dt, "year"):
            season = dt.year if (hasattr(dt, "month") and dt.month >= 9) else dt.year - 1
        else:
            season = prev_season

        if prev_season is not None and season != prev_season:
            for tid in elos:
                elos[tid] = INITIAL_ELO + MEAN_REVERT * (elos[tid] - INITIAL_ELO)
        prev_season = season

        home_r = elos.get(home_id, INITIAL_ELO)
        away_r = elos.get(away_id, INITIAL_ELO)

        home_elo_col[i] = home_r
        away_elo_col[i] = away_r

        expected_home = _expected_score(home_r, away_r, HOME_ADV)
        actual_home = float(row.get("home_win", int(row["homeScore"] > row["awayScore"])))

        elos[home_id] = home_r + K * (actual_home - expected_home)
        elos[away_id] = away_r + K * ((1 - actual_home) - (1 - expected_home))

    df["home_elo"] = home_elo_col
    df["away_elo"] = away_elo_col
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    return df
