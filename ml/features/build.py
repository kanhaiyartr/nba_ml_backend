"""
Assemble the full feature matrix for model training / inference.

Combines:
  1. Rolling team performance stats (from TeamStatistics.csv via team_stats module)
  2. Team-level chemistry aggregates (from PlayerStatistics.csv via chemistry module)
  3. Target variable ``home_win``
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .team_stats import build_game_features, enrich_games_with_team_stats
from .chemistry import (
    resolve_player_teams,
    compute_co_play,
    compute_shared_minutes,
    compute_team_net_rating,
    chemistry_score,
)


def _aggregate_team_chemistry(
    chem_df: pd.DataFrame,
    team_col: str = "teamId",
) -> pd.DataFrame:
    """
    Collapse pair-level chemistry scores to a single value per team:
    mean chemistry_score across all active pairs.
    """
    if chem_df.empty:
        return pd.DataFrame(columns=[team_col, "team_chemistry"])
    agg = chem_df.groupby(team_col)["chemistry_score"].mean().reset_index()
    agg.rename(columns={"chemistry_score": "team_chemistry"}, inplace=True)
    return agg


def build_feature_matrix(
    games_df: pd.DataFrame,
    team_stats_df: pd.DataFrame | None = None,
    player_stats_df: pd.DataFrame | None = None,
    rolling_window: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the model-ready feature matrix.

    Parameters
    ----------
    games_df : cleaned games DataFrame (from preprocess)
    team_stats_df : TeamStatistics.csv DataFrame (per-team-per-game box scores)
    player_stats_df : PlayerStatistics.csv DataFrame (per-player-per-game rows)
    rolling_window : number of prior games for rolling averages

    Returns
    -------
    (X_with_target, feature_columns)
    """
    # -- Enrich games with team box-score stats from TeamStatistics ----------
    enriched = games_df
    if team_stats_df is not None and not team_stats_df.empty:
        enriched = enrich_games_with_team_stats(games_df, team_stats_df)

    df = build_game_features(enriched, window=rolling_window)

    # -- Chemistry features --------------------------------------------------
    if player_stats_df is not None and not player_stats_df.empty:
        print("       Computing chemistry features …")
        player_team = resolve_player_teams(player_stats_df, games_df)

        co_play = compute_co_play(player_team)

        shared_min = compute_shared_minutes(player_team)

        team_nr = None
        if team_stats_df is not None and not team_stats_df.empty:
            team_nr = compute_team_net_rating(team_stats_df)

        chem = chemistry_score(co_play, shared_min, team_nr)
        team_chem = _aggregate_team_chemistry(chem)

        df = df.merge(
            team_chem.rename(columns={"teamId": "hometeamId", "team_chemistry": "home_chemistry"}),
            on="hometeamId",
            how="left",
        )
        df = df.merge(
            team_chem.rename(columns={"teamId": "awayteamId", "team_chemistry": "away_chemistry"}),
            on="awayteamId",
            how="left",
        )
        df["home_chemistry"] = df["home_chemistry"].fillna(0)
        df["away_chemistry"] = df["away_chemistry"].fillna(0)
    else:
        df["home_chemistry"] = 0.0
        df["away_chemistry"] = 0.0

    # Select feature columns (rolling stats + chemistry)
    feature_cols = sorted(
        [c for c in df.columns if c.startswith(("home_roll_", "away_roll_"))]
    ) + ["home_chemistry", "away_chemistry"]

    core_roll = [c for c in feature_cols if "roll_score" in c or "roll_opp_score" in c or "roll_win" in c]
    df.dropna(subset=core_roll, inplace=True)

    for c in feature_cols:
        df[c] = df[c].fillna(0)

    df.reset_index(drop=True, inplace=True)

    return df[feature_cols + ["home_win"]], feature_cols
