"""
Assemble the full feature matrix for model training / inference.

Combines:
  1. Rolling team performance stats (team_stats module)
  2. Team-level chemistry aggregates (chemistry module)
  3. Target variable ``home_win``
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .team_stats import build_game_features, enrich_games_with_player_stats
from .chemistry import compute_co_play, compute_shared_minutes, chemistry_score


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
    player_game_df: pd.DataFrame | None = None,
    stint_df: pd.DataFrame | None = None,
    rolling_window: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the model-ready feature matrix.

    Returns
    -------
    (X_with_target, feature_columns)
        DataFrame containing feature columns + ``home_win``;
        and the ordered list of feature column names.
    """
    # Merge per-player box-score stats into team-game aggregates so that
    # the rolling feature builder has fg_pct, reb, ast, tov, etc.
    enriched = games_df
    if player_game_df is not None and not player_game_df.empty:
        enriched = enrich_games_with_player_stats(games_df, player_game_df)

    df = build_game_features(enriched, window=rolling_window)

    # -- Chemistry features (optional) --------------------------------------
    if player_game_df is not None and not player_game_df.empty:
        co_play = compute_co_play(player_game_df)
        if stint_df is not None and not stint_df.empty:
            shared_min = compute_shared_minutes(stint_df)
        else:
            shared_min = co_play[["teamId", "playerA", "playerB"]].copy()
            shared_min["shared_minutes"] = 0.0

        chem = chemistry_score(co_play, shared_min)
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

    # Only keep rows where core rolling features (score, opp_score, win) exist;
    # fill optional box-score rolling cols with 0 if missing.
    core_roll = [c for c in feature_cols if "roll_score" in c or "roll_opp_score" in c or "roll_win" in c]
    df.dropna(subset=core_roll, inplace=True)

    for c in feature_cols:
        df[c] = df[c].fillna(0)

    df.reset_index(drop=True, inplace=True)

    return df[feature_cols + ["home_win"]], feature_cols
