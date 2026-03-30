"""
Assemble the full feature matrix for model training / inference.

Combines:
  1. Rolling team performance stats (from TeamStatistics.csv via team_stats module)
  2. Team-level chemistry aggregates (from PlayerStatistics.csv via chemistry module)
  3. Contextual features (rest days, streaks, season record)
  4. Differential features (home minus away for key metrics)
  5. Elo ratings -- running competitive strength estimate
  6. Target variable ``home_win``
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
from .elo import compute_elo_ratings

DIFF_PAIRS = [
    ("home_roll_score", "away_roll_score", "diff_ppg"),
    ("home_roll_opp_score", "away_roll_opp_score", "diff_opp_ppg"),
    ("home_roll_margin", "away_roll_margin", "diff_margin"),
    ("home_roll_win", "away_roll_win", "diff_win_pct"),
    ("home_roll_fg_pct", "away_roll_fg_pct", "diff_fg_pct"),
    ("home_roll_fg3_pct", "away_roll_fg3_pct", "diff_fg3_pct"),
    ("home_roll_reb", "away_roll_reb", "diff_reb"),
    ("home_roll_ast", "away_roll_ast", "diff_ast"),
    ("home_roll_tov", "away_roll_tov", "diff_tov"),
    ("home_roll_stl", "away_roll_stl", "diff_stl"),
    ("home_roll_blk", "away_roll_blk", "diff_blk"),
    ("home_season_win_pct", "away_season_win_pct", "diff_season_win_pct"),
    ("home_chemistry", "away_chemistry", "diff_chemistry"),
]


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

    Returns
    -------
    (X_with_target, feature_columns)
    """
    # -- Elo ratings (must be computed on full history before filtering) ------
    games_with_elo = compute_elo_ratings(games_df)

    # -- Filter to modern era (2010+) for training ----------------------------
    if "gameDateTimeEst" in games_with_elo.columns:
        cutoff = pd.Timestamp("2010-01-01", tz="UTC")
        dt_col = pd.to_datetime(games_with_elo["gameDateTimeEst"], errors="coerce", utc=True)
        before = len(games_with_elo)
        games_with_elo = games_with_elo[dt_col >= cutoff].copy()
        games_with_elo.reset_index(drop=True, inplace=True)
        print(f"       Filtered to modern era (2010+): {before} → {len(games_with_elo)} games")

    # -- Enrich games with team box-score stats from TeamStatistics ----------
    enriched = games_with_elo
    if team_stats_df is not None and not team_stats_df.empty:
        enriched = enrich_games_with_team_stats(games_with_elo, team_stats_df)

    df = build_game_features(enriched, window=rolling_window)

    # -- Chemistry features --------------------------------------------------
    if player_stats_df is not None and not player_stats_df.empty:
        print("       Computing chemistry features …")
        recent_games = games_df
        if "gameDateTimeEst" in games_df.columns:
            dt = pd.to_datetime(games_df["gameDateTimeEst"], errors="coerce", utc=True)
            recent_games = games_df[dt >= pd.Timestamp("2015-01-01", tz="UTC")]
            print(f"       Chemistry data: {len(games_df)} → {len(recent_games)} games (2015+)")
        player_team = resolve_player_teams(player_stats_df, recent_games)

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

    # -- Differential features (home - away) ---------------------------------
    for home_col, away_col, diff_name in DIFF_PAIRS:
        if home_col in df.columns and away_col in df.columns:
            df[diff_name] = df[home_col].fillna(0) - df[away_col].fillna(0)

    # -- Carry Elo columns through (they came from games_with_elo) -----------
    elo_cols = [c for c in ["home_elo", "away_elo", "elo_diff"] if c in df.columns]

    # -- Select feature columns ----------------------------------------------
    roll_cols = sorted([c for c in df.columns if c.startswith(("home_roll_", "away_roll_"))])
    context_cols = [c for c in [
        "home_streak", "away_streak",
        "home_rest_days", "away_rest_days",
        "home_season_win_pct", "away_season_win_pct",
    ] if c in df.columns]
    chem_cols = ["home_chemistry", "away_chemistry"]
    diff_cols = [name for _, _, name in DIFF_PAIRS if name in df.columns]

    feature_cols = roll_cols + context_cols + chem_cols + elo_cols + diff_cols

    core_roll = [c for c in feature_cols if "roll_score" in c or "roll_opp_score" in c or "roll_win" in c]
    df.dropna(subset=core_roll, inplace=True)

    for c in feature_cols:
        df[c] = df[c].fillna(0)

    df.reset_index(drop=True, inplace=True)

    return df[feature_cols + ["home_win"]], feature_cols
