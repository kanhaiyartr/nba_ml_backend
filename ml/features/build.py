"""
Assemble the full feature matrix for model training / inference.

Combines:
  1. Rolling team performance stats (from TeamStatistics.csv via team_stats module)
  2. Contextual features (rest days, streaks, season record)
  3. Elo rating differential -- running competitive strength estimate
  4. Target variable ``home_win``

Chemistry is computed here for the API display layer (services.py) but is
**excluded** from model features because it uses all-time aggregate data
that causes temporal leakage (future game data leaks into past training
samples). Differential features (home - away) are also excluded because
they are perfect linear combinations of the raw home/away columns and
cause multicollinearity that leads to overconfident predictions.
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

    # -- Chemistry features (for API display, NOT for model training) --------
    # Chemistry uses all-time aggregate co-play/shared-minutes data which
    # causes temporal leakage when used as a training feature. It is still
    # computed here so that _aggregate_team_chemistry() can be used by the
    # services layer to populate API responses.
    if player_stats_df is not None and not player_stats_df.empty:
        print("       Computing chemistry features (API display only, excluded from model) …")
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

    # -- Select feature columns ----------------------------------------------
    # Only rolling stats + contextual + elo_diff are used as model features.
    # Differential features (home - away) are excluded: they are exact linear
    # combinations of home/away columns and cause multicollinearity that
    # leads XGBoost to learn spurious correlations and produce overconfident
    # predictions. home_elo and away_elo are excluded for the same reason
    # (elo_diff = home_elo - away_elo); only elo_diff is retained.
    roll_cols = sorted([c for c in df.columns if c.startswith(("home_roll_", "away_roll_"))])
    context_cols = [c for c in [
        "home_streak", "away_streak",
        "home_rest_days", "away_rest_days",
        "home_season_win_pct", "away_season_win_pct",
    ] if c in df.columns]
    elo_cols = [c for c in ["elo_diff"] if c in df.columns]

    feature_cols = roll_cols + context_cols + elo_cols

    core_roll = [c for c in feature_cols if "roll_score" in c or "roll_opp_score" in c or "roll_win" in c]
    df.dropna(subset=core_roll, inplace=True)

    for c in feature_cols:
        df[c] = df[c].fillna(0)

    df.reset_index(drop=True, inplace=True)

    print(f"       Model features: {len(feature_cols)} (excluded chemistry + diffs to prevent leakage/multicollinearity)")

    return df[feature_cols + ["home_win"]], feature_cols
