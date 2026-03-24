"""
Rolling team-level performance features for game-outcome prediction.

Features (computed as rolling averages over the last *window* games):
  - points per game (PPG)
  - opponent points per game (OPP_PPG)
  - field goal percentage (FG_PCT)
  - three-point percentage (FG3_PCT)
  - free-throw percentage (FT_PCT)
  - rebounds per game (REB)
  - assists per game (AST)
  - turnovers per game (TOV)
  - win percentage over the window (WIN_PCT)

These are computed separately for the home and away team and prefixed with
``home_`` / ``away_`` before being merged back to the game row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLLING_WINDOW = 10

# Internal short names used throughout the pipeline
BOX_SCORE_AGG_COLS = ["fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov"]

# Mapping from TeamStatistics.csv real column names → internal short names
TEAM_STATS_COL_MAP = {
    "fieldGoalsPercentage": "fg_pct",
    "threePointersPercentage": "fg3_pct",
    "freeThrowsPercentage": "ft_pct",
    "reboundsTotal": "reb",
    "assists": "ast",
    "turnovers": "tov",
}


def enrich_games_with_team_stats(
    games_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join TeamStatistics.csv (2 rows per game, one per team) onto the
    games DataFrame so that each game row gets home/away box-score columns.
    """
    needed = ["gameId", "teamId", "home"] + list(TEAM_STATS_COL_MAP.keys())
    available = [c for c in needed if c in team_stats_df.columns]
    ts = team_stats_df[available].copy()

    ts.rename(columns=TEAM_STATS_COL_MAP, inplace=True)
    short_cols = [c for c in BOX_SCORE_AGG_COLS if c in ts.columns]

    home_ts = ts[ts["home"] == 1].copy()
    away_ts = ts[ts["home"] == 0].copy()

    home_ts = home_ts.rename(
        columns={c: f"home_{c}" for c in short_cols}
    ).rename(columns={"teamId": "hometeamId"})
    home_ts = home_ts[["gameId", "hometeamId"] + [f"home_{c}" for c in short_cols]]

    away_ts = away_ts.rename(
        columns={c: f"away_{c}" for c in short_cols}
    ).rename(columns={"teamId": "awayteamId"})
    away_ts = away_ts[["gameId", "awayteamId"] + [f"away_{c}" for c in short_cols]]

    games_df = games_df.copy()
    games_df["gameId"] = games_df["gameId"].astype(home_ts["gameId"].dtype)
    games_df = games_df.merge(home_ts, on=["gameId", "hometeamId"], how="left")
    games_df = games_df.merge(away_ts, on=["gameId", "awayteamId"], how="left")

    return games_df


def _team_game_log(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode each game into two rows — one per participating team — so we
    can compute per-team rolling stats regardless of home/away status.
    """
    home_extra = {c.replace("home_", ""): c for c in games_df.columns
                  if c.startswith("home_") and c.replace("home_", "") in BOX_SCORE_AGG_COLS}
    away_extra = {c.replace("away_", ""): c for c in games_df.columns
                  if c.startswith("away_") and c.replace("away_", "") in BOX_SCORE_AGG_COLS}

    home_rename = {
        "hometeamId": "teamId",
        "awayteamId": "oppTeamId",
        "homeScore": "score",
        "awayScore": "opp_score",
        **{v: k for k, v in home_extra.items()},
    }
    away_rename = {
        "awayteamId": "teamId",
        "hometeamId": "oppTeamId",
        "awayScore": "score",
        "homeScore": "opp_score",
        **{v: k for k, v in away_extra.items()},
    }

    home_drop = [c for c in games_df.columns if c.startswith("away_") and c in away_extra.values()]
    away_drop = [c for c in games_df.columns if c.startswith("home_") and c in home_extra.values()]

    home = games_df.drop(columns=home_drop, errors="ignore").rename(columns=home_rename).assign(is_home=1)
    away = games_df.drop(columns=away_drop, errors="ignore").rename(columns=away_rename).assign(is_home=0)

    log = pd.concat([home, away], ignore_index=True)
    log["win"] = (log["score"] > log["opp_score"]).astype(int)

    if "gameDateTimeEst" in log.columns:
        log.sort_values(["teamId", "gameDateTimeEst"], inplace=True)
    else:
        log.sort_index(inplace=True)

    return log


def _rolling_features(team_log: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add rolling-average columns to a per-team game log."""
    team_log = team_log.copy()
    team_log.sort_values(["teamId", "gameDateTimeEst"], inplace=True)

    candidates = ["score", "opp_score", "win"] + BOX_SCORE_AGG_COLS
    numeric_feats = [c for c in candidates if c in team_log.columns and team_log[c].notna().any()]

    shifted = (
        team_log
        .groupby("teamId")[numeric_feats]
        .shift(1)
    )

    rolled = (
        shifted
        .groupby(team_log["teamId"])
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolled.columns = [f"roll_{c}" for c in rolled.columns]
    return pd.concat([team_log, rolled], axis=1)


def build_game_features(
    games_df: pd.DataFrame,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Given a cleaned games DataFrame (enriched with team box-score stats),
    return one row per game with home/away rolling stats.
    """
    team_log = _team_game_log(games_df)
    team_log = _rolling_features(team_log, window)

    roll_cols = [c for c in team_log.columns if c.startswith("roll_")]
    id_cols = ["teamId", "gameDateTimeEst", "is_home"]

    home_feats = (
        team_log.loc[team_log["is_home"] == 1, id_cols + roll_cols]
        .rename(columns={c: f"home_{c}" for c in roll_cols})
        .rename(columns={"teamId": "hometeamId"})
    )

    away_feats = (
        team_log.loc[team_log["is_home"] == 0, id_cols + roll_cols]
        .rename(columns={c: f"away_{c}" for c in roll_cols})
        .rename(columns={"teamId": "awayteamId"})
    )

    merged = games_df.merge(
        home_feats.drop(columns=["is_home"]),
        on=["hometeamId", "gameDateTimeEst"],
        how="left",
    ).merge(
        away_feats.drop(columns=["is_home"]),
        on=["awayteamId", "gameDateTimeEst"],
        how="left",
    )

    return merged
