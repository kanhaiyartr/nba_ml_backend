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

BOX_SCORE_AGG_COLS = ["fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov"]


def enrich_games_with_player_stats(
    games_df: pd.DataFrame,
    player_game_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-player box-score rows into team-game averages and merge
    them onto the games DataFrame so rolling features can use shooting, rebound,
    assist, and turnover data.
    """
    available = [c for c in BOX_SCORE_AGG_COLS if c in player_game_df.columns]
    if not available or "teamId" not in player_game_df.columns:
        return games_df

    game_id_col = "gameId"
    if game_id_col not in player_game_df.columns:
        return games_df

    team_agg = (
        player_game_df
        .groupby(["teamId", game_id_col])[available]
        .mean()
        .reset_index()
    )

    # Build a game_id in the games df so we can join.
    # If no explicit gameId, manufacture one from (date, home, away).
    if "gameId" not in games_df.columns:
        # Same hash logic used by the sample data generator
        games_df = games_df.copy()
        games_df["_gameId"] = games_df.apply(
            lambda r: hash((r["gameDateTimeEst"], r["hometeamId"], r["awayteamId"])) % 10**8,
            axis=1,
        )
    else:
        games_df = games_df.copy()
        games_df["_gameId"] = games_df["gameId"]

    home_stats = team_agg.rename(
        columns={c: f"home_{c}" for c in available}
    ).rename(columns={"teamId": "hometeamId", game_id_col: "_gameId"})

    away_stats = team_agg.rename(
        columns={c: f"away_{c}" for c in available}
    ).rename(columns={"teamId": "awayteamId", game_id_col: "_gameId"})

    games_df = games_df.merge(home_stats, on=["hometeamId", "_gameId"], how="left")
    games_df = games_df.merge(away_stats, on=["awayteamId", "_gameId"], how="left")
    games_df.drop(columns=["_gameId"], inplace=True)

    return games_df


def _team_game_log(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode each game into two rows — one per participating team — so we
    can compute per-team rolling stats regardless of home/away status.
    """
    # Collect optional box-score columns that may have been merged in
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

    # Drop the "other side" prefixed cols to avoid column collisions
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
    Given a cleaned games DataFrame (optionally enriched with box-score
    aggregates), return one row per game with home/away rolling stats.
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

    # Only merge on columns that exist in both sides
    merge_home_on = ["hometeamId", "gameDateTimeEst"]
    merge_away_on = ["awayteamId", "gameDateTimeEst"]

    merged = games_df.merge(
        home_feats.drop(columns=["is_home"]),
        on=merge_home_on,
        how="left",
    ).merge(
        away_feats.drop(columns=["is_home"]),
        on=merge_away_on,
        how="left",
    )

    return merged
