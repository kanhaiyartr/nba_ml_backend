"""
Rolling team-level performance features for game-outcome prediction.

Features (computed as rolling averages over the last *window* games):
  - points per game (PPG) and opponent PPG
  - point differential (margin of victory)
  - field goal percentage (FG_PCT)
  - three-point percentage (FG3_PCT)
  - free-throw percentage (FT_PCT)
  - total / offensive / defensive rebounds
  - assists, turnovers, steals, blocks
  - points in the paint, fastbreak points, bench points
  - points from turnovers, second-chance points
  - win percentage over the window
  - current win/loss streak
  - rest days since last game
  - season win percentage

These are computed separately for the home and away team and prefixed with
``home_`` / ``away_`` before being merged back to the game row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLLING_WINDOW = 10

BOX_SCORE_AGG_COLS = [
    "fg_pct", "fg3_pct", "ft_pct",
    "reb", "oreb", "dreb",
    "ast", "tov", "stl", "blk",
    "pts_paint", "pts_fb", "pts_bench",
    "pts_from_tov", "pts_2nd_chance",
]

TEAM_STATS_COL_MAP = {
    "fieldGoalsPercentage": "fg_pct",
    "threePointersPercentage": "fg3_pct",
    "freeThrowsPercentage": "ft_pct",
    "reboundsTotal": "reb",
    "reboundsOffensive": "oreb",
    "reboundsDefensive": "dreb",
    "assists": "ast",
    "turnovers": "tov",
    "steals": "stl",
    "blocks": "blk",
    "pointsInThePaint": "pts_paint",
    "pointsFastBreak": "pts_fb",
    "benchPoints": "pts_bench",
    "pointsFromTurnovers": "pts_from_tov",
    "pointsSecondChance": "pts_2nd_chance",
    "seasonWins": "season_wins",
    "seasonLosses": "season_losses",
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
    short_cols = [c for c in BOX_SCORE_AGG_COLS + ["season_wins", "season_losses"]
                  if c in ts.columns]

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
    Explode each game into two rows -- one per participating team -- so we
    can compute per-team rolling stats regardless of home/away status.
    """
    home_extra = {c.replace("home_", ""): c for c in games_df.columns
                  if c.startswith("home_") and c.replace("home_", "") in
                  BOX_SCORE_AGG_COLS + ["season_wins", "season_losses"]}
    away_extra = {c.replace("away_", ""): c for c in games_df.columns
                  if c.startswith("away_") and c.replace("away_", "") in
                  BOX_SCORE_AGG_COLS + ["season_wins", "season_losses"]}

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
    log["margin"] = log["score"] - log["opp_score"]

    if "gameDateTimeEst" in log.columns:
        log.sort_values(["teamId", "gameDateTimeEst"], inplace=True)
    else:
        log.sort_index(inplace=True)

    return log


def _rolling_features(team_log: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add rolling-average columns to a per-team game log."""
    team_log = team_log.copy()
    team_log.sort_values(["teamId", "gameDateTimeEst"], inplace=True)

    candidates = ["score", "opp_score", "win", "margin"] + BOX_SCORE_AGG_COLS
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

    # --- Streak: consecutive wins (+) or losses (-) going into this game ---
    def _compute_streak(group):
        wins = group["win"].values
        streak = np.zeros(len(wins), dtype=float)
        for i in range(1, len(wins)):
            if wins[i - 1] == 1:
                streak[i] = max(streak[i - 1], 0) + 1
            else:
                streak[i] = min(streak[i - 1], 0) - 1
        return pd.Series(streak, index=group.index)

    team_log["streak"] = team_log.groupby("teamId", group_keys=False)[["win"]].apply(
        lambda g: _compute_streak(g)
    ).reset_index(level=0, drop=True)

    # --- Rest days since last game ---
    if "gameDateTimeEst" in team_log.columns:
        team_log["game_dt"] = pd.to_datetime(team_log["gameDateTimeEst"], errors="coerce")
        team_log["rest_days"] = (
            team_log.groupby("teamId")["game_dt"]
            .diff()
            .dt.total_seconds() / 86400.0
        )
        team_log["rest_days"] = team_log["rest_days"].fillna(3.0).clip(0, 10)
        team_log.drop(columns=["game_dt"], inplace=True)
    else:
        team_log["rest_days"] = 3.0

    # --- Season win percentage ---
    if "season_wins" in team_log.columns and "season_losses" in team_log.columns:
        total = team_log["season_wins"] + team_log["season_losses"]
        team_log["season_win_pct"] = np.where(total > 0, team_log["season_wins"] / total, 0.5)
    else:
        team_log["season_win_pct"] = 0.5

    return pd.concat([team_log, rolled], axis=1)


def build_game_features(
    games_df: pd.DataFrame,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Given a cleaned games DataFrame (enriched with team box-score stats),
    return one row per game with home/away rolling stats plus contextual features.
    """
    team_log = _team_game_log(games_df)
    team_log = _rolling_features(team_log, window)

    roll_cols = [c for c in team_log.columns if c.startswith("roll_")]
    extra_cols = ["streak", "rest_days", "season_win_pct"]
    carry_cols = [c for c in extra_cols if c in team_log.columns]
    id_cols = ["teamId", "gameDateTimeEst", "is_home"]

    home_feats = (
        team_log.loc[team_log["is_home"] == 1, id_cols + roll_cols + carry_cols]
        .rename(columns={c: f"home_{c}" for c in roll_cols + carry_cols})
        .rename(columns={"teamId": "hometeamId"})
    )

    away_feats = (
        team_log.loc[team_log["is_home"] == 0, id_cols + roll_cols + carry_cols]
        .rename(columns={c: f"away_{c}" for c in roll_cols + carry_cols})
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
