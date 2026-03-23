"""
Player chemistry proxy metrics (Appendix of the report).

Formulas
--------
CoPlay(i,j)        = games_together(i,j) / total_team_games
SharedMinutes(i,j) = minutes_together(i,j) / total_team_minutes
NetRating          = (pts_scored - pts_conceded) / minutes_played * 100
ChemistryScore     = w1*CoPlay + w2*SharedMinutes + w3*NetRating_norm

All values are normalised to [0, 1] before weighting.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _get_weights() -> dict[str, float]:
    try:
        from django.conf import settings
        return settings.CHEMISTRY_WEIGHTS
    except Exception:
        return {"co_play": 0.35, "shared_minutes": 0.35, "net_rating": 0.30}


# ── Pair-level helpers ──────────────────────────────────────────────────────

def compute_co_play(
    player_game_df: pd.DataFrame,
    team_col: str = "teamId",
    game_col: str = "gameId",
    player_col: str = "playerId",
) -> pd.DataFrame:
    """Compute normalised co-play frequency for every player pair per team."""
    grouped = player_game_df.groupby([team_col, game_col])[player_col].apply(set)

    records: list[dict] = []
    for (team, _game), players in grouped.items():
        for a, b in combinations(sorted(players), 2):
            records.append({"teamId": team, "playerA": a, "playerB": b})

    if not records:
        return pd.DataFrame(columns=["teamId", "playerA", "playerB", "co_play"])

    pairs = pd.DataFrame(records)
    pair_counts = (
        pairs.groupby(["teamId", "playerA", "playerB"])
        .size()
        .reset_index(name="games_together")
    )
    team_games = (
        player_game_df.groupby(team_col)[game_col].nunique().reset_index(name="team_games")
    )
    pair_counts = pair_counts.merge(team_games, left_on="teamId", right_on=team_col)
    pair_counts["co_play"] = pair_counts["games_together"] / pair_counts["team_games"]
    return pair_counts[["teamId", "playerA", "playerB", "co_play"]]


def compute_shared_minutes(
    stint_df: pd.DataFrame,
    team_col: str = "teamId",
    player_col: str = "playerId",
    minutes_col: str = "minutes",
    group_col: str = "stintId",
) -> pd.DataFrame:
    """Compute normalised shared minutes from stint/lineup data."""
    grouped = stint_df.groupby([team_col, group_col])

    records: list[dict] = []
    for (team, _stint), grp in grouped:
        players = sorted(grp[player_col].unique())
        stint_min = grp[minutes_col].iloc[0] if minutes_col in grp.columns else 0
        for a, b in combinations(players, 2):
            records.append({
                "teamId": team,
                "playerA": a,
                "playerB": b,
                "shared_min": stint_min,
            })

    if not records:
        return pd.DataFrame(columns=["teamId", "playerA", "playerB", "shared_minutes"])

    df = pd.DataFrame(records)
    pair_min = df.groupby(["teamId", "playerA", "playerB"])["shared_min"].sum().reset_index()
    team_min = stint_df.groupby(team_col)[minutes_col].sum().reset_index(name="team_minutes")
    pair_min = pair_min.merge(team_min, left_on="teamId", right_on=team_col)
    pair_min["shared_minutes"] = np.where(
        pair_min["team_minutes"] > 0,
        pair_min["shared_min"] / pair_min["team_minutes"],
        0,
    )
    return pair_min[["teamId", "playerA", "playerB", "shared_minutes"]]


def compute_lineup_net_rating(
    stint_df: pd.DataFrame,
    pts_scored_col: str = "ptsScored",
    pts_conceded_col: str = "ptsConceded",
    minutes_col: str = "minutes",
    group_col: str = "stintId",
    team_col: str = "teamId",
) -> pd.DataFrame:
    """Compute net rating per lineup stint, then average per team."""
    stints = stint_df.groupby([team_col, group_col]).agg(
        pts_scored=(pts_scored_col, "first"),
        pts_conceded=(pts_conceded_col, "first"),
        minutes=(minutes_col, "first"),
    ).reset_index()

    stints["net_rating"] = np.where(
        stints["minutes"] > 0,
        (stints["pts_scored"] - stints["pts_conceded"]) / stints["minutes"] * 100,
        0,
    )
    return stints[[team_col, group_col, "net_rating"]]


# ── Normalisation ───────────────────────────────────────────────────────────

def _min_max(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


# ── Composite chemistry score ──────────────────────────────────────────────

def chemistry_score(
    co_play: pd.DataFrame,
    shared_minutes: pd.DataFrame,
    net_rating_by_pair: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Combine pair-level metrics into a single ChemistryScore.

    Parameters
    ----------
    co_play : DataFrame with [teamId, playerA, playerB, co_play]
    shared_minutes : DataFrame with [teamId, playerA, playerB, shared_minutes]
    net_rating_by_pair : optional DataFrame with [teamId, playerA, playerB, net_rating]

    Returns
    -------
    DataFrame with columns [teamId, playerA, playerB, chemistry_score]
    """
    w = _get_weights()
    merge_keys = ["teamId", "playerA", "playerB"]
    df = co_play.merge(shared_minutes, on=merge_keys, how="outer")

    if net_rating_by_pair is not None and not net_rating_by_pair.empty:
        df = df.merge(net_rating_by_pair, on=merge_keys, how="left")
    else:
        df["net_rating"] = 0.0

    df.fillna(0, inplace=True)

    df["co_play_norm"] = _min_max(df["co_play"])
    df["shared_minutes_norm"] = _min_max(df["shared_minutes"])
    df["net_rating_norm"] = _min_max(df["net_rating"])

    df["chemistry_score"] = (
        w["co_play"] * df["co_play_norm"]
        + w["shared_minutes"] * df["shared_minutes_norm"]
        + w["net_rating"] * df["net_rating_norm"]
    )
    return df[merge_keys + ["co_play", "shared_minutes", "net_rating", "chemistry_score"]]
