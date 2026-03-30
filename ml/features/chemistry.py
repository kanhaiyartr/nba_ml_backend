"""
Player chemistry proxy metrics (Appendix of the report).

Formulas
--------
CoPlay(i,j)        = games_together(i,j) / total_team_games
SharedMinutes(i,j) = minutes_together(i,j) / total_team_minutes
NetRating          = (pts_scored - pts_conceded) / minutes_played * 100
ChemistryScore     = w1*CoPlay + w2*SharedMinutes + w3*NetRating_norm

All values are normalised to [0, 1] before weighting.

Adaptations for real data
-------------------------
- PlayerStatistics.csv has ``personId`` (not ``playerId``) and no ``teamId``.
  Team membership is resolved by joining on ``gameId`` + ``home`` flag against
  the cleaned Games.csv.
- No separate lineup/stint file exists. Shared minutes are approximated from
  per-player ``numMinutes``: for a pair in the same game on the same team,
  overlap ≈ min(minutes_a, minutes_b).
- Net rating proxy comes from TeamStatistics ``plusMinusPoints`` averaged
  per team.
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


# ── Resolve team membership ────────────────────────────────────────────────

def resolve_player_teams(
    player_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a ``teamId`` column to PlayerStatistics by joining on ``gameId``
    and using the ``home`` flag to pick the correct team.

    Only keeps rows whose ``gameId`` appears in the cleaned games DataFrame
    (i.e. regular-season games), which dramatically reduces the working set.

    Returns a DataFrame with columns [personId, gameId, teamId, numMinutes].
    """
    ps = player_df[["personId", "gameId", "home", "numMinutes"]].copy()
    ps["home"] = pd.to_numeric(ps["home"], errors="coerce")
    ps.dropna(subset=["home"], inplace=True)
    ps["home"] = ps["home"].astype(int)
    ps["numMinutes"] = pd.to_numeric(ps["numMinutes"], errors="coerce").fillna(0)

    # Inner join keeps only regular-season games present in games_df
    gm = games_df[["gameId", "hometeamId", "awayteamId"]].drop_duplicates("gameId")
    ps = ps.merge(gm, on="gameId", how="inner")

    ps["teamId"] = np.where(ps["home"] == 1, ps["hometeamId"], ps["awayteamId"])
    print(f"       Player-team rows after filtering: {len(ps)}")
    return ps[["personId", "gameId", "teamId", "numMinutes"]]


# ── Co-play frequency ──────────────────────────────────────────────────────

def compute_co_play(
    player_team_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute normalised co-play frequency for every player pair per team.

    Expects columns: [personId, gameId, teamId].
    """
    grouped = player_team_df.groupby(["teamId", "gameId"])["personId"].apply(set)

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
        player_team_df.groupby("teamId")["gameId"]
        .nunique()
        .reset_index(name="team_games")
    )
    pair_counts = pair_counts.merge(team_games, on="teamId")
    pair_counts["co_play"] = pair_counts["games_together"] / pair_counts["team_games"]
    return pair_counts[["teamId", "playerA", "playerB", "co_play"]]


# ── Shared minutes (approximated from per-player minutes) ──────────────────

def compute_shared_minutes(
    player_team_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Approximate shared minutes for every teammate pair, then normalise
    by total team minutes.

    For a pair (A, B) in the same game on the same team:
        shared_min_game = min(minutes_A, minutes_B)

    Processes per-team to avoid an explosive full self-join on 1M+ rows.
    """
    pt = player_team_df[["personId", "gameId", "teamId", "numMinutes"]].copy()

    results: list[pd.DataFrame] = []
    for team_id, team_group in pt.groupby("teamId"):
        # Self-join only within this team's games
        m = team_group.merge(team_group, on="gameId", suffixes=("_a", "_b"))
        m = m[m["personId_a"] < m["personId_b"]]
        if m.empty:
            continue
        m["shared_min"] = np.minimum(m["numMinutes_a"], m["numMinutes_b"])
        agg = (
            m.groupby(["personId_a", "personId_b"])["shared_min"]
            .sum()
            .reset_index()
        )
        agg["teamId"] = team_id
        results.append(agg)

    if not results:
        return pd.DataFrame(columns=["teamId", "playerA", "playerB", "shared_minutes"])

    pair_min = pd.concat(results, ignore_index=True)
    pair_min.rename(
        columns={"personId_a": "playerA", "personId_b": "playerB"},
        inplace=True,
    )

    team_min = (
        pt.groupby("teamId")["numMinutes"]
        .sum()
        .reset_index(name="team_minutes")
    )
    pair_min = pair_min.merge(team_min, on="teamId")
    pair_min["shared_minutes"] = np.where(
        pair_min["team_minutes"] > 0,
        pair_min["shared_min"] / pair_min["team_minutes"],
        0,
    )
    return pair_min[["teamId", "playerA", "playerB", "shared_minutes"]]


# ── Net rating proxy from TeamStatistics ────────────────────────────────────

def compute_team_net_rating(
    team_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Average plusMinusPoints per team across all games as a net-rating proxy.
    Returns DataFrame with [teamId, net_rating].
    """
    if "plusMinusPoints" not in team_stats_df.columns:
        return pd.DataFrame(columns=["teamId", "net_rating"])

    agg = (
        team_stats_df.groupby("teamId")["plusMinusPoints"]
        .mean()
        .reset_index(name="net_rating")
    )
    return agg


# ── Normalisation ───────────────────────────────────────────────────────────

def _min_max(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def _group_min_max(series: pd.Series) -> pd.Series:
    """Min-max normalize within a groupby context (per-team)."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


# ── Composite chemistry score ──────────────────────────────────────────────

def chemistry_score(
    co_play: pd.DataFrame,
    shared_minutes: pd.DataFrame,
    team_net_rating: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Combine pair-level metrics into a single ChemistryScore.

    Normalization is done **per-team** for co_play and shared_minutes so
    that the best pair within each team is close to 1.0 rather than being
    compressed by all-time historical outliers from other franchises.
    Net rating stays global since it's already team-level.

    Parameters
    ----------
    co_play : DataFrame with [teamId, playerA, playerB, co_play]
    shared_minutes : DataFrame with [teamId, playerA, playerB, shared_minutes]
    team_net_rating : optional DataFrame with [teamId, net_rating]
    """
    w = _get_weights()
    merge_keys = ["teamId", "playerA", "playerB"]
    df = co_play.merge(shared_minutes, on=merge_keys, how="outer")

    if team_net_rating is not None and not team_net_rating.empty:
        df = df.merge(team_net_rating, on="teamId", how="left")
    else:
        df["net_rating"] = 0.0

    df.fillna(0, inplace=True)

    df["co_play_norm"] = df.groupby("teamId")["co_play"].transform(_group_min_max)
    df["shared_minutes_norm"] = df.groupby("teamId")["shared_minutes"].transform(
        _group_min_max
    )
    df["net_rating_norm"] = _min_max(df["net_rating"])

    df["chemistry_score"] = (
        w["co_play"] * df["co_play_norm"]
        + w["shared_minutes"] * df["shared_minutes_norm"]
        + w["net_rating"] * df["net_rating_norm"]
    )
    return df[merge_keys + ["co_play", "shared_minutes", "net_rating", "chemistry_score"]]
