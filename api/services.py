"""
Singleton service that loads trained model artifacts and precomputed
team feature lookups. Provides prediction + explanation helpers consumed
by DRF views.

On first access the service:
  1. Loads the trained RF / XGBoost models from joblib.
  2. Builds a per-team rolling-stats lookup from historical data (the same
     feature engineering used at training time) so that the prediction
     endpoint can automatically compute features from just team IDs.
  3. Loads precomputed team-level chemistry scores.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from django.conf import settings

_lock = threading.Lock()
_state: dict = {}


class ModelNotReady(Exception):
    pass


def _artifacts_dir() -> Path:
    return Path(settings.MODEL_ARTIFACTS_DIR)


def _ensure_loaded() -> dict:
    """Load models + metadata + team lookups once (thread-safe)."""
    if _state:
        return _state

    with _lock:
        if _state:
            return _state

        adir = _artifacts_dir()
        fc_path = adir / "feature_columns.json"
        rf_path = adir / "model_rf.joblib"
        xgb_path = adir / "model_xgb.joblib"
        metrics_path = adir / "metrics.json"

        if not fc_path.exists():
            raise ModelNotReady(
                "Model artifacts not found. Run `python manage.py train_models` first."
            )

        with open(fc_path) as f:
            _state["feature_columns"] = json.load(f)

        if rf_path.exists():
            _state["rf"] = joblib.load(rf_path)
        if xgb_path.exists():
            _state["xgb"] = joblib.load(xgb_path)

        stack_path = adir / "model_stack.joblib"
        if stack_path.exists():
            _state["stack"] = joblib.load(stack_path)

        if metrics_path.exists():
            with open(metrics_path) as f:
                _state["metrics"] = json.load(f)
        else:
            _state["metrics"] = {}

        # Build the team-level lookup tables from historical data
        _build_team_lookups()

    return _state


def _build_team_lookups():
    """
    Precompute the latest rolling stats, contextual features, and Elo
    for every team. Stored in _state so prediction requests are instant.
    """
    from ml.preprocess import main as preprocess_main
    from ml.data_loader import load_team_statistics
    from ml.features.team_stats import (
        enrich_games_with_team_stats,
        _team_game_log,
        _rolling_features,
    )
    from ml.features.elo import compute_elo_ratings

    print("[services] Building team feature lookups from historical data …")

    games_df = preprocess_main()
    team_stats_df = load_team_statistics()

    # Compute Elo on full history
    games_with_elo = compute_elo_ratings(games_df)
    elo_latest = (
        games_with_elo.sort_values("gameDateTimeEst")
        .drop_duplicates("hometeamId", keep="last")[["hometeamId", "home_elo"]]
    )
    elo_away_latest = (
        games_with_elo.sort_values("gameDateTimeEst")
        .drop_duplicates("awayteamId", keep="last")[["awayteamId", "away_elo"]]
    )
    team_elo: dict[int, float] = {}
    for _, r in elo_latest.iterrows():
        team_elo[int(r["hometeamId"])] = float(r["home_elo"])
    for _, r in elo_away_latest.iterrows():
        tid = int(r["awayteamId"])
        if tid not in team_elo:
            team_elo[tid] = float(r["away_elo"])
    _state["team_elo"] = team_elo
    print(f"[services] Elo ratings loaded for {len(team_elo)} teams")

    enriched = enrich_games_with_team_stats(games_df, team_stats_df)

    team_log = _team_game_log(enriched)
    team_log = _rolling_features(team_log, window=10)

    roll_cols = [c for c in team_log.columns if c.startswith("roll_")]
    context_cols = [c for c in ["streak", "rest_days", "season_win_pct"]
                    if c in team_log.columns]

    carry_cols = roll_cols + context_cols

    team_log_sorted = team_log.sort_values(["teamId", "gameDateTimeEst"])
    latest = team_log_sorted.groupby("teamId")[carry_cols].last()
    _state["team_rolling"] = latest.to_dict(orient="index")

    # Precompute team chemistry scores
    try:
        from ml.data_loader import load_player_statistics
        from ml.features.chemistry import (
            resolve_player_teams,
            compute_co_play,
            compute_shared_minutes,
            compute_team_net_rating,
            chemistry_score,
        )
        from ml.features.build import _aggregate_team_chemistry

        player_stats_df = load_player_statistics()
        recent_games = games_df
        if "gameDateTimeEst" in games_df.columns:
            dt = pd.to_datetime(games_df["gameDateTimeEst"], errors="coerce", utc=True)
            recent_games = games_df[dt >= pd.Timestamp("2015-01-01", tz="UTC")]
            print(f"[services] Chemistry data: {len(games_df)} → {len(recent_games)} games (2015+)")
        player_team = resolve_player_teams(player_stats_df, recent_games)
        co_play = compute_co_play(player_team)
        shared_min = compute_shared_minutes(player_team)
        team_nr = compute_team_net_rating(team_stats_df)
        chem = chemistry_score(co_play, shared_min, team_nr)

        # Keep the full pair-level data for the /chemistry/pair/ endpoint
        _state["chemistry_pairs"] = chem

        team_chem = _aggregate_team_chemistry(chem)
        _state["team_chemistry"] = dict(
            zip(team_chem["teamId"], team_chem["team_chemistry"])
        )
        print(
            f"[services] Chemistry loaded: "
            f"{len(chem)} pairs across {len(_state['team_chemistry'])} teams"
        )
    except Exception as exc:
        print(f"[services] Chemistry not available: {exc}")
        _state["team_chemistry"] = {}
        _state["chemistry_pairs"] = pd.DataFrame()

    print(f"[services] Rolling stats loaded for {len(_state['team_rolling'])} teams")

    # Preload teams and roster data for the /teams/ and /players/ endpoints
    _build_roster_lookups()


def _build_roster_lookups():
    """
    Cache active NBA team list and current-season player rosters with
    aggregated stats so the /api/teams/ and /api/teams/<id>/players/
    endpoints are instant.
    """
    from ml.data_loader import load_team_histories, load_player_statistics

    print("[services] Building roster lookups …")

    # --- Active teams (only standard franchise IDs starting with 1610612) ----
    th = load_team_histories()
    active = th[
        (th["league"] == "NBA")
        & (th["seasonActiveTill"] >= 2025)
        & (th["teamId"].astype(str).str.startswith("1610612"))
    ]
    latest = (
        active.sort_values("seasonActiveTill", ascending=False)
        .drop_duplicates("teamId", keep="first")
    )
    teams_list = []
    for _, r in latest.iterrows():
        teams_list.append({
            "team_id": int(r["teamId"]),
            "city": str(r["teamCity"]).strip(),
            "name": str(r["teamName"]).strip(),
            "abbreviation": str(r["teamAbbrev"]).strip(),
        })
    teams_list.sort(key=lambda t: t["name"])
    _state["teams"] = teams_list
    team_name_map = {t["team_id"]: t for t in teams_list}
    _state["team_name_map"] = team_name_map
    print(f"[services] {len(teams_list)} active NBA teams cached")

    # --- Current-season player rosters with aggregated stats ----------------
    ps = load_player_statistics()
    ps["gameDateTimeEst"] = pd.to_datetime(ps["gameDateTimeEst"], errors="coerce")
    current = ps[
        (ps["gameDateTimeEst"] >= "2025-10-01")
        & (ps["gameType"] == "Regular Season")
    ].copy()

    games_df = _state.get("_games_df")
    if games_df is None:
        from ml.data_loader import load_games
        games_df = load_games()

    gm = games_df[["gameId", "hometeamId", "awayteamId"]].drop_duplicates("gameId")
    current["home"] = pd.to_numeric(current["home"], errors="coerce").fillna(0).astype(int)
    current = current.merge(gm, on="gameId", how="left")
    import numpy as _np
    current["teamId"] = _np.where(
        current["home"] == 1, current["hometeamId"], current["awayteamId"]
    )

    # For traded players keep only their most recent team
    latest_team = (
        current.sort_values("gameDateTimeEst")
        .drop_duplicates("personId", keep="last")[["personId", "teamId"]]
    )
    current = current.merge(latest_team.rename(columns={"teamId": "currentTeam"}), on="personId")
    current = current[current["teamId"] == current["currentTeam"]]

    for col in ["points", "assists", "reboundsTotal", "steals", "blocks"]:
        current[col] = pd.to_numeric(current[col], errors="coerce").fillna(0)

    agg = (
        current.groupby(["personId", "teamId"])
        .agg(
            first_name=("firstName", "first"),
            last_name=("lastName", "first"),
            ppg=("points", "mean"),
            apg=("assists", "mean"),
            rpg=("reboundsTotal", "mean"),
            spg=("steals", "mean"),
            bpg=("blocks", "mean"),
            games_played=("gameId", "nunique"),
        )
        .reset_index()
    )

    # Derive position from stat profile (Players.csv flags are unreliable for modern players)
    def _infer_position(row):
        apg, rpg = row["apg"], row["rpg"]
        if rpg >= 8:
            return "C" if apg < 4 else "PF"
        if rpg >= 6:
            return "PF" if apg < 4 else "SF"
        if apg >= 6:
            return "PG"
        if apg >= 4:
            return "SG" if rpg < 4 else "SF"
        return "SG" if rpg < 4 else "SF"

    agg["position"] = agg.apply(_infer_position, axis=1)

    roster_by_team: dict[int, list[dict]] = {}
    for _, r in agg.iterrows():
        tid = int(r["teamId"])
        roster_by_team.setdefault(tid, []).append({
            "person_id": int(r["personId"]),
            "first_name": str(r["first_name"]),
            "last_name": str(r["last_name"]),
            "position": r["position"],
            "ppg": round(float(r["ppg"]), 1),
            "apg": round(float(r["apg"]), 1),
            "rpg": round(float(r["rpg"]), 1),
            "spg": round(float(r["spg"]), 1),
            "bpg": round(float(r["bpg"]), 1),
            "games_played": int(r["games_played"]),
        })

    for tid in roster_by_team:
        roster_by_team[tid].sort(key=lambda p: p["ppg"], reverse=True)

    _state["roster_by_team"] = roster_by_team
    _state["player_names"] = {
        int(r["personId"]): f"{r['first_name']} {r['last_name']}"
        for _, r in agg.iterrows()
    }
    print(f"[services] Rosters cached for {len(roster_by_team)} teams, {len(agg)} players total")


# ── Public API helpers ──────────────────────────────────────────────────────

def get_feature_columns() -> list[str]:
    return _ensure_loaded()["feature_columns"]


def get_metrics() -> dict:
    return _ensure_loaded()["metrics"]


def get_team_ids() -> list[int]:
    """Return all team IDs that have rolling stats available."""
    state = _ensure_loaded()
    return sorted(state["team_rolling"].keys())


def get_teams() -> list[dict]:
    """Return list of active NBA teams for dropdown selectors."""
    return _ensure_loaded()["teams"]


def get_team_players(team_id: int) -> dict | None:
    """Return the current-season roster for *team_id*, or None."""
    state = _ensure_loaded()
    name_map = state.get("team_name_map", {})
    roster = state.get("roster_by_team", {}).get(team_id)
    if roster is None:
        return None
    team_info = name_map.get(team_id, {})
    return {
        "team_id": team_id,
        "team_city": team_info.get("city", ""),
        "team_name": team_info.get("name", ""),
        "players": roster,
    }


def compute_lineup_chemistry(
    team_id: int,
    player_ids: list[int],
) -> dict:
    """
    For a lineup of 2-5 players on the same team, compute:
      - lineup_chemistry (0-100 scale)
      - per-player bonus (sum of pair scores involving that player, scaled)
      - individual pair scores with percentile ranking within the team
    """
    from itertools import combinations

    state = _ensure_loaded()
    chem_df: pd.DataFrame = state.get("chemistry_pairs", pd.DataFrame())
    player_names = state.get("player_names", {})

    team_chem = chem_df[chem_df["teamId"] == team_id] if not chem_df.empty else pd.DataFrame()
    team_scores = team_chem["chemistry_score"].values if not team_chem.empty else np.array([])

    pairs_result = []
    player_totals: dict[int, float] = {pid: 0.0 for pid in player_ids}

    for a_id, b_id in combinations(player_ids, 2):
        a, b = sorted([a_id, b_id])
        score = 0.0
        percentile = 0.0
        if not chem_df.empty:
            mask = (
                (chem_df["playerA"] == a)
                & (chem_df["playerB"] == b)
                & (chem_df["teamId"] == team_id)
            )
            matches = chem_df.loc[mask, "chemistry_score"]
            if not matches.empty:
                score = float(matches.iloc[0])
                if len(team_scores) > 1:
                    percentile = float(np.mean(team_scores <= score) * 100)

        pairs_result.append({
            "player_a": a_id,
            "player_a_name": player_names.get(a_id, f"Player {a_id}"),
            "player_b": b_id,
            "player_b_name": player_names.get(b_id, f"Player {b_id}"),
            "chemistry_score": round(score * 100, 1),
            "percentile": round(percentile, 1),
        })
        player_totals[a_id] = player_totals.get(a_id, 0.0) + score
        player_totals[b_id] = player_totals.get(b_id, 0.0) + score

    pair_pcts = [p["chemistry_score"] for p in pairs_result]
    raw_avg = np.mean(pair_pcts) if pair_pcts else 0.0
    lineup_chemistry = round(min(raw_avg, 100), 1)

    max_bonus_raw = max(player_totals.values()) if player_totals else 1.0
    if max_bonus_raw == 0:
        max_bonus_raw = 1.0

    player_bonuses = []
    for pid in player_ids:
        raw = player_totals.get(pid, 0.0)
        bonus_pct = round(raw / max_bonus_raw * 100, 1)
        player_bonuses.append({
            "person_id": pid,
            "name": player_names.get(pid, f"Player {pid}"),
            "chemistry_contribution": bonus_pct,
        })

    return {
        "team_id": team_id,
        "lineup_chemistry": lineup_chemistry,
        "player_bonuses": player_bonuses,
        "pairs": pairs_result,
    }


def _compute_team_rating(team_id: int) -> float:
    """Normalized team strength (0-100) from Elo + season win% + form."""
    state = _ensure_loaded()
    elo = state.get("team_elo", {}).get(team_id, 1500.0)
    roll = state.get("team_rolling", {}).get(team_id, {})
    season_wp = roll.get("season_win_pct", 0.5)
    recent_wp = roll.get("roll_win", 0.5)

    elo_norm = max(0, min(100, (elo - 1300) / 4))
    season_norm = season_wp * 100
    form_norm = recent_wp * 100
    return round(0.4 * elo_norm + 0.3 * season_norm + 0.3 * form_norm, 1)


def build_features_for_matchup(
    home_team_id: int,
    away_team_id: int,
    home_player_ids: list[int] | None = None,
    away_player_ids: list[int] | None = None,
) -> tuple[list[float], dict]:
    """
    Look up the latest rolling stats, contextual features, Elo, and
    chemistry for both teams. When lineup player IDs are supplied,
    lineup-specific chemistry replaces the team average.

    Returns (feature_row, supporting_stats_dict).
    Raises ModelNotReady if team data is not available.
    """
    state = _ensure_loaded()
    columns = state["feature_columns"]
    team_rolling = state["team_rolling"]
    team_chemistry = state["team_chemistry"]
    team_elo = state.get("team_elo", {})

    if home_team_id not in team_rolling:
        raise ModelNotReady(
            f"No historical data for home team {home_team_id}. "
            f"Available teams: {sorted(team_rolling.keys())[:10]}…"
        )
    if away_team_id not in team_rolling:
        raise ModelNotReady(
            f"No historical data for away team {away_team_id}. "
            f"Available teams: {sorted(team_rolling.keys())[:10]}…"
        )

    home_roll = team_rolling[home_team_id]
    away_roll = team_rolling[away_team_id]
    home_elo_val = team_elo.get(home_team_id, 1500.0)
    away_elo_val = team_elo.get(away_team_id, 1500.0)

    # Lineup-specific chemistry overrides team average when lineups provided
    home_lineup_chem = None
    away_lineup_chem = None
    if home_player_ids and len(home_player_ids) >= 2:
        lc = compute_lineup_chemistry(home_team_id, home_player_ids)
        home_lineup_chem = lc
        home_chem = lc["lineup_chemistry"] / 100.0  # model feature is 0-1
    else:
        home_chem = team_chemistry.get(home_team_id, 0.0)

    if away_player_ids and len(away_player_ids) >= 2:
        lc = compute_lineup_chemistry(away_team_id, away_player_ids)
        away_lineup_chem = lc
        away_chem = lc["lineup_chemistry"] / 100.0  # model feature is 0-1
    else:
        away_chem = team_chemistry.get(away_team_id, 0.0)

    raw_features = {
        "home_chemistry": float(home_chem),
        "away_chemistry": float(away_chem),
        "home_elo": float(home_elo_val),
        "away_elo": float(away_elo_val),
        "elo_diff": float(home_elo_val - away_elo_val),
    }

    for col in columns:
        if col in raw_features:
            continue
        if col.startswith("diff_"):
            continue
        if col.startswith("home_"):
            raw_key = col.replace("home_", "")
            raw_features[col] = float(home_roll.get(raw_key, 0.0))
        elif col.startswith("away_"):
            raw_key = col.replace("away_", "")
            raw_features[col] = float(away_roll.get(raw_key, 0.0))
        else:
            raw_features[col] = 0.0

    diff_map = {
        "diff_ppg": ("home_roll_score", "away_roll_score"),
        "diff_opp_ppg": ("home_roll_opp_score", "away_roll_opp_score"),
        "diff_margin": ("home_roll_margin", "away_roll_margin"),
        "diff_win_pct": ("home_roll_win", "away_roll_win"),
        "diff_fg_pct": ("home_roll_fg_pct", "away_roll_fg_pct"),
        "diff_fg3_pct": ("home_roll_fg3_pct", "away_roll_fg3_pct"),
        "diff_reb": ("home_roll_reb", "away_roll_reb"),
        "diff_ast": ("home_roll_ast", "away_roll_ast"),
        "diff_tov": ("home_roll_tov", "away_roll_tov"),
        "diff_stl": ("home_roll_stl", "away_roll_stl"),
        "diff_blk": ("home_roll_blk", "away_roll_blk"),
        "diff_season_win_pct": ("home_season_win_pct", "away_season_win_pct"),
        "diff_chemistry": ("home_chemistry", "away_chemistry"),
    }
    for diff_col, (h_col, a_col) in diff_map.items():
        if diff_col in columns:
            raw_features[diff_col] = raw_features.get(h_col, 0.0) - raw_features.get(a_col, 0.0)

    for k, v in raw_features.items():
        if np.isnan(v):
            raw_features[k] = 0.0

    row = [raw_features.get(c, 0.0) for c in columns]

    home_stats = {
        "elo_rating": raw_features.get("home_elo", 1500),
        "win_pct_last_10": raw_features.get("home_roll_win", 0),
        "ppg_last_10": raw_features.get("home_roll_score", 0),
        "opp_ppg_last_10": raw_features.get("home_roll_opp_score", 0),
        "margin_last_10": raw_features.get("home_roll_margin", 0),
        "fg_pct_last_10": raw_features.get("home_roll_fg_pct", 0),
        "fg3_pct_last_10": raw_features.get("home_roll_fg3_pct", 0),
        "ft_pct_last_10": raw_features.get("home_roll_ft_pct", 0),
        "reb_last_10": raw_features.get("home_roll_reb", 0),
        "ast_last_10": raw_features.get("home_roll_ast", 0),
        "tov_last_10": raw_features.get("home_roll_tov", 0),
        "stl_last_10": raw_features.get("home_roll_stl", 0),
        "blk_last_10": raw_features.get("home_roll_blk", 0),
        "streak": raw_features.get("home_streak", 0),
        "rest_days": raw_features.get("home_rest_days", 0),
        "season_win_pct": raw_features.get("home_season_win_pct", 0),
        "chemistry_score": raw_features.get("home_chemistry", 0),
        "team_rating": _compute_team_rating(home_team_id),
    }
    away_stats = {
        "elo_rating": raw_features.get("away_elo", 1500),
        "win_pct_last_10": raw_features.get("away_roll_win", 0),
        "ppg_last_10": raw_features.get("away_roll_score", 0),
        "opp_ppg_last_10": raw_features.get("away_roll_opp_score", 0),
        "margin_last_10": raw_features.get("away_roll_margin", 0),
        "fg_pct_last_10": raw_features.get("away_roll_fg_pct", 0),
        "fg3_pct_last_10": raw_features.get("away_roll_fg3_pct", 0),
        "ft_pct_last_10": raw_features.get("away_roll_ft_pct", 0),
        "reb_last_10": raw_features.get("away_roll_reb", 0),
        "ast_last_10": raw_features.get("away_roll_ast", 0),
        "tov_last_10": raw_features.get("away_roll_tov", 0),
        "stl_last_10": raw_features.get("away_roll_stl", 0),
        "blk_last_10": raw_features.get("away_roll_blk", 0),
        "streak": raw_features.get("away_streak", 0),
        "rest_days": raw_features.get("away_rest_days", 0),
        "season_win_pct": raw_features.get("away_season_win_pct", 0),
        "chemistry_score": raw_features.get("away_chemistry", 0),
        "team_rating": _compute_team_rating(away_team_id),
    }

    if home_lineup_chem is not None:
        home_stats["lineup_chemistry"] = home_lineup_chem["lineup_chemistry"]
        home_stats["player_bonuses"] = home_lineup_chem["player_bonuses"]
    if away_lineup_chem is not None:
        away_stats["lineup_chemistry"] = away_lineup_chem["lineup_chemistry"]
        away_stats["player_bonuses"] = away_lineup_chem["player_bonuses"]

    supporting = {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_stats": home_stats,
        "away_stats": away_stats,
        "elo_diff": raw_features.get("elo_diff", 0),
    }

    return row, supporting


def lookup_pair_chemistry(
    player_a_id: int,
    player_b_id: int,
    team_id: int | None = None,
) -> list[dict]:
    """
    Look up pair-level chemistry for two players.

    Returns a list of matching records (could be multiple if the pair
    played on different teams). Each record contains co_play,
    shared_minutes, net_rating, and chemistry_score.
    """
    state = _ensure_loaded()
    chem_df: pd.DataFrame = state.get("chemistry_pairs", pd.DataFrame())

    if chem_df.empty:
        return []

    a, b = sorted([player_a_id, player_b_id])
    mask = (chem_df["playerA"] == a) & (chem_df["playerB"] == b)
    if team_id is not None:
        mask = mask & (chem_df["teamId"] == team_id)

    matches = chem_df.loc[mask]
    if matches.empty:
        return []

    records = matches.to_dict(orient="records")
    for rec in records:
        for k in ("teamId", "playerA", "playerB"):
            if k in rec:
                rec[k] = int(rec[k])
    return records


def predict(feature_values: list | np.ndarray, model_name: str = "xgboost") -> dict:
    """
    Run inference on a single feature row.

    Supported model_name values:
      - "xgboost"
      - "random_forest"
      - "ensemble" (stacking: logistic regression over RF + XGB probabilities)
      - "both" is handled in the view layer

    Returns dict with win probability, predicted winner, and SHAP summary.
    """
    state = _ensure_loaded()
    row = np.asarray(feature_values, dtype=np.float64).reshape(1, -1)

    if model_name == "ensemble":
        rf_model = state.get("rf")
        xgb_model = state.get("xgb")
        meta = state.get("stack")
        if rf_model is None or xgb_model is None or meta is None:
            raise ModelNotReady("Stacking ensemble requires RF + XGB + meta-learner artifacts.")

        rf_prob = float(rf_model.predict_proba(row)[0, 1])
        xgb_prob = float(xgb_model.predict_proba(row)[0, 1])
        meta_input = np.array([[rf_prob, xgb_prob]])
        prob = float(meta.predict_proba(meta_input)[0, 1])
        label = int(prob >= 0.5)

        return {
            "model": "ensemble",
            "home_win_probability": round(prob, 4),
            "predicted_winner": "home" if label == 1 else "away",
            "component_probabilities": {
                "random_forest": round(rf_prob, 4),
                "xgboost": round(xgb_prob, 4),
            },
        }

    key = "xgb" if model_name == "xgboost" else "rf"
    model = state.get(key)
    if model is None:
        raise ModelNotReady(f"Model '{model_name}' not found in artifacts.")

    prob = float(model.predict_proba(row)[0, 1])
    label = int(prob >= 0.5)

    from ml.explain import explain_single

    explanation = explain_single(row.flatten(), model_name=model_name, top_k=8)

    return {
        "model": model_name,
        "home_win_probability": round(prob, 4),
        "predicted_winner": "home" if label == 1 else "away",
        "shap_summary": {
            "base_value": explanation["base_value"],
            "top_positive": explanation["top_positive"],
            "top_negative": explanation["top_negative"],
        },
    }
