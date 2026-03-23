"""
Generate small synthetic CSV files that match the column contracts expected by
the data pipeline, so the training and API can be smoke-tested without real
NBA data.

Usage:
    python -m ml.generate_sample_data          # writes to data/samples/
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

TEAMS = list(range(1610612737, 1610612767))  # 30 NBA franchise IDs
PLAYERS_PER_TEAM = 12
random.seed(42)
np.random.seed(42)


def _player_pool() -> dict[int, list[int]]:
    pid = 200000
    pool: dict[int, list[int]] = {}
    for t in TEAMS:
        pool[t] = list(range(pid, pid + PLAYERS_PER_TEAM))
        pid += PLAYERS_PER_TEAM
    return pool


def generate_games(n: int = 600) -> pd.DataFrame:
    rows = []
    base_ts = pd.Timestamp("2020-10-01", tz="UTC")
    for i in range(n):
        home, away = random.sample(TEAMS, 2)
        home_score = random.randint(85, 130)
        away_score = random.randint(85, 130)
        winner = home if home_score > away_score else away
        rows.append({
            "gameDateTimeEst": base_ts + pd.Timedelta(hours=i * 4),
            "hometeamId": home,
            "awayteamId": away,
            "homeScore": home_score,
            "awayScore": away_score,
            "winner": winner,
            "gameType": 2,
        })
    return pd.DataFrame(rows)


def generate_player_game_stats(games_df: pd.DataFrame) -> pd.DataFrame:
    pool = _player_pool()
    rows = []
    for _, g in games_df.iterrows():
        game_id = hash((g["gameDateTimeEst"], g["hometeamId"], g["awayteamId"])) % 10**8
        for team_id in [g["hometeamId"], g["awayteamId"]]:
            players = pool.get(team_id, [])
            active = random.sample(players, min(8, len(players)))
            for pid in active:
                rows.append({
                    "gameId": game_id,
                    "teamId": team_id,
                    "playerId": pid,
                    "minutes": round(random.uniform(5, 38), 1),
                    "fg_pct": round(random.uniform(0.30, 0.60), 3),
                    "fg3_pct": round(random.uniform(0.20, 0.50), 3),
                    "ft_pct": round(random.uniform(0.60, 0.95), 3),
                    "reb": random.randint(0, 14),
                    "ast": random.randint(0, 12),
                    "tov": random.randint(0, 6),
                    "pts": random.randint(0, 35),
                })
    return pd.DataFrame(rows)


def generate_lineups(player_game_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stint_id = 0
    for (team, game), grp in player_game_df.groupby(["teamId", "gameId"]):
        players = grp["playerId"].tolist()
        for chunk_start in range(0, len(players), 5):
            stint_players = players[chunk_start:chunk_start + 5]
            if len(stint_players) < 2:
                continue
            stint_id += 1
            mins = round(random.uniform(2, 12), 1)
            scored = random.randint(2, 20)
            conceded = random.randint(2, 20)
            for pid in stint_players:
                rows.append({
                    "stintId": stint_id,
                    "teamId": team,
                    "gameId": game,
                    "playerId": pid,
                    "minutes": mins,
                    "ptsScored": scored,
                    "ptsConceded": conceded,
                })
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    games = generate_games()
    games.to_csv(OUTPUT_DIR / "Games.csv", index=False)
    print(f"Games: {len(games)} rows → {OUTPUT_DIR / 'Games.csv'}")

    pgs = generate_player_game_stats(games)
    pgs.to_csv(OUTPUT_DIR / "PlayerGameStats.csv", index=False)
    print(f"PlayerGameStats: {len(pgs)} rows")

    lineups = generate_lineups(pgs)
    lineups.to_csv(OUTPUT_DIR / "Lineups.csv", index=False)
    print(f"Lineups: {len(lineups)} rows")


if __name__ == "__main__":
    main()
