# NBA Game Outcome Prediction & Player Chemistry Analysis

A Django + Django REST Framework backend that predicts NBA game outcomes (home win / loss) using **Random Forest** and **XGBoost**, with player-chemistry proxy features derived from co-play frequency, shared minutes, and lineup net rating. Model transparency is provided via **SHAP** explanations.

## Quick start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run migrations
python manage.py migrate

# 4. (Optional) Generate synthetic sample data for smoke-testing
python -m ml.generate_sample_data

# 5. Train models
#    With sample data:
python manage.py train_models \
  --games data/samples/Games.csv \
  --player-stats data/samples/PlayerGameStats.csv \
  --lineups data/samples/Lineups.csv

#    With real data (place CSVs in data/):
python manage.py train_models

# 6. Start the dev server
python manage.py runserver
```

## Data requirements

Place the following CSV files under `data/`:

| File | Description |
|---|---|
| `Games.csv` | Game-level rows: `gameDateTimeEst`, `hometeamId`, `awayteamId`, `homeScore`, `awayScore`, `winner`, `gameType` |
| `PlayerGameStats.csv` | Per-player-per-game box scores: `gameId`, `teamId`, `playerId`, `minutes`, `fg_pct`, `fg3_pct`, `ft_pct`, `reb`, `ast`, `tov`, `pts` |
| `Lineups.csv` | Stint/lineup rows: `stintId`, `teamId`, `gameId`, `playerId`, `minutes`, `ptsScored`, `ptsConceded` |

Historical NBA data covering the 2004–2023 seasons can be sourced from public datasets (e.g. Kaggle NBA datasets or the `nba_api` Python package).

## API endpoints

All endpoints return JSON.

### `POST /api/predict/game/`

Predict the outcome of a game given rolling team stats and chemistry scores.

**Request body:**
```json
{
  "home_team_id": 1610612737,
  "away_team_id": 1610612738,
  "model": "xgboost",
  "features": {
    "home_roll_score": 112, "home_roll_opp_score": 101, "home_roll_win": 0.7,
    "home_roll_fg_pct": 0.47, "home_roll_fg3_pct": 0.37, "home_roll_ft_pct": 0.82,
    "home_roll_reb": 44, "home_roll_ast": 6.2, "home_roll_tov": 11,
    "away_roll_score": 108, "away_roll_opp_score": 105, "away_roll_win": 0.6,
    "away_roll_fg_pct": 0.45, "away_roll_fg3_pct": 0.35, "away_roll_ft_pct": 0.78,
    "away_roll_reb": 42, "away_roll_ast": 5.1, "away_roll_tov": 13,
    "home_chemistry": 0.55, "away_chemistry": 0.48
  }
}
```

**Response:** win probability, predicted winner, and SHAP feature attributions.

Set `"model": "both"` to get predictions from both Random Forest and XGBoost.

### `GET /api/models/metrics/`

Returns 10-fold cross-validation results (accuracy, F1, AUC) and the expected feature column list.

### `GET /api/chemistry/pair/?player_a_id=X&player_b_id=Y`

Returns precomputed chemistry metrics (co-play, shared minutes, net rating, chemistry score) for a player pair.

## Project structure

```
nba_ml_backend/
├── config/              # Django project settings, URLs, WSGI
├── api/                 # DRF app (views, serializers, services)
│   └── management/commands/train_models.py
├── ml/                  # ML package (usable standalone or from Django)
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── explain.py
│   ├── generate_sample_data.py
│   └── features/
│       ├── build.py       # Assembles final feature matrix
│       ├── team_stats.py  # Rolling team performance features
│       └── chemistry.py   # Player chemistry proxy metrics
├── data/                # Raw CSV files (gitignored except samples/)
│   └── samples/         # Small synthetic CSVs for testing
├── artifacts/           # Trained models + metrics (gitignored)
├── manage.py
├── requirements.txt
└── README.md
```

## Chemistry model (from the report appendix)

Player chemistry is approximated using three proxy metrics:

- **CoPlay(i,j)** = games played together / total team games
- **SharedMinutes(i,j)** = minutes on court together / total team minutes
- **NetRating** = (points scored − points conceded) / minutes × 100

These are normalised to [0, 1] and combined:

**ChemistryScore = w₁ × CoPlay + w₂ × SharedMinutes + w₃ × NetRating**

Weights default to `w₁=0.35, w₂=0.35, w₃=0.30` and are configurable via environment variables (`CHEM_W_COPLAY`, `CHEM_W_SHARED_MIN`, `CHEM_W_NET_RATING`) or in `config/settings.py`.

## ML evaluation

Models are evaluated with **10-fold stratified cross-validation**. Metrics reported:

- **Accuracy**
- **F1-score**
- **ROC AUC**

After cross-validation, both models are retrained on the full dataset and persisted as `.joblib` files for inference.
