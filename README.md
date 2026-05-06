# NBA Game Outcome Prediction & Player Chemistry Analysis

A Django REST Framework backend that predicts NBA game outcomes and analyzes player chemistry using Random Forest and XGBoost classifiers, trained on historical NBA data spanning 1946--2026.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Player Chemistry Analysis](#player-chemistry-analysis)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
- [Running on a New Machine](#running-on-a-new-machine)
- [Configuration](#configuration)
- [Evaluation Results](#evaluation-results)

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                        Frontend (any client)                      │
│       Selects teams → picks lineups → views chemistry →          │
│                    requests prediction                            │
└──────────────────────────┬────────────────────────────────────────┘
                           │  REST API (JSON)
┌──────────────────────────▼────────────────────────────────────────┐
│                   Django REST Framework (api/)                     │
│                                                                   │
│  GET  /api/teams/                  → list active NBA teams        │
│  GET  /api/teams/<id>/players/     → current-season roster+stats  │
│  POST /api/chemistry/lineup/       → lineup chemistry (2-5 pids) │
│  POST /api/predict/game/           → win probability + stats      │
│  GET  /api/chemistry/pair/         → pair-level chemistry lookup  │
│  GET  /api/models/metrics/         → CV metrics + feature list    │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                   Singleton Service Layer (services.py)            │
│                                                                   │
│  Loads on first request (thread-safe):                            │
│    • Trained models (RF, XGBoost, meta-learner) from joblib       │
│    • Precomputed per-team rolling stats, Elo, chemistry           │
│    • Current-season rosters with aggregated player stats          │
│                                                                   │
│  Provides O(1) lookups for all API endpoints.                     │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                   ML Pipeline (ml/)                                │
│                                                                   │
│  data_loader.py  → loads raw CSVs from data/                      │
│  preprocess.py   → cleans, filters Regular Season, derives target │
│  features/       → rolling stats, chemistry, Elo, differentials   │
│  train.py        → 10-fold stratified CV, persists artifacts      │
│  explain.py      → SHAP-based feature explanations                │
└───────────────────────────────────────────────────────────────────┘
```

The system follows a **train-once, serve-many** pattern: models are trained offline via a management command and persisted as `.joblib` artifacts. At server startup, the service layer loads models and precomputes all team-level lookups so that API requests require no heavy computation.

---

## Project Structure

```
nba_ml_backend/
├── manage.py                          # Django entry point
├── requirements.txt                   # Python dependencies
├── .gitignore
│
├── config/                            # Django project settings
│   ├── settings.py                    # DRF, CORS, data/artifact paths, chemistry weights
│   ├── urls.py                        # Root URL routing → api/urls.py
│   └── wsgi.py                        # WSGI application
│
├── api/                               # Django REST Framework application
│   ├── urls.py                        # API endpoint routing
│   ├── views.py                       # Request handling (6 view classes)
│   ├── serializers.py                 # Input validation
│   ├── services.py                    # Singleton model/data service
│   └── management/commands/
│       └── train_models.py            # `python manage.py train_models`
│
├── ml/                                # Machine learning pipeline
│   ├── data_loader.py                 # CSV loading functions
│   ├── preprocess.py                  # Data cleaning and target derivation
│   ├── train.py                       # Model training with 10-fold CV
│   ├── explain.py                     # SHAP-based explanations
│   └── features/
│       ├── build.py                   # Feature matrix assembly
│       ├── team_stats.py              # Rolling team performance stats
│       ├── chemistry.py               # Player chemistry proxy metrics
│       └── elo.py                     # Elo rating system
│
├── data/                              # Raw NBA CSV files (git-ignored)
│   ├── Games.csv                      # 73K+ game records (1946–2026)
│   ├── TeamStatistics.csv             # 146K team box-score records
│   ├── PlayerStatistics.csv           # 1.66M player game-level stats
│   ├── Players.csv                    # Player biographical data
│   ├── TeamHistories.csv              # Franchise history and naming
│   ├── LeagueSchedule24_25.csv        # Schedule data
│   └── LeagueSchedule25_26.csv
│
└── artifacts/                         # Generated model artifacts (git-ignored)
    ├── model_rf.joblib                # Trained Random Forest
    ├── model_xgb.joblib               # Trained XGBoost
    ├── model_stack.joblib             # Stacking meta-learner
    ├── feature_columns.json           # Ordered feature names
    ├── metrics.json                   # Cross-validation results
    └── games_clean.csv                # Preprocessed game data
```

---

## Data Pipeline

### Raw Data Sources

| File | Records | Description |
|------|---------|-------------|
| `Games.csv` | 73,033 | Every NBA game since 1946 with scores, teams, dates, and game type |
| `TeamStatistics.csv` | 146,068 | Per-team box-score stats per game (FG%, rebounds, assists, turnovers, etc.) |
| `PlayerStatistics.csv` | 1,663,409 | Per-player game-level stats (points, assists, rebounds, minutes, +/-) |
| `Players.csv` | 6,688 | Player biographical data (name, birth date, position flags) |
| `TeamHistories.csv` | 140 | Franchise naming, city, and active season ranges |

### Preprocessing (`ml/preprocess.py`)

1. **Column selection**: `gameId`, `gameDateTimeEst`, `hometeamId`, `awayteamId`, `homeScore`, `awayScore`, `winner`, `gameType`
2. **Filter**: Keep only `"Regular Season"` games (excludes playoffs, preseason, All-Star)
3. **Drop nulls**: Remove rows with missing scores, teams, dates, or winner
4. **Target derivation**: `home_win = 1` if `winner == hometeamId`, else `0`
5. **Datetime normalization**: Parse to UTC-aware timestamps, sort chronologically
6. **Persist**: Save to `artifacts/games_clean.csv` (66,613 regular-season games)

---

## Feature Engineering

The model uses **62 features** organized into 5 categories. All features are computed per-game so that training and inference use the exact same pipeline.

### 1. Rolling Team Performance (10-game window)

For each team, a sliding window over their last 10 games computes:

| Feature | Description |
|---------|-------------|
| `roll_win` | Win rate in last 10 games |
| `roll_score` | Average points scored |
| `roll_opp_score` | Average points conceded |
| `roll_margin` | Average point differential |
| `roll_fg_pct` | Field goal percentage |
| `roll_fg3_pct` | Three-point percentage |
| `roll_ft_pct` | Free throw percentage |
| `roll_reb` | Total rebounds per game |
| `roll_oreb` / `roll_dreb` | Offensive / defensive rebounds |
| `roll_ast` | Assists per game |
| `roll_tov` | Turnovers per game |
| `roll_stl` / `roll_blk` | Steals / blocks per game |
| `roll_pts_paint` | Points in the paint |
| `roll_pts_fb` | Fast break points |
| `roll_pts_bench` | Bench points |
| `roll_pts_from_tov` | Points from turnovers |
| `roll_pts_2nd_chance` | Second chance points |

Each feature is computed for both `home_` and `away_` teams.

### 2. Contextual Features

| Feature | Description |
|---------|-------------|
| `streak` | Current win/loss streak (positive = wins) |
| `rest_days` | Days since last game |
| `season_win_pct` | Season-to-date win percentage |

### 3. Elo Ratings (`ml/features/elo.py`)

A running competitive strength estimate computed chronologically over all historical games:

- **Initial Elo**: 1500.0 for all teams
- **K-factor**: 25.0 (update sensitivity)
- **Home advantage**: 70.0 Elo points
- **Season mean-reversion**: 60% reversion toward 1500 at season boundaries
- **Expected score formula**: `E = 1 / (1 + 10^(-(Elo_A - Elo_B + HomeAdv) / 400))`

Features: `home_elo`, `away_elo`, `elo_diff`

Elo is computed on the **full historical dataset** before the modern-era filter is applied, so teams carry historically calibrated strength into the training window.

### 4. Player Chemistry

Chemistry is computed from `PlayerStatistics.csv` using data from 2015 onward with **per-team normalization** (see [Player Chemistry Analysis](#player-chemistry-analysis) section).

Features: `home_chemistry`, `away_chemistry`

### 5. Differential Features (Home - Away)

For each key metric, the difference between home and away values is computed:

`diff_ppg`, `diff_opp_ppg`, `diff_margin`, `diff_win_pct`, `diff_fg_pct`, `diff_fg3_pct`, `diff_reb`, `diff_ast`, `diff_tov`, `diff_stl`, `diff_blk`, `diff_season_win_pct`, `diff_chemistry`

### Modern Era Filter

Training data is filtered to games from **2010 onward** (19,632 games) to focus on the modern NBA era where playing styles, rules, and pace are consistent with current basketball.

---

## Machine Learning Models

### Random Forest (Baseline)

```
n_estimators:        500
max_depth:           15
min_samples_split:   8
min_samples_leaf:    3
max_features:        sqrt
class_weight:        balanced_subsample
```

### XGBoost (Primary)

```
n_estimators:        800 (with early stopping at 50 rounds)
max_depth:           5
learning_rate:       0.03
subsample:           0.75
colsample_bytree:    0.7
colsample_bylevel:   0.7
reg_alpha:           0.3
reg_lambda:          2.0
min_child_weight:    5
gamma:               0.1
eval_metric:         logloss
```

### Training Process

1. **Preprocessing**: Clean raw games, filter to Regular Season
2. **Feature matrix**: Build 62 features across all categories
3. **10-fold Stratified Cross-Validation**: Ensures class balance (57.3% home-win rate) is preserved in each fold
4. **Per-fold evaluation**: Accuracy, F1-score, AUC recorded for each fold
5. **Full retrain**: After CV, both models are retrained on the complete dataset for deployment
6. **Artifact persistence**: Models saved as `.joblib`, feature columns and metrics saved as `.json`

### SHAP Explanations (`ml/explain.py`)

Every prediction from RF or XGBoost is accompanied by SHAP (SHapley Additive exPlanations) feature attributions:

- Uses `shap.TreeExplainer` (optimized for tree-based models)
- Returns the **top 8 positive** (pushing toward home win) and **top 8 negative** (pushing toward away win) features
- Provides the `base_value` (expected log-odds baseline)

---

## Player Chemistry Analysis

Chemistry measures how well a pair of players performs together, based on three proxy metrics from the report:

### Formula

```
ChemistryScore = 0.35 * CoPlay_norm + 0.35 * SharedMinutes_norm + 0.30 * NetRating_norm
```

| Component | Formula | What it captures |
|-----------|---------|-----------------|
| **CoPlay** | `games_together / total_team_games` | How often the pair plays together |
| **SharedMinutes** | `min(minutes_A, minutes_B) per game / total_team_minutes` | Estimated overlapping court time |
| **NetRating** | Team average `plusMinusPoints` | How well the team performs overall |

### Normalization Strategy

- **Per-team normalization**: CoPlay and SharedMinutes are min-max normalized **within each team**, not globally. This ensures the best pair on each team approaches 1.0 rather than being compressed by all-time outliers.
- **Net rating**: Normalized globally since it is a team-level metric.
- **Recent era focus**: Chemistry is computed from **2015 onward** to ensure relevance to current rosters.

### Lineup Chemistry (API)

When a user selects 2-5 players for a lineup:

1. **Pair scores**: Each pair's chemistry score is looked up and scaled to 0-100
2. **Percentile ranking**: Each pair is ranked against all pairs on that team
3. **Lineup chemistry**: Average of all pair scores (0-100 scale)
4. **Player contribution**: Each player's total chemistry contribution relative to the best contributor

Weights are configurable via environment variables:
- `CHEM_W_COPLAY` (default: 0.35)
- `CHEM_W_SHARED_MIN` (default: 0.35)
- `CHEM_W_NET_RATING` (default: 0.30)

---

## API Reference

Base URL: `http://localhost:8000/api/`

### 1. List Teams

```
GET /api/teams/
```

Returns all 30 active NBA teams for dropdown selection.

**Response:**
```json
[
  {
    "team_id": 1610612747,
    "city": "Los Angeles",
    "name": "Lakers",
    "abbreviation": "LAL"
  }
]
```

**cURL:**
```bash
curl http://localhost:8000/api/teams/
```

---

### 2. Get Team Roster

```
GET /api/teams/<team_id>/players/
```

Returns the current-season roster with per-game averages.

**Response:**
```json
{
  "team_id": 1610612747,
  "team_city": "Los Angeles",
  "team_name": "Lakers",
  "players": [
    {
      "person_id": 1629029,
      "first_name": "Luka",
      "last_name": "Doncic",
      "position": "PG",
      "ppg": 33.4,
      "apg": 9.8,
      "rpg": 8.3,
      "spg": 1.5,
      "bpg": 0.5,
      "games_played": 42
    }
  ]
}
```

**cURL:**
```bash
curl http://localhost:8000/api/teams/1610612747/players/
```

---

### 3. Lineup Chemistry

```
POST /api/chemistry/lineup/
```

Computes chemistry for a lineup of 2-5 players on the same team.

**Request body:**
```json
{
  "team_id": 1610612738,
  "player_ids": [1627759, 1628369, 1628401, 1630202, 202696]
}
```

**Response:**
```json
{
  "team_id": 1610612738,
  "lineup_chemistry": 56.1,
  "player_bonuses": [
    {
      "person_id": 1627759,
      "name": "Jaylen Brown",
      "chemistry_contribution": 100.0
    }
  ],
  "pairs": [
    {
      "player_a": 1627759,
      "player_a_name": "Jaylen Brown",
      "player_b": 1628369,
      "player_b_name": "Jayson Tatum",
      "chemistry_score": 100.0,
      "percentile": 100.0
    }
  ]
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/api/chemistry/lineup/ \
  -H "Content-Type: application/json" \
  -d '{"team_id": 1610612738, "player_ids": [1627759, 1628369, 1628401, 1630202, 202696]}'
```

---

### 4. Predict Game Outcome

```
POST /api/predict/game/
```

Predicts the probability of the home team winning. Automatically computes all features from team IDs and optional lineup selections.

**Request body:**
```json
{
  "home_team_id": 1610612747,
  "away_team_id": 1610612738,
  "model": "xgboost",
  "home_player_ids": [1629029, 1630559, 2544, 1629028, 1629060],
  "away_player_ids": [1627759, 1628369, 1628401, 1630202, 202696]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `home_team_id` | int | Yes | NBA team ID for the home team |
| `away_team_id` | int | Yes | NBA team ID for the away team |
| `model` | string | No | `"xgboost"` (default), `"random_forest"`, or `"both"` |
| `home_player_ids` | int[] | No | 2-5 player IDs for lineup-specific chemistry |
| `away_player_ids` | int[] | No | 2-5 player IDs for lineup-specific chemistry |

**Response:**
```json
{
  "prediction": {
    "model": "xgboost",
    "home_win_probability": 0.6234,
    "predicted_winner": "home",
    "confidence": 24.7,
    "shap_summary": {
      "base_value": 0.287,
      "top_positive": [["elo_diff", 0.142], ["diff_win_pct", 0.089]],
      "top_negative": [["away_roll_score", -0.051]]
    }
  },
  "supporting_stats": {
    "home_team_id": 1610612747,
    "away_team_id": 1610612738,
    "home_stats": {
      "elo_rating": 1587.3,
      "win_pct_last_10": 0.7,
      "ppg_last_10": 118.5,
      "team_rating": 72.1,
      "lineup_chemistry": 49.7
    },
    "away_stats": { "..." : "..." },
    "elo_diff": 45.2
  }
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/api/predict/game/ \
  -H "Content-Type: application/json" \
  -d '{
    "home_team_id": 1610612747,
    "away_team_id": 1610612738,
    "model": "xgboost",
    "home_player_ids": [1629029, 1630559, 2544, 1629028, 1629060],
    "away_player_ids": [1627759, 1628369, 1628401, 1630202, 202696]
  }'
```

---

### 5. Pair Chemistry Lookup

```
GET /api/chemistry/pair/?player_a=<id>&player_b=<id>[&team_id=<id>]
```

Returns the raw chemistry metrics for a specific player pair.

**cURL:**
```bash
curl "http://localhost:8000/api/chemistry/pair/?player_a=2544&player_b=1630559&team_id=1610612747"
```

---

### 6. Model Metrics

```
GET /api/models/metrics/
```

Returns cross-validation performance metrics and the feature column list.

**cURL:**
```bash
curl http://localhost:8000/api/models/metrics/
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
cd nba_ml_backend
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Place the Data Files

Copy the NBA CSV files into the `data/` directory:

```
data/
├── Games.csv
├── TeamStatistics.csv
├── PlayerStatistics.csv
├── Players.csv
├── TeamHistories.csv
├── LeagueSchedule24_25.csv
└── LeagueSchedule25_26.csv
```

### Train the Models

```bash
python manage.py train_models
```

This takes approximately 60 seconds and produces the following artifacts:

| File | Description |
|------|-------------|
| `artifacts/model_rf.joblib` | Trained Random Forest classifier |
| `artifacts/model_xgb.joblib` | Trained XGBoost classifier |
| `artifacts/model_stack.joblib` | Stacking meta-learner (Logistic Regression) |
| `artifacts/feature_columns.json` | Ordered list of 62 feature names |
| `artifacts/metrics.json` | 10-fold CV accuracy, F1, and AUC per model |
| `artifacts/games_clean.csv` | Preprocessed game data |

### Start the Server

```bash
python manage.py runserver 8000
```

The API is available at `http://localhost:8000/api/`.

### Tear Down

Stop the server with `Ctrl+C`. Deactivate the virtual environment:

```bash
deactivate
```

---

## Running on a New Machine

### Option A: Retrain from Scratch

```bash
git clone <repo-url> nba_ml_backend
cd nba_ml_backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Copy data files (not in git due to size)
scp -r user@source:/path/to/nba_ml_backend/data/ ./data/

# Train and serve
python manage.py train_models
python manage.py runserver 0.0.0.0:8000
```

### Option B: Use Pre-trained Artifacts

Copy both `data/` and `artifacts/` from the original machine:

```bash
scp -r user@source:/path/to/nba_ml_backend/data/ ./data/
scp -r user@source:/path/to/nba_ml_backend/artifacts/ ./artifacts/

# Serve immediately (no training needed)
python manage.py runserver 0.0.0.0:8000
```

> **Note:** The `data/` folder is always required because the server loads CSVs at startup to compute rolling stats, Elo ratings, and current-season rosters.

### Production Deployment

For a persistent deployment, use Gunicorn instead of the Django development server:

```bash
pip install gunicorn
gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 2
```

---

## Configuration

All settings are in `config/settings.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DJANGO_SECRET_KEY` | dev key | Django secret key (change in production) |
| `DJANGO_DEBUG` | `True` | Debug mode |
| `DJANGO_ALLOWED_HOSTS` | `*` | Comma-separated allowed hosts |
| `CORS_ALLOWED_ORIGINS` | `localhost:8080,5173` | Frontend origins for CORS |
| `CHEM_W_COPLAY` | `0.35` | Chemistry weight for CoPlay |
| `CHEM_W_SHARED_MIN` | `0.35` | Chemistry weight for SharedMinutes |
| `CHEM_W_NET_RATING` | `0.30` | Chemistry weight for NetRating |

---

## Evaluation Results

10-fold Stratified Cross-Validation on 19,632 modern-era games (2010--2026):

| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| Random Forest | 66.04% ± 1.17% | 0.705 ± 0.012 | 0.712 ± 0.015 |
| **XGBoost** | **66.84% ± 1.09%** | **0.730 ± 0.010** | **0.719 ± 0.014** |
| Stacking Ensemble | 66.74% ± 1.19% | 0.722 ± 0.011 | 0.718 ± 0.014 |

- **Training samples**: 19,632 games
- **Features**: 62
- **Class balance**: 57.3% home-win rate
- **Home-court advantage** is the most significant baseline predictor, reflected in the Elo home-advantage parameter (70 points)

> **Context:** NBA game prediction above 68% accuracy is considered competitive with professional oddsmakers. Key factors like real-time injury reports, travel fatigue, and motivation are not available in the historical box-score data, which sets an inherent ceiling on achievable accuracy.

---

## Common Team IDs

| Team | ID |
|------|-----|
| Atlanta Hawks | 1610612737 |
| Boston Celtics | 1610612738 |
| Brooklyn Nets | 1610612751 |
| Charlotte Hornets | 1610612766 |
| Chicago Bulls | 1610612741 |
| Cleveland Cavaliers | 1610612739 |
| Dallas Mavericks | 1610612742 |
| Denver Nuggets | 1610612743 |
| Detroit Pistons | 1610612765 |
| Golden State Warriors | 1610612744 |
| Houston Rockets | 1610612745 |
| Indiana Pacers | 1610612754 |
| LA Clippers | 1610612746 |
| Los Angeles Lakers | 1610612747 |
| Memphis Grizzlies | 1610612763 |
| Miami Heat | 1610612748 |
| Milwaukee Bucks | 1610612749 |
| Minnesota Timberwolves | 1610612750 |
| New Orleans Pelicans | 1610612740 |
| New York Knicks | 1610612752 |
| Oklahoma City Thunder | 1610612760 |
| Orlando Magic | 1610612753 |
| Philadelphia 76ers | 1610612755 |
| Phoenix Suns | 1610612756 |
| Portland Trail Blazers | 1610612757 |
| Sacramento Kings | 1610612758 |
| San Antonio Spurs | 1610612759 |
| Toronto Raptors | 1610612761 |
| Utah Jazz | 1610612762 |
| Washington Wizards | 1610612764 |
