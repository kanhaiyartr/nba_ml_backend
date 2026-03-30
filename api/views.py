"""
DRF views for NBA game prediction, model metrics, and chemistry analysis.
"""
from __future__ import annotations

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import (
    ChemistryPairRequestSerializer,
    GamePredictionRequestSerializer,
    LineupChemistryRequestSerializer,
)
from .services import (
    ModelNotReady,
    build_features_for_matchup,
    compute_lineup_chemistry,
    get_feature_columns,
    get_metrics,
    get_team_players,
    get_teams,
    lookup_pair_chemistry,
    predict,
    _compute_team_rating,
)


class TeamListView(APIView):
    """
    GET /api/teams/

    Returns all currently active NBA teams for dropdown selection.
    """

    def get(self, request):
        try:
            teams = get_teams()
        except ModelNotReady as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response(teams)


class TeamPlayersView(APIView):
    """
    GET /api/teams/<team_id>/players/

    Returns the current-season roster for a team with per-game averages.
    """

    def get(self, request, team_id: int):
        try:
            data = get_team_players(team_id)
        except ModelNotReady as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        if data is None:
            return Response(
                {"detail": f"No roster found for team {team_id} in the current season."},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(data)


class LineupChemistryView(APIView):
    """
    POST /api/chemistry/lineup/

    Computes chemistry for a lineup of 2-5 players on the same team.
    Returns overall lineup chemistry (0-100), per-player bonuses, and
    individual pair scores.
    """

    def post(self, request):
        ser = LineupChemistryRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)

        team_id = ser.validated_data["team_id"]
        player_ids = ser.validated_data["player_ids"]

        try:
            result = compute_lineup_chemistry(team_id, player_ids)
        except ModelNotReady as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        return Response(result)


class PredictGameView(APIView):
    """
    POST /api/predict/game/

    The user sends home_team_id, away_team_id, optional lineup player IDs
    (2-5 each), and which model to use. The backend automatically looks up
    each team's recent performance, computes rolling features and chemistry
    (lineup-specific when player IDs are provided), runs the ML model, and
    returns the prediction with supporting statistics, confidence, and
    team ratings.
    """

    def post(self, request):
        ser = GamePredictionRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)

        home_team_id = ser.validated_data["home_team_id"]
        away_team_id = ser.validated_data["away_team_id"]
        home_player_ids = ser.validated_data.get("home_player_ids") or []
        away_player_ids = ser.validated_data.get("away_player_ids") or []
        model_name = ser.validated_data["model"]

        manual_features: dict | None = request.data.get("features")

        if manual_features:
            try:
                columns = get_feature_columns()
            except ModelNotReady as exc:
                return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

            missing = [c for c in columns if c not in manual_features]
            if missing:
                return Response(
                    {"detail": f"Missing features: {missing}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            row = [float(manual_features[c]) for c in columns]
            supporting = None
        else:
            try:
                row, supporting = build_features_for_matchup(
                    home_team_id,
                    away_team_id,
                    home_player_ids=home_player_ids or None,
                    away_player_ids=away_player_ids or None,
                )
            except ModelNotReady as exc:
                return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        try:
            if model_name == "both":
                prediction = {
                    "xgboost": predict(row, "xgboost"),
                    "random_forest": predict(row, "random_forest"),
                }
            else:
                prediction = predict(row, model_name)
        except ModelNotReady as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        # Add confidence score
        if isinstance(prediction, dict) and "home_win_probability" in prediction:
            prob = prediction["home_win_probability"]
            prediction["confidence"] = round(abs(prob - 0.5) * 200, 1)

        result = {"prediction": prediction}
        if supporting:
            result["supporting_stats"] = supporting

        return Response(result)


class ModelMetricsView(APIView):
    """GET /api/models/metrics/"""

    def get(self, request):
        try:
            metrics = get_metrics()
            columns = get_feature_columns()
        except ModelNotReady as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        return Response({
            "metrics": metrics,
            "feature_columns": columns,
        })


class ChemistryPairView(APIView):
    """
    GET /api/chemistry/pair/?player_a_id=X&player_b_id=Y[&team_id=Z]
    """

    def get(self, request):
        ser = ChemistryPairRequestSerializer(data=request.query_params)
        ser.is_valid(raise_exception=True)

        player_a = ser.validated_data["player_a_id"]
        player_b = ser.validated_data["player_b_id"]
        team_id = ser.validated_data.get("team_id")

        try:
            results = lookup_pair_chemistry(player_a, player_b, team_id)
        except ModelNotReady as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        if not results:
            return Response(
                {
                    "detail": "No chemistry data found for this player pair. "
                              "They may not have been teammates in the dataset.",
                    "player_a_id": player_a,
                    "player_b_id": player_b,
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        if len(results) == 1:
            return Response(results[0])

        return Response({"pairs": results, "count": len(results)})
