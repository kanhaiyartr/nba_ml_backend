"""
DRF views for NBA game prediction, model metrics, and chemistry analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from django.conf import settings

from .serializers import (
    ChemistryPairRequestSerializer,
    GamePredictionRequestSerializer,
)
from .services import ModelNotReady, get_feature_columns, get_metrics, predict


class PredictGameView(APIView):
    """
    POST /api/predict/game/

    Accepts home/away team IDs (and optional player IDs) together with a
    feature vector. Returns win probability, predicted winner, and SHAP
    explanation.

    The request body may supply ``features`` (a dict mapping feature names to
    values) directly, or supply team/player IDs for a future auto-lookup path.
    """

    def post(self, request):
        ser = GamePredictionRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)

        model_name = ser.validated_data["model"]

        features_dict: dict | None = request.data.get("features")
        if not features_dict:
            return Response(
                {
                    "detail": (
                        "Provide a 'features' dict mapping feature column names "
                        "to numeric values. Run GET /api/models/metrics/ to see "
                        "the expected feature list."
                    ),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            columns = get_feature_columns()
        except ModelNotReady as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        missing = [c for c in columns if c not in features_dict]
        if missing:
            return Response(
                {"detail": f"Missing features: {missing}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        row = [float(features_dict[c]) for c in columns]

        try:
            if model_name == "both":
                result = {
                    "xgboost": predict(row, "xgboost"),
                    "random_forest": predict(row, "random_forest"),
                }
            else:
                result = predict(row, model_name)
        except ModelNotReady as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        return Response(result)


class ModelMetricsView(APIView):
    """
    GET /api/models/metrics/

    Returns persisted cross-validation metrics and the expected feature list.
    """

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

    Looks up precomputed chemistry data for a player pair.
    Falls back to a stub if chemistry artifacts are not available.
    """

    def get(self, request):
        ser = ChemistryPairRequestSerializer(data=request.query_params)
        ser.is_valid(raise_exception=True)

        player_a = ser.validated_data["player_a_id"]
        player_b = ser.validated_data["player_b_id"]
        team_id = ser.validated_data.get("team_id")

        chem_path = Path(settings.MODEL_ARTIFACTS_DIR) / "chemistry_pairs.json"
        if not chem_path.exists():
            return Response(
                {
                    "detail": (
                        "Chemistry data not precomputed. "
                        "Run train_models with lineup data first."
                    ),
                    "player_a_id": player_a,
                    "player_b_id": player_b,
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        with open(chem_path) as f:
            all_pairs: list[dict] = json.load(f)

        pair_key = tuple(sorted([player_a, player_b]))
        for entry in all_pairs:
            e_key = tuple(sorted([entry["playerA"], entry["playerB"]]))
            if e_key == pair_key:
                if team_id and entry.get("teamId") != team_id:
                    continue
                return Response(entry)

        return Response(
            {"detail": "Pair not found", "player_a_id": player_a, "player_b_id": player_b},
            status=status.HTTP_404_NOT_FOUND,
        )
