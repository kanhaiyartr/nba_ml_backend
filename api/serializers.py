from rest_framework import serializers


class GamePredictionRequestSerializer(serializers.Serializer):
    home_team_id = serializers.IntegerField(help_text="NBA team ID for the home team")
    away_team_id = serializers.IntegerField(help_text="NBA team ID for the away team")
    home_player_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False,
        default=list,
        help_text="Optional list of player IDs in the home lineup",
    )
    away_player_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False,
        default=list,
        help_text="Optional list of player IDs in the away lineup",
    )
    model = serializers.ChoiceField(
        choices=["xgboost", "random_forest", "both"],
        default="xgboost",
        help_text="Which trained model to use for inference",
    )


class ChemistryPairRequestSerializer(serializers.Serializer):
    player_a_id = serializers.IntegerField()
    player_b_id = serializers.IntegerField()
    team_id = serializers.IntegerField(required=False)
