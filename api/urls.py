from django.urls import path
from . import views

urlpatterns = [
    path("teams/", views.TeamListView.as_view(), name="team-list"),
    path("teams/<int:team_id>/players/", views.TeamPlayersView.as_view(), name="team-players"),
    path("chemistry/lineup/", views.LineupChemistryView.as_view(), name="chemistry-lineup"),
    path("chemistry/pair/", views.ChemistryPairView.as_view(), name="chemistry-pair"),
    path("predict/game/", views.PredictGameView.as_view(), name="predict-game"),
    path("models/metrics/", views.ModelMetricsView.as_view(), name="model-metrics"),
]
