from django.urls import path
from . import views

urlpatterns = [
    path("predict/game/", views.PredictGameView.as_view(), name="predict-game"),
    path("models/metrics/", views.ModelMetricsView.as_view(), name="model-metrics"),
    path("chemistry/pair/", views.ChemistryPairView.as_view(), name="chemistry-pair"),
]
