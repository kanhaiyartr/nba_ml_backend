"""
Django management command to train the ML models.

Usage:
    python manage.py train_models
    python manage.py train_models --games data/Games.csv
"""
from django.core.management.base import BaseCommand

from ml.train import train


class Command(BaseCommand):
    help = "Train Random Forest and XGBoost models on NBA game data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--games",
            type=str,
            default=None,
            help="Path to raw Games.csv (defaults to data/Games.csv)",
        )
        parser.add_argument(
            "--team-stats",
            type=str,
            default=None,
            help="Path to TeamStatistics.csv for box-score features",
        )
        parser.add_argument(
            "--player-stats",
            type=str,
            default=None,
            help="Path to PlayerStatistics.csv for chemistry features",
        )

    def handle(self, *args, **options):
        from pathlib import Path

        games = Path(options["games"]) if options["games"] else None
        ts = Path(options["team_stats"]) if options["team_stats"] else None
        ps = Path(options["player_stats"]) if options["player_stats"] else None

        self.stdout.write("Starting model training …\n")
        metrics = train(
            games_path=games,
            team_stats_path=ts,
            player_stats_path=ps,
        )
        self.stdout.write(self.style.SUCCESS("Training complete."))
