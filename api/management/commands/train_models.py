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
            help="Path to raw Games.csv (defaults to DATA_DIR/Games.csv)",
        )
        parser.add_argument(
            "--player-stats",
            type=str,
            default=None,
            help="Path to PlayerGameStats.csv for chemistry features",
        )
        parser.add_argument(
            "--lineups",
            type=str,
            default=None,
            help="Path to Lineups.csv for shared-minutes chemistry",
        )

    def handle(self, *args, **options):
        from pathlib import Path

        games = Path(options["games"]) if options["games"] else None
        pgs = Path(options["player_stats"]) if options["player_stats"] else None
        lups = Path(options["lineups"]) if options["lineups"] else None

        self.stdout.write("Starting model training …\n")
        metrics = train(
            games_path=games,
            player_game_path=pgs,
            stint_path=lups,
        )
        self.stdout.write(self.style.SUCCESS("Training complete."))
