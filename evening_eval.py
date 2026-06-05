"""
Hyperion V10 — Pipeline Évaluation Soir (20h00)
Point d'entrée : python evening_eval.py [--date YYYY-MM-DD]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logger import logger, log_section
from src.evaluation.agent_h import AgentH


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperion V10 — Évaluation soir")
    parser.add_argument("--date", type=str, help="Date YYYY-MM-DD (défaut : aujourd'hui)")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    log_section(f"HYPERION V10 — Évaluation soir {date_str}")

    # Créer les dossiers nécessaires
    for d in ["data/backups", "logs", "backup/telegram"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    agent = AgentH()
    success = agent.run_evening_pipeline(date_str=date_str)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
