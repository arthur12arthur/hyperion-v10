"""
Hyperion V10 — Système de logging centralisé.
Format : [HYPERION][MODULE][LEVEL] message
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import config


def setup_logger(name: str = "hyperion") -> logging.Logger:
    """
    Configure et retourne un logger avec handler fichier + console.

    Args:
        name: Nom du logger (défaut : 'hyperion')

    Returns:
        Logger configuré
    """
    level_str = config.get("logging.level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)

    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(level)

    # Éviter les handlers dupliqués (appels multiples)
    if logger_instance.handlers:
        return logger_instance

    log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_format = "%H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # ── Handler console ───────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger_instance.addHandler(console_handler)

    # ── Handler fichier rotatif ───────────────────────────────
    log_file = config.get("logging.file", "logs/hyperion_v10.log")
    log_path = config.get_path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        file_handler = RotatingFileHandler(
            str(log_path),
            maxBytes=config.get("logging.max_bytes", 10_485_760),
            backupCount=config.get("logging.backup_count", 5),
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger_instance.addHandler(file_handler)
    except Exception as e:
        logger_instance.warning(f"Impossible d'ouvrir le fichier de log {log_path} : {e}")

    return logger_instance


# ── Fonctions helpers ─────────────────────────────────────────

logger = setup_logger()


def log_success(msg: str) -> None:
    logger.info(f"✅ {msg}")


def log_warning(msg: str) -> None:
    logger.warning(f"⚠️  {msg}")


def log_error(msg: str) -> None:
    logger.error(f"❌ {msg}")


def log_processing(msg: str) -> None:
    logger.info(f"🔄 {msg}")


def log_section(title: str) -> None:
    sep = "═" * 55
    logger.info(sep)
    logger.info(f"  {title}")
    logger.info(sep)
