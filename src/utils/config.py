"""
Hyperion V10 — Gestionnaire de configuration centralisé
Supporte les clés imbriquées via notation pointée.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional


class ConfigManager:
    """Gestionnaire de configuration centralisé — singleton."""

    _instance: Optional["ConfigManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._config: dict = {}
        self._sources: dict = {}
        self._load_all()
        self._initialized = True

    # ── Chargement ───────────────────────────────────────────

    def _load_all(self) -> None:
        """Charge config.yaml + sources.yaml depuis la racine projet."""
        root = self._project_root()
        self._config = self._load_yaml(root / "config" / "config.yaml")
        self._sources = self._load_yaml(root / "config" / "sources.yaml")

    def _load_yaml(self, path: Path) -> dict:
        if not path.exists():
            print(f"[CONFIG] ⚠️  Fichier introuvable : {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _project_root(self) -> Path:
        """Remonte jusqu'à la racine du projet (contient config/)."""
        here = Path(__file__).resolve().parent
        for candidate in [here, here.parent, here.parent.parent, here.parent.parent.parent]:
            if (candidate / "config" / "config.yaml").exists():
                return candidate
        # Fallback : répertoire courant
        return Path.cwd()

    # ── Accesseurs ───────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur via notation pointée.
        Ex : config.get('gemini.model_primary')
        """
        parts = key.split(".")
        value = self._config
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

    def get_sources(self) -> dict:
        """Retourne la configuration sources.yaml complète."""
        return self._sources

    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Récupère une variable d'environnement."""
        return os.environ.get(key, default)

    def get_path(self, *parts: str) -> Path:
        """
        Construit un Path absolu depuis la racine projet.
        Ex : config.get_path('data', 'cache') → /projet/data/cache
        """
        root = self._project_root()
        if len(parts) == 1 and Path(parts[0]).is_absolute():
            return Path(parts[0])
        if len(parts) == 1:
            # Accepte 'data/cache' ou Path('data/cache')
            return root / parts[0]
        return root.joinpath(*parts)

    def reload(self) -> None:
        """Recharge la configuration (utile en tests)."""
        self._load_all()

    # ── Raccourcis pratiques ──────────────────────────────────

    @property
    def version(self) -> str:
        return self.get("hyperion.version", "10.0.0")

    @property
    def gemini_model(self) -> str:
        return self.get("gemini.model_primary", "gemini-2.5-flash")

    @property
    def telegram_enabled(self) -> bool:
        token = self.get_env("TELEGRAM_BOT_TOKEN")
        chat = self.get_env("TELEGRAM_CHAT_ID")
        return bool(token and chat)


# Instance globale unique
config = ConfigManager()
