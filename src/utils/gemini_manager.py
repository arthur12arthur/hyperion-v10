"""
Hyperion V10 — GeminiManager : rotation automatique de 2 clés Gemini.
Utilise le nouveau SDK google-genai (GA depuis mai 2025).

SDK : pip install google-genai
Import : from google import genai
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import config
from .logger import log_warning, log_error, logger


@dataclass
class GeminiKey:
    """État d'une clé Gemini."""
    key_id: str
    value: str
    calls_today: int = 0
    active: bool = True
    last_error: Optional[str] = None
    last_error_time: float = 0.0


class GeminiManager:
    """
    Gestionnaire de rotation automatique des clés Gemini.
    Transparent pour tous les modules — ils appellent call() sans
    connaître la clé active.

    Gère :
    - Rotation sur clé 2 si clé 1 épuise son quota (429)
    - Reset quotidien automatique
    - Mode template statique si les deux clés sont KO

    Utilise le nouveau SDK : from google import genai
    """

    def __init__(self):
        # Charger les deux clés depuis l'environnement
        key1 = os.environ.get("GEMINI_API_KEY_1") or os.environ.get("GEMINI_API_KEY")
        key2 = os.environ.get("GEMINI_API_KEY_2")

        self._keys: list[GeminiKey] = []

        if key1:
            self._keys.append(GeminiKey(key_id="KEY1", value=key1))
        if key2 and key2 != key1:
            self._keys.append(GeminiKey(key_id="KEY2", value=key2))

        if not self._keys:
            raise ValueError(
                "Aucune clé Gemini trouvée. "
                "Définissez GEMINI_API_KEY_1 (et optionnellement GEMINI_API_KEY_2)."
            )

        self._current_idx: int = 0
        self._model_primary: str = config.get("gemini.model_primary", "gemini-2.5-flash")
        self._rpm: int = config.get("gemini.rate_limit.requests_per_minute", 8)
        self._max_retries: int = config.get("gemini.rate_limit.max_retries", 3)
        self._retry_after: int = config.get("gemini.rate_limit.retry_after_seconds", 10)
        self._last_call_time: float = 0.0

        # Import du nouveau SDK ici pour un message d'erreur clair si absent
        try:
            from google import genai
            from google.genai import types as genai_types
            self._genai = genai
            self._genai_types = genai_types
        except ImportError:
            raise ImportError(
                "Le nouveau SDK Google GenAI n'est pas installé. "
                "Exécutez : pip install google-genai"
            )

        logger.info(f"✅ GeminiManager initialisé — {len(self._keys)} clé(s) — modèle : {self._model_primary}")

    # ── Propriétés ────────────────────────────────────────────

    @property
    def _current_key(self) -> GeminiKey:
        return self._keys[self._current_idx]

    @property
    def quota_status(self) -> dict:
        """Résumé du statut quota pour le rapport santé."""
        return {
            k.key_id: {
                "calls_today": k.calls_today,
                "active": k.active,
                "last_error": k.last_error,
            }
            for k in self._keys
        }

    # ── Client ────────────────────────────────────────────────

    def _get_client(self, key_value: str):
        """Crée un client genai avec la clé donnée."""
        return self._genai.Client(api_key=key_value)

    # ── Appel principal ───────────────────────────────────────

    def call(
        self,
        prompt: Any,
        model: Optional[str] = None,
        temperature: float = 0.4,
        max_output_tokens: int = 2000,
        extra_contents: Optional[list] = None,
        use_search: bool = False,
    ) -> Optional[str]:
        """
        Appelle Gemini avec rotation automatique si quota épuisé.

        Args:
            prompt: Texte du prompt (str) ou contenu Gemini
            model: Modèle à utiliser (défaut : config)
            temperature: Température
            max_output_tokens: Tokens max
            extra_contents: Contenus supplémentaires (ex: fichier uploadé)
            use_search: Active Google Search grounding

        Returns:
            Texte de la réponse ou None si échec complet
        """
        target_model = model or self._model_primary

        for attempt in range(self._max_retries * len(self._keys)):
            key = self._current_key

            if not key.active:
                if not self._switch_key("Clé désactivée"):
                    break
                continue

            try:
                self._rate_limit_wait()
                result = self._do_call(
                    key=key,
                    prompt=prompt,
                    model=target_model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    extra_contents=extra_contents,
                    use_search=use_search,
                )
                key.calls_today += 1
                self._last_call_time = time.time()
                return result

            except Exception as e:
                err_str = str(e)
                is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                is_unavailable = "503" in err_str or "UNAVAILABLE" in err_str

                if is_quota:
                    log_warning(f"[GeminiManager] Quota épuisé sur {key.key_id}")
                    key.active = False
                    key.last_error = "QUOTA_EXHAUSTED"
                    key.last_error_time = time.time()

                    if not self._switch_key("Quota épuisé"):
                        log_error("[GeminiManager] Toutes les clés épuisées — mode statique")
                        return None

                elif is_unavailable:
                    wait = self._retry_after * (attempt + 1)
                    log_warning(f"[GeminiManager] Service indisponible, attente {wait}s")
                    time.sleep(wait)

                else:
                    log_error(f"[GeminiManager] Erreur inattendue ({key.key_id}) : {e}")
                    # Erreur non-quota : ne pas tourner, juste retenter
                    if attempt < self._max_retries - 1:
                        time.sleep(self._retry_after)
                    else:
                        return None

        return None

    def _do_call(
        self,
        key: GeminiKey,
        prompt: Any,
        model: str,
        temperature: float,
        max_output_tokens: int,
        extra_contents: Optional[list],
        use_search: bool,
    ) -> str:
        """Effectue l'appel Gemini réel avec le nouveau SDK."""
        client = self._get_client(key.value)

        generation_config = self._genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        # Construire les contenus
        if extra_contents:
            contents = extra_contents + [prompt]
        else:
            contents = prompt

        # Ajout Google Search si demandé
        if use_search:
            generation_config = self._genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tools=[self._genai_types.Tool(google_search=self._genai_types.GoogleSearch())],
            )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generation_config,
        )

        # Extraction texte robuste (le tool use peut supprimer response.text)
        text = self._extract_text(response)
        if not text:
            raise ValueError("Réponse Gemini vide ou sans texte")
        return text

    def _extract_text(self, response) -> Optional[str]:
        """Extrait le texte d'une réponse Gemini de manière robuste."""
        # Cas simple
        try:
            if response.text and response.text.strip():
                return response.text.strip()
        except Exception:
            pass

        # Parcours des candidates → parts (cas tool_use)
        try:
            for candidate in (response.candidates or []):
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                parts_text = []
                for part in (getattr(content, "parts", None) or []):
                    t = getattr(part, "text", None)
                    if t and t.strip():
                        parts_text.append(t.strip())
                if parts_text:
                    return "\n".join(parts_text)
        except Exception:
            pass

        return None

    # ── Upload fichier ─────────────────────────────────────────

    def upload_file(self, file_path: str, mime_type: str = "application/pdf"):
        """
        Upload un fichier vers Gemini Files API.

        Returns:
            Objet fichier uploadé (à passer dans extra_contents)
        """
        key = self._current_key
        client = self._get_client(key.value)

        self._rate_limit_wait()

        with open(file_path, "rb") as f:
            uploaded = client.files.upload(
                file=f,
                config=self._genai_types.UploadFileConfig(mime_type=mime_type),
            )

        # Attendre le traitement
        import time as _time
        max_wait = 60
        waited = 0
        while getattr(uploaded, "state", None) and uploaded.state.name == "PROCESSING":
            _time.sleep(2)
            waited += 2
            uploaded = client.files.get(name=uploaded.name)
            if waited > max_wait:
                raise TimeoutError(f"PDF toujours en traitement après {max_wait}s")

        if getattr(uploaded, "state", None) and uploaded.state.name == "FAILED":
            raise RuntimeError("Traitement PDF échoué par Gemini Files API")

        self._last_call_time = time.time()
        key.calls_today += 1
        return uploaded

    def delete_file(self, uploaded_file) -> None:
        """Supprime un fichier uploadé pour libérer le quota Files API."""
        try:
            key = self._current_key
            client = self._get_client(key.value)
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass  # Non critique

    # ── Rotation clés ─────────────────────────────────────────

    def _switch_key(self, reason: str) -> bool:
        """
        Bascule sur la prochaine clé active.

        Returns:
            True si une clé active a été trouvée, False si toutes épuisées
        """
        prev_id = self._current_key.key_id
        start = self._current_idx

        for i in range(1, len(self._keys) + 1):
            next_idx = (start + i) % len(self._keys)
            if self._keys[next_idx].active:
                self._current_idx = next_idx
                logger.info(
                    f"🔄 [GeminiManager] Rotation : {prev_id} → {self._keys[next_idx].key_id} ({reason})"
                )
                return True

        return False  # Toutes les clés KO

    def reset_daily_quota(self) -> None:
        """Réinitialise les compteurs quotidiens (à appeler à minuit)."""
        for key in self._keys:
            key.calls_today = 0
            key.active = True
            key.last_error = None
        logger.info("🔄 [GeminiManager] Quota journalier réinitialisé")

    # ── Rate limiting ──────────────────────────────────────────

    def _rate_limit_wait(self) -> None:
        """Attend si nécessaire pour respecter les RPM."""
        min_interval = 60.0 / self._rpm
        elapsed = time.time() - self._last_call_time
        if elapsed < min_interval:
            wait = min_interval - elapsed
            logger.debug(f"[GeminiManager] Rate limit : attente {wait:.1f}s")
            time.sleep(wait)


# Instance globale
gemini_manager = GeminiManager.__new__(GeminiManager)
gemini_manager._initialized = False
