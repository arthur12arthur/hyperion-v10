"""
Hyperion V10 — GeminiExtractor
Extraction structurée des données course depuis PDF via Gemini Vision.

Utilise GeminiManager pour la rotation automatique des clés.
SDK : google-genai (nouveau, GA mai 2025)
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from ..utils.config import config
from ..utils.logger import logger, log_success, log_error, log_warning, log_processing
from ..utils.gemini_manager import gemini_manager


_EXTRACTION_PROMPT = """
Tu es un expert en extraction de données hippiques depuis des programmes PMU/LONAB.

Analyse ce document PDF et extrais TOUTES les informations de la course principale du jour.

INSTRUCTIONS CRITIQUES :
1. Retourne UNIQUEMENT un objet JSON valide, sans texte avant ou après
2. Pas de blocs markdown (pas de ```json)
3. Pas de commentaires dans le JSON
4. Si une information est absente : mets null (jamais de chaîne vide)
5. Les cotes : format décimal — "5/1" → 6.0, "3/2" → 2.5, "Evens" → 2.0
6. La forme : chaîne brute ex "1p2p3p4a5p" (p=place, a=arrêté, D=disqualifié, T=tombé)
7. Les gains en FCFA
8. Extrais TOUS les partants sans exception
9. JSON complet jusqu'au dernier partant et la fermeture des accolades

FORMAT JSON ATTENDU :
{
  "course": {
    "nom": "Nom complet de la course",
    "hippodrome": "Nom hippodrome",
    "date": "YYYY-MM-DD",
    "heure": "HH:MM",
    "distance": 2700,
    "terrain": "BON|SOUPLE|LOURD|PSF",
    "discipline": "TROT|GALOP|ATTELE|MONTE",
    "nb_partants": 12
  },
  "partants": [
    {
      "numero": 1,
      "nom": "NOM_CHEVAL",
      "age": 5,
      "sexe": "M",
      "poids": 58.5,
      "corde": 1,
      "forme": "1p2p3p4a5p",
      "gains_totaux": 125000,
      "jockey": "NOM Prénom",
      "entraineur": "NOM Prénom",
      "proprietaire": "Nom",
      "cote": 3.5
    }
  ]
}

RÈGLES COTES : "2/1"→3.0, "5/2"→3.5, "3/1"→4.0, "10/1"→11.0, "Evens"→2.0

Extrait maintenant et retourne UNIQUEMENT le JSON.
"""


class GeminiExtractor:
    """
    Extracteur de données hippiques via Gemini Vision.
    Délègue les appels à GeminiManager pour la gestion des clés.
    """

    def __init__(self):
        self._temperature = config.get("gemini.extraction.temperature", 0.1)
        self._max_tokens = config.get("gemini.extraction.max_output_tokens", 16000)
        logger.info("✅ GeminiExtractor initialisé (via GeminiManager)")

    # ── API publique ──────────────────────────────────────────

    def extract_with_fallback(
        self,
        pdf_path: Path,
        max_retries: int = 2,
    ) -> Optional[dict[str, Any]]:
        """
        Extrait les données d'un PDF avec retry automatique.

        Args:
            pdf_path: Chemin du PDF programme LONAB
            max_retries: Nombre de tentatives

        Returns:
            Dict données structurées ou None si échec
        """
        for attempt in range(1, max_retries + 1):
            logger.info(f"🔄 Tentative extraction {attempt}/{max_retries} : {pdf_path.name}")

            try:
                data = self._extract_once(pdf_path)
                if data and self._validate(data):
                    log_success(f"Données extraites : {len(data.get('partants', []))} partants")
                    return data
                log_warning(f"Données invalides à la tentative {attempt}")

            except Exception as e:
                log_warning(f"Échec tentative {attempt} : {e}")

            if attempt < max_retries:
                wait = config.get("gemini.rate_limit.retry_after_seconds", 10)
                log_warning(f"Nouvelle tentative dans {wait}s...")
                time.sleep(wait)

        log_error(f"Extraction échouée après {max_retries} tentatives")
        return None

    # ── Extraction ────────────────────────────────────────────

    def _extract_once(self, pdf_path: Path) -> Optional[dict[str, Any]]:
        """Effectue une extraction complète (upload + appel + parse)."""
        log_processing(f"Upload PDF vers Gemini : {pdf_path.name}")

        # Upload du fichier
        uploaded = gemini_manager.upload_file(str(pdf_path), mime_type="application/pdf")
        log_success("PDF uploadé et traité par Gemini Files API")

        try:
            # Appel extraction avec le fichier uploadé en contenu supplémentaire
            response_text = gemini_manager.call(
                prompt=_EXTRACTION_PROMPT,
                model=config.get("gemini.model_primary", "gemini-2.5-flash"),
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
                extra_contents=[uploaded],
            )

            if not response_text:
                log_error("Réponse Gemini vide pour l'extraction")
                return None

            return self._parse_json(response_text)

        finally:
            # Toujours nettoyer le fichier uploadé
            gemini_manager.delete_file(uploaded)

    # ── Parsing JSON ──────────────────────────────────────────

    def _parse_json(self, text: str) -> Optional[dict[str, Any]]:
        """
        Parse la réponse Gemini en JSON avec 5 tentatives de nettoyage.
        """
        original = text.strip()

        # Tentative 1 : JSON direct
        try:
            return json.loads(original)
        except json.JSONDecodeError:
            pass

        # Tentative 2 : Retirer blocs markdown
        cleaned = re.sub(r"```(?:json)?\s*", "", original)
        cleaned = re.sub(r"```", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Tentative 3 : Supprimer commentaires // et /* */
        cleaned2 = re.sub(r"//[^\n]*", "", cleaned)
        cleaned2 = re.sub(r"/\*.*?\*/", "", cleaned2, flags=re.DOTALL)
        try:
            return json.loads(cleaned2)
        except json.JSONDecodeError:
            pass

        # Tentative 4 : Supprimer virgules trailing
        cleaned3 = re.sub(r",\s*([}\]])", r"\1", cleaned2)
        try:
            return json.loads(cleaned3)
        except json.JSONDecodeError:
            pass

        # Tentative 5 : Extraire premier bloc JSON complet
        match = re.search(r"\{.*\}", cleaned3, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        log_error("Impossible de parser le JSON Gemini après 5 tentatives")
        logger.debug(f"Début réponse : {original[:300]}")
        logger.debug(f"Fin réponse : {original[-200:]}")
        return None

    # ── Validation ────────────────────────────────────────────

    def _validate(self, data: dict[str, Any]) -> bool:
        """Valide la structure minimale des données extraites."""
        if not isinstance(data, dict):
            log_error("Les données ne sont pas un dictionnaire")
            return False

        if "course" not in data or "partants" not in data:
            log_error("Structure invalide : 'course' ou 'partants' manquant")
            return False

        course = data["course"]
        partants = data["partants"]

        # Champs course obligatoires
        for field_name in ["nom", "hippodrome", "date", "nb_partants"]:
            if not course.get(field_name):
                log_error(f"Champ course manquant : {field_name}")
                return False

        # Minimum 3 partants
        if len(partants) < 3:
            log_error(f"Trop peu de partants : {len(partants)}")
            return False

        # Cohérence nb_partants
        if len(partants) != course.get("nb_partants", 0):
            log_warning(
                f"Incohérence nb_partants : annoncé {course.get('nb_partants')}, "
                f"extrait {len(partants)} — correction automatique"
            )
            data["course"]["nb_partants"] = len(partants)

        # Champs partants obligatoires
        for i, p in enumerate(partants):
            if not p.get("numero") or not p.get("nom"):
                log_error(f"Partant #{i+1} : numéro ou nom manquant")
                return False

        # Numéros uniques
        nums = [p.get("numero") for p in partants]
        if len(nums) != len(set(nums)):
            log_error("Numéros de partants non uniques")
            return False

        return True
