"""
Hyperion V10 — FirebaseManager
Stockage Firestore des prédictions, résultats et évaluations.
Fallback automatique vers JSON local si Firebase KO.

SDK : firebase-admin 7.1.0 (fév 2026)
Import : import firebase_admin
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..utils.config import config
from ..utils.logger import logger, log_success, log_warning, log_error


class FirebaseManager:
    """
    Gestionnaire Firebase Firestore avec fallback JSON local.

    Collections Firestore :
    - predictions/{date}/{course_id}  → prédictions du matin
    - results/{date}/{course_id}      → résultats officiels du soir
    - evaluations/{date}              → scores J/30
    - pipeline_runs/{date}            → logs techniques
    - hades_analysis/{date}           → alertes HADES
    """

    def __init__(self):
        self._db = None
        self._initialized = False
        self._backup_dir = config.get_path("backup")
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._try_init_firebase()

    def _try_init_firebase(self) -> None:
        """Tente l'initialisation Firebase. Passe en mode local si impossible."""
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            # Credentials depuis variable d'environnement (JSON string)
            creds_json = os.environ.get("FIREBASE_CREDENTIALS")
            if creds_json:
                creds_dict = json.loads(creds_json)
                cred = credentials.Certificate(creds_dict)
            else:
                # Fichier local (développement)
                creds_path = config.get_path("config/firebase_credentials.json")
                if not creds_path.exists():
                    log_warning("Firebase : credentials introuvables — mode local uniquement")
                    return
                cred = credentials.Certificate(str(creds_path))

            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)

            self._db = firestore.client()
            self._initialized = True
            log_success("Firebase Firestore initialisé")

        except ImportError:
            log_warning("firebase-admin non installé — mode local uniquement")
        except Exception as e:
            log_warning(f"Firebase init échoué : {e} — mode local uniquement")

    # ── Sauvegarde prédictions ────────────────────────────────

    def save_prediction(
        self,
        date_str: str,
        course_id: str,
        top5_final: list[dict[str, Any]],
        hades_result: dict[str, Any],
        ev_kelly_data: dict[str, Any],
        course_info: dict[str, Any],
    ) -> bool:
        """
        Sauvegarde la prédiction du matin.
        Retourne True si succès (Firebase ou local).
        """
        doc = {
            "date": date_str,
            "course_id": course_id,
            "hippodrome": course_info.get("hippodrome", "?"),
            "is_lonab": course_info.get("is_lonab", True),
            "predicted_top5": [ch.get("numero") for ch in top5_final],
            "predicted_winner": top5_final[0].get("numero") if top5_final else None,
            "top5_details": [
                {
                    "numero": ch.get("numero"),
                    "nom": ch.get("nom"),
                    "meta_score": ch.get("meta_score"),
                    "win_prob": ch.get("win_prob"),
                    "robuste": ch.get("robuste"),
                }
                for ch in top5_final
            ],
            "hades_niveau": hades_result.get("niveau_global", "green"),
            "nb_value_bets": ev_kelly_data.get("nb_value_bets", 0),
            "generated_at": datetime.now().isoformat(),
        }

        # Firebase
        if self._initialized and self._db:
            try:
                self._db.collection("predictions").document(date_str).collection("courses").document(
                    course_id
                ).set(doc)
                log_success(f"Prédiction sauvegardée Firestore : {course_id}")
                return True
            except Exception as e:
                log_warning(f"Firebase write échoué : {e} — fallback local")

        # Fallback local
        return self._save_local("predictions", date_str, course_id, doc)

    # ── Sauvegarde résultats (soir) ───────────────────────────

    def save_result(
        self,
        date_str: str,
        course_id: str,
        ordre_arrivee: list[int],
        gagnant_numero: int,
        gagnant_nom: str,
        top3: list[int],
    ) -> bool:
        """Sauvegarde le résultat officiel PMU."""
        doc = {
            "date": date_str,
            "course_id": course_id,
            "ordre_arrivee": ordre_arrivee,
            "gagnant_numero": gagnant_numero,
            "gagnant_nom": gagnant_nom,
            "top3": top3,
            "saved_at": datetime.now().isoformat(),
        }

        if self._initialized and self._db:
            try:
                self._db.collection("results").document(date_str).collection("courses").document(
                    course_id
                ).set(doc)
                log_success(f"Résultat sauvegardé Firestore : {course_id}")
                return True
            except Exception as e:
                log_warning(f"Firebase write résultat échoué : {e} — fallback local")

        return self._save_local("results", date_str, course_id, doc)

    # ── Sauvegarde évaluation J/30 ────────────────────────────

    def save_evaluation(
        self,
        date_str: str,
        day_number: int,
        score_top1: float,
        score_top3: float,
        running_top1: float,
        running_top3: float,
        details: dict[str, Any],
    ) -> bool:
        """Sauvegarde le score d'évaluation quotidien."""
        doc = {
            "date": date_str,
            "day_number": day_number,
            "score_top1_jour": round(score_top1, 4),
            "score_top3_jour": round(score_top3, 4),
            "running_top1": round(running_top1, 4),
            "running_top3": round(running_top3, 4),
            "details": details,
            "saved_at": datetime.now().isoformat(),
        }

        if self._initialized and self._db:
            try:
                self._db.collection("evaluations").document(date_str).set(doc)
                log_success(f"Évaluation J{day_number} sauvegardée Firestore")
                return True
            except Exception as e:
                log_warning(f"Firebase write évaluation échoué : {e} — fallback local")

        return self._save_local("evaluations", date_str, "eval", doc)

    # ── Lecture prédictions ───────────────────────────────────

    def load_predictions(self, date_str: str) -> dict[str, Any]:
        """
        Charge les prédictions d'une date donnée.
        Retourne dict {course_id: doc} ou {} si rien trouvé.
        """
        if self._initialized and self._db:
            try:
                courses_ref = (
                    self._db.collection("predictions").document(date_str).collection("courses")
                )
                docs = courses_ref.stream()
                result = {}
                for doc in docs:
                    result[doc.id] = doc.to_dict()
                if result:
                    logger.info(f"Prédictions chargées Firestore : {len(result)} courses")
                    return result
            except Exception as e:
                log_warning(f"Firebase read prédictions échoué : {e} — fallback local")

        # Fallback local
        return self._load_local("predictions", date_str)

    def load_results(self, date_str: str) -> dict[str, Any]:
        """Charge les résultats d'une date."""
        if self._initialized and self._db:
            try:
                courses_ref = (
                    self._db.collection("results").document(date_str).collection("courses")
                )
                docs = courses_ref.stream()
                result = {}
                for doc in docs:
                    result[doc.id] = doc.to_dict()
                if result:
                    return result
            except Exception as e:
                log_warning(f"Firebase read résultats échoué : {e} — fallback local")

        return self._load_local("results", date_str)

    def load_running_scores(self) -> dict[str, Any]:
        """Charge les scores cumulés J/30."""
        if self._initialized and self._db:
            try:
                doc = self._db.collection("evaluations").document("_running_scores").get()
                if doc.exists:
                    return doc.to_dict() or {}
            except Exception as e:
                log_warning(f"Firebase read scores cumulés échoué : {e}")

        # Fallback local
        path = self._backup_dir / "running_scores.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"running_top1": 0.0, "running_top3": 0.0, "days_evaluated": 0}

    def save_running_scores(self, data: dict[str, Any]) -> bool:
        """Sauvegarde les scores cumulés."""
        if self._initialized and self._db:
            try:
                self._db.collection("evaluations").document("_running_scores").set(data)
                return True
            except Exception as e:
                log_warning(f"Firebase write scores cumulés échoué : {e}")

        path = self._backup_dir / "running_scores.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True

    # ── Log pipeline ──────────────────────────────────────────

    def save_pipeline_run(self, date_str: str, run_data: dict[str, Any]) -> None:
        """Log le run pipeline (monitoring)."""
        if self._initialized and self._db:
            try:
                self._db.collection("pipeline_runs").document(date_str).set(run_data)
                return
            except Exception as e:
                log_warning(f"Firebase write pipeline_run échoué : {e}")
        self._save_local("pipeline_runs", date_str, "run", run_data)

    # ── Fallback local JSON ───────────────────────────────────

    def _save_local(self, collection: str, date_str: str, doc_id: str, data: dict) -> bool:
        """Sauvegarde JSON locale de secours."""
        try:
            folder = self._backup_dir / collection / date_str
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{doc_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Backup local : {path}")
            return True
        except Exception as e:
            log_error(f"Impossible de sauvegarder localement : {e}")
            return False

    def _load_local(self, collection: str, date_str: str) -> dict[str, Any]:
        """Charge les JSONs locaux de secours."""
        folder = self._backup_dir / collection / date_str
        if not folder.exists():
            return {}
        result = {}
        for path in folder.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    result[path.stem] = json.load(f)
            except Exception:
                pass
        return result

    @property
    def is_online(self) -> bool:
        return self._initialized
