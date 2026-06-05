"""
Hyperion V10 — Pipeline Principal (matin 09h00)
Orchestration complète : LONAB → Extraction → Filtrage → Scoring →
Monte Carlo → Consensus → HADES → EV/Kelly → Firebase → Telegram

Point d'entrée : python main.py [--date YYYY-MM-DD] [--test]
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

# Assurer que la racine du projet est dans le path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Utils ─────────────────────────────────────────────────────
from src.utils.config import config
from src.utils.logger import logger, log_success, log_warning, log_error, log_section
from src.utils.gemini_manager import gemini_manager

# ── Modules pipeline ──────────────────────────────────────────
from src.scraper.lonab_scraper import LONABScraper, ScraperStatus
from src.scraper.gemini_extractor import GeminiExtractor
from src.analysis.data_filter import DataFilter
from src.analysis.base_scorer import BaseScorer
from src.analysis.monte_carlo import MonteCarloSimulator
from src.analysis.consensus import InternalConsensusBuilder, ExternalConsensusBuilder, MetaFusion
from src.analysis.hades import HADESDetector
from src.financial.ev_kelly import EVKellyCalculator
from src.output.report_generator import ReportGenerator
from src.output.telegram_bot import TelegramBot
from src.storage.firebase_manager import FirebaseManager


class HyperionV10Pipeline:
    """
    Orchestrateur principal Hyperion V10.
    Pipeline 12 étapes déterministe et résilient.
    """

    def __init__(self):
        log_section("HYPERION V10 — Initialisation")

        # Initialisation des modules
        self.scraper = LONABScraper()
        self.extractor = GeminiExtractor()
        self.data_filter = DataFilter()
        self.scorer = BaseScorer()
        self.monte_carlo = MonteCarloSimulator()
        self.internal_consensus = InternalConsensusBuilder()
        self.external_consensus = ExternalConsensusBuilder()
        self.meta_fusion = MetaFusion()
        self.hades = HADESDetector()
        self.ev_kelly = EVKellyCalculator()
        self.report_gen = ReportGenerator()
        self.telegram = TelegramBot()
        self.firebase = FirebaseManager()

        self.max_courses = config.get("pipeline.max_courses_par_jour", 1)
        self.use_external = config.get("pipeline.use_external_consensus", True)
        self.use_gemini_rapport = config.get("pipeline.generate_rapport_gemini", True)

        log_success("Pipeline V10 prêt")

    # ── Point d'entrée ────────────────────────────────────────

    def run(self, target_date: Optional[date] = None, test_mode: bool = False) -> dict[str, Any]:
        """
        Lance le pipeline complet pour une journée.

        Args:
            target_date: Date cible (défaut : aujourd'hui)
            test_mode: Si True, désactive Telegram et Firebase

        Returns:
            Résumé du run
        """
        start_time = time.time()
        target_date = target_date or date.today()
        date_str = target_date.strftime("%Y-%m-%d")
        date_fr = target_date.strftime("%d/%m/%Y")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_section(f"HYPERION V10 — Pipeline matin {date_fr}")

        summary: dict[str, Any] = {
            "date": date_str,
            "run_id": run_id,
            "started_at": datetime.now().isoformat(),
            "nb_courses": 0,
            "nb_pronostics": 0,
            "errors": [],
            "results": [],
            "test_mode": test_mode,
        }

        # ── Étape 1 : Téléchargement PDF LONAB ────────────────
        logger.info("Étape 1/10 : Téléchargement programme LONAB...")
        scraper_result = self.scraper.get_program_status(
            date=datetime.combine(target_date, datetime.min.time())
        )

        if scraper_result.status == ScraperStatus.UNAVAILABLE:
            msg = f"Scraper LONAB indisponible : {scraper_result.reason}"
            log_error(msg)
            summary["errors"].append(msg)
            if not test_mode:
                self.telegram.send_message_sync(
                    self.report_gen.build_error_message("LONABScraper", scraper_result.reason)
                )
            self._save_summary(summary, target_date)
            return summary

        if scraper_result.status == ScraperStatus.NOT_FOUND:
            logger.info(f"Aucun programme LONAB publié pour {date_fr}")
            if not test_mode:
                self.telegram.send_message_sync(
                    self.report_gen.build_no_courses_message(date_fr)
                )
            self._save_summary(summary, target_date)
            return summary

        # ── Étape 2 : Extraction Gemini ───────────────────────
        logger.info("Étape 2/10 : Extraction des données via Gemini Vision...")
        raw_data = self.extractor.extract_with_fallback(
            scraper_result.pdf_path, max_retries=2
        )

        if not raw_data:
            msg = "Extraction Gemini échouée — données inutilisables"
            log_error(msg)
            summary["errors"].append(msg)
            if not test_mode:
                self.telegram.send_message_sync(
                    self.report_gen.build_error_message("GeminiExtractor", msg)
                )
            self._save_summary(summary, target_date)
            return summary

        # Normaliser les données extraites
        courses = self._normalize_extracted_data(raw_data, date_str)
        if not courses:
            summary["errors"].append("Données extraites invalides après normalisation")
            self._save_summary(summary, target_date)
            return summary

        courses = courses[: self.max_courses]
        summary["nb_courses"] = len(courses)

        if not test_mode:
            self.telegram.send_message_sync(
                self.report_gen.build_start_message(len(courses), date_fr, run_id)
            )

        # ── Analyse de chaque course ──────────────────────────
        all_results: list[dict[str, Any]] = []

        for i, course in enumerate(courses, 1):
            course_id = course.get("id_course", f"R{i}_{date_str}")
            hippodrome = course.get("hippodrome", "?")
            logger.info(f"\n[{i}/{len(courses)}] Analyse : {hippodrome} — {course_id}")

            try:
                result = self._analyse_course(course, course_id)
                if result:
                    all_results.append(result)
                    summary["nb_pronostics"] += 1

                    if not test_mode:
                        # Envoi Telegram
                        messages = self.report_gen.build_course_report(
                            course=course,
                            top5_final=result["top5_final"],
                            hades_result=result["hades"],
                            ev_kelly_data=result["ev_kelly"],
                            elimines=result["elimines"],
                            rapport_gemini=result.get("rapport_gemini"),
                        )
                        self.telegram.send_messages_sync(messages)

                        # Firebase
                        self.firebase.save_prediction(
                            date_str=date_str,
                            course_id=course_id,
                            top5_final=result["top5_final"],
                            hades_result=result["hades"],
                            ev_kelly_data=result["ev_kelly"],
                            course_info=course,
                        )

            except Exception as e:
                err_msg = f"{course_id} : {str(e)[:150]}"
                log_warning(f"Erreur analyse : {err_msg}")
                summary["errors"].append(err_msg)

        # ── Résumé final ──────────────────────────────────────
        duree = time.time() - start_time
        summary["results"] = all_results
        summary["duration_sec"] = round(duree, 1)
        summary["finished_at"] = datetime.now().isoformat()
        summary["quota_status"] = gemini_manager.quota_status

        if not test_mode and all_results:
            self.telegram.send_message_sync(
                self.report_gen.build_daily_summary(all_results, date_fr, duree)
            )

        # Log pipeline run
        if not test_mode:
            self.firebase.save_pipeline_run(date_str, summary)

        log_success(
            f"Pipeline terminé — {len(courses)} cours(es), "
            f"{len(all_results)} pronostic(s), "
            f"{duree:.0f}s, "
            f"{len(summary['errors'])} erreur(s)"
        )

        self._save_summary(summary, target_date)
        return summary

    # ── Analyse d'une course ──────────────────────────────────

    def _analyse_course(
        self, course: dict[str, Any], course_id: str
    ) -> Optional[dict[str, Any]]:
        """
        Analyse complète d'une course (étapes 3 à 10).
        """
        chevaux = course.get("chevaux", [])
        if not chevaux:
            log_warning(f"Aucun cheval pour {course_id}")
            return None

        # Étape 3 : Filtrage
        logger.info(f"  Étape 3 : Filtrage ({len(chevaux)} partants)")
        chevaux_retenus, elimines = self.data_filter.filter_chevaux(chevaux)
        if not chevaux_retenus:
            log_warning(f"Tous les chevaux filtrés pour {course_id}")
            return None

        # Étape 4 : Scoring multicritère
        logger.info(f"  Étape 4 : Scoring ({len(chevaux_retenus)} chevaux)")
        chevaux_scores = self.scorer.score_chevaux(chevaux_retenus, course)

        # Étape 5 : Monte Carlo 50 000 simulations
        logger.info(f"  Étape 5 : Monte Carlo")
        mc_result = self.monte_carlo.simulate(chevaux_scores, course_id=course_id)
        consensus_mc = mc_result["consensus"]
        simulations = mc_result["simulations"]

        # Étape 6 : Consensus interne (Borda)
        logger.info(f"  Étape 6 : Consensus interne (Borda)")
        int_consensus = self.internal_consensus.build(chevaux_scores, course)
        consensus_borda = int_consensus["consensus_borda"]

        # Étape 7 : Consensus externe (optionnel)
        external_aggregation: dict[str, Any] = {}
        if self.use_external:
            logger.info(f"  Étape 7 : Consensus externe (Gemini Search)")
            try:
                external_aggregation = self.external_consensus.collect(course)
            except Exception as e:
                log_warning(f"Consensus externe échoué : {e}")

        # Étape 8 : Méta-fusion
        logger.info(f"  Étape 8 : Méta-fusion")
        top5_final = self.meta_fusion.fuse(
            consensus_mc=consensus_mc,
            consensus_borda=consensus_borda,
            external_aggregation=external_aggregation,
            simulations=simulations,
        )

        # Enrichir top5 avec données originales (cote, forme, etc.)
        ch_by_num = {int(ch["numero"]): ch for ch in chevaux}
        for ch in top5_final:
            orig = ch_by_num.get(int(ch["numero"]), {})
            for key in ("cote", "forme", "age", "poids", "jockey"):
                if key not in ch or ch[key] is None:
                    ch[key] = orig.get(key)

        # Étape 9 : HADES
        logger.info(f"  Étape 9 : Analyse HADES")
        hades_result = self.hades.analyze(chevaux, consensus_mc, external_aggregation)

        # Étape 10 : EV/Kelly
        logger.info(f"  Étape 10 : EV/Kelly")
        ev_kelly_data = self.ev_kelly.calculate_all(top5_final, chevaux)

        # Rapport Gemini (facultatif)
        rapport_gemini: Optional[str] = None
        if self.use_gemini_rapport:
            try:
                confiance = self._compute_confiance(top5_final)
                rapport_gemini = self.report_gen.generate_gemini_narrative(
                    course=course,
                    top5_final=top5_final,
                    hades_result=hades_result,
                    elimines=elimines,
                    confiance=confiance,
                )
            except Exception as e:
                log_warning(f"Rapport Gemini échoué : {e}")

        return {
            "course": course,
            "top5_final": top5_final,
            "elimines": elimines,
            "hades": hades_result,
            "ev_kelly": ev_kelly_data,
            "rapport_gemini": rapport_gemini,
            "mc_result": {
                "is_robust": mc_result.get("is_robust"),
                "stability": mc_result.get("stability"),
                "total_simulations": mc_result.get("total_simulations"),
            },
            "generated_at": datetime.now().isoformat(),
        }

    # ── Normalisation données Gemini ──────────────────────────

    def _normalize_extracted_data(
        self, raw_data: dict[str, Any], date_str: str
    ) -> list[dict[str, Any]]:
        """
        Normalise les données extraites par GeminiExtractor en
        liste de courses avec structure uniforme.
        """
        if not isinstance(raw_data, dict):
            return []

        course_meta = raw_data.get("course", {})
        partants = raw_data.get("partants", [])

        if not partants:
            return []

        # Convertir partants en format interne
        chevaux = []
        for p in partants:
            chevaux.append({
                "numero": p.get("numero"),
                "nom": p.get("nom", f"Cheval {p.get('numero', 0)}"),
                "age": p.get("age"),
                "sexe": p.get("sexe"),
                "poids": p.get("poids"),
                "corde": p.get("corde"),
                "forme": p.get("forme"),
                "gains_totaux": p.get("gains_totaux"),
                "jockey": p.get("jockey"),
                "entraineur": p.get("entraineur"),
                "proprietaire": p.get("proprietaire"),
                "cote": p.get("cote"),
            })

        course = {
            "id_course": course_meta.get("nom", f"Course_{date_str}"),
            "hippodrome": course_meta.get("hippodrome", "LONAB"),
            "date": course_meta.get("date", date_str),
            "heure": course_meta.get("heure"),
            "distance": course_meta.get("distance"),
            "terrain": course_meta.get("terrain"),
            "discipline": course_meta.get("discipline"),
            "nb_partants": len(chevaux),
            "chevaux": chevaux,
            "is_lonab": True,
        }

        return [course]

    # ── Utilitaires ───────────────────────────────────────────

    def _compute_confiance(self, top5: list[dict[str, Any]]) -> float:
        """Indice de confiance basé sur l'écart entre #1 et #2."""
        if not top5 or len(top5) < 2:
            return 50.0
        s1 = float(top5[0].get("meta_score", top5[0].get("win_prob", 0)))
        s2 = float(top5[1].get("meta_score", top5[1].get("win_prob", 0)))
        return round(max(0.0, min(100.0, 50.0 + (s1 - s2) * 200)), 1)

    def _save_summary(self, summary: dict[str, Any], target_date: date) -> None:
        """Sauvegarde locale du résumé du run."""
        try:
            out_dir = Path("outputs")
            out_dir.mkdir(exist_ok=True)
            filename = out_dir / f"hyperion_{target_date.strftime('%Y%m%d')}.json"

            light = {
                "date": summary["date"],
                "run_id": summary.get("run_id"),
                "duration_sec": summary.get("duration_sec"),
                "nb_courses": summary["nb_courses"],
                "nb_pronostics": summary["nb_pronostics"],
                "errors": summary["errors"],
                "quota_status": summary.get("quota_status", {}),
                "top5_par_course": [
                    {
                        "course": r["course"].get("id_course"),
                        "hippodrome": r["course"].get("hippodrome"),
                        "top5": [
                            f"N°{ch['numero']} {ch.get('nom', '?')}"
                            for ch in r["top5_final"][:5]
                        ],
                        "nb_value_bets": r["ev_kelly"].get("nb_value_bets", 0),
                        "hades": r["hades"].get("niveau_global", "green"),
                        "is_robust": r["mc_result"].get("is_robust", False),
                    }
                    for r in summary.get("results", [])
                ],
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(light, f, ensure_ascii=False, indent=2)
            log_success(f"Résumé sauvegardé : {filename}")

        except Exception as e:
            log_warning(f"Sauvegarde résumé échouée : {e}")


# ── CLI ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperion V10 — Pipeline matin")
    parser.add_argument("--date", type=str, help="Date cible YYYY-MM-DD (défaut : aujourd'hui)")
    parser.add_argument("--test", action="store_true", help="Mode test (sans Telegram/Firebase)")
    args = parser.parse_args()

    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"Format de date invalide : {args.date} (attendu YYYY-MM-DD)", file=sys.stderr)
            sys.exit(1)
    else:
        target_date = date.today()

    # Créer les dossiers nécessaires
    for d in ["data/cache", "data/backups", "logs", "outputs", "backup/telegram"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    pipeline = HyperionV10Pipeline()
    result = pipeline.run(target_date=target_date, test_mode=args.test)

    # Code de sortie : 1 si aucun pronostic ET des erreurs
    if result["errors"] and result["nb_pronostics"] == 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
