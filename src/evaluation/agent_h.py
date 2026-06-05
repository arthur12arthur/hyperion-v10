"""
Hyperion V10 — AgentH (Auto-Évaluation)
Pipeline soir 20h00 : scraping résultats PMU → comparaison → score J/30.
Nouveau module absent dans les versions précédentes.
"""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup

from ..utils.config import config
from ..utils.logger import logger, log_success, log_warning, log_error
from ..utils.gemini_manager import gemini_manager
from ..storage.firebase_manager import FirebaseManager
from ..output.telegram_bot import TelegramBot


@dataclass
class CourseEvaluation:
    course_id: str
    is_lonab: bool
    predicted_winner: Optional[int]
    official_winner: Optional[int]
    predicted_top3: list[int]
    official_top3: list[int]
    top1_correct: Optional[bool] = None
    top3_score: int = 0  # 0-3

    def compute(self) -> None:
        if self.predicted_winner and self.official_winner:
            self.top1_correct = self.predicted_winner == self.official_winner
        self.top3_score = len(set(self.predicted_top3) & set(self.official_top3))


@dataclass
class DayEvaluation:
    date_str: str
    day_number: int
    courses: list[CourseEvaluation] = field(default_factory=list)
    score_top1_jour: float = 0.0
    score_top3_jour: float = 0.0
    running_top1: float = 0.0
    running_top3: float = 0.0
    lonab_top1_correct: Optional[bool] = None


class AgentH:
    """
    Agent d'auto-évaluation quotidienne (pipeline soir 20h00).

    Étapes :
    1. Charger prédictions matin depuis Firebase
    2. Scraper résultats officiels PMU.fr
    3. Comparer et calculer scores
    4. Mettre à jour scores cumulés J/30
    5. Envoyer rapport soir Telegram
    """

    RESULTATS_URL = "https://www.pmu.fr/turf/today/resultats"

    def __init__(self):
        self.firebase = FirebaseManager()
        self.telegram = TelegramBot()
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "fr-FR,fr;q=0.9",
        }
        logger.info("✅ AgentH (auto-évaluation) initialisé")

    # ── API publique ──────────────────────────────────────────

    def run_evening_pipeline(self, date_str: Optional[str] = None) -> bool:
        """
        Lance le pipeline complet d'évaluation du soir.

        Args:
            date_str: Date YYYY-MM-DD (défaut : aujourd'hui)

        Returns:
            True si succès, False si échec
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"📊 AgentH — Évaluation du soir : {date_str}")

        # 1. Charger prédictions
        predictions = self.firebase.load_predictions(date_str)
        if not predictions:
            log_warning(f"Aucune prédiction trouvée pour {date_str}")
            self.telegram.send_message_sync(
                f"ℹ️ <b>HYPERION V10</b>\nAucune prédiction à évaluer pour le {date_str}."
            )
            return False

        # 2. Scraper résultats (avec retry)
        results = self._scrape_results_with_retry(date_str, list(predictions.keys()))
        if not results:
            log_warning("Résultats PMU indisponibles — évaluation reportée")
            self.telegram.send_message_sync(
                f"⚠️ <b>HYPERION V10</b>\n"
                f"Résultats PMU non disponibles pour le {date_str}.\n"
                "Évaluation reportée."
            )
            return False

        # 3. Scores cumulés précédents
        running = self.firebase.load_running_scores()
        day_number = int(running.get("days_evaluated", 0)) + 1

        # 4. Évaluer
        day_eval = self._evaluate(date_str, day_number, predictions, results, running)

        # 5. Sauvegarder
        self.firebase.save_evaluation(
            date_str=date_str,
            day_number=day_number,
            score_top1=day_eval.score_top1_jour,
            score_top3=day_eval.score_top3_jour,
            running_top1=day_eval.running_top1,
            running_top3=day_eval.running_top3,
            details={"courses": [self._eval_to_dict(e) for e in day_eval.courses]},
        )

        new_running = {
            "running_top1": day_eval.running_top1,
            "running_top3": day_eval.running_top3,
            "days_evaluated": day_number,
        }
        self.firebase.save_running_scores(new_running)

        # 6. Rapport soir Telegram
        msg = self._build_evening_report(day_eval)
        self.telegram.send_message_sync(msg)

        log_success(f"Évaluation J{day_number}/30 terminée")
        return True

    # ── Scraping résultats PMU ────────────────────────────────

    def _scrape_results_with_retry(
        self, date_str: str, course_ids: list[str], max_attempts: int = 3
    ) -> dict[str, Any]:
        """Scrape les résultats avec 3 tentatives espacées."""
        for attempt in range(1, max_attempts + 1):
            logger.info(f"🔍 Scraping résultats PMU (tentative {attempt}/{max_attempts})")
            results = self._scrape_results(date_str, course_ids)
            if results:
                log_success(f"{len(results)} résultat(s) récupéré(s)")
                return results
            if attempt < max_attempts:
                wait = 7200 * attempt  # +2h, +4h
                log_warning(f"Résultats non disponibles — retry dans {wait//3600}h")
                time.sleep(min(wait, 14400))  # max 4h en pratique

        return {}

    def _scrape_results(self, date_str: str, course_ids: list[str]) -> dict[str, Any]:
        """
        Scrape les résultats PMU.fr pour la date donnée.
        Utilise optionnellement Gemini pour l'extraction si le HTML est complexe.
        """
        try:
            resp = requests.get(
                self.RESULTATS_URL,
                headers=self.headers,
                timeout=20,
            )
            if resp.status_code != 200:
                log_warning(f"PMU.fr résultats HTTP {resp.status_code}")
                return {}

            html = resp.text
        except Exception as e:
            log_warning(f"Impossible de scraper PMU.fr : {e}")
            return {}

        # Parsing HTML basique
        results = self._parse_results_html(html, course_ids)

        # Fallback Gemini si aucun résultat parsé
        if not results and course_ids:
            results = self._extract_results_via_gemini(html, course_ids)

        return results

    def _parse_results_html(self, html: str, course_ids: list[str]) -> dict[str, Any]:
        """Parse HTML PMU.fr pour extraire les ordres d'arrivée."""
        results: dict[str, Any] = {}
        soup = BeautifulSoup(html, "html.parser")

        # Structure typique PMU.fr : div.race-result ou table.arrivals
        # Plusieurs sélecteurs pour robustesse
        selectors = [
            "div.race-result",
            "div.resultats-course",
            "table.classement",
            "[data-race-result]",
        ]

        for sel in selectors:
            blocks = soup.select(sel)
            if not blocks:
                continue
            for block in blocks:
                cid = self._extract_course_id_from_block(block)
                if cid:
                    ordre = self._extract_ordre_arrivee(block)
                    if ordre:
                        results[cid] = {
                            "ordre_arrivee": ordre,
                            "gagnant_numero": ordre[0] if ordre else None,
                            "gagnant_nom": self._extract_gagnant_nom(block),
                            "top3": ordre[:3],
                        }

        return results

    def _extract_course_id_from_block(self, block) -> Optional[str]:
        """Extrait l'ID de course depuis un bloc HTML."""
        # Chercher attributs data-race, data-course, ou texte "R1", "C1"
        for attr in ["data-race-id", "data-course", "id"]:
            val = block.get(attr, "")
            if val:
                return str(val)
        text = block.get_text()
        match = re.search(r"R\d+C?\d*", text)
        if match:
            return match.group()
        return None

    def _extract_ordre_arrivee(self, block) -> list[int]:
        """Extrait l'ordre d'arrivée (numéros) depuis un bloc."""
        nums = []
        # Chercher les numéros de partants dans le bloc
        spans = block.find_all(["span", "td", "div"], class_=re.compile(r"num|cheval|horse"))
        for s in spans:
            t = s.get_text(strip=True)
            if t.isdigit() and 1 <= int(t) <= 30:
                nums.append(int(t))
        return nums[:10]  # Top 10 max

    def _extract_gagnant_nom(self, block) -> str:
        """Extrait le nom du gagnant."""
        # Chercher le nom du premier cheval
        name_tags = block.find_all(class_=re.compile(r"name|nom"))
        if name_tags:
            return name_tags[0].get_text(strip=True)
        return "Inconnu"

    def _extract_results_via_gemini(self, html: str, course_ids: list[str]) -> dict[str, Any]:
        """Utilise Gemini pour extraire les résultats si le parsing HTML échoue."""
        html_excerpt = html[:8000]  # Limiter la taille
        prompt = f"""Voici le HTML de la page des résultats PMU.fr.

Extrais les ordres d'arrivée pour les courses : {course_ids}

Retourne UNIQUEMENT ce JSON :
{{
  "results": {{
    "R1C1": {{
      "ordre_arrivee": [4, 7, 2, 9, 1],
      "gagnant_numero": 4,
      "gagnant_nom": "NOM CHEVAL",
      "top3": [4, 7, 2]
    }}
  }}
}}

Si une course est introuvable, omets-la. Commence par {{ et termine par }}.

HTML (extrait) :
{html_excerpt}"""

        response = gemini_manager.call(prompt=prompt, temperature=0.1, max_output_tokens=1000)
        if not response:
            return {}

        try:
            import json
            clean = re.sub(r"```(?:json)?", "", response)
            clean = re.sub(r"```", "", clean).strip()
            start = clean.find("{")
            end = clean.rfind("}")
            if start == -1 or end == -1:
                return {}
            data = json.loads(clean[start : end + 1])
            return data.get("results", {})
        except Exception as e:
            log_warning(f"Parse résultats Gemini échoué : {e}")
            return {}

    # ── Évaluation ────────────────────────────────────────────

    def _evaluate(
        self,
        date_str: str,
        day_number: int,
        predictions: dict[str, Any],
        results: dict[str, Any],
        running: dict[str, Any],
    ) -> DayEvaluation:
        """Compare prédictions vs résultats et calcule les scores."""
        ev = DayEvaluation(date_str=date_str, day_number=day_number)
        nb = 0
        top1_ok = 0
        top3_total = 0

        for cid, pred in predictions.items():
            res = results.get(cid)
            if not res:
                continue

            pred_top5 = pred.get("predicted_top5", [])
            pred_winner = pred.get("predicted_winner")
            official_winner = res.get("gagnant_numero")
            official_top3 = res.get("top3", [])

            ce = CourseEvaluation(
                course_id=cid,
                is_lonab=pred.get("is_lonab", False),
                predicted_winner=pred_winner,
                official_winner=official_winner,
                predicted_top3=pred_top5[:3],
                official_top3=official_top3,
            )
            ce.compute()
            ev.courses.append(ce)

            nb += 1
            if ce.top1_correct:
                top1_ok += 1
            top3_total += ce.top3_score

            if ce.is_lonab:
                ev.lonab_top1_correct = ce.top1_correct

        if nb > 0:
            ev.score_top1_jour = round(top1_ok / nb, 4)
            ev.score_top3_jour = round(top3_total / (nb * 3), 4)
        else:
            ev.score_top1_jour = 0.0
            ev.score_top3_jour = 0.0

        # Scores cumulés (moyenne glissante)
        days_done = int(running.get("days_evaluated", 0))
        r_top1 = float(running.get("running_top1", 0.0))
        r_top3 = float(running.get("running_top3", 0.0))

        if days_done > 0:
            ev.running_top1 = round((r_top1 * days_done + ev.score_top1_jour) / (days_done + 1), 4)
            ev.running_top3 = round((r_top3 * days_done + ev.score_top3_jour) / (days_done + 1), 4)
        else:
            ev.running_top1 = ev.score_top1_jour
            ev.running_top3 = ev.score_top3_jour

        logger.info(
            f"J{day_number}: Top1={ev.score_top1_jour:.1%} ({top1_ok}/{nb}) | "
            f"Top3={ev.score_top3_jour:.1%} | "
            f"Cumulé Top1={ev.running_top1:.1%}"
        )
        return ev

    # ── Rapport soir ──────────────────────────────────────────

    def _build_evening_report(self, ev: DayEvaluation) -> str:
        """Construit le message Telegram du rapport soir."""
        seuils = config.get("evaluation.seuils", {})
        t_min = float(seuils.get("top1_minimum", 0.25))
        t_bon = float(seuils.get("top1_bon", 0.35))
        t_exc = float(seuils.get("top1_excellent", 0.45))

        # Tendance
        if ev.running_top1 >= t_exc:
            tendance = "🌟 EXCELLENT"
        elif ev.running_top1 >= t_bon:
            tendance = "✅ BON"
        elif ev.running_top1 >= t_min:
            tendance = "📊 ACCEPTABLE"
        else:
            tendance = "⚠️ À AMÉLIORER"

        # Section LONAB
        lonab_section = ""
        lonab_ev = next((c for c in ev.courses if c.is_lonab), None)
        if lonab_ev:
            lonab_icon = "✅" if lonab_ev.top1_correct else "❌"
            lonab_section = (
                f"\n⭐ <b>Course LONAB :</b>\n"
                f"  Prédit N°{lonab_ev.predicted_winner} → "
                f"{lonab_icon} {'CORRECT' if lonab_ev.top1_correct else f'Réel : N°{lonab_ev.official_winner}'}\n"
                f"  Top3 : {lonab_ev.top3_score}/3 corrects\n"
            )

        # Section autres courses
        nb = len(ev.courses)
        top1_ok = sum(1 for c in ev.courses if c.top1_correct)
        top3_total = sum(c.top3_score for c in ev.courses)

        msg = (
            f"📊 <b>ÉVALUATION J{ev.day_number}/30</b> — {ev.date_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ <b>RÉSULTATS OFFICIELS</b>\n"
            f"{lonab_section}"
            f"\n📌 Toutes courses ({nb}) :\n"
            f"  Top1 correct : <b>{top1_ok}/{nb}</b> ({ev.score_top1_jour:.1%})\n"
            f"  Top3 correct : <b>{top3_total}/{nb*3}</b> ({ev.score_top3_jour:.1%})\n"
            f"\n━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 <b>SCORE DU JOUR</b>\n"
            f"  Top1 : {ev.score_top1_jour:.1%} | Top3 : {ev.score_top3_jour:.1%}\n"
            f"\n🎯 <b>SCORE CUMULÉ J{ev.day_number}/30</b>\n"
            f"  Top1 global : <b>{ev.running_top1:.1%}</b>\n"
            f"  Top3 global : <b>{ev.running_top3:.1%}</b>\n"
            f"  Tendance : {tendance}\n"
        )

        # Bilan final à J30
        if ev.day_number >= 30:
            verdict = (
                "🌟 EXCELLENT — Système validé" if ev.running_top1 >= t_exc
                else "✅ BON — Ouvrir au marché" if ev.running_top1 >= t_bon
                else "📊 ACCEPTABLE — Continuer le test" if ev.running_top1 >= t_min
                else "❌ INSUFFISANT — Ajuster les poids"
            )
            msg += (
                f"\n{'═'*25}\n"
                f"🏁 <b>BILAN FINAL 30 JOURS</b>\n"
                f"  {verdict}\n"
                f"  Top1 ≥ {t_min:.0%} (min) : {'✅' if ev.running_top1 >= t_min else '❌'}\n"
                f"  Top1 ≥ {t_bon:.0%} (bon) : {'✅' if ev.running_top1 >= t_bon else '❌'}\n"
                f"  Top1 ≥ {t_exc:.0%} (excellent) : {'✅' if ev.running_top1 >= t_exc else '❌'}\n"
            )

        msg += (
            "\n━━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ <i>HYPERION est un outil d'analyse statistique. "
            "Les paris comportent des risques financiers.</i>"
        )
        return msg

    def _eval_to_dict(self, ce: CourseEvaluation) -> dict:
        return {
            "course_id": ce.course_id,
            "is_lonab": ce.is_lonab,
            "predicted_winner": ce.predicted_winner,
            "official_winner": ce.official_winner,
            "top1_correct": ce.top1_correct,
            "top3_score": ce.top3_score,
        }
