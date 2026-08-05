"""
Hyperion V10 — AgentH (Auto-Évaluation)
Pipeline soir 20h00 : scraping résultats PMU → comparaison → score J/30.
Nouveau module absent dans les versions précédentes.
"""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Tuple

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

        # 2. Récupérer résultats officiels (avec retry)
        results = self._scrape_results_with_retry(date_str, predictions)
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

    # ── Récupération résultats via l'API officielle PMU ───────
    #
    # www.pmu.fr est une application JS (Angular) : un simple
    # requests.get() ne renvoie jamais le contenu affiché à l'écran,
    # donc le scraping HTML ne pouvait pas fonctionner. On utilise à
    # la place l'API JSON publique du PMU (utilisée par de nombreux
    # outils tiers), qui ne nécessite aucune clé.

    PROGRAMME_API = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme/{ddmmyyyy}"
    PARTICIPANTS_API = (
        "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme/"
        "{ddmmyyyy}/R{r}/C{c}/participants"
    )

    def _scrape_results_with_retry(
        self, date_str: str, predictions: dict[str, Any], max_attempts: int = 3
    ) -> dict[str, Any]:
        """
        Récupère les résultats avec quelques tentatives rapprochées.
        IMPORTANT : le job GitHub Actions a un timeout de 30 min — on ne peut
        donc pas attendre des heures entre deux tentatives (bug de l'ancienne
        version, qui faisait échouer le job à coup sûr). On espace les
        tentatives de quelques minutes seulement.
        """
        for attempt in range(1, max_attempts + 1):
            logger.info(f"🔍 Récupération résultats PMU (tentative {attempt}/{max_attempts})")
            results = self._fetch_results_via_pmu_api(date_str, predictions)
            if results:
                log_success(f"{len(results)} résultat(s) récupéré(s)")
                return results
            if attempt < max_attempts:
                wait = 180 * attempt  # 3 min, puis 6 min — reste bien sous les 30 min du job
                log_warning(f"Résultats non disponibles — retry dans {wait // 60} min")
                time.sleep(wait)

        return {}

    def _fetch_results_via_pmu_api(
        self, date_str: str, predictions: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Pour chaque course prédite, retrouve la réunion/course PMU officielle
        correspondante (via le nom de l'hippodrome sauvegardé le matin) et lit
        l'ordre d'arrivée réel depuis l'API PMU.
        """
        ddmmyyyy = datetime.strptime(date_str, "%Y-%m-%d").strftime("%d%m%Y")
        programme = self._fetch_json(self.PROGRAMME_API.format(ddmmyyyy=ddmmyyyy))
        if not programme:
            log_warning("Programme PMU du jour indisponible")
            return {}

        reunions = programme.get("programme", {}).get("reunions", [])
        if not reunions:
            log_warning("Programme PMU reçu mais vide (aucune réunion)")
            return {}

        results: dict[str, Any] = {}

        for course_id, pred in predictions.items():
            num_reunion = pred.get("numero_reunion")
            num_course = pred.get("numero_course")

            if isinstance(num_reunion, int) and isinstance(num_course, int):
                # Numéro exact capturé le matin depuis le PDF LONAB — le cas fiable.
                match = (num_reunion, num_course)
            else:
                # Prédiction plus ancienne (avant capture du numéro) ou PDF sans
                # mention explicite : on retombe sur la déduction par hippodrome.
                hippodrome = (pred.get("hippodrome") or "").strip()
                if not hippodrome or hippodrome in ("?", "LONAB"):
                    log_warning(f"{course_id} : ni numéro réunion/course, ni hippodrome exploitable")
                    continue
                match = self._locate_course(reunions, hippodrome)
                if not match:
                    log_warning(f"{course_id} : hippodrome '{hippodrome}' introuvable au programme PMU du {date_str}")
                    continue
                log_warning(f"{course_id} : numéro R/C non capturé le matin — déduction par hippodrome (best-effort)")

            num_reunion, num_course = match
            participants = self._fetch_json(
                self.PARTICIPANTS_API.format(ddmmyyyy=ddmmyyyy, r=num_reunion, c=num_course)
            )
            if not participants:
                log_warning(f"{course_id} : participants R{num_reunion}C{num_course} indisponibles (course pas encore courue ?)")
                continue

            ordre = self._extract_ordre_arrivee_api(participants)
            if not ordre:
                log_warning(f"{course_id} : R{num_reunion}C{num_course} trouvée mais pas encore d'arrivée")
                continue

            results[course_id] = {
                "ordre_arrivee": ordre,
                "gagnant_numero": ordre[0],
                "top3": ordre[:3],
                "reunion_course": f"R{num_reunion}C{num_course}",
            }
            log_success(f"{course_id} → R{num_reunion}C{num_course} : arrivée {ordre[:5]}")

        return results

    def _locate_course(self, reunions: list[dict], hippodrome: str) -> Optional[Tuple[int, int]]:
        """
        Retrouve (numéro réunion, numéro course) pour un hippodrome donné.

        Le marché LONAB republie une seule course française par jour — dans
        l'immense majorité des cas il s'agit du Quinté+, qui est par
        convention PMU la réunion 1 / course 1 (R1C1). On vérifie donc R1C1
        en priorité, puis on cherche l'hippodrome ailleurs dans le programme
        si besoin (dans ce cas on prend par défaut la 1ère course de la
        réunion trouvée — best-effort, voir note plus bas).
        """
        target = self._normalize(hippodrome)

        def hippo_of(reunion: dict) -> str:
            h = reunion.get("hippodrome")
            if isinstance(h, dict):
                return h.get("libelleLong") or h.get("libelleCourt") or ""
            return str(h or "")

        # 1) Vérifier R1C1 (cas Quinté+, très majoritaire)
        if reunions:
            h0 = self._normalize(hippo_of(reunions[0]))
            if h0 and (h0 in target or target in h0):
                return (1, 1)

        # 2) Chercher l'hippodrome ailleurs dans le programme du jour
        for i, reunion in enumerate(reunions, start=1):
            h = self._normalize(hippo_of(reunion))
            if h and (h in target or target in h):
                # Best-effort : on prend la 1ère course de cette réunion.
                # Si un hippodrome a plusieurs courses LONAB par jour, il
                # faudra enregistrer le numéro de course exact dès le matin
                # (voir remarque envoyée dans le chat).
                return (i, 1)

        return None

    @staticmethod
    def _normalize(s: str) -> str:
        import unicodedata
        s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
        return s.lower().strip()

    def _fetch_json(self, url: str) -> Optional[dict]:
        try:
            resp = requests.get(url, headers=self.headers, timeout=20)
            if resp.status_code != 200:
                log_warning(f"API PMU HTTP {resp.status_code} : {url}")
                return None
            return resp.json()
        except Exception as e:
            log_warning(f"Erreur appel API PMU ({url}) : {e}")
            return None

    @staticmethod
    def _extract_ordre_arrivee_api(participants_doc: dict) -> list[int]:
        """Extrait l'ordre d'arrivée depuis la réponse JSON /participants."""
        finishers = []
        for p in participants_doc.get("participants", []):
            pos = p.get("ordreArrivee")
            num = p.get("numPmu")
            if isinstance(pos, int) and pos > 0 and num:
                finishers.append((pos, num))
        finishers.sort(key=lambda x: x[0])
        return [num for _, num in finishers]

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
