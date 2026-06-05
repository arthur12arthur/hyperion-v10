"""
Hyperion V10 — ReportGenerator
Génération rapport narratif via Gemini (1 appel batch) + fallback template statique.
"""

from datetime import datetime
from typing import Any, Optional

from ..utils.config import config
from ..utils.logger import logger, log_success, log_warning
from ..utils.gemini_manager import gemini_manager
from ..financial.ev_kelly import EVKellyCalculator


_RAPPORT_PROMPT_TEMPLATE = """Tu es l'analyste hippique en chef d'HYPERION V10.
Rédige un commentaire professionnel et concis en français sur la course suivante.

COURSE : {hippodrome} — {distance}m — {terrain}
TOP 5 HYPERION :
{top5_details}

CHEVAUX ÉLIMINÉS :
{elimines_details}

SCORE DE CONFIANCE : {confiance:.0f}%
HADES : {hades_niveau} — {nb_signaux} signal(aux)

Rédige une analyse fluide de 6-10 lignes :
1. Présente le favori et ses atouts
2. Mentionne 1-2 outsiders intéressants
3. Précise si le classement est robuste ou incertain
4. Donne un conseil de combinaison (simple / couplé / trio)

Réponds en français, texte fluide, pas de listes à puces.
"""


class ReportGenerator:
    """Génère les rapports Telegram formatés Hyperion V10."""

    def __init__(self):
        self.ev_calc = EVKellyCalculator()
        logger.info("✅ ReportGenerator initialisé")

    # ── Rapport principal course ──────────────────────────────

    def build_course_report(
        self,
        course: dict[str, Any],
        top5_final: list[dict[str, Any]],
        hades_result: dict[str, Any],
        ev_kelly_data: dict[str, Any],
        elimines: list[dict[str, Any]],
        rapport_gemini: Optional[str] = None,
    ) -> list[str]:
        """
        Construit la liste de messages Telegram pour une course.

        Returns:
            Liste de chaînes HTML, à envoyer séquentiellement
        """
        messages: list[str] = []

        # Message 1 : En-tête + Top 5 + HADES
        messages.append(self._build_header(course, top5_final, hades_result))

        # Message 2 : Analyse financière
        messages.append(self.ev_calc.format_summary(ev_kelly_data, top3_only=True))

        # Message 3 : Narration Gemini (si disponible)
        if rapport_gemini:
            gemini_msg = (
                "🤖 <b>ANALYSE IA</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"{rapport_gemini[:2800]}"
            )
            messages.append(gemini_msg)

        return messages

    def _build_header(
        self,
        course: dict[str, Any],
        top5: list[dict[str, Any]],
        hades: dict[str, Any],
    ) -> str:
        hippodrome = course.get("hippodrome", "?")
        heure = course.get("heure", "")
        distance = course.get("distance", 0)
        nb_partants = course.get("nb_partants", 0)
        nom_course = course.get("id_course") or course.get("nom", "")

        niveau = hades.get("niveau_global", "green")
        hades_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(niveau, "⚪")
        nb_suspects = len(hades.get("chevaux_suspects", []))
        hades_line = (
            f"{hades_emoji} HADES : {'OK' if niveau == 'green' else f'{nb_suspects} suspect(s)'}"
        )
        if hades.get("mode_test"):
            hades_line += " <i>(mode test)</i>"

        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        top5_lines = []
        for i, ch in enumerate(top5[:5]):
            medal = medals[i] if i < 5 else f"{i+1}."
            robuste = " 🛡️" if ch.get("robuste") else ""
            score = ch.get("meta_score", ch.get("win_prob", 0))
            cote = ch.get("cote")
            cote_str = f" | cote {cote:.1f}" if cote else ""
            top5_lines.append(
                f"{medal} <b>N°{ch['numero']} {ch.get('nom', '?')}</b>{robuste}"
                f" — {score:.3f}{cote_str}"
            )

        heure_str = f" — {heure}" if heure else ""
        return (
            f"🏇 <b>{hippodrome}</b>{heure_str}\n"
            f"<i>{nom_course}</i>\n"
            f"📏 {distance}m | {nb_partants} partants\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>TOP 5 HYPERION V10</b>\n"
            + "\n".join(top5_lines)
            + f"\n━━━━━━━━━━━━━━━━━━━━━\n"
            f"{hades_line}\n"
            f"<i>🛡️ = Cheval ROBUSTE (méta-score ≥ {config.get('consensus.meta_fusion.threshold_robuste', 0.80):.0%})</i>"
        )

    # ── Rapport narratif Gemini ───────────────────────────────

    def generate_gemini_narrative(
        self,
        course: dict[str, Any],
        top5_final: list[dict[str, Any]],
        hades_result: dict[str, Any],
        elimines: list[dict[str, Any]],
        confiance: float,
    ) -> str:
        """
        Génère l'analyse narrative via Gemini (1 appel).
        Retourne un texte statique de fallback si Gemini échoue.
        """
        if not config.get("pipeline.generate_rapport_gemini", True):
            return self._fallback_narrative(top5_final, confiance)

        top5_lines = []
        for i, ch in enumerate(top5_final[:5], 1):
            score = ch.get("meta_score", ch.get("win_prob", 0))
            cote = ch.get("cote", "?")
            forme = ch.get("forme", "?")
            top5_lines.append(
                f"{i}. N°{ch['numero']} {ch.get('nom', '?')} — score {score:.3f}, "
                f"cote {cote}, forme [{forme}]"
            )

        elimines_lines = []
        for e in elimines[:3]:
            ch = e.get("cheval", e)
            nom = ch.get("nom", "?")
            motifs = ", ".join(str(m) for m in e.get("motifs", [])[:2])
            elimines_lines.append(f"- {nom} : {motifs[:80]}")

        prompt = _RAPPORT_PROMPT_TEMPLATE.format(
            hippodrome=course.get("hippodrome", "?"),
            distance=course.get("distance", 0),
            terrain=course.get("terrain", "?"),
            top5_details="\n".join(top5_lines) or "Non disponible",
            elimines_details="\n".join(elimines_lines) or "Aucun",
            confiance=confiance,
            hades_niveau=hades_result.get("niveau_global", "green").upper(),
            nb_signaux=hades_result.get("nb_signaux", 0),
        )

        text = gemini_manager.call(
            prompt=prompt,
            temperature=config.get("gemini.report.temperature", 0.6),
            max_output_tokens=config.get("gemini.report.max_output_tokens", 3000),
        )

        if not text or len(text.strip()) < 100:
            log_warning("Narration Gemini vide ou trop courte — fallback statique")
            return self._fallback_narrative(top5_final, confiance)

        log_success(f"Narration Gemini générée ({len(text)} chars)")
        return text.strip()

    def _fallback_narrative(self, top5: list[dict], confiance: float) -> str:
        favori = top5[0].get("nom", "?") if top5 else "?"
        second = top5[1].get("nom", "?") if len(top5) > 1 else "?"
        return (
            f"[Rapport automatique — Gemini indisponible]\n"
            f"Favori Hyperion : {favori}. "
            f"Outsider à surveiller : {second}. "
            f"Confiance globale : {confiance:.0f}%."
        )

    # ── Résumé journalier ─────────────────────────────────────

    def build_daily_summary(
        self,
        all_results: list[dict[str, Any]],
        date_str: str,
        duree_sec: float,
    ) -> str:
        nb = len(all_results)
        vbs: list[str] = []
        for r in all_results:
            for vb_num in r.get("ev_kelly", {}).get("value_bets", []):
                hipp = r["course"].get("hippodrome", "?")
                nom = next(
                    (ch["nom"] for ch in r.get("top5_final", []) if ch["numero"] == vb_num),
                    f"#{vb_num}",
                )
                vbs.append(f"  💎 {hipp} → {nom}")

        alerts = [r for r in all_results if r.get("hades", {}).get("niveau_global", "green") != "green"]

        msg = (
            f"📊 <b>RÉSUMÉ HYPERION V10 — {date_str}</b>\n"
            f"{'═'*28}\n"
            f"🏇 Courses : <b>{nb}</b>\n"
            f"⏱️ Durée : <b>{duree_sec:.0f}s</b>\n"
        )
        if vbs:
            msg += f"\n💎 <b>Value bets ({len(vbs)}) :</b>\n" + "\n".join(vbs[:5]) + "\n"
        if alerts:
            msg += f"\n⚠️ <b>Alertes HADES : {len(alerts)}</b>\n"
            for r in alerts[:3]:
                h = r["course"].get("hippodrome", "?")
                niv = r["hades"]["niveau_global"]
                e = "🔴" if niv == "red" else "🟡"
                msg += f"  {e} {h}\n"
        if not vbs and not alerts:
            msg += "\nℹ️ Pas de signal particulier aujourd'hui.\n"

        return msg

    # ── Messages système ──────────────────────────────────────

    def build_start_message(self, nb: int, date_str: str, run_id: str) -> str:
        return (
            f"🚀 <b>HYPERION V10 — DÉMARRAGE</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 {date_str}\n"
            f"🏇 Courses : <b>{nb}</b>\n"
            f"⏱️ {datetime.now().strftime('%H:%M:%S')}\n"
            f"<code>Run #{run_id}</code>"
        )

    def build_error_message(self, context: str, error: str) -> str:
        return (
            f"❌ <b>ERREUR HYPERION V10</b>\n"
            f"📍 <code>{context}</code>\n"
            f"💬 <code>{error[:300]}</code>\n"
            f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        )

    def build_no_courses_message(self, date_str: str) -> str:
        return f"ℹ️ <b>HYPERION V10</b>\nAucun programme LONAB pour le <b>{date_str}</b>."
