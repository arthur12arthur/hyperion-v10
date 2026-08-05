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
        6 messages courts et thématiques (plus lisible sur mobile
        qu'un message unique très long) :
        1. En-tête + Top5 détaillé + HADES
        2. Pourquoi ce classement (détail du scoring)
        3. Chevaux éliminés et motifs
        4. Analyse financière
        5. Recommandation finale (toujours présente, sans dépendre de Gemini)
        6. Analyse IA Gemini (commentaire, si disponible)

        Returns:
            Liste de chaînes HTML, à envoyer séquentiellement
        """
        messages: list[str] = []

        messages.append(self._build_header(course, top5_final, hades_result))
        messages.append(self._build_scoring_explanation(top5_final))
        messages.append(self._build_elimines_message(elimines))
        messages.append(self.ev_calc.format_summary(ev_kelly_data, top3_only=True))
        messages.append(self._build_recommendation(top5_final, hades_result, ev_kelly_data))

        if rapport_gemini:
            messages.append(
                "🤖 <b>ANALYSE IA</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"{rapport_gemini[:2800]}"
            )

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
        terrain = course.get("terrain")
        discipline = course.get("discipline")
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

            details = []
            if ch.get("jockey"):
                details.append(f"🏇 {ch['jockey']}")
            if ch.get("entraineur"):
                details.append(f"🎓 {ch['entraineur']}")
            if ch.get("corde"):
                details.append(f"📍 corde {ch['corde']}")
            if ch.get("forme"):
                details.append(f"📈 [{ch['forme']}]")
            details_line = f"\n   <i>{' | '.join(details)}</i>" if details else ""

            top5_lines.append(
                f"{medal} <b>N°{ch['numero']} {ch.get('nom', '?')}</b>{robuste}"
                f" — {score:.3f}{cote_str}{details_line}"
            )

        heure_str = f" — {heure}" if heure else ""
        terrain_str = f" | {terrain}" if terrain else ""
        discipline_str = f" | {discipline}" if discipline else ""
        return (
            f"🏇 <b>{hippodrome}</b>{heure_str}\n"
            f"<i>{nom_course}</i>\n"
            f"📏 {distance}m{terrain_str}{discipline_str} | {nb_partants} partants\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>TOP 5 HYPERION V10</b>\n"
            + "\n".join(top5_lines)
            + f"\n━━━━━━━━━━━━━━━━━━━━━\n"
            f"{hades_line}\n"
            f"<i>🛡️ = Cheval ROBUSTE (méta-score ≥ {config.get('consensus.meta_fusion.threshold_robuste', 0.80):.0%})</i>"
        )

    # ── Explication du scoring ─────────────────────────────────

    _DIMENSIONS = [
        ("historique", "score_historique", "Historique"),
        ("forme_recente", "score_forme", "Forme"),
        ("terrain_distance", "score_terrain_distance", "Terrain/dist."),
        ("handicap", "score_handicap", "Handicap"),
        ("fraicheur", "score_fraicheur", "Fraîcheur"),
    ]

    def _build_scoring_explanation(self, top5: list[dict[str, Any]]) -> str:
        """Détaille pourquoi chaque cheval du top5 est classé où il est."""
        weights = config.get("scoring.weights", {})
        w_internal = config.get("consensus.meta_fusion.weight_internal", 0.55)
        w_external = config.get("consensus.meta_fusion.weight_external", 0.45)

        lines = [
            "🧠 <b>POURQUOI CE CLASSEMENT</b>",
            "━━━━━━━━━━━━━━━━━━━━━",
        ]
        for i, ch in enumerate(top5[:5], 1):
            lines.append(f"\n<b>{i}. N°{ch['numero']} {ch.get('nom', '?')}</b>")

            sub_parts = []
            for weight_key, score_key, label in self._DIMENSIONS:
                if ch.get(score_key) is not None:
                    w = weights.get(weight_key, 0)
                    sub_parts.append(f"{label} {ch[score_key]:.1f}/10 (×{w:.0%})")
            if sub_parts:
                lines.append("  " + " · ".join(sub_parts))

            score_global = ch.get("score_global")
            if score_global is not None:
                lines.append(f"  → Score interne : <b>{score_global:.1f}/10</b>")

            s_mc = ch.get("score_mc")
            s_ext = ch.get("score_externe")
            if s_mc is not None:
                ext_part = f" + externe {s_ext:.2f} (×{w_external:.0%})" if s_ext is not None else ""
                lines.append(
                    f"  → Consensus interne {s_mc:.2f} (×{w_internal:.0%}){ext_part} "
                    f"= <b>méta-score {ch.get('meta_score', 0):.2f}</b>"
                )

            if ch.get("robuste"):
                lines.append("  🛡️ Classement stable sur la majorité des simulations Monte Carlo")

        return "\n".join(lines)

    # ── Chevaux éliminés ────────────────────────────────────────

    def _build_elimines_message(self, elimines: list[dict[str, Any]]) -> str:
        lines = [
            "🚫 <b>CHEVAUX ÉLIMINÉS</b>",
            "━━━━━━━━━━━━━━━━━━━━━",
        ]
        if not elimines:
            lines.append("Aucun cheval éliminé à l'analyse préalable.")
            return "\n".join(lines)

        lines.append(f"{len(elimines)} cheval(aux) écarté(s) avant scoring :\n")
        for e in elimines[:10]:
            ch = e.get("cheval", {})
            motifs = e.get("motifs", [])
            nom = ch.get("nom", "?")
            numero = ch.get("numero", "?")
            motifs_str = ", ".join(motifs) if motifs else "motif non précisé"
            lines.append(f"• <b>N°{numero} {nom}</b>\n   <i>{motifs_str}</i>")

        if len(elimines) > 10:
            lines.append(f"\n… et {len(elimines) - 10} autre(s).")

        return "\n".join(lines)

    # ── Recommandation finale ───────────────────────────────────

    def _build_recommendation(
        self,
        top5: list[dict[str, Any]],
        hades: dict[str, Any],
        ev_kelly_data: dict[str, Any],
    ) -> str:
        """
        Recommandation déterministe, toujours présente même si Gemini
        est indisponible (contrairement à l'ancienne narration IA seule).
        """
        if not top5:
            return "🎯 <b>RECOMMANDATION</b>\n━━━━━━━━━━━━━━━━━━━━━\nPas assez de données pour une recommandation."

        favori = top5[0]
        nb_robustes = sum(1 for ch in top5 if ch.get("robuste"))
        confiance = self._compute_confiance(top5)
        niveau_hades = hades.get("niveau_global", "green")
        value_bets = ev_kelly_data.get("value_bets", [])

        if confiance >= 70 and niveau_hades == "green":
            niveau_txt = "🌟 CONFIANCE ÉLEVÉE"
        elif confiance >= 50:
            niveau_txt = "✅ CONFIANCE CORRECTE"
        else:
            niveau_txt = "⚠️ CONFIANCE FAIBLE"

        if nb_robustes >= 3:
            pari_conseille = "Trio / Couplé (top 3 stable sur les simulations)"
        elif nb_robustes >= 1:
            pari_conseille = "Simple gagnant/placé sur le favori, éviter les combinaisons larges"
        else:
            pari_conseille = "Classement incertain — jouer petit ou s'abstenir"

        lines = [
            "🎯 <b>RECOMMANDATION</b>",
            "━━━━━━━━━━━━━━━━━━━━━",
            f"Favori : <b>N°{favori['numero']} {favori.get('nom', '?')}</b>",
            f"Niveau de confiance : {niveau_txt} ({confiance:.0f}%)",
            f"Pari suggéré : {pari_conseille}",
        ]

        if value_bets:
            noms_vb = ", ".join(
                next((ch['nom'] for ch in top5 if ch['numero'] == n), f"N°{n}")
                for n in value_bets[:3]
            )
            lines.append(f"💎 Value bet(s) détecté(s) : {noms_vb}")

        if niveau_hades != "green":
            lines.append("⚠️ HADES a détecté une anomalie — prudence renforcée sur ce favori.")

        lines.append("\n<i>⚠️ Outil d'analyse statistique — pas un conseil financier.</i>")
        return "\n".join(lines)

    @staticmethod
    def _compute_confiance(top5: list[dict[str, Any]]) -> float:
        """Indice de confiance basé sur l'écart entre #1 et #2 (identique à main.py)."""
        if not top5 or len(top5) < 2:
            return 50.0
        s1 = float(top5[0].get("meta_score", top5[0].get("win_prob", 0)))
        s2 = float(top5[1].get("meta_score", top5[1].get("win_prob", 0)))
        return round(max(0.0, min(100.0, 50.0 + (s1 - s2) * 200)), 1)

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
