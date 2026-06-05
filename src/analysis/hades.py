"""
Hyperion V10 — HADESDetector
Système anti-manipulation : détection anomalies de cotes et favoris artificiels.
En mode test (J1-J30) : alertes loggées sans blocage des recommandations.
"""

from datetime import datetime
from typing import Any

from ..utils.config import config
from ..utils.logger import logger, log_warning


class HADESDetector:
    """
    Détecteur d'anomalies hippiques.

    Détections :
    1. Écart cote affichée vs cote théorique (> 20%)
    2. Favori artificiel : haute hype externe, faible prob interne
    """

    def __init__(self):
        self.enabled = config.get("hades.enabled", True)
        self.mode_test = config.get("hades.mode_test", True)
        self.cote_dev_thresh = config.get("hades.cote_deviation_threshold_pct", 0.20)
        self.art_prob_max = config.get("hades.artificial_favorite.prob_internal_max", 0.10)
        self.art_freq_min = config.get("hades.artificial_favorite.freq_external_min", 0.50)
        mode_label = "MODE TEST (alertes loggées sans blocage)" if self.mode_test else "MODE PRODUCTION"
        logger.info(f"✅ HADESDetector initialisé — {mode_label}")

    # ── API publique ──────────────────────────────────────────

    def analyze(
        self,
        chevaux: list[dict[str, Any]],
        consensus_mc: list[dict[str, Any]],
        external_aggregation: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Analyse complète anti-manipulation.

        Returns:
            {
                'niveau_global': 'green'|'yellow'|'red',
                'signaux': [...],
                'chevaux_suspects': [...],
                'nb_signaux': int,
                'mode_test': bool,
                'detected_at': str,
            }
        """
        if not self.enabled:
            return self._clean_result()

        logger.info("🔍 Analyse HADES en cours...")

        signaux: list[dict] = []
        signaux += self._detect_cote_deviations(chevaux, consensus_mc)
        signaux += self._detect_artificial_favorites(chevaux, consensus_mc, external_aggregation)

        niveau = self._global_level(signaux)
        suspects = list(
            {s["cheval_numero"] for s in signaux if s["niveau"] in ("yellow", "red")}
        )

        if niveau != "green":
            log_warning(f"⚠️  HADES : {niveau.upper()} — {len(signaux)} signal(aux)")
            if self.mode_test:
                log_warning("ℹ️  Mode test actif — aucun blocage de recommandation")
        else:
            logger.info("✅ HADES : aucune anomalie")

        return {
            "niveau_global": niveau,
            "signaux": signaux,
            "chevaux_suspects": suspects,
            "nb_signaux": len(signaux),
            "mode_test": self.mode_test,
            "detected_at": datetime.now().isoformat(),
        }

    # ── Détections ────────────────────────────────────────────

    def _detect_cote_deviations(
        self,
        chevaux: list[dict[str, Any]],
        consensus_mc: list[dict[str, Any]],
    ) -> list[dict]:
        """Écart cote affichée vs cote théorique (calculée depuis prob MC)."""
        signaux = []

        mc_by_num = {int(ch["numero"]): ch for ch in consensus_mc}

        for ch in chevaux:
            num = int(ch.get("numero", 0))
            nom = ch.get("nom", f"Cheval {num}")
            cote_aff = ch.get("cote")
            if not cote_aff or float(cote_aff) <= 1.0:
                continue

            # Cote théorique depuis prob MC
            mc = mc_by_num.get(num, {})
            win_prob = float(mc.get("win_prob", 0.0))
            if win_prob <= 0:
                continue

            cote_theo = round(1.0 / win_prob, 2)
            ecart = abs(float(cote_aff) - cote_theo) / cote_theo

            if ecart > self.cote_dev_thresh:
                direction = "surévalué" if float(cote_aff) > cote_theo else "sous-évalué"
                niveau = "red" if ecart > 0.50 else "yellow"
                signaux.append({
                    "type": "ECART_COTE",
                    "niveau": niveau,
                    "cheval_numero": num,
                    "cheval_nom": nom,
                    "message": (
                        f"N°{num} {nom} : écart cote {ecart*100:.0f}% "
                        f"(affichée {cote_aff:.1f}, théorique {cote_theo:.1f}) — {direction}"
                    ),
                    "cote_affichee": float(cote_aff),
                    "cote_theorique": cote_theo,
                    "ecart_pct": round(ecart, 3),
                })

        return signaux

    def _detect_artificial_favorites(
        self,
        chevaux: list[dict[str, Any]],
        consensus_mc: list[dict[str, Any]],
        external_aggregation: dict[str, Any],
    ) -> list[dict]:
        """Favori artificiel : forte hype externe, faible probabilité interne."""
        signaux = []
        mc_by_num = {int(ch["numero"]): ch for ch in consensus_mc}
        ext_stats = external_aggregation.get("chevaux_stats", {})

        for ch in chevaux:
            num = int(ch.get("numero", 0))
            nom = ch.get("nom", f"Cheval {num}")

            prob_int = float(mc_by_num.get(num, {}).get("win_prob", 0.0))
            freq_ext = float(ext_stats.get(num, {}).get("external_frequency", 0.0))

            if prob_int < self.art_prob_max and freq_ext > self.art_freq_min:
                signaux.append({
                    "type": "FAVORI_ARTIFICIEL",
                    "niveau": "yellow",
                    "cheval_numero": num,
                    "cheval_nom": nom,
                    "message": (
                        f"N°{num} {nom} : hype artificielle possible — "
                        f"prob. interne {prob_int*100:.0f}% mais {freq_ext*100:.0f}% sources externes"
                    ),
                    "prob_internal": round(prob_int, 3),
                    "freq_external": round(freq_ext, 3),
                })

        return signaux

    # ── Niveau global ─────────────────────────────────────────

    def _global_level(self, signaux: list[dict]) -> str:
        if not signaux:
            return "green"
        nb_red = sum(1 for s in signaux if s["niveau"] == "red")
        nb_yellow = sum(1 for s in signaux if s["niveau"] == "yellow")
        if nb_red >= 1:
            return "red"
        if nb_yellow >= 1:
            return "yellow"
        return "green"

    def _clean_result(self) -> dict[str, Any]:
        return {
            "niveau_global": "green",
            "signaux": [],
            "chevaux_suspects": [],
            "nb_signaux": 0,
            "mode_test": self.mode_test,
            "detected_at": datetime.now().isoformat(),
        }

    # ── Formatage Telegram ────────────────────────────────────

    def format_alert(self, hades_result: dict[str, Any]) -> str:
        niveau = hades_result["niveau_global"]
        signaux = hades_result["signaux"]
        emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(niveau, "⚪")
        texte = {"green": "NORMAL", "yellow": "PRUDENCE", "red": "FORTE SUSPICION"}.get(niveau, "?")

        msg = f"{emoji} <b>HADES : {texte}</b>"
        if hades_result.get("mode_test"):
            msg += " <i>(mode test)</i>"
        msg += "\n\n"

        if not signaux:
            msg += "Aucune anomalie détectée.\n"
        else:
            msg += f"<b>{len(signaux)} signal(aux) :</b>\n\n"
            for s in sorted(signaux, key=lambda x: 0 if x["niveau"] == "red" else 1)[:3]:
                e = "🔴" if s["niveau"] == "red" else "🟡"
                msg += f"{e} {s['message']}\n"
            if len(signaux) > 3:
                msg += f"\n… et {len(signaux) - 3} autre(s)\n"

        return msg
