"""
Hyperion V10 — EVKellyCalculator
Calcul Expected Value (EV) et critère de Kelly fractionné.
"""

from typing import Any

import numpy as np

from ..utils.config import config
from ..utils.logger import logger, log_success


class EVKellyCalculator:
    """Calcul EV et Kelly pour les chevaux du Top 5."""

    def __init__(self):
        self.bookmaker_margin = config.get("finance.ev.bookmaker_margin", 0.15)
        self.ev_threshold = config.get("finance.ev.threshold_value_bet", 0.05)
        self.kelly_fraction = config.get("finance.kelly.fraction", 0.25)
        self.max_bet_pct = config.get("finance.kelly.max_bet_pct", 0.05)
        self.capital_refs = config.get("finance.capital_references", [1000, 10000])
        self.risk_levels = config.get("finance.risk_levels", {"faible": 0.75, "modere": 0.45})
        logger.info("✅ EVKellyCalculator initialisé")

    # ── API publique ──────────────────────────────────────────

    def calculate_all(
        self,
        top5_final: list[dict[str, Any]],
        chevaux: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Calcule EV et Kelly pour chaque cheval du Top 5.

        Returns:
            {
                'chevaux': {num: {...}},
                'value_bets': [num, ...],
                'nb_value_bets': int,
            }
        """
        logger.info("💰 Calcul EV & Kelly")

        results: dict[int, dict] = {}
        value_bets: list[int] = []

        ch_by_num = {int(ch["numero"]): ch for ch in chevaux}

        for ch_cons in top5_final:
            num = int(ch_cons["numero"])
            ch_full = ch_by_num.get(num, ch_cons)

            cote = ch_full.get("cote") or ch_cons.get("cote")
            if not cote or float(cote) <= 1.0:
                continue

            cote = float(cote)
            prob_reelle = float(ch_cons.get("meta_score", ch_cons.get("win_prob", 0.0)))

            if prob_reelle <= 0:
                continue

            prob_implicite = float(np.clip((1.0 / cote) * (1.0 - self.bookmaker_margin), 0, 1))
            cote_theorique = round(1.0 / prob_reelle, 2)

            # EV = (p × (cote-1)) − (1−p)
            ev = (prob_reelle * (cote - 1.0)) - (1.0 - prob_reelle)

            # Kelly f* = (p×b − q) / b
            b = cote - 1.0
            q = 1.0 - prob_reelle
            kelly_raw = float(np.clip((prob_reelle * b - q) / b, 0, 1))
            kelly_frac = kelly_raw * self.kelly_fraction
            kelly_cap = min(kelly_frac, self.max_bet_pct)

            mises = {
                f"capital_{cap}": round(cap * kelly_cap, 0)
                for cap in self.capital_refs
            }

            # Niveau risque
            confiance = float(ch_cons.get("confiance_mc", ch_cons.get("score_mc", 0.0)))
            if confiance >= self.risk_levels["faible"]:
                risque = "FAIBLE"
            elif confiance >= self.risk_levels["modere"]:
                risque = "MODÉRÉ"
            else:
                risque = "ÉLEVÉ"

            is_vb = ev > self.ev_threshold

            results[num] = {
                "numero": num,
                "nom": ch_full.get("nom", ch_cons.get("nom", f"Cheval {num}")),
                "cote_affichee": cote,
                "cote_theorique": cote_theorique,
                "prob_reelle": round(prob_reelle, 3),
                "prob_implicite": round(prob_implicite, 3),
                "ev": round(ev, 3),
                "ev_pct": round(ev * 100, 1),
                "kelly_raw": round(kelly_raw, 4),
                "kelly_fractionne": round(kelly_frac, 4),
                "kelly_cappe": round(kelly_cap, 4),
                "mises": mises,
                "risque": risque,
                "is_value_bet": is_vb,
            }

            if is_vb:
                value_bets.append(num)

        if value_bets:
            log_success(f"💎 {len(value_bets)} value bet(s) : {value_bets}")

        return {
            "chevaux": results,
            "value_bets": value_bets,
            "nb_value_bets": len(value_bets),
        }

    # ── Formatage Telegram ────────────────────────────────────

    def format_summary(self, data: dict[str, Any], top3_only: bool = True) -> str:
        """Formate le résumé financier pour Telegram (HTML)."""
        chevaux = data.get("chevaux", {})
        value_bets = data.get("value_bets", [])
        limit = 3 if top3_only else 5

        msg = "💰 <b>ANALYSE FINANCIÈRE</b>\n\n"
        sorted_ch = sorted(chevaux.values(), key=lambda x: x["ev"], reverse=True)

        for ch in sorted_ch[:limit]:
            e = "💎" if ch["is_value_bet"] else "📊"
            msg += f"{e} <b>N°{ch['numero']} {ch['nom']}</b>\n"
            msg += f"   Cote : {ch['cote_affichee']:.1f} (théorique {ch['cote_theorique']:.1f})\n"
            msg += f"   EV : {ch['ev_pct']:+.1f}%\n"
            if ch["is_value_bet"]:
                msg += "   ⚡ VALUE BET !\n"
                msg += f"   Kelly : {ch['kelly_cappe']*100:.1f}%\n"
                cap_key = f"capital_{self.capital_refs[0]}"
                mise = ch["mises"].get(cap_key, 0)
                msg += f"   Mise ({self.capital_refs[0]} FCFA) : {mise:.0f} FCFA\n"
            msg += f"   Risque : {ch['risque']}\n\n"

        if not value_bets:
            msg += "ℹ️ Aucun value bet identifié.\n"

        msg += "\n⚠️ <i>Outil d'analyse — aucune recommandation de mise.</i>\n"
        return msg
