"""
Hyperion V10 — DataFilter
Filtrage préalable des partants avant analyse lourde.
Élimine les outsiders irréalistes, conserve au minimum N chevaux.
"""

from collections import Counter
from typing import Any

from ..utils.config import config
from ..utils.logger import logger, log_success, log_warning


class DataFilter:
    """Filtre préalable des partants."""

    def __init__(self):
        self.min_retenus = config.get("filter.min_chevaux_retenus", 5)
        self.force_keep_top = config.get("filter.force_keep_top_by_cote", 5)
        self.gains_min = config.get("filter.elimination.gains_min", 10000)
        self.place_max = config.get("filter.elimination.derniere_place_max", 8)
        self.nb_courses_check = config.get("filter.elimination.nb_courses_forme_check", 3)
        self.cote_max = config.get("filter.elimination.cote_implicite_max", 50.0)
        logger.info("✅ DataFilter initialisé")

    # ── API publique ──────────────────────────────────────────

    def filter_chevaux(
        self,
        chevaux: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Filtre les chevaux selon les critères d'élimination.

        Args:
            chevaux: Liste brute des partants

        Returns:
            (chevaux_retenus, chevaux_elimines)
            Chaque éliminé : {'cheval': dict, 'motifs': [str], 'score': int}
        """
        if not chevaux:
            return [], []

        logger.info(f"🔍 Filtrage préalable sur {len(chevaux)} partants")

        # Évaluation individuelle
        evaluations = []
        for ch in chevaux:
            score, motifs = self._evaluate(ch)
            evaluations.append({
                "cheval": ch,
                "score": score,
                "motifs": motifs,
                "elimine": score >= 3,
            })

        # Trier par cote croissante (favoris en premier)
        evaluations.sort(key=lambda x: x["cheval"].get("cote") or 999)

        # Forcer conservation du top N par cote
        for i in range(min(self.force_keep_top, len(evaluations))):
            if evaluations[i]["elimine"]:
                evaluations[i]["elimine"] = False
                evaluations[i]["motifs"].append("⚠️ CONSERVÉ (top cote)")

        retenus = [e["cheval"] for e in evaluations if not e["elimine"]]
        elimines = [
            {"cheval": e["cheval"], "motifs": e["motifs"], "score": e["score"]}
            for e in evaluations if e["elimine"]
        ]

        # Garantir le minimum
        if len(retenus) < self.min_retenus:
            manquants = self.min_retenus - len(retenus)
            elimines.sort(key=lambda x: x["score"])
            for i in range(min(manquants, len(elimines))):
                reintegre = elimines.pop(0)
                retenus.append(reintegre["cheval"])
                log_warning(
                    f"⚠️ Réintégration {reintegre['cheval'].get('nom', '?')} "
                    f"(minimum {self.min_retenus} chevaux requis)"
                )

        log_success(f"Filtre : {len(retenus)} retenus, {len(elimines)} éliminés")
        return retenus, elimines

    # ── Évaluation individuelle ───────────────────────────────

    def _evaluate(self, cheval: dict[str, Any]) -> tuple[int, list[str]]:
        """
        Évalue un cheval et retourne (score_elimination, motifs).
        score >= 3 → éliminé.
        """
        motifs: list[str] = []
        score = 0

        forme = cheval.get("forme", "") or ""
        gains = cheval.get("gains_totaux", 0) or 0
        cote = cheval.get("cote")

        # Critère 1 : Gains faibles ET mauvaises places récentes
        if gains < self.gains_min and forme:
            positions = _parse_forme(forme)
            if positions:
                dernieres = positions[: self.nb_courses_check]
                if all(p > self.place_max for p in dernieres):
                    motifs.append(
                        f"Gains < {self.gains_min} FCFA ET "
                        f"aucun top {self.place_max} sur {len(dernieres)} dernières courses"
                    )
                    score += 1

        # Critère 2 : Cote outsider extrême
        if cote and cote > self.cote_max:
            motifs.append(f"Cote > {self.cote_max} (outsider extrême)")
            score += 1

        # Critère 3 : Aucune forme disponible
        if not forme or len(forme.strip()) < 2:
            motifs.append("Aucune forme récente disponible")
            score += 1

        # Critère 4 : Forme catastrophique (tout > 10e)
        if forme:
            positions = _parse_forme(forme)
            if positions and len(positions) >= 3 and all(p > 10 for p in positions[:5]):
                motifs.append("Forme catastrophique (toutes places > 10e)")
                score += 2

        return score, motifs

    # ── Stats filtrage ────────────────────────────────────────

    def get_stats(self, elimines: list[dict[str, Any]]) -> dict[str, Any]:
        """Génère des statistiques de filtrage."""
        if not elimines:
            return {"nb_elimines": 0, "motifs_frequents": []}
        all_motifs = []
        for e in elimines:
            all_motifs.extend(e.get("motifs", []))
        motifs_count = Counter(all_motifs)
        return {
            "nb_elimines": len(elimines),
            "motifs_frequents": motifs_count.most_common(5),
            "score_moyen": sum(e.get("score", 0) for e in elimines) / len(elimines),
        }


# ── Utilitaires ────────────────────────────────────────────────

def _parse_forme(forme: str) -> list[int]:
    """
    Parse une chaîne de forme en liste de positions entières.
    Abandons/disqualifications → 99.
    Ex: "1p2p3p4a5p" → [1, 2, 3, 99, 5]
    """
    text = forme
    text = text.replace("p", " ").replace("P", " ")
    text = re.sub(r"[aAdDtT]", " 99 ", text)
    positions = []
    for token in text.split():
        try:
            positions.append(int(token))
        except ValueError:
            continue
    return positions


import re  # noqa: E402 — placé après pour éviter un import circulaire
