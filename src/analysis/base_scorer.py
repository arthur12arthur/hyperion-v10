"""
Hyperion V10 — BaseScorer
Scoring multicritère 5 dimensions, échelle 0-10.
Les poids sont chargés depuis config.yaml et peuvent être
ajustés automatiquement par le BacktestEngine après J+30.
"""

from typing import Any

import numpy as np

from ..utils.config import config
from ..utils.logger import logger, log_success
from .data_filter import _parse_forme


class BaseScorer:
    """
    Calculateur de scores multicritères pour chaque partant.

    Dimensions :
    - historique (35%)  : cote inverse + gains + corde
    - forme_recente (25%) : musique pondérée + ajustements
    - terrain_distance (20%) : distance + aptitude piste
    - handicap (10%)    : écart poids vs référence
    - fraicheur (10%)   : âge optimal
    """

    def __init__(self):
        self._load_weights()
        self._forme_params = config.get("scoring.forme", {})
        self._handicap_params = config.get("scoring.handicap", {})
        self._fraicheur_params = config.get("scoring.fraicheur", {})
        logger.info("✅ BaseScorer initialisé")

    def _load_weights(self) -> None:
        w = config.get("scoring.weights", {})
        self.w_historique = float(w.get("historique", 0.35))
        self.w_forme = float(w.get("forme_recente", 0.25))
        self.w_terrain = float(w.get("terrain_distance", 0.20))
        self.w_handicap = float(w.get("handicap", 0.10))
        self.w_fraicheur = float(w.get("fraicheur", 0.10))
        total = self.w_historique + self.w_forme + self.w_terrain + self.w_handicap + self.w_fraicheur
        if not np.isclose(total, 1.0, atol=0.01):
            logger.warning(f"⚠️  Somme des poids = {total:.3f} ≠ 1.0")

    def get_weights(self) -> dict[str, float]:
        return {
            "historique": self.w_historique,
            "forme_recente": self.w_forme,
            "terrain_distance": self.w_terrain,
            "handicap": self.w_handicap,
            "fraicheur": self.w_fraicheur,
        }

    # ── API publique ──────────────────────────────────────────

    def score_chevaux(
        self,
        chevaux: list[dict[str, Any]],
        course_info: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Calcule les 5 scores pour chaque cheval et retourne
        la liste triée par score_global décroissant.

        Args:
            chevaux: Partants filtrés
            course_info: Infos course (distance, terrain, ...)

        Returns:
            Liste de chevaux enrichis, triés par score_global DESC
        """
        logger.info(f"📊 Calcul scores pour {len(chevaux)} chevaux")
        scored = []

        for ch in chevaux:
            s_hist = self._score_historique(ch)
            s_forme = self._score_forme(ch)
            s_terrain = self._score_terrain(ch, course_info)
            s_handi = self._score_handicap(ch)
            s_fraich = self._score_fraicheur(ch)

            s_global = (
                s_hist * self.w_historique
                + s_forme * self.w_forme
                + s_terrain * self.w_terrain
                + s_handi * self.w_handicap
                + s_fraich * self.w_fraicheur
            )

            ch_scored = ch.copy()
            ch_scored.update(
                {
                    "score_historique": round(s_hist, 2),
                    "score_forme": round(s_forme, 2),
                    "score_terrain_distance": round(s_terrain, 2),
                    "score_handicap": round(s_handi, 2),
                    "score_fraicheur": round(s_fraich, 2),
                    "score_global": round(float(np.clip(s_global, 0, 10)), 3),
                }
            )
            scored.append(ch_scored)

        scored.sort(key=lambda x: x["score_global"], reverse=True)

        if scored:
            best = scored[0]
            log_success(f"Meilleur score : {best.get('nom', '?')} ({best['score_global']}/10)")

        return scored

    # ── Dimensions ────────────────────────────────────────────

    def _score_historique(self, ch: dict[str, Any]) -> float:
        """Cote inverse (50%) + gains logarithmiques (30%) + corde (20%)."""
        score = 0.0

        # Cote
        cote = ch.get("cote")
        if cote and float(cote) > 1.0:
            s_cote = 10.0 * np.exp(-0.15 * (float(cote) - 1.5))
            score += float(np.clip(s_cote, 0, 10)) * 0.50
        else:
            score += 5.0 * 0.50

        # Gains
        gains = float(ch.get("gains_totaux") or 0)
        if gains > 0:
            s_gains = 10.0 * (1 - np.exp(-gains / 100_000))
            score += float(np.clip(s_gains, 0, 10)) * 0.30
        else:
            score += 3.0 * 0.30

        # Corde
        numero = int(ch.get("numero") or 10)
        corde = int(ch.get("corde") or numero)
        if corde <= 3:
            s_corde = 8.0
        elif corde <= 6:
            s_corde = 6.0
        elif corde <= 10:
            s_corde = 4.0
        else:
            s_corde = 2.0
        score += s_corde * 0.20

        return float(np.clip(score, 0, 10))

    def _score_forme(self, ch: dict[str, Any]) -> float:
        """Musique des 5 dernières courses, avec ajustements cote/âge/poids."""
        forme = ch.get("forme", "") or ""
        if not forme:
            return 5.0

        positions = _parse_forme(forme)
        if not positions:
            return 5.0

        fp = self._forme_params
        mapping = {
            1: fp.get("position_1", 10.0),
            2: fp.get("position_2", 8.5),
            3: fp.get("position_3", 7.0),
            4: fp.get("position_4", 5.5),
            5: fp.get("position_5", 4.0),
            6: fp.get("position_6", 2.5),
        }

        scores_pos = []
        for pos in positions[:5]:
            if pos in mapping:
                scores_pos.append(mapping[pos])
            else:
                scores_pos.append(fp.get("position_sup_6", 1.5))

        base = float(np.mean(scores_pos))

        # Ajustement cote
        cote = ch.get("cote")
        if cote:
            c = float(cote)
            if base > 8.0 and c > 10.0:
                base *= 0.90
            elif base < 4.0 and c < 4.0:
                base *= 1.10

        # Ajustement âge
        age = ch.get("age")
        if age is not None:
            a = int(age)
            if a < 3:
                base *= 0.95
            elif a > 10:
                base *= 0.90

        # Ajustement poids
        poids = ch.get("poids")
        if poids is not None and float(poids) > 60:
            base *= 0.95

        return float(np.clip(base, 0, 10))

    def _score_terrain(self, ch: dict[str, Any], course_info: dict[str, Any]) -> float:
        """Adéquation distance + corde."""
        distance = int(course_info.get("distance") or 2400)

        if distance > 2500:
            s_dist = 7.5
        elif distance > 2000:
            s_dist = 6.5
        elif distance > 1600:
            s_dist = 5.5
        else:
            s_dist = 5.0

        score = s_dist * 0.60

        numero = int(ch.get("numero") or 10)
        corde = int(ch.get("corde") or numero)
        if corde <= 2:
            s_corde = 8.0
        elif corde <= 5:
            s_corde = 6.5
        elif corde <= 8:
            s_corde = 5.0
        else:
            s_corde = 3.5

        score += s_corde * 0.40
        return float(np.clip(score, 0, 10))

    def _score_handicap(self, ch: dict[str, Any]) -> float:
        """Pénalité/bonus selon poids vs référence."""
        poids = ch.get("poids")
        if poids is None:
            return 7.5

        hp = self._handicap_params
        poids_ref = float(hp.get("poids_reference", 58.0))
        penalite = float(hp.get("penalite_par_kg", 0.4))
        ecart = float(poids) - poids_ref

        if ecart <= 0:
            score = 7.5 + abs(ecart) * 0.2
        else:
            score = 7.5 - ecart * penalite

        return float(np.clip(score, 0, 10))

    def _score_fraicheur(self, ch: dict[str, Any]) -> float:
        """Score d'âge optimal."""
        age = ch.get("age")
        if age is None:
            return 6.5

        fp = self._fraicheur_params
        age_min = int(fp.get("age_ideal_min", 4))
        age_max = int(fp.get("age_ideal_max", 6))
        s_ideal = float(fp.get("score_ideal", 7.5))
        s_3ans = float(fp.get("score_3_ans", 6.5))
        s_7ans = float(fp.get("score_7_ans", 6.0))
        degr = float(fp.get("degressif_apres_7", 0.5))

        a = int(age)
        if age_min <= a <= age_max:
            return s_ideal
        elif a == 3:
            return s_3ans
        elif a == 7:
            return s_7ans
        elif a > 7:
            return max(2.0, s_7ans - (a - 7) * degr)
        else:
            return 5.0

    # ── Variante pour consensus interne ──────────────────────

    def create_variant(self, seed: int) -> "BaseScorer":
        """
        Crée une copie avec poids perturbés de ±5%.
        Utilisé par InternalConsensus pour les variantes Borda.
        """
        rng = np.random.default_rng(seed)
        pct = float(config.get("consensus.internal.variation_weights_pct", 0.05))

        variant = BaseScorer.__new__(BaseScorer)
        variant._forme_params = self._forme_params
        variant._handicap_params = self._handicap_params
        variant._fraicheur_params = self._fraicheur_params

        raw = rng.uniform(1 - pct, 1 + pct, size=5)
        weights = np.array([
            self.w_historique * raw[0],
            self.w_forme * raw[1],
            self.w_terrain * raw[2],
            self.w_handicap * raw[3],
            self.w_fraicheur * raw[4],
        ])
        weights /= weights.sum()

        variant.w_historique = float(weights[0])
        variant.w_forme = float(weights[1])
        variant.w_terrain = float(weights[2])
        variant.w_handicap = float(weights[3])
        variant.w_fraicheur = float(weights[4])

        return variant
