"""
Hyperion V10 — MonteCarloSimulator
10 000 sims × 5 seeds = 50 000 simulations totales, vectorisé NumPy.
Déterministe : même input → même output garanti.
"""

from typing import Any

import hashlib
import numpy as np

from ..utils.config import config
from ..utils.logger import logger, log_success, log_processing


class MonteCarloSimulator:
    """
    Simulateur Monte Carlo vectorisé haute performance.

    Architecture :
    - 5 seeds indépendants (42-46)
    - 10 000 sims par seed = 50 000 totales
    - Perturbation gaussienne sur les scores de base
    - Consensus Borda sur les 5 résultats
    - Détection robustesse (top3 stable >= 4/5 seeds)
    """

    def __init__(self):
        self.nb_simulations = config.get("monte_carlo.nb_simulations", 10000)
        self.seeds = config.get("monte_carlo.seeds", [42, 43, 44, 45, 46])
        self.variance_std = config.get("monte_carlo.variance_std", 0.25)
        self.deterministic = config.get("monte_carlo.deterministic", True)
        self.robustesse_threshold = config.get("monte_carlo.robustesse_threshold", 0.80)
        logger.info(
            f"✅ MonteCarloSimulator initialisé — "
            f"{self.nb_simulations} sims × {len(self.seeds)} seeds = "
            f"{self.nb_simulations * len(self.seeds):,} totales"
        )

    # ── API publique ──────────────────────────────────────────

    def simulate(
        self,
        chevaux: list[dict[str, Any]],
        course_id: str = "default",
    ) -> dict[str, Any]:
        """
        Lance les simulations et retourne consensus + statistiques.

        Args:
            chevaux: Partants avec 'score_global'
            course_id: ID pour seed déterministe

        Returns:
            Dict {
                'consensus': [...],     ← Top 5 enrichi pour MetaFusion
                'statistiques': {...},  ← Stats par numéro
                'simulations': [...],   ← Exemples de simulations (100 max)
                'is_robust': bool,
                'seeds': [...],
            }
        """
        if not chevaux:
            raise ValueError("Aucun cheval à simuler")

        log_processing(f"🎲 Monte Carlo : {len(chevaux)} partants, course_id={course_id}")

        numeros = np.array([int(ch["numero"]) for ch in chevaux])
        noms = [ch.get("nom", f"Cheval {ch['numero']}") for ch in chevaux]
        scores_base = np.array([float(ch.get("score_global", 5.0)) for ch in chevaux])

        # Seed de base basé sur course_id (déterminisme)
        base_seed = self._course_seed(course_id) if self.deterministic else None

        # ── Lancement 5 seeds ─────────────────────────────────
        seeds_rankings: list[list[int]] = []
        win_counts_all = np.zeros(len(chevaux))
        place_counts_all = np.zeros(len(chevaux))
        top3_counts_all = np.zeros(len(chevaux))
        score_pondere_all = np.zeros(len(chevaux))

        sample_simulations: list[dict] = []

        for seed_offset in self.seeds:
            seed = ((base_seed or 0) + seed_offset) % (2**32) if base_seed else seed_offset
            np.random.seed(seed)

            # Matrice [nb_simulations × nb_chevaux] de scores perturbés
            noise = np.random.normal(0.0, self.variance_std, size=(self.nb_simulations, len(chevaux)))
            scores_mat = scores_base + noise  # broadcast

            # Rang de chaque cheval dans chaque simulation (argsort desc)
            sorted_indices = np.argsort(scores_mat, axis=1)[:, ::-1]

            # Compteurs vectorisés
            for pos in range(min(5, len(chevaux))):
                col = sorted_indices[:, pos]  # index du cheval en position `pos`
                pts = len(chevaux) - pos  # Borda points
                score_pondere_all += np.bincount(col, minlength=len(chevaux)) * pts

            win_counts_all += np.bincount(sorted_indices[:, 0], minlength=len(chevaux))
            if len(chevaux) >= 2:
                place_counts_all += np.bincount(sorted_indices[:, 0], minlength=len(chevaux))
                place_counts_all += np.bincount(sorted_indices[:, 1], minlength=len(chevaux))
            for p in range(min(3, len(chevaux))):
                top3_counts_all += np.bincount(sorted_indices[:, p], minlength=len(chevaux))

            # Ranking Borda pour cette seed
            borda_seed = np.argsort(score_pondere_all)[::-1]
            seeds_rankings.append([int(numeros[i]) for i in borda_seed[:5]])

            # Quelques simulations en exemple (50 par seed, max 100 total)
            if len(sample_simulations) < 100:
                for sim_idx in range(min(50, self.nb_simulations)):
                    top5_idx = sorted_indices[sim_idx, :5]
                    sample_simulations.append({
                        "top5": [int(numeros[i]) for i in top5_idx],
                        "top5_noms": [noms[i] for i in top5_idx],
                    })

        nb_seeds = len(self.seeds)
        total_sims = self.nb_simulations * nb_seeds

        # ── Consensus Borda global ─────────────────────────────
        final_order = np.argsort(score_pondere_all)[::-1]
        consensus_numeros = [int(numeros[i]) for i in final_order[:5]]

        # ── Robustesse ─────────────────────────────────────────
        top3_per_seed = [tuple(r[:3]) for r in seeds_rankings]
        most_common_top3 = max(set(top3_per_seed), key=top3_per_seed.count)
        stability = top3_per_seed.count(most_common_top3) / nb_seeds
        is_robust = stability >= self.robustesse_threshold

        # ── Statistiques par cheval ────────────────────────────
        stats: dict[int, dict] = {}
        ch_by_num = {int(ch["numero"]): ch for ch in chevaux}

        for idx, num in enumerate(numeros):
            num_int = int(num)
            win_prob = float(win_counts_all[idx]) / total_sims
            place_prob = float(place_counts_all[idx]) / total_sims
            top3_prob = float(top3_counts_all[idx]) / total_sims
            score_pond = float(score_pondere_all[idx])

            stats[num_int] = {
                "numero": num_int,
                "nom": noms[idx],
                "win_prob": round(win_prob, 4),
                "place_prob": round(place_prob, 4),
                "top3_prob": round(top3_prob, 4),
                "confiance_mc": round(top3_prob, 3),
                "score_pondere": round(score_pond, 1),
                "borda_rank": int(np.where(final_order == idx)[0][0]) + 1,
            }

        # ── Consensus enrichi (pour MetaFusion) ───────────────
        consensus: list[dict[str, Any]] = []
        for rank, num in enumerate(consensus_numeros, start=1):
            ch = ch_by_num.get(num, {})
            st = stats.get(num, {})
            consensus.append({
                **ch,
                **st,
                "position": rank,
                "robuste": is_robust and rank <= 3,
            })

        log_success(
            f"Monte Carlo terminé — {total_sims:,} sims, "
            f"robustesse {stability:.0%} ({'✅' if is_robust else '⚠️'})"
        )

        return {
            "consensus": consensus,
            "statistiques": stats,
            "simulations": sample_simulations,
            "is_robust": is_robust,
            "stability": round(stability, 3),
            "seeds": self.seeds,
            "total_simulations": total_sims,
        }

    # ── Utilitaire seed ───────────────────────────────────────

    def _course_seed(self, course_id: str) -> int:
        """Génère un seed entier 32-bit déterministe depuis l'ID de course."""
        h = hashlib.md5(course_id.encode()).hexdigest()
        return int(h, 16) % (2**32)
