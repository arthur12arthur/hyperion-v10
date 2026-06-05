"""
Hyperion V10 — Consensus (interne + externe + méta-fusion)

Regroupe en un seul module :
- InternalConsensusBuilder : Borda sur 3 variantes du scorer
- ExternalConsensusBuilder : Gemini Search → pronostics experts
- MetaFusion : fusion pondérée interne/externe + pairwise tie-breaking
"""

import json
import re
from typing import Any, Optional

import numpy as np

from ..utils.config import config
from ..utils.logger import logger, log_success, log_warning, log_error, log_processing
from ..utils.gemini_manager import gemini_manager
from .base_scorer import BaseScorer


# ═══════════════════════════════════════════════════════════════
# CONSENSUS INTERNE — Borda sur variantes du scorer
# ═══════════════════════════════════════════════════════════════

class InternalConsensusBuilder:
    """Génère N variantes du scorer (poids ±5%) et agrège par Borda."""

    def __init__(self):
        self.nb_variantes = config.get("consensus.internal.nb_variantes", 3)
        self.borda_pts = {
            1: config.get("consensus.internal.borda.points_position_1", 5),
            2: config.get("consensus.internal.borda.points_position_2", 4),
            3: config.get("consensus.internal.borda.points_position_3", 3),
            4: config.get("consensus.internal.borda.points_position_4", 2),
            5: config.get("consensus.internal.borda.points_position_5", 1),
        }
        logger.info(f"✅ InternalConsensusBuilder — {self.nb_variantes} variantes")

    def build(
        self,
        chevaux: list[dict[str, Any]],
        course_info: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Construit le consensus interne.

        Returns:
            {
                'consensus_borda': [num1, num2, ...],  ← liste numéros top5
                'scores_borda': {num: score, ...},
                'variantes': [...],
            }
        """
        log_processing(f"🔄 Génération {self.nb_variantes} variantes scorer")

        base_scorer = BaseScorer()
        variantes_rankings: list[list[int]] = []
        variantes_details: list[dict] = []

        # Variante 0 = poids nominaux
        for i in range(self.nb_variantes):
            scorer = base_scorer if i == 0 else base_scorer.create_variant(seed=42 + i)
            chevaux_variant = scorer.score_chevaux(chevaux, course_info)
            top5_nums = [ch["numero"] for ch in chevaux_variant[:5]]
            variantes_rankings.append(top5_nums)
            variantes_details.append({
                "variante": i + 1,
                "top5": top5_nums,
                "weights": scorer.get_weights(),
            })

        # Agrégation Borda
        scores_borda: dict[int, int] = {}
        for ranking in variantes_rankings:
            for pos, num in enumerate(ranking, start=1):
                pts = self.borda_pts.get(pos, 0)
                scores_borda[num] = scores_borda.get(num, 0) + pts

        consensus_borda = sorted(scores_borda, key=scores_borda.get, reverse=True)[:5]  # type: ignore[arg-type]

        log_success(f"Consensus Borda interne : {consensus_borda}")

        return {
            "consensus_borda": consensus_borda,
            "scores_borda": scores_borda,
            "variantes": variantes_details,
            "nb_variantes": self.nb_variantes,
        }


# ═══════════════════════════════════════════════════════════════
# CONSENSUS EXTERNE — Gemini Search + pronostics experts
# ═══════════════════════════════════════════════════════════════

class ExternalConsensusBuilder:
    """Collecte pronostics externes via Gemini Search."""

    def __init__(self):
        self.max_sources = config.get("consensus.external.max_sources", 7)
        self.enabled = config.get("consensus.external.gemini_search_enabled", True)
        sources_cfg = config.get_sources().get("press_sources", [])
        self.sources_names = [s["name"] for s in sources_cfg if s.get("enabled", True)]
        logger.info(
            f"✅ ExternalConsensusBuilder — {len(self.sources_names)} sources, "
            f"Gemini Search {'activé' if self.enabled else 'désactivé'}"
        )

    def collect(self, course_info: dict[str, Any]) -> dict[str, Any]:
        """
        Collecte et agrège les pronostics externes.

        Returns:
            {
                'chevaux_stats': {num: {external_frequency, external_score, ...}},
                'consensus_externe': [num1, ...],
                'nb_sources': int,
                'qualite': 'DISPONIBLE'|'INDISPONIBLE',
            }
        """
        if not self.enabled:
            return self._empty_result()

        nom = course_info.get("id_course") or course_info.get("nom", "")
        hippodrome = course_info.get("hippodrome", "")
        date = course_info.get("date", "")

        log_processing(f"🌐 Recherche pronostics externes : {nom}")

        query = self._build_query(nom, hippodrome, date)
        raw = self._call_gemini_search(query)

        if not raw:
            log_warning("Aucun pronostic externe obtenu")
            return self._empty_result()

        sources = self._parse(raw)
        if not sources:
            return self._empty_result()

        aggregation = self._aggregate(sources)
        log_success(f"{len(sources)} sources externes collectées")

        aggregation["qualite"] = "DISPONIBLE"
        aggregation["nb_sources"] = len(sources)
        return aggregation

    def _build_query(self, nom: str, hippodrome: str, date: str) -> str:
        nom_court = " ".join(nom.split()[:5])
        try:
            from datetime import datetime
            d = datetime.strptime(date, "%Y-%m-%d")
            date_fr = d.strftime("%d/%m/%Y")
        except Exception:
            date_fr = date
        return f"pronostic {nom_court} {hippodrome} {date_fr} turf PMU"

    def _call_gemini_search(self, query: str) -> Optional[str]:
        prompt = f"""Recherche les pronostics hippiques PMU pour : "{query}"

Sources prioritaires : {', '.join(self.sources_names)}

RÈGLE ABSOLUE : réponds UNIQUEMENT avec le JSON suivant, rien d'autre.
Commence ta réponse par {{ et termine par }}

{{"sources":[{{"nom":"NomSite","url":"https://...","top5":[1,2,3,4,5]}}]}}

Si aucun pronostic trouvé : {{"sources":[]}}
Maximum {self.max_sources} sources. Les numéros dans top5 doivent être des entiers."""

        return gemini_manager.call(
            prompt=prompt,
            temperature=0.1,
            max_output_tokens=1000,
            use_search=True,
        )

    def _parse(self, text: str) -> list[dict[str, Any]]:
        """Parse la réponse JSON du consensus externe."""
        if not text:
            return []

        clean = text.strip()
        clean = re.sub(r"```(?:json)?", "", clean)
        clean = re.sub(r"```", "", clean).strip()
        start = clean.find("{")
        end = clean.rfind("}")
        if start == -1 or end == -1:
            return []
        clean = clean[start : end + 1]

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            log_error(f"Erreur parsing JSON externe : {e}")
            return []

        sources_valid = []
        for s in data.get("sources", []):
            nom = s.get("nom")
            top5 = s.get("top5", [])
            if not nom or not isinstance(top5, list) or len(top5) == 0:
                continue
            # Forcer entiers
            top5_int = []
            for x in top5:
                try:
                    top5_int.append(int(x))
                except (ValueError, TypeError):
                    pass
            if top5_int:
                sources_valid.append({"source": nom, "url": s.get("url", ""), "top5": top5_int})

        return sources_valid

    def _aggregate(self, sources: list[dict[str, Any]]) -> dict[str, Any]:
        """Agrège les classements par fréquence et score Borda pondéré."""
        freq: dict[int, int] = {}
        positions: dict[int, list[int]] = {}

        for s in sources:
            for pos, num in enumerate(s["top5"], start=1):
                freq[num] = freq.get(num, 0) + 1
                positions.setdefault(num, []).append(pos)

        nb = len(sources)
        stats: dict[int, dict] = {}
        for num, count in freq.items():
            pos_list = positions[num]
            ext_freq = count / nb
            pts = sum(6 - p for p in pos_list)
            max_pts = 5 * count
            ext_score = pts / max_pts if max_pts > 0 else 0.0
            stats[num] = {
                "numero": num,
                "external_frequency": round(ext_freq, 3),
                "external_score": round(ext_score, 3),
                "apparitions": count,
                "position_moyenne": round(float(np.mean(pos_list)), 2),
            }

        consensus_ext = sorted(stats, key=lambda n: stats[n]["external_score"], reverse=True)[:5]

        return {
            "chevaux_stats": stats,
            "consensus_externe": consensus_ext,
        }

    def _empty_result(self) -> dict[str, Any]:
        return {
            "chevaux_stats": {},
            "consensus_externe": [],
            "nb_sources": 0,
            "qualite": "INDISPONIBLE",
        }


# ═══════════════════════════════════════════════════════════════
# MÉTA-FUSION — Fusion interne + externe + pairwise
# ═══════════════════════════════════════════════════════════════

class MetaFusion:
    """
    Fusionne le consensus interne (Monte Carlo + Borda) avec
    le consensus externe (sources experts).

    RÈGLE FONDAMENTALE :
    Le classement final = consensus interne UNIQUEMENT.
    L'externe ne peut que modifier le score de CONFIANCE.
    """

    def __init__(self):
        self.w_internal = config.get("consensus.meta_fusion.weight_internal", 0.55)
        self.w_external = config.get("consensus.meta_fusion.weight_external", 0.45)
        self.threshold_robuste = config.get("consensus.meta_fusion.threshold_robuste", 0.80)
        self.use_pairwise = config.get("consensus.meta_fusion.use_pairwise", True)
        logger.info("✅ MetaFusion initialisé")

    def fuse(
        self,
        consensus_mc: list[dict[str, Any]],       # Sortie MonteCarlo.simulate()['consensus']
        consensus_borda: list[int],                # Sortie InternalConsensus['consensus_borda']
        external_aggregation: dict[str, Any],      # Sortie ExternalConsensus.collect()
        simulations: list[dict[str, Any]],         # Sortie MonteCarlo['simulations']
    ) -> list[dict[str, Any]]:
        """
        Retourne le Top 5 final méta-fusionné.

        Returns:
            Liste de 5 dicts : {position, numero, nom, meta_score, robuste, ...}
        """
        log_processing("🔀 Méta-fusion consensus interne/externe")

        # Pool de candidats : union MC + Borda + top 3 externe
        candidats: set[int] = set()
        for ch in consensus_mc:
            candidats.add(int(ch["numero"]))
        for num in consensus_borda[:5]:
            candidats.add(num)
        ext_stats = external_aggregation.get("chevaux_stats", {})
        for num in external_aggregation.get("consensus_externe", [])[:3]:
            candidats.add(num)

        log_processing(f"🎯 {len(candidats)} candidats pour méta-fusion")

        # Calcul méta-score par candidat
        meta_scores: list[dict] = []
        for num in candidats:
            s_mc = self._get_mc_score(num, consensus_mc)
            s_ext = self._get_ext_score(num, ext_stats)
            meta = round(s_mc * self.w_internal + s_ext * self.w_external, 3)
            nom = self._get_nom(num, consensus_mc)

            meta_scores.append({
                "numero": num,
                "nom": nom,
                "score_mc": s_mc,
                "score_externe": s_ext,
                "meta_score": meta,
                "robuste": meta >= self.threshold_robuste,
            })

        meta_scores.sort(key=lambda x: x["meta_score"], reverse=True)

        # Tie-breaking pairwise
        if self.use_pairwise:
            meta_scores = self._pairwise_tiebreak(meta_scores, simulations)

        # Top 5 final
        top5: list[dict[str, Any]] = []
        for i, ch in enumerate(meta_scores[:5], start=1):
            # Enrichir avec les données complètes du cheval MC
            mc_data = next((c for c in consensus_mc if int(c["numero"]) == ch["numero"]), {})
            top5.append({
                **mc_data,
                "position": i,
                "numero": ch["numero"],
                "nom": ch["nom"],
                "meta_score": ch["meta_score"],
                "robuste": ch["robuste"],
                "score_mc": ch["score_mc"],
                "score_externe": ch["score_externe"],
            })

        nb_robustes = sum(1 for ch in top5 if ch["robuste"])
        log_success(f"Top 5 final — {nb_robustes}/5 ROBUSTES")

        return top5

    # ── Helpers ───────────────────────────────────────────────

    def _get_mc_score(self, num: int, consensus_mc: list[dict]) -> float:
        for ch in consensus_mc:
            if int(ch["numero"]) == num:
                return float(ch.get("confiance_mc", ch.get("win_prob", 0.0)))
        return 0.0

    def _get_ext_score(self, num: int, ext_stats: dict) -> float:
        s = ext_stats.get(num, {})
        return float(s.get("external_frequency", 0.0))

    def _get_nom(self, num: int, consensus_mc: list[dict]) -> str:
        for ch in consensus_mc:
            if int(ch["numero"]) == num:
                return str(ch.get("nom", f"Cheval {num}"))
        return f"Cheval {num}"

    def _pairwise_tiebreak(
        self,
        meta_scores: list[dict],
        simulations: list[dict],
    ) -> list[dict]:
        """
        Tie-breaking : si écart < 0.05, compter qui bat qui
        dans les simulations Monte Carlo.
        """
        n = len(meta_scores)
        for i in range(n - 1):
            for j in range(i + 1, n):
                ch1 = meta_scores[i]
                ch2 = meta_scores[j]
                if abs(ch1["meta_score"] - ch2["meta_score"]) >= 0.05:
                    continue

                w1 = w2 = 0
                for sim in simulations:
                    top5 = sim.get("top5", [])
                    if ch1["numero"] in top5 and ch2["numero"] in top5:
                        p1 = top5.index(ch1["numero"])
                        p2 = top5.index(ch2["numero"])
                        if p1 < p2:
                            w1 += 1
                        elif p2 < p1:
                            w2 += 1

                if w2 > w1:
                    meta_scores[i], meta_scores[j] = meta_scores[j], meta_scores[i]

        return meta_scores
