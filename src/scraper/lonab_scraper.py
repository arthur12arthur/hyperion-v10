"""
Hyperion V10 — LONABScraper
Téléchargement programme hippique LONAB depuis https://lonab.bf/programme-pmub

Stratégie :
1. Charger la page index HTML
2. Parser tous les liens .pdf avec BeautifulSoup
3. Matcher la date du jour avec plusieurs patterns
4. Télécharger + valider le PDF
5. Cache local pour éviter le re-téléchargement
"""

import re
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ..utils.config import config
from ..utils.logger import logger, log_success, log_error, log_warning, log_processing


class ScraperStatus(Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"


class ScraperResult:
    def __init__(
        self,
        status: ScraperStatus,
        pdf_path: Optional[Path] = None,
        url: Optional[str] = None,
        reason: str = "",
    ):
        self.status = status
        self.pdf_path = pdf_path
        self.url = url
        self.reason = reason

    def __bool__(self) -> bool:
        return self.status == ScraperStatus.FOUND


class LONABScraper:
    """Scraper robuste pour le programme LONAB quotidien."""

    INDEX_URL = "https://lonab.bf/programme-pmub"
    BASE_SITE = "https://lonab.bf"

    def __init__(self):
        self.timeout = config.get("scraper.timeout_seconds", 30)
        self.retry_attempts = config.get("scraper.retry_attempts", 3)
        self.retry_delay = config.get("scraper.retry_delay_seconds", 8)

        self.headers = {
            "User-Agent": config.get("scraper.user_agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.5",
            "Connection": "keep-alive",
        }

        self.cache_dir = config.get_path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"✅ LONABScraper initialisé — {self.INDEX_URL}")

    # ── API publique ──────────────────────────────────────────

    def get_program_status(
        self,
        date: Optional[datetime] = None,
        force_download: bool = False,
    ) -> ScraperResult:
        """
        Point d'entrée principal.

        Returns:
            ScraperResult avec status FOUND / NOT_FOUND / UNAVAILABLE
        """
        if date is None:
            date = datetime.now()

        log_processing(f"Recherche programme LONAB pour le {date.strftime('%d/%m/%Y')}")

        # 1. Cache
        if not force_download:
            cached = self._get_cached_pdf(date)
            if cached:
                log_success(f"Programme trouvé en cache : {cached.name}")
                return ScraperResult(ScraperStatus.FOUND, pdf_path=cached, url=self.INDEX_URL)

        # 2. Charger la page index
        html = self._fetch_index_page()
        if html is None:
            return ScraperResult(
                ScraperStatus.UNAVAILABLE,
                reason="Impossible de joindre lonab.bf/programme-pmub",
            )

        # 3. Trouver le lien PDF pour la date
        pdf_url = self._find_pdf_url_for_date(html, date)
        if pdf_url is None:
            log_warning(f"Aucun PDF trouvé pour le {date.strftime('%d/%m/%Y')}")
            return ScraperResult(
                ScraperStatus.NOT_FOUND,
                reason=f"Aucun lien PDF exact pour {date.strftime('%d/%m/%Y')}",
            )

        # 4. Télécharger
        pdf_path = self._download_pdf(pdf_url, date)
        if pdf_path is None:
            return ScraperResult(
                ScraperStatus.UNAVAILABLE,
                reason=f"URL trouvée ({pdf_url}) mais téléchargement échoué",
            )

        log_success(f"PDF téléchargé : {pdf_path.name}")
        return ScraperResult(ScraperStatus.FOUND, pdf_path=pdf_path, url=pdf_url)

    # ── Page index ────────────────────────────────────────────

    def _fetch_index_page(self) -> Optional[str]:
        """Charge la page index LONAB. Retourne HTML ou None."""
        for attempt in range(1, self.retry_attempts + 1):
            try:
                resp = requests.get(
                    self.INDEX_URL,
                    headers=self.headers,
                    timeout=self.timeout,
                    allow_redirects=True,
                )
                if resp.status_code == 200:
                    return resp.text
                log_warning(f"Page index HTTP {resp.status_code} (tentative {attempt})")

            except requests.exceptions.Timeout:
                log_warning(f"Timeout page index (tentative {attempt}/{self.retry_attempts})")
            except requests.exceptions.RequestException as e:
                log_error(f"Erreur réseau page index : {e}")
                return None

            if attempt < self.retry_attempts:
                time.sleep(self.retry_delay)

        return None

    # ── Extraction lien PDF ───────────────────────────────────

    def _build_date_patterns(self, date: datetime) -> list[str]:
        """
        Patterns de date observés sur lonab.bf :
        - 13-05-2026 (standard)
        - 12-05_2026 (tiret+underscore)
        - 12_05_2026 (tout underscores)
        """
        dd = date.strftime("%d")
        mm = date.strftime("%m")
        yyyy = date.strftime("%Y")
        return [
            f"{dd}-{mm}-{yyyy}",
            f"{dd}-{mm}_{yyyy}",
            f"{dd}_{mm}_{yyyy}",
            f"{dd}_{mm}-{yyyy}",
        ]

    def _find_pdf_url_for_date(self, html: str, date: datetime) -> Optional[str]:
        """
        Parse le HTML et retourne l'URL PDF exacte pour la date donnée.
        Pas de fallback sur date voisine — NOT_FOUND propre si introuvable.
        """
        soup = BeautifulSoup(html, "html.parser")
        patterns = self._build_date_patterns(date)

        # Collecter tous les liens PDF
        all_pdf_links: list[str] = []
        for tag in soup.find_all(["a", "iframe", "embed", "object"]):
            href = tag.get("href") or tag.get("src") or tag.get("data") or ""
            if ".pdf" not in href.lower():
                continue
            full_url = href if href.startswith("http") else urljoin(self.BASE_SITE, href)
            all_pdf_links.append(full_url)

        logger.info(f"🔍 {len(all_pdf_links)} lien(s) PDF trouvé(s) sur la page LONAB")
        for url in all_pdf_links:
            logger.debug(f"   PDF disponible : {url}")

        # Recherche pattern exact
        for url in all_pdf_links:
            for pattern in patterns:
                if pattern in url:
                    logger.info(f"✅ PDF exact trouvé (pattern '{pattern}') : {url}")
                    return url

        log_warning(
            f"Aucun PDF exact pour {date.strftime('%d/%m/%Y')}. "
            f"Patterns testés : {patterns}. "
            f"Liens disponibles : {all_pdf_links}"
        )
        return None

    # ── Téléchargement ────────────────────────────────────────

    def _download_pdf(self, url: str, date: datetime) -> Optional[Path]:
        """Télécharge, sauvegarde et valide un PDF depuis une URL."""
        for attempt in range(1, self.retry_attempts + 1):
            try:
                resp = requests.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout,
                    allow_redirects=True,
                )
                if resp.status_code != 200:
                    log_warning(f"HTTP {resp.status_code} pour {url}")
                    return None

                content = resp.content
                if len(content) < 5000:
                    log_warning(f"Fichier trop petit ({len(content)} octets) — probablement erreur HTML")
                    return None

                pdf_path = self._save_pdf(content, date)

                if self._validate_pdf(pdf_path):
                    log_success(f"PDF validé ({len(content)} octets)")
                    return pdf_path

                pdf_path.unlink(missing_ok=True)
                log_warning(f"PDF invalide après validation : {url}")
                return None

            except requests.exceptions.Timeout:
                log_warning(f"Timeout PDF (tentative {attempt}/{self.retry_attempts})")
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay)
            except requests.exceptions.RequestException as e:
                log_error(f"Erreur téléchargement PDF : {e}")
                return None

        return None

    # ── Utilitaires ───────────────────────────────────────────

    def _save_pdf(self, content: bytes, date: datetime) -> Path:
        filename = f"lonab_{date.strftime('%Y%m%d')}.pdf"
        pdf_path = self.cache_dir / filename
        with open(pdf_path, "wb") as f:
            f.write(content)
        return pdf_path

    def _validate_pdf(self, pdf_path: Path) -> bool:
        """Validation en 2 étapes : taille + magic bytes PDF."""
        try:
            if pdf_path.stat().st_size < 1024:
                return False
            with open(pdf_path, "rb") as f:
                if f.read(4) != b"%PDF":
                    return False
            return True
        except Exception:
            return False

    def _get_cached_pdf(self, date: datetime) -> Optional[Path]:
        filename = f"lonab_{date.strftime('%Y%m%d')}.pdf"
        cached = self.cache_dir / filename
        if cached.exists() and self._validate_pdf(cached):
            return cached
        return None

    def clear_cache(self, older_than_days: int = 7) -> None:
        """Supprime les PDFs en cache plus vieux que N jours."""
        limit = datetime.now() - timedelta(days=older_than_days)
        deleted = 0
        for f in self.cache_dir.glob("lonab_*.pdf"):
            try:
                ds = f.stem.replace("lonab_", "")
                fd = datetime.strptime(ds, "%Y%m%d")
                if fd < limit:
                    f.unlink()
                    deleted += 1
            except Exception:
                continue
        if deleted:
            logger.info(f"🗑️ {deleted} PDF anciens supprimés du cache")
