"""
Hyperion V10 — TelegramBot
Envoi de messages HTML vers Telegram avec retry × 3 et fallback texte brut.
"""

import asyncio
import time
from typing import Optional

import httpx

from ..utils.config import config
from ..utils.logger import logger, log_success, log_warning, log_error


class TelegramBot:
    """
    Bot Telegram pour notifications Hyperion V10.

    Fonctionnalités :
    - Envoi HTML (parse_mode=HTML)
    - Retry × 3 avec backoff exponentiel
    - Fallback texte brut si HTML refusé (400)
    - Découpage automatique si message > 4096 chars
    """

    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self):
        self.token = config.get_env("TELEGRAM_BOT_TOKEN")
        self.chat_id = config.get_env("TELEGRAM_CHAT_ID")

        if not self.token or not self.chat_id:
            log_warning(
                "⚠️  Telegram non configuré "
                "(TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquants) — mode console"
            )
            self.enabled = False
        else:
            self.enabled = True
            self.api_url = self.BASE_URL.format(token=self.token)
            logger.info("✅ TelegramBot initialisé")

        self.max_length = config.get("telegram.max_message_length", 4096)
        self.retry_attempts = config.get("telegram.retry_attempts", 3)

    # ── Envoi principal ───────────────────────────────────────

    def send_message_sync(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Envoie un message de façon synchrone.
        Si le message dépasse 4096 chars, il est découpé automatiquement.
        """
        if not self.enabled:
            logger.info(f"[TELEGRAM CONSOLE]\n{text[:200]}...")
            return True

        # Découpage si nécessaire
        chunks = self._split_message(text)
        all_ok = True
        for chunk in chunks:
            ok = self._send_with_retry(chunk, parse_mode)
            if not ok:
                all_ok = False
            if len(chunks) > 1:
                time.sleep(0.5)
        return all_ok

    def send_messages_sync(self, messages: list[str]) -> bool:
        """Envoie plusieurs messages séquentiellement."""
        all_ok = True
        for msg in messages:
            ok = self.send_message_sync(msg)
            if not ok:
                all_ok = False
            time.sleep(0.3)
        return all_ok

    # ── Retry ─────────────────────────────────────────────────

    def _send_with_retry(self, text: str, parse_mode: str) -> bool:
        """Tente l'envoi avec retry × 3 et fallback texte brut."""
        for attempt in range(1, self.retry_attempts + 1):
            try:
                status, body = self._http_send(text, parse_mode)

                if status == 200:
                    log_success(f"Message Telegram envoyé ({len(text)} chars)")
                    return True

                if status == 400 and parse_mode != "":
                    # Erreur de formatage HTML : réessayer en texte brut
                    log_warning(f"Telegram 400 — fallback texte brut (tentative {attempt})")
                    clean = self._strip_html(text)
                    status2, _ = self._http_send(clean, "")
                    if status2 == 200:
                        log_success("Message envoyé en texte brut (fallback)")
                        return True

                log_warning(f"Telegram erreur {status} (tentative {attempt}/{self.retry_attempts})")

            except Exception as e:
                log_warning(f"Telegram exception (tentative {attempt}) : {e}")

            if attempt < self.retry_attempts:
                wait = 30 * attempt  # 30s, 60s, 90s
                time.sleep(wait)

        log_error("Échec envoi Telegram après 3 tentatives — archivage local")
        self._archive_failed(text)
        return False

    # ── HTTP ──────────────────────────────────────────────────

    def _http_send(self, text: str, parse_mode: str) -> tuple[int, str]:
        """Effectue la requête HTTP POST vers l'API Telegram."""
        payload: dict = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        with httpx.Client(timeout=15) as client:
            resp = client.post(f"{self.api_url}/sendMessage", json=payload)
            return resp.status_code, resp.text

    # ── Utilitaires ───────────────────────────────────────────

    def _split_message(self, text: str) -> list[str]:
        """Découpe un message trop long en morceaux de max_length chars."""
        if len(text) <= self.max_length:
            return [text]
        chunks = []
        while text:
            if len(text) <= self.max_length:
                chunks.append(text)
                break
            # Couper sur un saut de ligne si possible
            cut = text.rfind("\n", 0, self.max_length)
            if cut == -1:
                cut = self.max_length
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return chunks

    def _strip_html(self, text: str) -> str:
        """Retire les balises HTML pour le fallback texte brut."""
        import re
        clean = re.sub(r"<[^>]+>", "", text)
        # Nettoyer les espaces multiples
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        return clean.strip()

    def _archive_failed(self, text: str) -> None:
        """Archive le message échoué localement."""
        import os
        from datetime import datetime
        try:
            os.makedirs("backup/telegram", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"backup/telegram/failed_{ts}.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Message archivé : {path}")
        except Exception as e:
            log_error(f"Impossible d'archiver le message : {e}")

    def test_connection(self) -> bool:
        """Test de connexion Telegram."""
        if not self.enabled:
            return False
        from datetime import datetime
        msg = (
            f"✅ <b>HYPERION V10</b> — Test connexion OK\n"
            f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        )
        return self.send_message_sync(msg)
