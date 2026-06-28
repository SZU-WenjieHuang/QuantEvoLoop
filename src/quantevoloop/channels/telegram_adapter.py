"""Telegram channel adapter — sends notifications via Telegram Bot API."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("quantevoloop.channels.telegram")


class TelegramChannel:
    """Send messages via Telegram Bot API (python-telegram-bot or raw HTTP)."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def send(self, title: str, message: str) -> None:
        if not self._enabled:
            return
        try:
            import httpx
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": f"*{title}*\n{message}",
                "parse_mode": "Markdown",
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=10)
                resp.raise_for_status()
        except ImportError:
            # Fallback: use urllib (stdlib)
            import urllib.request
            import urllib.parse
            import json
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id,
                "text": f"*{title}*\n{message}",
                "parse_mode": "Markdown",
            }).encode()
            req = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
            raise
