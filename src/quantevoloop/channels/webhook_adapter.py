"""Generic webhook channel — works with DingTalk, Feishu, WeChat Work, Slack, etc."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("quantevoloop.channels.webhook")


class WebhookChannel:
    """Send messages via generic HTTP webhook POST.

    Supports common IM webhook formats:
      - DingTalk: {"msgtype": "text", "text": {"content": ...}}
      - Feishu:   {"msg_type": "text", "content": {"text": ...}}
      - Slack:    {"text": ...}
      - Generic:  {"title": ..., "message": ...}
    """

    def __init__(self, url: str, format: str = "generic"):
        self.url = url
        self.format = format
        self._enabled = bool(url)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _build_payload(self, title: str, message: str) -> dict:
        if self.format == "dingtalk":
            return {"msgtype": "text", "text": {"content": f"{title}\n{message}"}}
        elif self.format == "feishu":
            return {"msg_type": "text", "content": {"text": f"{title}\n{message}"}}
        elif self.format == "slack":
            return {"text": f"*{title}*\n{message}"}
        else:
            return {"title": title, "message": message}

    async def send(self, title: str, message: str) -> None:
        if not self._enabled:
            return
        payload = self._build_payload(title, message)
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.url, json=payload, timeout=10,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
        except ImportError:
            import urllib.request
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                self.url, data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning(f"Webhook send failed: {e}")
            raise
