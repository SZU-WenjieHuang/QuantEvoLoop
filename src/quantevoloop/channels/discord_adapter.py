"""Discord channel adapter — sends notifications via Discord Webhook/Bot."""

from __future__ import annotations

import logging

logger = logging.getLogger("quantevoloop.channels.discord")


class DiscordChannel:
    """Send messages via Discord webhook URL or Bot API."""

    def __init__(self, bot_token: str, channel_id: str):
        self.bot_token = bot_token
        self.channel_id = channel_id
        self._enabled = bool(bot_token and channel_id)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def send(self, title: str, message: str) -> None:
        if not self._enabled:
            return
        try:
            import httpx
            # Use webhook-style URL (works without bot gateway)
            url = f"https://discord.com/api/v10/channels/{self.channel_id}/messages"
            headers = {
                "Authorization": f"Bot {self.bot_token}",
                "Content-Type": "application/json",
            }
            payload = {"content": f"**{title}**\n{message}"}
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=10)
                resp.raise_for_status()
        except ImportError:
            import urllib.request
            import urllib.parse
            import json
            url = f"https://discord.com/api/v10/channels/{self.channel_id}/messages"
            data = json.dumps({"content": f"**{title}**\n{message}"}).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={
                    "Authorization": f"Bot {self.bot_token}",
                    "Content-Type": "application/json",
                },
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning(f"Discord send failed: {e}")
            raise
