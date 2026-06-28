"""IM channels — Telegram, Discord, Webhook notification adapters.

Follows OpenClaw's Channel Adapter pattern:
  Each channel is an independent adapter that can send messages.
  The Notifier dispatches to all enabled channels.
"""

from .notifier import IMNotifier, NotifyLevel
from .telegram_adapter import TelegramChannel
from .discord_adapter import DiscordChannel
from .webhook_adapter import WebhookChannel

__all__ = [
    "IMNotifier", "NotifyLevel",
    "TelegramChannel", "DiscordChannel", "WebhookChannel",
]
