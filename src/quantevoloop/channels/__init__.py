"""IM channels — Telegram, Discord, Webhook notification adapters.

Follows OpenClaw's Channel Adapter pattern:
  Each channel is an independent adapter that can send messages.
  The Notifier dispatches to all enabled channels.
  The BotCommandProcessor handles interactive commands (/pause, /diagnose, etc.)
"""

from .notifier import IMNotifier, NotifyLevel
from .telegram_adapter import TelegramChannel
from .discord_adapter import DiscordChannel
from .webhook_adapter import WebhookChannel
from .bot_commands import CommandProcessor, CommandResult, TelegramBotPoller

__all__ = [
    "IMNotifier", "NotifyLevel",
    "TelegramChannel", "DiscordChannel", "WebhookChannel",
    "CommandProcessor", "CommandResult", "TelegramBotPoller",
]
