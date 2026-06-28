"""IM Notifier — dispatches notifications to all enabled channels.

Follows OpenClaw's Channel Adapter pattern:
  - Each channel implements a simple send() interface
  - The Notifier dispatches to all enabled channels concurrently
  - Failed channels are logged but don't crash the pipeline
"""

from __future__ import annotations

import asyncio
import logging
from enum import IntEnum
from typing import Protocol

from ..config import IMConfig

logger = logging.getLogger("quantevoloop.channels")


class NotifyLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    IMPORTANT = 2
    CRITICAL = 3


class ChannelAdapter(Protocol):
    """Protocol that all channel adapters must implement."""

    async def send(self, title: str, message: str) -> None: ...

    @property
    def is_enabled(self) -> bool: ...


class IMNotifier:
    """Dispatches notifications to all enabled IM channels."""

    def __init__(self, config: IMConfig):
        self.config = config
        self._channels: list[ChannelAdapter] = []
        self._min_level = NotifyLevel[config.min_notify_level.upper()]
        self._init_channels()

    def _init_channels(self) -> None:
        if self.config.telegram_enabled:
            try:
                from .telegram_adapter import TelegramChannel
                self._channels.append(TelegramChannel(
                    bot_token=self.config.telegram_bot_token,
                    chat_id=self.config.telegram_chat_id,
                ))
            except ImportError:
                logger.warning("telegram bot library not installed, skipping Telegram")

        if self.config.discord_enabled:
            try:
                from .discord_adapter import DiscordChannel
                self._channels.append(DiscordChannel(
                    bot_token=self.config.discord_bot_token,
                    channel_id=self.config.discord_channel_id,
                ))
            except ImportError:
                logger.warning("discord library not installed, skipping Discord")

        if self.config.webhook_enabled:
            from .webhook_adapter import WebhookChannel
            self._channels.append(WebhookChannel(url=self.config.webhook_url))

    async def send(self, level: str, title: str, message: str) -> None:
        """Send notification to all enabled channels."""
        notify_level = NotifyLevel[level.upper()] if isinstance(level, str) else level
        if notify_level < self._min_level:
            return

        formatted = f"[QuantEvoLoop] {title}\n{message}"
        tasks = []
        for channel in self._channels:
            if channel.is_enabled:
                tasks.append(self._safe_send(channel, title, formatted))

        if tasks:
            await asyncio.gather(*tasks)

    async def _safe_send(self, channel: ChannelAdapter, title: str, message: str) -> None:
        try:
            await channel.send(title, message)
        except Exception as e:
            logger.warning(f"Failed to send via {type(channel).__name__}: {e}")
