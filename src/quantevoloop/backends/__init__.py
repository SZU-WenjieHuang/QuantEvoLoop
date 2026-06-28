"""Backend factory — create the appropriate CodeAgentBackend from config."""

from quantevoloop.backends.base import (
    AnalysisResult,
    CodeAgentBackend,
    JudgeResult,
    MutationResult,
)
from quantevoloop.config import BackendConfig

__all__ = [
    "AnalysisResult",
    "CodeAgentBackend",
    "JudgeResult",
    "MutationResult",
    "create_backend",
]

_REGISTRY: dict[str, type[CodeAgentBackend]] = {}


def _lazy_register():
    """Lazily import backend classes to avoid import errors for uninstalled backends."""
    if _REGISTRY:
        return

    from quantevoloop.backends.claude_code import ClaudeCodeBackend
    from quantevoloop.backends.codex import CodexBackend
    from quantevoloop.backends.qoder_cli import QoderCliBackend

    _REGISTRY.update({
        "claude-code": ClaudeCodeBackend,
        "codex": CodexBackend,
        "qoder-cli": QoderCliBackend,
    })


def create_backend(config: BackendConfig) -> CodeAgentBackend:
    """Create a backend instance from configuration.

    Args:
        config: BackendConfig specifying type, cli_path, and options.

    Returns:
        Concrete CodeAgentBackend instance.

    Raises:
        ValueError: If backend type is not supported.
    """
    _lazy_register()

    cls = _REGISTRY.get(config.type)
    if cls is None:
        supported = ", ".join(_REGISTRY.keys())
        raise ValueError(
            f"Unsupported backend: {config.type!r}. Supported: {supported}"
        )
    return cls(config)
