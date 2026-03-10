"""Configuration loader for Voz Bridge.

Reads environment variables with sensible defaults for local development.
All config is immutable after load (frozen dataclass).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BridgeConfig:
    """Immutable bridge configuration."""

    # Bridge server
    host: str = "0.0.0.0"
    port: int = 8089
    token: str = ""  # empty = auth disabled
    log_level: str = "info"

    # CoPaw upstream
    copaw_host: str = "127.0.0.1"
    copaw_port: int = 8088
    copaw_timeout: int = 120
    copaw_base_url: str = ""  # override, computed if empty

    # Session
    session_prefix: str = "voz-"

    @property
    def copaw_url(self) -> str:
        """Effective CoPaw base URL."""
        if self.copaw_base_url:
            return self.copaw_base_url.rstrip("/")
        return f"http://{self.copaw_host}:{self.copaw_port}"

    @property
    def copaw_process_endpoint(self) -> str:
        """Full URL for CoPaw agent process endpoint."""
        return f"{self.copaw_url}/api/agent/process"

    @property
    def auth_enabled(self) -> bool:
        """Whether Bearer token auth is active."""
        return bool(self.token)


def load_config() -> BridgeConfig:
    """Load configuration from environment variables."""
    return BridgeConfig(
        host=os.getenv("VOZ_BRIDGE_HOST", "0.0.0.0"),
        port=int(os.getenv("VOZ_BRIDGE_PORT", "8089")),
        token=os.getenv("VOZ_BRIDGE_TOKEN", ""),
        log_level=os.getenv("VOZ_BRIDGE_LOG_LEVEL", "info"),
        copaw_host=os.getenv("VOZ_COPAW_HOST", "127.0.0.1"),
        copaw_port=int(os.getenv("VOZ_COPAW_PORT", "8088")),
        copaw_timeout=int(os.getenv("VOZ_COPAW_TIMEOUT", "120")),
        copaw_base_url=os.getenv("VOZ_COPAW_BASE_URL", ""),
        session_prefix=os.getenv("VOZ_DEFAULT_SESSION_PREFIX", "voz-"),
    )
