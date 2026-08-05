"""The resolved configuration for one commander turn.

Pulls together the three places configuration comes from:

* :class:`~game.settings.Settings` -- everything shareable, which is saved with
  the campaign like any other setting.
* :class:`~game.ai_commander.secretstore.SecretStore` -- the API key, which is
  deliberately *not* a setting.
* The environment -- ``OPENROUTER_API_KEY`` for the key, plus
  ``RETRIBUTION_AI_AUDIT_DIR`` for the decision log, so the dry-run harness and
  the test suite never need a configured install.

:class:`AiCommanderConfig` is a value object: it carries the key but never logs
it, never serialises it and reports it only as a mask.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from game.ai_commander.enums import (
    CommanderPersonality,
    IntelPolicy,
)
from game.ai_commander.llmclient import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
)
from game.ai_commander.secretstore import SecretStore, mask

if TYPE_CHECKING:
    from game.settings import Settings


#: Endpoints that never need a key and never cost anything. Matched on the host
#: so a user running Ollama or llama.cpp locally is not nagged for a key.
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal")


def is_local_endpoint(base_url: str) -> bool:
    """Whether ``base_url`` points at something on this machine."""

    from urllib.parse import urlsplit

    try:
        host = (urlsplit(base_url).hostname or "").lower()
    except ValueError:  # pragma: no cover - defensive
        return False
    return host in _LOCAL_HOSTS


@dataclass(frozen=True)
class AiCommanderConfig:
    """Everything one commander turn needs to know, already validated."""

    enabled: bool = False
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: Optional[str] = None
    personality: CommanderPersonality = CommanderPersonality.BALANCED
    intel_policy: IntelPolicy = IntelPolicy.REALISTIC
    cost_cap_per_turn: float = 0.5
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    log_prompts: bool = True
    fallback_to_builtin: bool = True
    #: Where the decision log is written. ``None`` means "work it out from the
    #: Retribution save directory", which may itself be unavailable.
    audit_directory: Optional[Path] = None
    #: Populated by :meth:`from_settings` when configuration is unusable.
    problems: tuple[str, ...] = field(default_factory=tuple)

    # -- derived ----------------------------------------------------------

    @property
    def is_local(self) -> bool:
        return is_local_endpoint(self.base_url)

    @property
    def requires_api_key(self) -> bool:
        return not self.is_local

    @property
    def is_usable(self) -> bool:
        """Whether a request could actually be attempted."""

        if not self.enabled or self.problems:
            return False
        if not self.model.strip() or not self.base_url.strip():
            return False
        if self.requires_api_key and not self.api_key:
            return False
        return True

    @property
    def allows_paid_requests(self) -> bool:
        """A zero cap forbids anything that could be billed."""

        return self.is_local or self.cost_cap_per_turn > 0.0

    def describe(self) -> str:
        """One-line summary safe to write to the log or show in the UI."""

        return (
            f"model={self.model} base_url={self.base_url} "
            f"key={mask(self.api_key)} personality={self.personality.value} "
            f"intel={self.intel_policy.value} cap=${self.cost_cap_per_turn:.2f}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form with the key removed, not masked-in-place."""

        return {
            "enabled": self.enabled,
            "model": self.model,
            "base_url": self.base_url,
            "personality": self.personality.value,
            "intel_policy": self.intel_policy.value,
            "cost_cap_per_turn": self.cost_cap_per_turn,
            "timeout_seconds": self.timeout_seconds,
            "max_output_tokens": self.max_output_tokens,
            "log_prompts": self.log_prompts,
            "fallback_to_builtin": self.fallback_to_builtin,
            "api_key_configured": self.api_key is not None,
            "problems": list(self.problems),
        }

    # -- construction -----------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        secret_store: Optional[SecretStore] = None,
        audit_directory: Optional[Path] = None,
    ) -> AiCommanderConfig:
        """Read configuration from ``settings`` plus the key store.

        Never raises. A setting that is missing (an older save loaded by a newer
        build, or vice versa) falls back to the documented default, and anything
        unusable is recorded in :attr:`problems` so the caller can log a single
        clear reason for falling back instead of failing mysteriously.
        """

        problems: list[str] = []

        enabled = bool(getattr(settings, "ai_commander_enabled", False))
        model = str(
            getattr(settings, "ai_commander_model", DEFAULT_MODEL) or ""
        ).strip()
        base_url = str(
            getattr(settings, "ai_commander_base_url", DEFAULT_BASE_URL) or ""
        ).strip()
        if not model:
            problems.append("no model identifier is configured")
        if not base_url:
            problems.append("no provider base URL is configured")
        elif not base_url.startswith(("http://", "https://")):
            problems.append(f"provider base URL {base_url!r} is not an http(s) URL")

        personality = _enum_or_default(
            CommanderPersonality,
            getattr(settings, "ai_commander_personality", None),
            CommanderPersonality.BALANCED,
            "commander personality",
        )
        intel_policy = _enum_or_default(
            IntelPolicy,
            getattr(settings, "ai_commander_intel_policy", None),
            IntelPolicy.REALISTIC,
            "intel policy",
        )

        try:
            cap = float(getattr(settings, "ai_commander_cost_cap_per_turn", 0.5))
        except (TypeError, ValueError):
            cap = 0.5
        cap = max(0.0, cap)

        timeout = _bounded_int(
            getattr(settings, "ai_commander_timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            DEFAULT_TIMEOUT_SECONDS,
            10,
            600,
        )
        max_output_tokens = _bounded_int(
            getattr(
                settings, "ai_commander_max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS
            ),
            DEFAULT_MAX_OUTPUT_TOKENS,
            256,
            16000,
        )

        store = secret_store if secret_store is not None else SecretStore()
        try:
            api_key = store.load()
        except Exception:  # pragma: no cover - defensive
            logging.warning("Could not read the AI commander key store", exc_info=True)
            api_key = None

        config = cls(
            enabled=enabled,
            model=model or DEFAULT_MODEL,
            base_url=base_url or DEFAULT_BASE_URL,
            api_key=api_key,
            personality=personality,
            intel_policy=intel_policy,
            cost_cap_per_turn=cap,
            timeout_seconds=timeout,
            max_output_tokens=max_output_tokens,
            log_prompts=bool(getattr(settings, "ai_commander_log_prompts", True)),
            fallback_to_builtin=bool(
                getattr(settings, "ai_commander_fallback_to_builtin", True)
            ),
            audit_directory=audit_directory,
            problems=tuple(problems),
        )
        if enabled and config.requires_api_key and api_key is None:
            problems.append(
                "no API key is configured; enter one in the AI Opponent settings "
                "page or set the OPENROUTER_API_KEY environment variable"
            )
        if enabled and not config.allows_paid_requests:
            problems.append(
                "the per-turn spending cap is $0.00, so no billable request can "
                "be made; raise the cap or use a local provider"
            )
        if problems == list(config.problems):
            return config
        return cls(
            enabled=config.enabled,
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            personality=config.personality,
            intel_policy=config.intel_policy,
            cost_cap_per_turn=config.cost_cap_per_turn,
            timeout_seconds=config.timeout_seconds,
            max_output_tokens=config.max_output_tokens,
            log_prompts=config.log_prompts,
            fallback_to_builtin=config.fallback_to_builtin,
            audit_directory=config.audit_directory,
            problems=tuple(problems),
        )


def _enum_or_default(enum_type: Any, raw: Any, default: Any, label: str) -> Any:
    if raw is None:
        return default
    if isinstance(raw, enum_type):
        return raw
    try:
        return enum_type(str(raw))
    except ValueError:
        logging.warning(
            "Unknown %s %r in settings; using %s", label, raw, default.value
        )
        return default


def _bounded_int(raw: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))
