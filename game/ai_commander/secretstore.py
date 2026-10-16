"""Storage for the provider API key, kept out of every other artefact.

The key must not end up in any of the places Retribution already writes:

* ``Settings`` is pickled into the campaign save **and** dumped to JSON by
  ``QSettingsWindow.save_settings``, so the key cannot be a ``Settings`` field.
* The decision log is meant to be shareable for auditing, so the key never
  reaches :mod:`game.ai_commander.audit`.
* Log records go to the plain-text Retribution log, so the key is never passed
  to ``logging`` and :class:`SecretStore` has a redacting ``__repr__``.

It therefore lives in its own file in the per-user data directory, alongside
``retribution_preferences.json``, with owner-only permissions where the platform
supports them. An ``OPENROUTER_API_KEY`` environment variable takes precedence,
which is how the dry-run harness and CI avoid needing a stored key at all.

The path logic is duplicated from ``qt_ui.liberation_install`` on purpose:
``game`` must not import ``qt_ui``, and the headless dry-run harness must work
without Qt installed.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import Any, Optional

#: Environment variable checked before the stored file.
ENV_VAR = "OPENROUTER_API_KEY"

#: Name of the file inside the per-user data directory.
SECRETS_FILENAME = "ai_commander_secrets.json"

#: Key inside that file. A dict rather than a bare string so a future provider
#: can be added without a migration.
_KEY_FIELD = "openrouter_api_key"

#: What the key is replaced with anywhere it might be displayed.
REDACTED = "<redacted>"


def user_data_path() -> Path:
    """Same directory Retribution already keeps its preferences in."""

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "DCSRetribution"
        return Path.home() / "AppData" / "Local" / "DCSRetribution"

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "DCSRetribution"
    return Path.home() / ".local" / "share" / "DCSRetribution"


def mask(secret: Optional[str]) -> str:
    """A safe-to-display fingerprint of a key.

    Shows only enough to let a user confirm *which* key is stored. Short values
    are fully masked rather than partially revealed.
    """

    if not secret:
        return "(not set)"
    trimmed = secret.strip()
    if len(trimmed) <= 12:
        return "*" * len(trimmed)
    return f"{trimmed[:6]}...{trimmed[-4:]} ({len(trimmed)} chars)"


class SecretStore:
    """Reads and writes the provider key. Never logs its value."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = (
            Path(path) if path is not None else user_data_path() / SECRETS_FILENAME
        )

    def __repr__(self) -> str:
        return f"SecretStore(path={self.path!s}, key={REDACTED})"

    # -- reading ----------------------------------------------------------

    @property
    def env_override(self) -> Optional[str]:
        value = os.environ.get(ENV_VAR)
        return value.strip() or None if value else None

    def load(self) -> Optional[str]:
        """The key to use, environment first, then the stored file."""

        override = self.env_override
        if override is not None:
            return override
        return self.load_stored()

    def load_stored(self) -> Optional[str]:
        """Only the stored key, ignoring the environment."""

        import json

        if not self.path.is_file():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Deliberately does not include the file body in the message.
            logging.warning("Could not read %s", self.path)
            return None
        if not isinstance(payload, dict):
            return None
        value = payload.get(_KEY_FIELD)
        if not isinstance(value, str):
            return None
        return value.strip() or None

    @property
    def source(self) -> str:
        """Where the key came from, for the settings UI."""

        if self.env_override is not None:
            return f"environment ({ENV_VAR})"
        if self.load_stored() is not None:
            return str(self.path)
        return "not configured"

    @property
    def is_configured(self) -> bool:
        return self.load() is not None

    def describe(self) -> str:
        return f"{mask(self.load())} from {self.source}"

    # -- writing ----------------------------------------------------------

    def save(self, secret: Optional[str]) -> bool:
        """Store ``secret``, or delete the stored key when it is empty.

        Returns whether the operation succeeded. Failure is reported without the
        value ever being included in the message.
        """

        import json

        cleaned = (secret or "").strip()
        if not cleaned:
            return self.clear()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logging.warning("Could not create %s", self.path.parent, exc_info=True)
            return False

        payload: dict[str, Any] = {}
        if self.path.is_file():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload = existing
            except (OSError, ValueError):
                payload = {}
        payload[_KEY_FIELD] = cleaned

        try:
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _restrict_permissions(temporary)
            temporary.replace(self.path)
            _restrict_permissions(self.path)
        except OSError:
            logging.warning("Could not write the AI commander key file")
            return False
        return True

    def clear(self) -> bool:
        if not self.path.exists():
            return True
        try:
            self.path.unlink()
        except OSError:
            logging.warning("Could not remove %s", self.path, exc_info=True)
            return False
        return True


def _restrict_permissions(path: Path) -> None:
    """Owner read/write only, where the platform supports it.

    Windows ignores POSIX mode bits; the call is harmless there and the file
    still sits inside the user's own ``LOCALAPPDATA``.
    """

    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:  # pragma: no cover - platform dependent
        logging.debug("Could not restrict permissions on %s", path, exc_info=True)
