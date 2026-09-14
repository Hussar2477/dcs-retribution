"""JSON helpers for the AI commander's audit records.

The audit log has to be stable, human readable and free of live game objects, so
everything written through here is reduced to plain JSON types.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from typing import Any


def jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, enums and mappings to JSON types."""

    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(jsonable(k)): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """Deterministic JSON text, suitable for hashing."""

    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any, length: int = 16) -> str:
    """A short, stable, non-reversible digest of ``value``."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length]
