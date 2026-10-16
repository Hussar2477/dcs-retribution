"""Normalisation and de-duplication of enemy unit and weapon type names.

The intel channels that feed the RED commander collect type names from several
places at once: ``unit.type.id`` for observed objectives, the DCS
``getTypeName()`` initiator recorded against each kill, and Retribution's own
:class:`AircraftType` display names for confirmed air losses. These disagree on
spelling, so the *same* real type can appear two or three times -- a raw DCS id
(``FA-18C_hornet``) alongside a human display name (``F/A-18C Hornet (Lot 20)``),
or a weapon carried as its full internal path (``weapons.shells.M256_120_HE``).

This module collapses those duplicates to a single clean name so the memory the
commander reads is a tidy list of distinct types rather than a noisy one. It is
purely cosmetic: it never adds, removes or counts anything, so it cannot affect
the fairness boundary. Everything degrades to a tidied form of the raw id when a
proper display name cannot be resolved -- a name is never dropped.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Iterable, Optional

#: Default cap on the length of any normalised type/weapon list. Matches the cap
#: the raw channels already applied, so normalisation never grows a list.
DEFAULT_LIMIT = 24

# Trailing parenthetical qualifier, e.g. the "(Lot 20)" in "F/A-18C Hornet
# (Lot 20)". Two names that differ only by such a qualifier are the same type.
_TRAILING_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")

#: Lazily built map of raw DCS type id -> proper display name, from pydcs. Cached
#: because the maps are large and never change within a process.
_TYPE_DISPLAY_LOOKUP: Optional[dict[str, str]] = None


def _tidy(raw: str) -> str:
    """A readable form of a raw id: underscores to spaces, whitespace collapsed."""

    return _WHITESPACE_RE.sub(" ", str(raw).replace("_", " ")).strip()


def _type_display_lookup() -> dict[str, str]:
    """Map every known DCS type id to its cleanest display name.

    Built from pydcs's own type maps, which load without a DCS installation. A
    class that carries no display ``name`` (most aircraft) maps to a tidied form
    of its id. If pydcs is unavailable for any reason the lookup is simply empty
    and every name falls back to its tidied raw id.
    """

    global _TYPE_DISPLAY_LOOKUP
    if _TYPE_DISPLAY_LOOKUP is not None:
        return _TYPE_DISPLAY_LOOKUP

    lookup: dict[str, str] = {}
    try:
        from dcs.helicopters import helicopter_map
        from dcs.planes import plane_map
        from dcs.ships import ship_map
        from dcs.vehicles import vehicle_map

        for type_map in (plane_map, helicopter_map, ship_map, vehicle_map):
            for key, unit_type in type_map.items():
                display = getattr(unit_type, "name", None)
                text = str(display).strip() if display else ""
                lookup[str(key)] = text or _tidy(str(key))
    except Exception:  # pragma: no cover - defensive; pydcs is a hard dependency
        logging.debug(
            "pydcs type maps unavailable; type names will be tidied raw ids",
            exc_info=True,
        )

    _TYPE_DISPLAY_LOOKUP = lookup
    return lookup


def normalise_weapon_name(raw: str) -> str:
    """A readable weapon name: drop the ``weapons.<category>.`` path, tidy the rest.

    ``weapons.shells.M256_120_HE`` becomes ``M256 120 HE``; a name that is
    already clean (``AIM-120C``) is returned essentially unchanged.
    """

    text = str(raw).strip()
    if not text:
        return ""
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return _tidy(text)


def normalise_type_name(raw: str) -> str:
    """A clean display name for a unit/aircraft type id.

    Resolves the id through pydcs where possible; otherwise falls back to a
    tidied form of the raw id. A weapon path that has strayed into the type
    channel is routed through :func:`normalise_weapon_name` rather than dropped.
    """

    text = str(raw).strip()
    if not text:
        return ""
    if text.startswith("weapons."):
        return normalise_weapon_name(text)
    lookup = _type_display_lookup()
    resolved = lookup.get(text)
    if resolved:
        return resolved
    return _tidy(text)


def _signature(name: str) -> str:
    """A comparison key that ignores spelling, casing and trailing qualifiers."""

    base = _TRAILING_PAREN_RE.sub("", name)
    signature = _NON_ALNUM_RE.sub("", base.lower())
    return signature or name.strip().lower()


def _prefer(candidate: str, current: str) -> bool:
    """Whether ``candidate`` is the nicer display of two names for one type.

    Prefers the more descriptive spelling: the longer name once any trailing
    qualifier is removed, then one that reads as words (contains a space), then
    the alphabetically earlier one so the choice is deterministic.
    """

    cand = _TRAILING_PAREN_RE.sub("", candidate).strip()
    curr = _TRAILING_PAREN_RE.sub("", current).strip()
    if len(cand) != len(curr):
        return len(cand) > len(curr)
    if (" " in cand) != (" " in curr):
        return " " in cand
    return cand < curr


def _dedupe(
    names: Iterable[str], normaliser: Callable[[str], str], limit: int
) -> tuple[str, ...]:
    chosen: dict[str, str] = {}
    for raw in names:
        display = normaliser(raw)
        if not display:
            continue
        signature = _signature(display)
        existing = chosen.get(signature)
        if existing is None or _prefer(display, existing):
            chosen[signature] = display
    cleaned = {
        (_TRAILING_PAREN_RE.sub("", value).strip() or value)
        for value in chosen.values()
    }
    return tuple(sorted(cleaned))[:limit]


def dedupe_type_names(
    names: Iterable[str], limit: int = DEFAULT_LIMIT
) -> tuple[str, ...]:
    """Distinct, normalised unit/aircraft type names, sorted and capped.

    Raw DCS ids and their display-name duplicates collapse to one clean name.
    """

    return _dedupe(names, normalise_type_name, limit)


def dedupe_weapon_names(
    names: Iterable[str], limit: int = DEFAULT_LIMIT
) -> tuple[str, ...]:
    """Distinct, normalised weapon type names, sorted and capped."""

    return _dedupe(names, normalise_weapon_name, limit)
