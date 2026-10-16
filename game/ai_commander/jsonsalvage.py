"""Recover the complete part of a JSON object that was cut off mid-stream.

A reasoning model asked for a large, list-based plan (the logistics stage is the
worst offender: procurement, ground transfers, squadron relocations, runway
repairs and squadron auto-tasking, each an independent list) can overrun its
output budget and stop partway through. The provider reports
``finish_reason == "length"`` and the reply ends inside a half-written array
element, so it will not parse and the whole stage would otherwise be discarded --
throwing away every order the model *did* manage to emit.

:func:`salvage_truncated_json` recovers the largest valid JSON-object prefix from
such a reply: it drops the trailing incomplete element, closes the still-open
brackets and parses the result. It is deliberately conservative -- it only ever
returns something ``json.loads`` accepts as an object -- and it is pure and never
raises, so the caller can treat a ``None`` result as "nothing usable" without any
error handling. It repairs *structure* only (missing closing brackets from a
clean cut); it never guesses at values, so it cannot invent content the model did
not write.
"""

from __future__ import annotations

import json
from typing import Any, Optional

__all__ = ["salvage_truncated_json"]

#: A hard bound on how many candidate cut points we will try to parse. Each try
#: is a full ``json.loads``; capping the count keeps salvage cheap even on a very
#: large truncated reply. Boundaries are tried newest-first, so the largest
#: recoverable prefix is found well within this many attempts in practice.
_MAX_CANDIDATES = 400


def salvage_truncated_json(text: str) -> Optional[dict[str, Any]]:
    """Return the largest valid JSON object recoverable from ``text``.

    Handles a fenced code block and surrounding prose, then walks the object
    tracking bracket depth and string state, recording every point at which a
    complete value has just been emitted. Working back from the end, it closes
    the brackets that were still open at each such point and returns the first
    result that parses to a ``dict``. Returns ``None`` when nothing complete can
    be recovered. Never raises.
    """

    try:
        return _salvage(text)
    except Exception:  # pragma: no cover - defensive; salvage must never raise
        return None


def _salvage(text: str) -> Optional[dict[str, Any]]:
    candidate = _strip_to_object(text)
    if candidate is None:
        return None

    # Every index at which a complete value has just closed, paired with the
    # bracket stack still open at that point (the closers needed to finish).
    boundaries = _value_boundaries(candidate)
    if not boundaries:
        return None

    # Try the largest recoverable prefixes first.
    for cut, closers in reversed(boundaries[-_MAX_CANDIDATES:]):
        prefix = candidate[:cut].rstrip()
        # A trailing comma (from a dropped, half-written next element) would make
        # the closed object invalid, so strip any run of them and whitespace.
        while prefix and prefix[-1] in ",":
            prefix = prefix[:-1].rstrip()
        completed = prefix + "".join(reversed(closers))
        try:
            parsed = json.loads(completed)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _strip_to_object(text: str) -> Optional[str]:
    """Strip a code fence and any prose, returning text from the first ``{``."""

    candidate = text.strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        without_fence = candidate[3:]
        newline = without_fence.find("\n")
        if newline != -1:
            without_fence = without_fence[newline + 1 :]
        # A truncated reply usually has no closing fence; tolerate its absence.
        closing = without_fence.rfind("```")
        if closing != -1:
            without_fence = without_fence[:closing]
        candidate = without_fence.strip()
    start = candidate.find("{")
    if start == -1:
        return None
    return candidate[start:]


def _value_boundaries(candidate: str) -> list[tuple[int, list[str]]]:
    """Find every index where a complete value has just closed.

    Returns ``(index_after_value, open_closers)`` pairs, where ``open_closers``
    is the list of ``}``/``]`` characters (in the order they were opened) still
    needed to finish the document at that point. Only boundaries that sit inside
    the top-level object (stack non-empty, bottom ``}``) are returned, since the
    salvaged result must be an object.
    """

    boundaries: list[tuple[int, list[str]]] = []
    stack: list[str] = []
    in_string = False
    escape = False

    def record(index_after: int) -> None:
        if stack and stack[0] == "}":
            boundaries.append((index_after, list(stack)))

    for i, ch in enumerate(candidate):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                # A closed string is a complete value unless it is an object key,
                # i.e. unless the next significant character is a colon. We do
                # not know that yet, so record it and let a failed parse (a key
                # with no value, once closed) fall through to an earlier
                # boundary.
                record(i + 1)
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
            continue
        if ch in "}]":
            if stack:
                stack.pop()
            record(i + 1)
            continue
        if ch.isspace() or ch in ",:":
            continue
        # A bare token: number, or true/false/null. Its final character marks a
        # complete value, so record after each -- the last one in a run is the
        # real boundary and earlier partial ones simply fail to parse.
        record(i + 1)

    return boundaries
