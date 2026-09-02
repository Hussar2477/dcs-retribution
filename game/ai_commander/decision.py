"""The RED commander decision contract and its deterministic validator.

The model returns JSON. This module is the *only* thing that turns that JSON
into an object the rest of the code will look at, and it is deliberately
paranoid:

* Unknown keys are rejected, so a model cannot smuggle extra instructions in.
* Every identifier must appear in the brief the decision was produced for.
* Duplicate identifiers and duplicate ranks are rejected.
* A decision whose ``campaign_revision`` or ``turn_id`` does not match the brief
  is rejected as stale.
* Enum values are checked against the enums the prompt was generated from.
* Nothing here mutates game state, and nothing here is allowed to fall back to a
  "best guess" interpretation. Rejections are collected and reported.

Validation is *lenient about omissions and strict about content*: a decision may
leave a list empty (the deterministic planner then keeps its own ordering for
that dimension), but any element it does supply must be completely legal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Type, TypeVar

from game.ai_commander.enums import (
    FrontPosture,
    MissionPurpose,
    RedStrategy,
    ReservePolicy,
)
from game.ai_commander.intel import RedCommanderBrief
from game.ai_commander.serialization import jsonable

DECISION_SCHEMA_VERSION = "red-commander-decision/1"

#: Hard ceiling on ``commander_intent`` so a chatty model cannot bloat the audit
#: log or the UI. Longer intents are truncated (and the truncation is recorded).
MAX_INTENT_CHARACTERS = 600

#: Upper bound on list sizes, independent of the brief, as a cheap guard against
#: pathological responses.
MAX_LIST_ENTRIES = 32

_EnumT = TypeVar("_EnumT", RedStrategy, FrontPosture, ReservePolicy, MissionPurpose)


class DecisionValidationError(Exception):
    """Raised when a response cannot be salvaged into any legal decision."""

    def __init__(self, message: str, rejections: Sequence[Rejection] = ()) -> None:
        super().__init__(message)
        self.message = message
        self.rejections = list(rejections)


@dataclass(frozen=True)
class Rejection:
    """One thing the model asked for that was refused, and why.

    ``element`` is a dotted path into the response so the audit viewer can show
    exactly which part of the plan was dropped.
    """

    element: str
    reason: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "element": self.element,
            "reason": self.reason,
            "value": jsonable(self.value),
        }

    def __str__(self) -> str:
        if self.value is None:
            return f"{self.element}: {self.reason}"
        return f"{self.element}: {self.reason} ({self.value!r})"


@dataclass(frozen=True)
class RankedId:
    """An identifier with the rank the commander gave it (1 = highest)."""

    id: str
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "rank": self.rank}


@dataclass(frozen=True)
class FrontPostureRequest:
    front_id: str
    posture: FrontPosture

    def to_dict(self) -> dict[str, Any]:
        return {"front_id": self.front_id, "posture": self.posture.value}


@dataclass(frozen=True)
class TargetSetPriority:
    target_set_id: str
    rank: int
    purpose: MissionPurpose

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_set_id": self.target_set_id,
            "rank": self.rank,
            "purpose": self.purpose.value,
        }


@dataclass(frozen=True)
class RedCommanderDecision:
    """A validated, schema-clean commander decision.

    Being an instance of this class means the decision was internally consistent
    and referred only to the brief it was produced for. It does **not** yet mean
    every element is applicable to live game state: that is re-checked by
    :mod:`game.ai_commander.legality` immediately before anything is applied.
    """

    schema_version: str
    campaign_revision: str
    turn_id: int
    strategy: RedStrategy
    front_priorities: tuple[RankedId, ...]
    push_postures: tuple[FrontPostureRequest, ...]
    spending_priorities: tuple[RankedId, ...]
    target_set_priorities: tuple[TargetSetPriority, ...]
    reserve_policy: ReservePolicy
    commander_intent: str = ""

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @property
    def ordered_front_ids(self) -> tuple[str, ...]:
        return tuple(r.id for r in sorted(self.front_priorities, key=lambda r: r.rank))

    @property
    def ordered_spending_ids(self) -> tuple[str, ...]:
        return tuple(
            r.id for r in sorted(self.spending_priorities, key=lambda r: r.rank)
        )

    @property
    def ordered_target_set_ids(self) -> tuple[str, ...]:
        return tuple(
            t.target_set_id
            for t in sorted(self.target_set_priorities, key=lambda t: t.rank)
        )

    def posture_for(self, front_id: str) -> Optional[FrontPosture]:
        for request in self.push_postures:
            if request.front_id == front_id:
                return request.posture
        return None


@dataclass
class ValidationOutcome:
    """The result of validating one model response."""

    decision: Optional[RedCommanderDecision]
    rejections: list[Rejection] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.decision is not None

    @property
    def rejection_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.rejections]

    def error_summary(self, limit: int = 12) -> str:
        """Compact text used to build the single repair prompt."""

        if not self.rejections:
            return "no specific errors recorded"
        return "; ".join(str(r) for r in self.rejections[:limit])


# ---------------------------------------------------------------------------
# JSON schema, generated from the enums so prompt and validator cannot diverge.
# ---------------------------------------------------------------------------


def decision_json_schema(brief: RedCommanderBrief) -> dict[str, Any]:
    """A JSON Schema for ``brief``'s action space.

    Sent as ``response_format`` when the provider supports structured output.
    The schema is *advisory*: :func:`validate_decision` re-checks everything
    locally, because provider-side enforcement varies by model.
    """

    front_ids = sorted(brief.front_ids)
    target_ids = sorted(brief.target_set_ids)
    procurement_ids = sorted(brief.procurement_ids)

    def ranked(id_values: Sequence[str], id_key: str) -> dict[str, Any]:
        id_schema: dict[str, Any] = {"type": "string"}
        if id_values:
            id_schema["enum"] = list(id_values)
        return {
            "type": "array",
            "maxItems": max(len(id_values), 1),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [id_key, "rank"],
                "properties": {
                    id_key: id_schema,
                    "rank": {"type": "integer", "minimum": 1},
                },
            },
        }

    target_items = ranked(target_ids, "target_set_id")
    target_items["items"]["required"] = ["target_set_id", "rank", "purpose"]
    target_items["items"]["properties"]["purpose"] = {
        "type": "string",
        "enum": [p.value for p in MissionPurpose],
    }

    posture_id_schema: dict[str, Any] = {"type": "string"}
    if front_ids:
        posture_id_schema["enum"] = front_ids

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "campaign_revision",
            "turn_id",
            "strategy",
            "front_priorities",
            "push_postures",
            "spending_priorities",
            "target_set_priorities",
            "reserve_policy",
            "commander_intent",
        ],
        "properties": {
            "schema_version": {"type": "string", "const": DECISION_SCHEMA_VERSION},
            "campaign_revision": {"type": "string", "const": brief.campaign_revision},
            "turn_id": {"type": "integer", "const": brief.turn_id},
            "strategy": {"type": "string", "enum": [s.value for s in RedStrategy]},
            "front_priorities": ranked(front_ids, "front_id"),
            "push_postures": {
                "type": "array",
                "maxItems": max(len(front_ids), 1),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["front_id", "posture"],
                    "properties": {
                        "front_id": posture_id_schema,
                        "posture": {
                            "type": "string",
                            "enum": [p.value for p in FrontPosture],
                        },
                    },
                },
            },
            "spending_priorities": ranked(procurement_ids, "category_id"),
            "target_set_priorities": target_items,
            "reserve_policy": {
                "type": "string",
                "enum": [r.value for r in ReservePolicy],
            },
            "commander_intent": {
                "type": "string",
                "maxLength": MAX_INTENT_CHARACTERS,
            },
        },
    }


def decision_schema_hash(brief: RedCommanderBrief) -> str:
    from game.ai_commander.serialization import stable_hash

    return stable_hash(decision_json_schema(brief))


def example_decision_json(brief: RedCommanderBrief) -> str:
    """A minimal well-formed example, used in the prompt."""

    front = sorted(brief.front_ids)[:1]
    target = sorted(brief.target_set_ids)[:1]
    procurement = sorted(brief.procurement_ids)[:1]
    payload: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "campaign_revision": brief.campaign_revision,
        "turn_id": brief.turn_id,
        "strategy": RedStrategy.DEFEND.value,
        "front_priorities": [{"front_id": f, "rank": 1} for f in front],
        "push_postures": [
            {"front_id": f, "posture": FrontPosture.HOLD.value} for f in front
        ],
        "spending_priorities": [{"category_id": p, "rank": 1} for p in procurement],
        "target_set_priorities": [
            {
                "target_set_id": t,
                "rank": 1,
                "purpose": MissionPurpose.PROTECT_OWN_FORCES.value,
            }
            for t in target
        ],
        "reserve_policy": ReservePolicy.BALANCED.value,
        "commander_intent": "Hold the line while air defences are rebuilt.",
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "campaign_revision",
        "turn_id",
        "strategy",
        "front_priorities",
        "push_postures",
        "spending_priorities",
        "target_set_priorities",
        "reserve_policy",
        "commander_intent",
    }
)


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from model output.

    Handles the two harmless deviations that even well-behaved models produce:
    a fenced code block, and prose around the object. Anything else is an error
    -- we do not try to repair broken JSON locally, because guessing what the
    model meant is exactly the kind of leniency that lets cheating slip through.
    """

    if not isinstance(text, str) or not text.strip():
        raise DecisionValidationError("response body was empty")

    candidate = text.strip()
    if candidate.startswith("```"):
        # Strip a fenced block, tolerating a language tag on the opening fence.
        without_fence = candidate[3:]
        newline = without_fence.find("\n")
        if newline != -1:
            without_fence = without_fence[newline + 1 :]
        closing = without_fence.rfind("```")
        if closing != -1:
            without_fence = without_fence[:closing]
        candidate = without_fence.strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise DecisionValidationError("response did not contain a JSON object")
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as err:
            raise DecisionValidationError(f"response was not valid JSON: {err.msg}")

    if not isinstance(parsed, dict):
        raise DecisionValidationError(
            f"response JSON was {type(parsed).__name__}, expected an object"
        )
    return parsed


def _coerce_enum(
    enum_type: Type[_EnumT], raw: Any, element: str, rejections: list[Rejection]
) -> Optional[_EnumT]:
    if not isinstance(raw, str):
        rejections.append(Rejection(element, "expected a string enum value", raw))
        return None
    normalised = raw.strip().lower().replace(" ", "_").replace("-", "_")
    for member in enum_type:
        if member.value == normalised:
            return member
    rejections.append(
        Rejection(
            element,
            "not one of " + ", ".join(m.value for m in enum_type),
            raw,
        )
    )
    return None


def _as_list(raw: Any, element: str, rejections: list[Rejection]) -> list[Any]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        rejections.append(Rejection(element, "expected a list", raw))
        return []
    if len(raw) > MAX_LIST_ENTRIES:
        rejections.append(
            Rejection(
                element,
                f"list longer than the {MAX_LIST_ENTRIES}-entry limit; extra "
                "entries dropped",
                len(raw),
            )
        )
        return raw[:MAX_LIST_ENTRIES]
    return raw


def _rank_of(raw: Any, element: str, rejections: list[Rejection]) -> Optional[int]:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        rejections.append(Rejection(element, "rank must be an integer", raw))
        return None
    if isinstance(raw, float) and not raw.is_integer():
        rejections.append(Rejection(element, "rank must be a whole number", raw))
        return None
    rank = int(raw)
    if rank < 1:
        rejections.append(Rejection(element, "ranks start at 1", raw))
        return None
    return rank


def _validate_ranked_ids(
    raw_entries: Any,
    id_key: str,
    legal_ids: frozenset[str],
    element_prefix: str,
    rejections: list[Rejection],
) -> tuple[RankedId, ...]:
    accepted: list[RankedId] = []
    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    for index, entry in enumerate(_as_list(raw_entries, element_prefix, rejections)):
        element = f"{element_prefix}[{index}]"
        if not isinstance(entry, dict):
            rejections.append(Rejection(element, "expected an object", entry))
            continue
        extra = set(entry) - {id_key, "rank"}
        if extra:
            rejections.append(
                Rejection(element, "unexpected keys", sorted(str(k) for k in extra))
            )
            continue
        identifier = entry.get(id_key)
        if not isinstance(identifier, str):
            rejections.append(
                Rejection(element, f"{id_key} must be a string", identifier)
            )
            continue
        identifier = identifier.strip()
        if identifier not in legal_ids:
            rejections.append(
                Rejection(element, "identifier is not in the brief", identifier)
            )
            continue
        if identifier in seen_ids:
            rejections.append(Rejection(element, "duplicate identifier", identifier))
            continue
        rank = _rank_of(entry.get("rank"), f"{element}.rank", rejections)
        if rank is None:
            continue
        if rank in seen_ranks:
            rejections.append(Rejection(f"{element}.rank", "duplicate rank", rank))
            continue
        seen_ids.add(identifier)
        seen_ranks.add(rank)
        accepted.append(RankedId(id=identifier, rank=rank))
    return tuple(sorted(accepted, key=lambda r: r.rank))


def _validate_postures(
    raw_entries: Any, brief: RedCommanderBrief, rejections: list[Rejection]
) -> tuple[FrontPostureRequest, ...]:
    accepted: list[FrontPostureRequest] = []
    seen: set[str] = set()
    for index, entry in enumerate(_as_list(raw_entries, "push_postures", rejections)):
        element = f"push_postures[{index}]"
        if not isinstance(entry, dict):
            rejections.append(Rejection(element, "expected an object", entry))
            continue
        extra = set(entry) - {"front_id", "posture"}
        if extra:
            rejections.append(
                Rejection(element, "unexpected keys", sorted(str(k) for k in extra))
            )
            continue
        front_id = entry.get("front_id")
        if not isinstance(front_id, str):
            rejections.append(Rejection(element, "front_id must be a string", front_id))
            continue
        front_id = front_id.strip()
        front = brief.front(front_id)
        if front is None:
            rejections.append(
                Rejection(element, "identifier is not in the brief", front_id)
            )
            continue
        if front_id in seen:
            rejections.append(Rejection(element, "duplicate front", front_id))
            continue
        posture = _coerce_enum(
            FrontPosture, entry.get("posture"), f"{element}.posture", rejections
        )
        if posture is None:
            continue
        if posture not in front.legal_postures:
            rejections.append(
                Rejection(
                    f"{element}.posture",
                    "posture is not legal on this front given its force balance; "
                    "legal values were "
                    + (", ".join(p.value for p in front.legal_postures) or "none"),
                    posture.value,
                )
            )
            continue
        seen.add(front_id)
        accepted.append(FrontPostureRequest(front_id=front_id, posture=posture))
    return tuple(accepted)


def _validate_target_sets(
    raw_entries: Any, brief: RedCommanderBrief, rejections: list[Rejection]
) -> tuple[TargetSetPriority, ...]:
    accepted: list[TargetSetPriority] = []
    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    for index, entry in enumerate(
        _as_list(raw_entries, "target_set_priorities", rejections)
    ):
        element = f"target_set_priorities[{index}]"
        if not isinstance(entry, dict):
            rejections.append(Rejection(element, "expected an object", entry))
            continue
        extra = set(entry) - {"target_set_id", "rank", "purpose"}
        if extra:
            rejections.append(
                Rejection(element, "unexpected keys", sorted(str(k) for k in extra))
            )
            continue
        target_id = entry.get("target_set_id")
        if not isinstance(target_id, str):
            rejections.append(
                Rejection(element, "target_set_id must be a string", target_id)
            )
            continue
        target_id = target_id.strip()
        if target_id not in brief.target_set_ids:
            rejections.append(
                Rejection(element, "identifier is not in the brief", target_id)
            )
            continue
        if target_id in seen_ids:
            rejections.append(Rejection(element, "duplicate identifier", target_id))
            continue
        rank = _rank_of(entry.get("rank"), f"{element}.rank", rejections)
        if rank is None:
            continue
        if rank in seen_ranks:
            rejections.append(Rejection(f"{element}.rank", "duplicate rank", rank))
            continue
        purpose = _coerce_enum(
            MissionPurpose, entry.get("purpose"), f"{element}.purpose", rejections
        )
        if purpose is None:
            continue
        seen_ids.add(target_id)
        seen_ranks.add(rank)
        accepted.append(
            TargetSetPriority(target_set_id=target_id, rank=rank, purpose=purpose)
        )
    return tuple(sorted(accepted, key=lambda t: t.rank))


def validate_decision(payload: Any, brief: RedCommanderBrief) -> ValidationOutcome:
    """Validate a parsed response against ``brief``.

    Never raises for content problems: fatal problems produce an outcome with
    ``decision is None`` plus the rejections that explain it, so the caller can
    log everything and fall back.
    """

    rejections: list[Rejection] = []

    if not isinstance(payload, dict):
        rejections.append(
            Rejection("<root>", "expected a JSON object", type(payload).__name__)
        )
        return ValidationOutcome(None, rejections)

    unexpected = set(payload) - _TOP_LEVEL_KEYS
    if unexpected:
        # Not fatal, but recorded: an unknown key is either a hallucination or an
        # attempt to act outside the contract, and either way it is discarded.
        rejections.append(
            Rejection(
                "<root>",
                "unexpected top-level keys were ignored",
                sorted(str(k) for k in unexpected),
            )
        )

    schema_version = payload.get("schema_version")
    if schema_version != DECISION_SCHEMA_VERSION:
        rejections.append(
            Rejection(
                "schema_version",
                f"expected {DECISION_SCHEMA_VERSION}",
                schema_version,
            )
        )
        return ValidationOutcome(None, rejections)

    revision = payload.get("campaign_revision")
    if revision != brief.campaign_revision:
        rejections.append(
            Rejection(
                "campaign_revision",
                "decision was produced for a different campaign state",
                revision,
            )
        )
        return ValidationOutcome(None, rejections)

    turn_id = payload.get("turn_id")
    if isinstance(turn_id, bool) or not isinstance(turn_id, int):
        rejections.append(Rejection("turn_id", "must be an integer", turn_id))
        return ValidationOutcome(None, rejections)
    if turn_id != brief.turn_id:
        rejections.append(
            Rejection("turn_id", f"expected turn {brief.turn_id}", turn_id)
        )
        return ValidationOutcome(None, rejections)

    strategy = _coerce_enum(
        RedStrategy, payload.get("strategy"), "strategy", rejections
    )
    reserve_policy = _coerce_enum(
        ReservePolicy, payload.get("reserve_policy"), "reserve_policy", rejections
    )
    if strategy is None or reserve_policy is None:
        # Strategy and reserve policy are the only two mandatory scalars; without
        # them there is no directive worth applying.
        return ValidationOutcome(None, rejections)

    front_priorities = _validate_ranked_ids(
        payload.get("front_priorities"),
        "front_id",
        brief.front_ids,
        "front_priorities",
        rejections,
    )
    spending_priorities = _validate_ranked_ids(
        payload.get("spending_priorities"),
        "category_id",
        brief.procurement_ids,
        "spending_priorities",
        rejections,
    )
    push_postures = _validate_postures(payload.get("push_postures"), brief, rejections)
    target_set_priorities = _validate_target_sets(
        payload.get("target_set_priorities"), brief, rejections
    )

    intent_raw = payload.get("commander_intent", "")
    if intent_raw is None:
        intent = ""
    elif isinstance(intent_raw, str):
        intent = intent_raw.strip()
    else:
        rejections.append(
            Rejection(
                "commander_intent", "expected a string", type(intent_raw).__name__
            )
        )
        intent = ""
    limit = brief.commander_constraints.max_intent_characters or MAX_INTENT_CHARACTERS
    if len(intent) > limit:
        rejections.append(
            Rejection(
                "commander_intent",
                f"longer than {limit} characters; truncated",
                len(intent),
            )
        )
        intent = intent[:limit]

    decision = RedCommanderDecision(
        schema_version=DECISION_SCHEMA_VERSION,
        campaign_revision=brief.campaign_revision,
        turn_id=brief.turn_id,
        strategy=strategy,
        front_priorities=front_priorities,
        push_postures=push_postures,
        spending_priorities=spending_priorities,
        target_set_priorities=target_set_priorities,
        reserve_policy=reserve_policy,
        commander_intent=intent,
    )
    return ValidationOutcome(decision, rejections)


def parse_decision(text: str, brief: RedCommanderBrief) -> ValidationOutcome:
    """Parse raw model text and validate it in one step.

    Raises :class:`DecisionValidationError` only when the text is not JSON at
    all; content problems come back as a failed :class:`ValidationOutcome`.
    """

    payload = extract_json_object(text)
    return validate_decision(payload, brief)
