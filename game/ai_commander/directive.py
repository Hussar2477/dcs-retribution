"""The normalised directive: the only thing the rest of the game ever sees.

A :class:`RedCommanderDecision` is model output shaped by a brief. A
:class:`CommanderDirective` is what survived validation *and* legality checking,
expressed in terms the deterministic planner understands, with every opaque
identifier already resolved to a stable game-side key.

Two properties matter:

* It is a plain value object -- no live game references, fully serialisable, so
  the audit log records exactly what was applied.
* Fronts are keyed by ``(own base name, enemy base name)`` rather than by the
  turn-scoped ``FRONT-n`` identifier, so the directive stays meaningful if the
  front enumeration order changes between projection and application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING

from game.ai_commander.enums import (
    FrontPosture,
    MissionPurpose,
    ProcurementCategory,
    RedStrategy,
    ReservePolicy,
    TargetSetCategory,
)
from game.ai_commander.serialization import jsonable

if TYPE_CHECKING:
    from game.theater import FrontLine
    from game.theater.player import Player


#: A front, identified by the two bases facing each other across it.
FrontKey = tuple[str, str]


def front_key(front_line: FrontLine, player: Player) -> FrontKey:
    """Stable key for ``front_line`` from ``player``'s point of view."""

    return (
        str(front_line.control_point_friendly_to(player).name),
        str(front_line.control_point_hostile_to(player).name),
    )


@dataclass(frozen=True)
class CommanderDirective:
    """A validated, legal, fully resolved strategic directive for one turn."""

    turn_id: int
    campaign_revision: str
    strategy: RedStrategy
    reserve_policy: ReservePolicy
    #: Target-set categories in the commander's priority order. Categories the
    #: commander did not rank are absent, and keep their stock relative order
    #: after the ranked ones.
    target_set_order: tuple[TargetSetCategory, ...] = ()
    #: Advisory purpose per ranked category, for the audit log and UI.
    target_set_purposes: Mapping[str, str] = field(default_factory=dict)
    #: Spending categories in the commander's priority order.
    procurement_order: tuple[ProcurementCategory, ...] = ()
    #: Fronts in the commander's priority order.
    front_order: tuple[FrontKey, ...] = ()
    #: Requested posture per front. Only fronts whose posture passed the game's
    #: own advantage predicate appear here.
    front_postures: Mapping[str, str] = field(default_factory=dict)
    commander_intent: str = ""

    # -- lookups ----------------------------------------------------------

    @staticmethod
    def encode_front(key: FrontKey) -> str:
        """Mapping-friendly encoding of a :data:`FrontKey`.

        Dataclass mappings have to be JSON-serialisable for the audit log, and
        JSON object keys must be strings, so the tuple is flattened with a
        separator that cannot appear in a DCS base name.
        """

        return f"{key[0]} -> {key[1]}"

    def posture_for(self, key: FrontKey) -> Optional[FrontPosture]:
        raw = self.front_postures.get(self.encode_front(key))
        if raw is None:
            return None
        try:
            return FrontPosture(raw)
        except ValueError:  # pragma: no cover - written only by this module
            return None

    def purpose_for(self, category: TargetSetCategory) -> Optional[MissionPurpose]:
        raw = self.target_set_purposes.get(category.value)
        if raw is None:
            return None
        try:
            return MissionPurpose(raw)
        except ValueError:  # pragma: no cover - written only by this module
            return None

    def front_rank(self, key: FrontKey) -> int:
        """Priority index of a front; unranked fronts sort last."""

        try:
            return self.front_order.index(key)
        except ValueError:
            return len(self.front_order) + 1

    @property
    def prioritises_reserves(self) -> bool:
        return self.reserve_policy is ReservePolicy.BUILD_RESERVES

    @property
    def has_content(self) -> bool:
        """Whether anything at all survived validation.

        A directive with no orderings and no postures would change nothing, so
        the controller treats it as a fallback rather than pretending the AI
        made a decision.
        """

        return bool(
            self.target_set_order
            or self.procurement_order
            or self.front_postures
            or self.front_order
        )

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CommanderDirective:
        """Rebuild a directive from :meth:`to_dict` output.

        Used to replay an already-accepted directive from the audit log when
        ``initialize_turn`` runs more than once for the same turn, so a
        re-entrant turn is applied identically without paying for a second call.
        Anything unrecognised is dropped rather than guessed at.
        """

        def _front_key(raw: Any) -> Optional[FrontKey]:
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                return (str(raw[0]), str(raw[1]))
            return None

        target_order: list[TargetSetCategory] = []
        for raw in payload.get("target_set_order") or ():
            try:
                target_order.append(TargetSetCategory(raw))
            except ValueError:
                continue

        procurement_order: list[ProcurementCategory] = []
        for raw in payload.get("procurement_order") or ():
            try:
                procurement_order.append(ProcurementCategory(raw))
            except ValueError:
                continue

        front_order: list[FrontKey] = []
        for raw in payload.get("front_order") or ():
            key = _front_key(raw)
            if key is not None:
                front_order.append(key)

        purposes = {
            str(k): str(v)
            for k, v in (payload.get("target_set_purposes") or {}).items()
        }
        postures = {
            str(k): str(v) for k, v in (payload.get("front_postures") or {}).items()
        }
        return cls(
            turn_id=int(payload["turn_id"]),
            campaign_revision=str(payload["campaign_revision"]),
            strategy=RedStrategy(payload["strategy"]),
            reserve_policy=ReservePolicy(payload["reserve_policy"]),
            target_set_order=tuple(target_order),
            target_set_purposes=purposes,
            procurement_order=tuple(procurement_order),
            front_order=tuple(front_order),
            front_postures=postures,
            commander_intent=str(payload.get("commander_intent") or ""),
        )

    def render_summary(self) -> str:
        """Human-readable one-paragraph summary for the decision-log viewer."""

        lines = [
            f"Strategy: {self.strategy.value}",
            f"Reserve policy: {self.reserve_policy.value}",
        ]
        if self.target_set_order:
            lines.append(
                "Target priorities: "
                + " > ".join(c.value for c in self.target_set_order)
            )
        if self.procurement_order:
            lines.append(
                "Spending priorities: "
                + " > ".join(c.value for c in self.procurement_order)
            )
        if self.front_order:
            lines.append(
                "Front priorities: "
                + " > ".join(self.encode_front(k) for k in self.front_order)
            )
        for encoded, posture in self.front_postures.items():
            lines.append(f"Posture {encoded}: {posture}")
        if self.commander_intent:
            lines.append(f"Intent: {self.commander_intent}")
        return "\n".join(lines)


def build_directive(
    *,
    turn_id: int,
    campaign_revision: str,
    strategy: RedStrategy,
    reserve_policy: ReservePolicy,
    target_sets: Sequence[tuple[TargetSetCategory, MissionPurpose]] = (),
    procurement: Sequence[ProcurementCategory] = (),
    fronts: Sequence[FrontKey] = (),
    postures: Sequence[tuple[FrontKey, FrontPosture]] = (),
    commander_intent: str = "",
) -> CommanderDirective:
    """Construct a directive, de-duplicating while preserving order."""

    seen_categories: set[TargetSetCategory] = set()
    ordered_targets: list[TargetSetCategory] = []
    purposes: dict[str, str] = {}
    for category, purpose in target_sets:
        if category in seen_categories:
            continue
        seen_categories.add(category)
        ordered_targets.append(category)
        purposes[category.value] = purpose.value

    seen_procurement: set[ProcurementCategory] = set()
    ordered_procurement: list[ProcurementCategory] = []
    for spend in procurement:
        if spend in seen_procurement:
            continue
        seen_procurement.add(spend)
        ordered_procurement.append(spend)

    seen_fronts: set[FrontKey] = set()
    ordered_fronts: list[FrontKey] = []
    for key in fronts:
        if key in seen_fronts:
            continue
        seen_fronts.add(key)
        ordered_fronts.append(key)

    posture_map: dict[str, str] = {}
    for key, posture in postures:
        posture_map.setdefault(CommanderDirective.encode_front(key), posture.value)

    return CommanderDirective(
        turn_id=turn_id,
        campaign_revision=campaign_revision,
        strategy=strategy,
        reserve_policy=reserve_policy,
        target_set_order=tuple(ordered_targets),
        target_set_purposes=purposes,
        procurement_order=tuple(ordered_procurement),
        front_order=tuple(ordered_fronts),
        front_postures=posture_map,
        commander_intent=commander_intent,
    )
