"""Operational-level brief for the ACTIVE RED commander.

:mod:`game.ai_commander.intel` projects the *strategic* picture: budget, force
totals, fronts, aggregated target sets. That is everything the COMMANDER mode needs,
because COMMANDER mode only nudges Retribution's own automation.

ACTIVE mode plans individual purchases, repairs, transfers and air packages, so it
needs stable identifiers for the concrete things it can act on:

* ``BASE-n`` — a control point RED owns, with its parking, supply and runway state.
* ``SQN-n`` — a squadron in RED's air wing, with strength, pilots and legal tasks.
* ``TGT-n`` — one individually identified enemy objective that RED may attack.

The same observability gate used by the strategic brief is applied here, so under the
``REALISTIC`` policy only objectives within observation range of RED forces are
enumerated, and no coordinates are ever emitted — a target carries only its category
and the RED base or front it is nearest to. Own-force entries are unrestricted
because a commander obviously knows its own order of battle.

This module is additive: it does not modify :class:`~game.ai_commander.intel.
RedCommanderBrief`, so the strategic brief, its content hash and its leak tests are
unchanged by ACTIVE mode.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional, Sequence

from game.ato.flighttype import FlightType
from game.config import RUNWAY_REPAIR_COST
from game.utils import meters

from .enums import IntelPolicy, TargetSetCategory
from .serialization import jsonable

if TYPE_CHECKING:
    from game.game import Game
    from game.theater import ControlPoint


#: Bumped whenever the wire format changes.
OPERATIONS_SCHEMA_VERSION = "red-operations-brief/1"

#: Hard ceiling on individually enumerated targets. Large theaters can present
#: hundreds of objectives; enumerating all of them would blow the token budget for no
#: planning benefit, because the commander can only fly a handful of packages a turn.
#: Targets are ranked before truncation, and the truncation is reported in the brief
#: so the model knows the list is not exhaustive.
MAX_ENUMERATED_TARGETS = 40

#: Hard ceiling on enumerated own bases and squadrons. Both are RED's own property so
#: there is no fairness concern, only a token one.
MAX_ENUMERATED_BASES = 40
MAX_ENUMERATED_SQUADRONS = 40

#: Mission types the commander may request in an ACTIVE air tasking order. Deliberately
#: excludes the mission families Retribution plans for itself or that make no sense
#: under commander control (ferry flights, Pretense cargo, recovery tankers).
PLANNABLE_MISSION_TYPES: tuple[FlightType, ...] = (
    FlightType.BARCAP,
    FlightType.TARCAP,
    FlightType.CAS,
    FlightType.BAI,
    FlightType.ARMED_RECON,
    FlightType.STRIKE,
    FlightType.SEAD,
    FlightType.DEAD,
    FlightType.SEAD_SWEEP,
    FlightType.SWEEP,
    FlightType.OCA_AIRCRAFT,
    FlightType.OCA_RUNWAY,
    FlightType.ANTISHIP,
    FlightType.ESCORT,
    FlightType.SEAD_ESCORT,
    FlightType.AEWC,
    FlightType.REFUELING,
    FlightType.AIR_ASSAULT,
)

#: Mission types that only make sense as part of a package built around another
#: mission. The validator rejects a package whose primary flight is one of these.
ESCORT_MISSION_TYPES: frozenset[FlightType] = frozenset(
    {FlightType.ESCORT, FlightType.SEAD_ESCORT}
)

#: Which mission types each target category can legitimately be attacked with. This
#: mirrors what Retribution's own package planning tasks propose for each objective
#: type, so an ACTIVE plan cannot ask for something the flight planner would refuse.
TARGET_CATEGORY_MISSIONS: dict[TargetSetCategory, frozenset[FlightType]] = {
    TargetSetCategory.ENEMY_AIR_DEFENCES: frozenset(
        {FlightType.SEAD, FlightType.DEAD, FlightType.SEAD_SWEEP}
    ),
    TargetSetCategory.ENEMY_INFRASTRUCTURE: frozenset({FlightType.STRIKE}),
    TargetSetCategory.ENEMY_MOTORPOOLS: frozenset(
        {FlightType.BAI, FlightType.ARMED_RECON}
    ),
    TargetSetCategory.ENEMY_BATTLE_POSITIONS: frozenset(
        {FlightType.CAS, FlightType.BAI, FlightType.ARMED_RECON}
    ),
    TargetSetCategory.ENEMY_REINFORCEMENTS: frozenset(
        {FlightType.BAI, FlightType.ARMED_RECON}
    ),
    TargetSetCategory.ENEMY_SHIPPING: frozenset({FlightType.ANTISHIP}),
    TargetSetCategory.ENEMY_AIRBASES: frozenset(
        {FlightType.OCA_AIRCRAFT, FlightType.OCA_RUNWAY}
    ),
    TargetSetCategory.AIR_SUPERIORITY: frozenset({FlightType.SWEEP, FlightType.TARCAP}),
    TargetSetCategory.BASE_DEFENCE: frozenset({FlightType.BARCAP, FlightType.TARCAP}),
    TargetSetCategory.BASE_CAPTURE: frozenset({FlightType.AIR_ASSAULT, FlightType.CAS}),
}


#: Categories whose objectives are actively dangerous to RED forces, which the
#: commander needs in order to prioritise defensive counter-air and SEAD.
_THREATENING_CATEGORIES: frozenset[TargetSetCategory] = frozenset(
    {
        TargetSetCategory.ENEMY_AIR_DEFENCES,
        TargetSetCategory.ENEMY_MOTORPOOLS,
        TargetSetCategory.ENEMY_BATTLE_POSITIONS,
        TargetSetCategory.ENEMY_SHIPPING,
    }
)


@dataclass(frozen=True)
class BaseView:
    """One control point RED owns, described the way the base menu describes it."""

    id: str
    name: str
    kind: str
    is_front_line_base: bool
    runway_operational: bool
    runway_repairable: bool
    runway_repair_turns_remaining: Optional[int]
    aircraft_present: int
    aircraft_on_order: int
    parking_free: Optional[int]
    ground_units_present: int
    ground_units_on_order: int
    can_recruit_ground_units: bool
    has_ground_unit_source: bool
    squadron_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    def render(self) -> str:
        aircraft_on_order = (
            f" on_order={self.aircraft_on_order}" if self.aircraft_on_order else ""
        )
        ground_on_order = (
            f" on_order={self.ground_units_on_order}"
            if self.ground_units_on_order
            else ""
        )
        parts = [
            f"{self.id} {self.name}",
            self.kind,
            f"aircraft={self.aircraft_present}{aircraft_on_order}",
            f"ground={self.ground_units_present}{ground_on_order}",
        ]
        if self.parking_free is not None:
            parts.append(f"parking_free={self.parking_free}")
        if not self.runway_operational:
            if self.runway_repair_turns_remaining is not None:
                parts.append(
                    f"runway=repairing({self.runway_repair_turns_remaining} turns)"
                )
            elif self.runway_repairable:
                parts.append(f"runway=damaged,repairable(${RUNWAY_REPAIR_COST})")
            else:
                parts.append("runway=damaged,not_repairable")
        parts.append(
            f"recruit_ground={'yes' if self.can_recruit_ground_units else 'no'}"
        )
        if self.is_front_line_base:
            parts.append("front_line")
        return " | ".join(parts)


@dataclass(frozen=True)
class SquadronView:
    """One RED squadron, described the way the squadron dialog describes it."""

    id: str
    name: str
    aircraft_id: str
    base_id: str
    base_name: str
    aircraft_on_hand: int
    aircraft_untasked: int
    aircraft_on_order: int
    pilots_available: int
    pilot_limit_enabled: bool
    max_fulfillable_aircraft: int
    price_per_aircraft: int
    relocating_to_base_id: Optional[str]
    capable_tasks: tuple[str, ...]
    auto_assignable_tasks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    def render(self) -> str:
        tasks = ",".join(self.capable_tasks) or "none"
        auto = ",".join(self.auto_assignable_tasks) or "none"
        relocation = (
            f" | relocating_to={self.relocating_to_base_id}"
            if self.relocating_to_base_id
            else ""
        )
        on_order = (
            f" on_order={self.aircraft_on_order}" if self.aircraft_on_order else ""
        )
        return (
            f"{self.id} {self.name} | {self.aircraft_id} | at={self.base_id} "
            f"| onhand={self.aircraft_on_hand} untasked={self.aircraft_untasked}"
            f"{on_order} | pilots={self.pilots_available} "
            f"| plannable_now={self.max_fulfillable_aircraft} "
            f"| ${self.price_per_aircraft}/airframe | can_fly={tasks} | auto={auto}"
            f"{relocation}"
        )


@dataclass(frozen=True)
class TargetView:
    """One individually identified enemy objective RED may attack.

    No coordinates are carried. ``near`` names the RED base the objective is closest
    to, which is information RED necessarily has because the detection came from that
    base's own forces.
    """

    id: str
    category: TargetSetCategory
    label: str
    near: str
    threatens_own_forces: bool
    legal_missions: tuple[str, ...]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    def render(self) -> str:
        missions = ",".join(self.legal_missions) or "none"
        threat = " | threatens_own_forces" if self.threatens_own_forces else ""
        note = f" | {self.notes}" if self.notes else ""
        return (
            f"{self.id} | {self.category.value} | {self.label} | near={self.near} "
            f"| missions={missions}{threat}{note}"
        )


@dataclass(frozen=True)
class OperationsBrief:
    """The operational picture handed to the ACTIVE stages."""

    schema_version: str
    campaign_id_hash: str
    campaign_revision: str
    turn_id: int
    intel_policy: IntelPolicy
    budget_available: float
    runway_repair_cost: int
    bases: tuple[BaseView, ...]
    squadrons: tuple[SquadronView, ...]
    targets: tuple[TargetView, ...]
    plannable_mission_types: tuple[str, ...]
    targets_truncated: int = 0
    bases_truncated: int = 0
    squadrons_truncated: int = 0
    withheld_fields: tuple[str, ...] = field(default_factory=tuple)

    @property
    def base_ids(self) -> frozenset[str]:
        return frozenset(base.id for base in self.bases)

    @property
    def squadron_ids(self) -> frozenset[str]:
        return frozenset(squadron.id for squadron in self.squadrons)

    @property
    def target_ids(self) -> frozenset[str]:
        return frozenset(target.id for target in self.targets)

    def base(self, base_id: str) -> Optional[BaseView]:
        for entry in self.bases:
            if entry.id == base_id:
                return entry
        return None

    def squadron(self, squadron_id: str) -> Optional[SquadronView]:
        for entry in self.squadrons:
            if entry.id == squadron_id:
                return entry
        return None

    def target(self, target_id: str) -> Optional[TargetView]:
        for entry in self.targets:
            if entry.id == target_id:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    def content_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def render_compact(self) -> str:
        lines: list[str] = [
            f"OPERATIONS {self.schema_version}",
            f"campaign={self.campaign_id_hash} revision={self.campaign_revision} "
            f"turn={self.turn_id} intel_policy={self.intel_policy.value}",
            f"budget={self.budget_available:.0f}M "
            f"runway_repair_cost={self.runway_repair_cost}M",
            "",
            "[OWN BASES]",
        ]
        if self.bases:
            lines.extend(f"{base.render()}" for base in self.bases)
        else:
            lines.append("(none)")
        if self.bases_truncated:
            lines.append(f"(+{self.bases_truncated} further bases not listed)")

        lines += ["", "[OWN SQUADRONS]"]
        if self.squadrons:
            lines.extend(f"{squadron.render()}" for squadron in self.squadrons)
        else:
            lines.append("(none)")
        if self.squadrons_truncated:
            lines.append(f"(+{self.squadrons_truncated} further squadrons not listed)")

        lines += [
            "",
            "[IDENTIFIED ENEMY OBJECTIVES] (only these may be attacked; "
            "no coordinates are available to you)",
        ]
        if self.targets:
            lines.extend(f"{target.render()}" for target in self.targets)
        else:
            lines.append("(none identified)")
        if self.targets_truncated:
            lines.append(
                f"(+{self.targets_truncated} lower-priority objectives not listed)"
            )

        lines += [
            "",
            "[PLANNABLE MISSION TYPES]",
            ",".join(self.plannable_mission_types),
        ]
        if self.withheld_fields:
            lines += ["", "[WITHHELD]", ",".join(self.withheld_fields)]
        return "\n".join(lines)


@dataclass
class OperationsResolver:
    """Maps brief identifiers back to the live objects they were projected from.

    The brief itself is deliberately object-free so it can be hashed, logged and
    replayed. Execution however needs the real :class:`ControlPoint`,
    :class:`Squadron` and mission-target instances, and re-deriving them by name
    would silently pick the wrong one when two bases share a name. So the projector
    records the mapping as it goes and hands it over here.

    Only identifiers that actually appear in the brief are registered, which means a
    resolver lookup failing is itself a legality signal: the model referred to
    something it was never shown.
    """

    bases: dict[str, Any] = field(default_factory=dict)
    squadrons: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, Any] = field(default_factory=dict)

    def base(self, base_id: str) -> Optional[Any]:
        return self.bases.get(base_id)

    def squadron(self, squadron_id: str) -> Optional[Any]:
        return self.squadrons.get(squadron_id)

    def target(self, target_id: str) -> Optional[Any]:
        return self.targets.get(target_id)

    def base_id_of(self, control_point: Any) -> Optional[str]:
        for base_id, candidate in self.bases.items():
            if candidate is control_point:
                return base_id
        return None


class OperationsProjector:
    """Builds an :class:`OperationsBrief` for the RED coalition.

    Mirrors :class:`game.ai_commander.intel.IntelProjector`: the policy is a required
    argument, every enemy lookup passes through :meth:`_is_observable`, and every
    enumeration is wrapped so a theater quirk degrades to an empty list rather than
    breaking the turn.
    """

    #: Same observation radius the strategic brief uses, so the two briefs can never
    #: disagree about what RED can see.
    OBSERVATION_RANGE_METERS = 120_000.0

    def __init__(self, game: Game, policy: IntelPolicy) -> None:
        from game.theater.player import Player

        self.game = game
        self.policy = policy
        self.player = Player.RED
        self.coalition = game.coalition_for(self.player)
        #: Populated by :meth:`project`; empty until then.
        self.resolver = OperationsResolver()

    # -- entry point ------------------------------------------------------

    def project(self, campaign_id_hash: str, campaign_revision: str) -> OperationsBrief:
        self.resolver = OperationsResolver()
        base_ids: dict[int, str] = {}
        squadron_ids: dict[int, str] = {}
        bases, bases_truncated = self._project_bases(base_ids, squadron_ids)
        squadrons, squadrons_truncated = self._project_squadrons(base_ids, squadron_ids)
        targets, targets_truncated = self._project_targets(bases)
        withheld = (
            (
                "enemy_unit_coordinates",
                "enemy_planned_flights",
                "unobserved_enemy_objectives",
            )
            if self.policy is IntelPolicy.REALISTIC
            else ("enemy_unit_coordinates", "enemy_planned_flights")
        )
        return OperationsBrief(
            schema_version=OPERATIONS_SCHEMA_VERSION,
            campaign_id_hash=campaign_id_hash,
            campaign_revision=campaign_revision,
            turn_id=int(self.game.turn),
            intel_policy=self.policy,
            budget_available=float(self._coalition().budget),
            runway_repair_cost=RUNWAY_REPAIR_COST,
            bases=bases,
            squadrons=squadrons,
            targets=targets,
            plannable_mission_types=tuple(t.value for t in PLANNABLE_MISSION_TYPES),
            targets_truncated=targets_truncated,
            bases_truncated=bases_truncated,
            squadrons_truncated=squadrons_truncated,
            withheld_fields=withheld,
        )

    # -- helpers ----------------------------------------------------------

    def _coalition(self) -> Any:
        return self.coalition

    def _all_control_points(self) -> list[Any]:
        try:
            return list(self.game.theater.controlpoints)
        except Exception:  # pragma: no cover - defensive
            logging.debug("Control point enumeration failed", exc_info=True)
            return []

    def _own_control_points(self) -> list[Any]:
        return [cp for cp in self._all_control_points() if cp.captured == self.player]

    @staticmethod
    def _safe(call: Any) -> list[Any]:
        try:
            return list(call())
        except Exception:  # pragma: no cover - defensive
            logging.debug("Operational enumeration failed", exc_info=True)
            return []

    @staticmethod
    def _int_or_zero(value: Any) -> int:
        return value if isinstance(value, int) else 0

    def _is_observable(self, position: Any) -> bool:
        """True when RED could plausibly have detected something at ``position``."""

        if self.policy is IntelPolicy.FULL_PARITY:
            return True
        if position is None:
            return False
        limit = self.OBSERVATION_RANGE_METERS
        for control_point in self._own_control_points():
            try:
                if control_point.position.distance_to_point(position) <= limit:
                    return True
            except Exception:  # pragma: no cover - defensive
                continue
        return False

    def _base_kind(self, control_point: Any) -> str:
        for attribute, label in (
            ("is_carrier", "carrier"),
            ("is_lha", "lha"),
            ("is_fleet", "fleet"),
            ("is_global", "off-map"),
        ):
            if bool(getattr(control_point, attribute, False)):
                return label
        return (
            "airbase"
            if getattr(control_point, "runway_is_operational", None)
            else "base"
        )

    # -- own forces -------------------------------------------------------

    def _project_bases(
        self, base_ids: dict[int, str], squadron_ids: dict[int, str]
    ) -> tuple[tuple[BaseView, ...], int]:
        from game.theater import ParkingType

        parking_type = ParkingType(
            fixed_wing=True, fixed_wing_stol=True, rotary_wing=True
        )
        front_line_bases = self._front_line_base_ids()

        own = self._own_control_points()
        views: list[BaseView] = []
        for index, control_point in enumerate(own, start=1):
            base_id = f"BASE-{index}"
            base_ids[id(control_point)] = base_id
            if len(views) >= MAX_ENUMERATED_BASES:
                continue

            aircraft_present = 0
            aircraft_ordered = 0
            try:
                allocation = control_point.allocated_aircraft(parking_type)
                aircraft_present = int(allocation.total_present)
                aircraft_ordered = int(allocation.total_ordered)
            except Exception:  # pragma: no cover - defensive
                logging.debug("Aircraft allocation failed", exc_info=True)

            ground_present = 0
            ground_ordered = 0
            try:
                ground = control_point.allocated_ground_units(
                    self._coalition().transfers
                )
                ground_present = int(ground.total_present)
                ground_ordered = int(ground.total_ordered)
            except Exception:  # pragma: no cover - defensive
                logging.debug("Ground allocation failed", exc_info=True)

            parking_free: Optional[int] = None
            try:
                parking_free = int(control_point.unclaimed_parking(parking_type))
            except Exception:  # pragma: no cover - defensive
                parking_free = None

            squadron_id_list: list[str] = []
            for squadron in self._safe(lambda cp=control_point: cp.squadrons):
                key = id(squadron)
                if key not in squadron_ids:
                    squadron_ids[key] = f"SQN-{len(squadron_ids) + 1}"
                squadron_id_list.append(squadron_ids[key])

            views.append(
                BaseView(
                    id=base_id,
                    name=str(control_point.name),
                    kind=self._base_kind(control_point),
                    is_front_line_base=base_id in front_line_bases
                    or id(control_point) in front_line_bases,
                    runway_operational=bool(
                        self._call_bool(control_point, "runway_is_operational", True)
                    ),
                    runway_repairable=bool(
                        getattr(control_point, "runway_can_be_repaired", False)
                    ),
                    runway_repair_turns_remaining=self._repair_turns(control_point),
                    aircraft_present=aircraft_present,
                    aircraft_on_order=aircraft_ordered,
                    parking_free=parking_free,
                    ground_units_present=ground_present,
                    ground_units_on_order=ground_ordered,
                    can_recruit_ground_units=self._can_recruit(control_point),
                    has_ground_unit_source=bool(
                        self._call_bool_with_game(
                            control_point, "has_ground_unit_source", False
                        )
                    ),
                    squadron_ids=tuple(squadron_id_list),
                )
            )
            self.resolver.bases[base_id] = control_point
        return tuple(views), max(0, len(own) - len(views))

    def _front_line_base_ids(self) -> set[Any]:
        result: set[Any] = set()
        try:
            for front in self.game.theater.conflicts():
                for attribute in ("blue_cp", "red_cp"):
                    control_point = getattr(front, attribute, None)
                    if control_point is not None:
                        result.add(id(control_point))
        except Exception:  # pragma: no cover - defensive
            logging.debug("Front line enumeration failed", exc_info=True)
        return result

    @staticmethod
    def _call_bool(target: Any, name: str, default: bool) -> bool:
        attribute = getattr(target, name, None)
        if attribute is None:
            return default
        if callable(attribute):
            try:
                return bool(attribute())
            except Exception:  # pragma: no cover - defensive
                return default
        return bool(attribute)

    def _call_bool_with_game(self, target: Any, name: str, default: bool) -> bool:
        attribute = getattr(target, name, None)
        if attribute is None:
            return default
        try:
            return bool(attribute(self.game))
        except Exception:  # pragma: no cover - defensive
            return default

    def _can_recruit(self, control_point: Any) -> bool:
        if bool(getattr(control_point, "is_global", False)):
            return False
        return self._call_bool_with_game(
            control_point, "can_recruit_ground_units", False
        )

    @staticmethod
    def _repair_turns(control_point: Any) -> Optional[int]:
        status = getattr(control_point, "runway_status", None)
        if status is None:
            return None
        value = getattr(status, "repair_turns_remaining", None)
        return value if isinstance(value, int) else None

    def _project_squadrons(
        self, base_ids: dict[int, str], squadron_ids: dict[int, str]
    ) -> tuple[tuple[SquadronView, ...], int]:
        coalition = self._coalition()
        squadrons = self._safe(coalition.air_wing.iter_squadrons)
        views: list[SquadronView] = []
        for squadron in squadrons:
            key = id(squadron)
            if key not in squadron_ids:
                squadron_ids[key] = f"SQN-{len(squadron_ids) + 1}"
            if len(views) >= MAX_ENUMERATED_SQUADRONS:
                continue
            location = getattr(squadron, "location", None)
            base_id = base_ids.get(id(location)) if location is not None else None
            if base_id is None:
                # A squadron parked somewhere RED no longer holds cannot be tasked or
                # reinforced, so leaving it out of the brief is correct rather than
                # lossy.
                continue
            destination = getattr(squadron, "destination", None)
            capable = tuple(
                task.value
                for task in PLANNABLE_MISSION_TYPES
                if self._squadron_capable(squadron, task)
            )
            auto = tuple(
                task.value
                for task in PLANNABLE_MISSION_TYPES
                if task
                in set(
                    getattr(squadron, "auto_assignable_mission_types", set()) or set()
                )
            )
            aircraft = getattr(squadron, "aircraft", None)
            views.append(
                SquadronView(
                    id=squadron_ids[key],
                    name=str(getattr(squadron, "name", squadron)),
                    aircraft_id=str(getattr(aircraft, "variant_id", "unknown")),
                    base_id=base_id,
                    base_name=str(getattr(location, "name", "unknown")),
                    aircraft_on_hand=self._int_or_zero(
                        getattr(squadron, "owned_aircraft", 0)
                    ),
                    aircraft_untasked=self._int_or_zero(
                        getattr(squadron, "untasked_aircraft", 0)
                    ),
                    aircraft_on_order=self._int_or_zero(
                        getattr(squadron, "pending_deliveries", 0)
                    ),
                    pilots_available=self._int_or_zero(
                        getattr(squadron, "number_of_available_pilots", 0)
                    ),
                    pilot_limit_enabled=bool(
                        getattr(squadron, "pilot_limits_enabled", False)
                    ),
                    max_fulfillable_aircraft=self._int_or_zero(
                        getattr(squadron, "max_fulfillable_aircraft", 0)
                    ),
                    price_per_aircraft=self._int_or_zero(
                        getattr(aircraft, "price", 0) if aircraft is not None else 0
                    ),
                    relocating_to_base_id=(
                        base_ids.get(id(destination))
                        if destination is not None
                        else None
                    ),
                    capable_tasks=capable,
                    auto_assignable_tasks=auto,
                )
            )
            self.resolver.squadrons[squadron_ids[key]] = squadron
        return tuple(views), max(0, len(squadrons) - len(views))

    @staticmethod
    def _squadron_capable(squadron: Any, task: FlightType) -> bool:
        checker = getattr(squadron, "capable_of", None)
        if not callable(checker):
            return False
        try:
            return bool(checker(task))
        except Exception:  # pragma: no cover - defensive
            return False

    # -- enemy objectives -------------------------------------------------

    def _project_targets(
        self, bases: Sequence[BaseView]
    ) -> tuple[tuple[TargetView, ...], int]:
        from game.commander.objectivefinder import ObjectiveFinder

        finder = ObjectiveFinder(self.game, self.player)
        own = self._own_control_points()
        base_lookup = {base.name: base.id for base in bases}

        groups: list[tuple[TargetSetCategory, list[Any], str]] = [
            (
                TargetSetCategory.ENEMY_AIR_DEFENCES,
                self._safe(finder.enemy_air_defenses),
                "",
            ),
            (
                TargetSetCategory.ENEMY_INFRASTRUCTURE,
                self._safe(finder.strike_targets),
                "",
            ),
            (
                TargetSetCategory.ENEMY_MOTORPOOLS,
                self._safe(finder.motorpool_targets),
                "",
            ),
            (TargetSetCategory.ENEMY_SHIPPING, self._safe(finder.enemy_ships), ""),
            (
                TargetSetCategory.ENEMY_REINFORCEMENTS,
                self._safe(finder.convoys),
                "road convoy",
            ),
            (
                TargetSetCategory.ENEMY_SHIPPING,
                self._safe(finder.cargo_ships),
                "sealift",
            ),
            (
                TargetSetCategory.ENEMY_AIRBASES,
                self._safe(finder.enemy_control_points),
                "",
            ),
        ]

        ranked: list[tuple[int, TargetSetCategory, Any, str]] = []
        for category, objects, note in groups:
            for objective in objects:
                position = getattr(objective, "position", None)
                if not self._is_observable(position):
                    continue
                distance = self._closest_own_distance(position, own)
                if distance is None:
                    continue
                ranked.append((distance, category, objective, note))

        ranked.sort(key=lambda entry: (entry[0], entry[1].value, str(entry[2])))
        views: list[TargetView] = []
        for index, (_distance, category, objective, note) in enumerate(ranked, start=1):
            if len(views) >= MAX_ENUMERATED_TARGETS:
                break
            near = self._nearest_base_id(
                getattr(objective, "position", None), own, base_lookup
            )
            missions = TARGET_CATEGORY_MISSIONS.get(category, frozenset())
            self.resolver.targets[f"TGT-{index}"] = objective
            views.append(
                TargetView(
                    id=f"TGT-{index}",
                    category=category,
                    label=self._label_for(objective, category),
                    near=near,
                    threatens_own_forces=category in _THREATENING_CATEGORIES,
                    legal_missions=tuple(sorted(m.value for m in missions)),
                    notes=note,
                )
            )
        return tuple(views), max(0, len(ranked) - len(views))

    @staticmethod
    def _closest_own_distance(position: Any, own: Sequence[Any]) -> Optional[int]:
        best: Optional[float] = None
        for control_point in own:
            try:
                distance = control_point.position.distance_to_point(position)
            except Exception:  # pragma: no cover - defensive
                continue
            if best is None or distance < best:
                best = distance
        if best is None:
            return None
        return int(meters(best).nautical_miles)

    @staticmethod
    def _nearest_base_id(
        position: Any, own: Sequence[Any], base_lookup: dict[str, str]
    ) -> str:
        best: Optional[float] = None
        best_name: Optional[str] = None
        for control_point in own:
            try:
                distance = control_point.position.distance_to_point(position)
            except Exception:  # pragma: no cover - defensive
                continue
            if best is None or distance < best:
                best = distance
                best_name = str(control_point.name)
        if best_name is None:
            return "unknown"
        return base_lookup.get(best_name, best_name)

    @staticmethod
    def _label_for(objective: Any, category: TargetSetCategory) -> str:
        """A short human label that carries no position information."""

        for attribute in ("category", "name"):
            value = getattr(objective, attribute, None)
            if isinstance(value, str) and value:
                if attribute == "name":
                    # Objective names embed the airbase or objective they belong to,
                    # which is exactly the level of detail a reconnaissance report
                    # would carry, so it is safe. Trim to keep the prompt small.
                    return value[:48]
                return value
        return category.value


def target_ids_for_missions(
    brief: OperationsBrief, missions: Iterable[FlightType]
) -> Iterator[str]:
    """Target ids that at least one of ``missions`` may legally be flown against."""

    wanted = {m.value for m in missions}
    for target in brief.targets:
        if wanted & set(target.legal_missions):
            yield target.id
