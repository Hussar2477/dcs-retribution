"""ACTIVE-mode plan schemas.

COMMANDER mode asks the model for one strategic decision and lets Retribution's own
automation carry it out. ACTIVE mode gives the model the controls a human player has,
which means it has to name concrete things: buy *these* airframes into *that*
squadron, repair *that* runway, move *those* tanks to *that* base, fly *this* package
against *that* objective.

The turn is therefore split into three stages, each its own bounded LLM call:

1. **COMMAND** — reuses :mod:`game.ai_commander.decision` unchanged. Aggression per
   front, overall strategy, reserve policy, spending and target priorities.
2. **LOGISTICS** — this module. Procurement, runway repair, ground transfers,
   squadron relocation and squadron tasking permissions.
3. **AIR TASKING** — this module. Individual packages and the flights inside them.

Validation here is *structural and referential*: does the plan use identifiers from
the brief, are the quantities sane, is the mission type one the target category can
legally be attacked with, is the airframe one this faction actually has. Anything that
depends on live campaign state (budget, parking, pilots, supply routes) is checked
separately by :mod:`game.ai_commander.planlegality` immediately before execution, so
a plan can never be accepted against a stale snapshot.

Every rejection is a machine-readable :class:`game.ai_commander.decision.Rejection`
carrying the offending element path, a reason and the offending value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Optional, Sequence

from game.ato.flighttype import FlightType

from .capabilities import CapabilityIndex
from .decision import MAX_LIST_ENTRIES, Rejection
from .operations import (
    ESCORT_MISSION_TYPES,
    PLANNABLE_MISSION_TYPES,
    OperationsBrief,
)

#: Bumped whenever the wire format of a stage changes.
LOGISTICS_SCHEMA_VERSION = "red-commander-logistics/1"
AIR_TASKING_SCHEMA_VERSION = "red-commander-air-tasking/1"

#: Upper bounds. These are not balance decisions, they are denial-of-service limits: a
#: model that emits a thousand orders must not be able to make turn processing hang,
#: and a validated plan must stay small enough to audit by eye.
MAX_ORDERS_PER_LIST = MAX_LIST_ENTRIES
MAX_QUANTITY_PER_ORDER = 24
MAX_PACKAGES_PER_TURN = 12
MAX_FLIGHTS_PER_PACKAGE = 6
MAX_AIRCRAFT_PER_FLIGHT = 4
MAX_UNIT_TYPES_PER_TRANSFER = 8
MAX_INTENT_CHARACTERS = 400

_MISSION_BY_VALUE: dict[str, FlightType] = {t.value: t for t in PLANNABLE_MISSION_TYPES}


@unique
class CommanderStage(Enum):
    """The stages of an ACTIVE turn, in the order they run."""

    COMMAND = "command"
    LOGISTICS = "logistics"
    AIR_TASKING = "air_tasking"

    @property
    def schema_version(self) -> str:
        if self is CommanderStage.LOGISTICS:
            return LOGISTICS_SCHEMA_VERSION
        if self is CommanderStage.AIR_TASKING:
            return AIR_TASKING_SCHEMA_VERSION
        from .decision import DECISION_SCHEMA_VERSION

        return DECISION_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Logistics orders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AircraftOrder:
    """Buy ``quantity`` more airframes for an existing squadron."""

    squadron_id: str
    quantity: int

    def describe(self) -> str:
        return f"buy {self.quantity} airframes for {self.squadron_id}"


@dataclass(frozen=True)
class GroundUnitOrder:
    """Buy ``quantity`` ground units of ``unit_id`` at ``base_id``."""

    base_id: str
    unit_id: str
    quantity: int

    def describe(self) -> str:
        return f"buy {self.quantity}x {self.unit_id} at {self.base_id}"


@dataclass(frozen=True)
class RunwayRepairOrder:
    """Begin runway repair at ``base_id``."""

    base_id: str

    def describe(self) -> str:
        return f"repair runway at {self.base_id}"


@dataclass(frozen=True)
class SquadronRelocationOrder:
    """Move a squadron to another RED base, or cancel a pending move."""

    squadron_id: str
    #: ``None`` cancels an existing relocation order.
    base_id: Optional[str]

    def describe(self) -> str:
        if self.base_id is None:
            return f"cancel relocation of {self.squadron_id}"
        return f"relocate {self.squadron_id} to {self.base_id}"


@dataclass(frozen=True)
class SquadronTaskingOrder:
    """Set which mission types Retribution's automation may assign a squadron."""

    squadron_id: str
    mission_types: tuple[FlightType, ...]

    def describe(self) -> str:
        tasks = ",".join(t.value for t in self.mission_types) or "none"
        return f"limit {self.squadron_id} to {tasks}"


@dataclass(frozen=True)
class GroundTransferOrder:
    """Move existing ground units between two RED bases."""

    origin_base_id: str
    destination_base_id: str
    units: tuple[tuple[str, int], ...]

    @property
    def size(self) -> int:
        return sum(count for _, count in self.units)

    def describe(self) -> str:
        units = ",".join(f"{count}x{unit}" for unit, count in self.units)
        return (
            f"transfer {units} from {self.origin_base_id} "
            f"to {self.destination_base_id}"
        )


@dataclass(frozen=True)
class LogisticsPlan:
    """A structurally valid logistics stage output."""

    schema_version: str
    turn_id: int
    campaign_revision: str
    intent: str
    aircraft_orders: tuple[AircraftOrder, ...] = field(default=())
    ground_orders: tuple[GroundUnitOrder, ...] = field(default=())
    runway_repairs: tuple[RunwayRepairOrder, ...] = field(default=())
    squadron_relocations: tuple[SquadronRelocationOrder, ...] = field(default=())
    squadron_tasking: tuple[SquadronTaskingOrder, ...] = field(default=())
    ground_transfers: tuple[GroundTransferOrder, ...] = field(default=())

    @property
    def has_content(self) -> bool:
        return bool(
            self.aircraft_orders
            or self.ground_orders
            or self.runway_repairs
            or self.squadron_relocations
            or self.squadron_tasking
            or self.ground_transfers
        )

    @property
    def order_count(self) -> int:
        return (
            len(self.aircraft_orders)
            + len(self.ground_orders)
            + len(self.runway_repairs)
            + len(self.squadron_relocations)
            + len(self.squadron_tasking)
            + len(self.ground_transfers)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "turn_id": self.turn_id,
            "campaign_revision": self.campaign_revision,
            "intent": self.intent,
            "aircraft_orders": [
                {"squadron_id": o.squadron_id, "quantity": o.quantity}
                for o in self.aircraft_orders
            ],
            "ground_orders": [
                {
                    "base_id": o.base_id,
                    "unit_id": o.unit_id,
                    "quantity": o.quantity,
                }
                for o in self.ground_orders
            ],
            "runway_repairs": [{"base_id": o.base_id} for o in self.runway_repairs],
            "squadron_relocations": [
                {"squadron_id": o.squadron_id, "base_id": o.base_id}
                for o in self.squadron_relocations
            ],
            "squadron_tasking": [
                {
                    "squadron_id": o.squadron_id,
                    "mission_types": [t.value for t in o.mission_types],
                }
                for o in self.squadron_tasking
            ],
            "ground_transfers": [
                {
                    "origin_base_id": o.origin_base_id,
                    "destination_base_id": o.destination_base_id,
                    "units": [
                        {"unit_id": unit, "quantity": count} for unit, count in o.units
                    ],
                }
                for o in self.ground_transfers
            ],
        }

    def render_summary(self) -> str:
        lines = [f"logistics intent: {self.intent}" if self.intent else "logistics"]
        for group in (
            self.aircraft_orders,
            self.ground_orders,
            self.runway_repairs,
            self.squadron_relocations,
            self.squadron_tasking,
            self.ground_transfers,
        ):
            lines.extend(f"  {order.describe()}" for order in group)
        if self.order_count == 0:
            lines.append("  (no orders)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Air tasking orders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposedFlightOrder:
    """One flight inside a package."""

    mission_type: FlightType
    aircraft_count: int
    #: Optional preferred airframe. ``None`` lets Retribution's own squadron selection
    #: pick the best available airframe, which is usually the better outcome.
    aircraft_id: Optional[str] = None

    def describe(self) -> str:
        airframe = f" ({self.aircraft_id})" if self.aircraft_id else ""
        return f"{self.aircraft_count}x {self.mission_type.value}{airframe}"


@dataclass(frozen=True)
class ProposedPackageOrder:
    """One package: a target plus the flights sent against it."""

    target_id: str
    priority: int
    flights: tuple[ProposedFlightOrder, ...]
    asap: bool = False

    @property
    def primary_mission(self) -> FlightType:
        return self.flights[0].mission_type

    def describe(self) -> str:
        flights = ", ".join(flight.describe() for flight in self.flights)
        asap = " asap" if self.asap else ""
        return f"#{self.priority} {self.target_id}{asap}: {flights}"


@dataclass(frozen=True)
class AirTaskingPlan:
    """A structurally valid air tasking stage output."""

    schema_version: str
    turn_id: int
    campaign_revision: str
    intent: str
    packages: tuple[ProposedPackageOrder, ...] = field(default=())

    @property
    def has_content(self) -> bool:
        return bool(self.packages)

    @property
    def order_count(self) -> int:
        return len(self.packages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "turn_id": self.turn_id,
            "campaign_revision": self.campaign_revision,
            "intent": self.intent,
            "packages": [
                {
                    "target_id": package.target_id,
                    "priority": package.priority,
                    "asap": package.asap,
                    "flights": [
                        {
                            "mission_type": flight.mission_type.value,
                            "aircraft_count": flight.aircraft_count,
                            "aircraft_id": flight.aircraft_id,
                        }
                        for flight in package.flights
                    ],
                }
                for package in self.packages
            ],
        }

    def render_summary(self) -> str:
        lines = [f"air tasking intent: {self.intent}" if self.intent else "air tasking"]
        if not self.packages:
            lines.append("  (no packages)")
        lines.extend(f"  {package.describe()}" for package in self.packages)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _fatal(element: str, reason: str, value: Any = None) -> list[Rejection]:
    return [Rejection(element=element, reason=reason, value=value)]


def _check_envelope(
    payload: Any, expected_schema: str, brief: OperationsBrief
) -> list[Rejection]:
    """Validates the fields every stage payload must carry.

    A mismatch here is fatal for the stage: without a matching schema version, turn id
    and campaign revision we cannot tell whether the model was answering the question
    we asked or an earlier one.
    """

    if not isinstance(payload, dict):
        return _fatal("root", "response was not a JSON object")

    schema_version = payload.get("schema_version")
    if schema_version != expected_schema:
        return _fatal(
            "schema_version",
            f"expected {expected_schema}",
            schema_version,
        )

    turn_id = payload.get("turn_id")
    if not isinstance(turn_id, int) or isinstance(turn_id, bool):
        return _fatal("turn_id", "turn_id must be an integer", turn_id)
    if turn_id != brief.turn_id:
        return _fatal("turn_id", f"expected turn {brief.turn_id}", turn_id)

    revision = payload.get("campaign_revision")
    if revision != brief.campaign_revision:
        return _fatal(
            "campaign_revision",
            "campaign state changed between briefing and application",
            revision,
        )
    return []


def _intent_of(payload: dict[str, Any]) -> str:
    intent = payload.get("intent")
    if not isinstance(intent, str):
        return ""
    return intent.strip()[:MAX_INTENT_CHARACTERS]


def _entries(
    payload: dict[str, Any],
    key: str,
    rejections: list[Rejection],
) -> list[Any]:
    """Pulls a list field, rejecting non-lists and truncating over-long ones."""

    raw = payload.get(key)
    if raw is None:
        return []
    if not isinstance(raw, list):
        rejections.append(
            Rejection(element=key, reason="value must be a list", value=raw)
        )
        return []
    if len(raw) > MAX_ORDERS_PER_LIST:
        rejections.append(
            Rejection(
                element=key,
                reason=f"list is longer than the {MAX_ORDERS_PER_LIST} entry limit; "
                "the extra entries were dropped",
                value=len(raw),
            )
        )
        return raw[:MAX_ORDERS_PER_LIST]
    return raw


def _quantity(
    element: str,
    value: Any,
    limit: int,
    rejections: list[Rejection],
) -> Optional[int]:
    if not isinstance(value, int) or isinstance(value, bool):
        rejections.append(
            Rejection(
                element=element, reason="quantity must be an integer", value=value
            )
        )
        return None
    if value < 1:
        rejections.append(
            Rejection(
                element=element, reason="quantity must be at least 1", value=value
            )
        )
        return None
    if value > limit:
        rejections.append(
            Rejection(
                element=element,
                reason=f"quantity exceeds the {limit} per-order limit",
                value=value,
            )
        )
        return None
    return value


def _known_id(
    element: str,
    value: Any,
    known: frozenset[str],
    label: str,
    rejections: list[Rejection],
) -> Optional[str]:
    if not isinstance(value, str) or not value:
        rejections.append(
            Rejection(element=element, reason=f"{label} must be a string", value=value)
        )
        return None
    if value not in known:
        rejections.append(
            Rejection(
                element=element,
                reason=f"{label} is not in the brief",
                value=value,
            )
        )
        return None
    return value


def _unexpected_keys(
    payload: dict[str, Any], allowed: Sequence[str], rejections: list[Rejection]
) -> None:
    extra = sorted(set(payload) - set(allowed))
    if extra:
        rejections.append(
            Rejection(
                element="root",
                reason="unexpected keys were ignored",
                value=",".join(extra),
            )
        )


# ---------------------------------------------------------------------------
# Logistics validation
# ---------------------------------------------------------------------------

_LOGISTICS_KEYS = (
    "schema_version",
    "turn_id",
    "campaign_revision",
    "intent",
    "aircraft_orders",
    "ground_orders",
    "runway_repairs",
    "squadron_relocations",
    "squadron_tasking",
    "ground_transfers",
)


def validate_logistics_plan(
    payload: Any, brief: OperationsBrief, capabilities: CapabilityIndex
) -> tuple[Optional[LogisticsPlan], list[Rejection]]:
    """Structurally validates a logistics stage payload.

    Returns ``(plan, rejections)``. ``plan`` is ``None`` only when the payload was
    unusable as a whole; individual bad orders are dropped with a rejection each and
    the remaining orders are kept, which is what lets a partly-hallucinated plan still
    contribute something rather than costing the whole turn.
    """

    fatal = _check_envelope(payload, LOGISTICS_SCHEMA_VERSION, brief)
    if fatal:
        return None, fatal
    assert isinstance(payload, dict)

    rejections: list[Rejection] = []
    _unexpected_keys(payload, _LOGISTICS_KEYS, rejections)

    aircraft_orders = _validate_aircraft_orders(payload, brief, rejections)
    ground_orders = _validate_ground_orders(payload, brief, capabilities, rejections)
    repairs = _validate_runway_repairs(payload, brief, rejections)
    relocations = _validate_relocations(payload, brief, rejections)
    tasking = _validate_squadron_tasking(payload, brief, rejections)
    transfers = _validate_ground_transfers(payload, brief, capabilities, rejections)

    plan = LogisticsPlan(
        schema_version=LOGISTICS_SCHEMA_VERSION,
        turn_id=brief.turn_id,
        campaign_revision=brief.campaign_revision,
        intent=_intent_of(payload),
        aircraft_orders=aircraft_orders,
        ground_orders=ground_orders,
        runway_repairs=repairs,
        squadron_relocations=relocations,
        squadron_tasking=tasking,
        ground_transfers=transfers,
    )
    return plan, rejections


def _validate_aircraft_orders(
    payload: dict[str, Any], brief: OperationsBrief, rejections: list[Rejection]
) -> tuple[AircraftOrder, ...]:
    orders: list[AircraftOrder] = []
    seen: set[str] = set()
    for index, entry in enumerate(_entries(payload, "aircraft_orders", rejections)):
        element = f"aircraft_orders[{index}]"
        if not isinstance(entry, dict):
            rejections.append(
                Rejection(
                    element=element, reason="entry must be an object", value=entry
                )
            )
            continue
        squadron_id = _known_id(
            f"{element}.squadron_id",
            entry.get("squadron_id"),
            brief.squadron_ids,
            "squadron identifier",
            rejections,
        )
        quantity = _quantity(
            f"{element}.quantity",
            entry.get("quantity"),
            MAX_QUANTITY_PER_ORDER,
            rejections,
        )
        if squadron_id is None or quantity is None:
            continue
        if squadron_id in seen:
            rejections.append(
                Rejection(
                    element=f"{element}.squadron_id",
                    reason="duplicate identifier",
                    value=squadron_id,
                )
            )
            continue
        seen.add(squadron_id)
        orders.append(AircraftOrder(squadron_id=squadron_id, quantity=quantity))
    return tuple(orders)


def _validate_ground_orders(
    payload: dict[str, Any],
    brief: OperationsBrief,
    capabilities: CapabilityIndex,
    rejections: list[Rejection],
) -> tuple[GroundUnitOrder, ...]:
    purchasable = frozenset(capabilities.purchasable_ground_unit_ids)
    orders: list[GroundUnitOrder] = []
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(_entries(payload, "ground_orders", rejections)):
        element = f"ground_orders[{index}]"
        if not isinstance(entry, dict):
            rejections.append(
                Rejection(
                    element=element, reason="entry must be an object", value=entry
                )
            )
            continue
        base_id = _known_id(
            f"{element}.base_id",
            entry.get("base_id"),
            brief.base_ids,
            "base identifier",
            rejections,
        )
        unit_id = entry.get("unit_id")
        quantity = _quantity(
            f"{element}.quantity",
            entry.get("quantity"),
            MAX_QUANTITY_PER_ORDER,
            rejections,
        )
        if not isinstance(unit_id, str) or unit_id not in purchasable:
            rejections.append(
                Rejection(
                    element=f"{element}.unit_id",
                    reason="unit is not a ground unit this faction can purchase",
                    value=unit_id,
                )
            )
            continue
        if base_id is None or quantity is None:
            continue
        key = (base_id, unit_id)
        if key in seen:
            rejections.append(
                Rejection(
                    element=f"{element}.unit_id",
                    reason="duplicate identifier",
                    value=unit_id,
                )
            )
            continue
        seen.add(key)
        orders.append(
            GroundUnitOrder(base_id=base_id, unit_id=unit_id, quantity=quantity)
        )
    return tuple(orders)


def _validate_runway_repairs(
    payload: dict[str, Any], brief: OperationsBrief, rejections: list[Rejection]
) -> tuple[RunwayRepairOrder, ...]:
    orders: list[RunwayRepairOrder] = []
    seen: set[str] = set()
    for index, entry in enumerate(_entries(payload, "runway_repairs", rejections)):
        element = f"runway_repairs[{index}]"
        base_value = entry.get("base_id") if isinstance(entry, dict) else entry
        base_id = _known_id(
            f"{element}.base_id",
            base_value,
            brief.base_ids,
            "base identifier",
            rejections,
        )
        if base_id is None:
            continue
        if base_id in seen:
            rejections.append(
                Rejection(
                    element=f"{element}.base_id",
                    reason="duplicate identifier",
                    value=base_id,
                )
            )
            continue
        seen.add(base_id)
        orders.append(RunwayRepairOrder(base_id=base_id))
    return tuple(orders)


def _validate_relocations(
    payload: dict[str, Any], brief: OperationsBrief, rejections: list[Rejection]
) -> tuple[SquadronRelocationOrder, ...]:
    orders: list[SquadronRelocationOrder] = []
    seen: set[str] = set()
    for index, entry in enumerate(
        _entries(payload, "squadron_relocations", rejections)
    ):
        element = f"squadron_relocations[{index}]"
        if not isinstance(entry, dict):
            rejections.append(
                Rejection(
                    element=element, reason="entry must be an object", value=entry
                )
            )
            continue
        squadron_id = _known_id(
            f"{element}.squadron_id",
            entry.get("squadron_id"),
            brief.squadron_ids,
            "squadron identifier",
            rejections,
        )
        if squadron_id is None:
            continue
        raw_base = entry.get("base_id")
        base_id: Optional[str]
        if raw_base is None:
            base_id = None
        else:
            base_id = _known_id(
                f"{element}.base_id",
                raw_base,
                brief.base_ids,
                "base identifier",
                rejections,
            )
            if base_id is None:
                continue
        if squadron_id in seen:
            rejections.append(
                Rejection(
                    element=f"{element}.squadron_id",
                    reason="duplicate identifier",
                    value=squadron_id,
                )
            )
            continue
        seen.add(squadron_id)
        orders.append(SquadronRelocationOrder(squadron_id=squadron_id, base_id=base_id))
    return tuple(orders)


def _validate_squadron_tasking(
    payload: dict[str, Any], brief: OperationsBrief, rejections: list[Rejection]
) -> tuple[SquadronTaskingOrder, ...]:
    orders: list[SquadronTaskingOrder] = []
    seen: set[str] = set()
    for index, entry in enumerate(_entries(payload, "squadron_tasking", rejections)):
        element = f"squadron_tasking[{index}]"
        if not isinstance(entry, dict):
            rejections.append(
                Rejection(
                    element=element, reason="entry must be an object", value=entry
                )
            )
            continue
        squadron_id = _known_id(
            f"{element}.squadron_id",
            entry.get("squadron_id"),
            brief.squadron_ids,
            "squadron identifier",
            rejections,
        )
        if squadron_id is None:
            continue
        view = brief.squadron(squadron_id)
        capable = set(view.capable_tasks) if view is not None else set()
        raw_types = entry.get("mission_types")
        if not isinstance(raw_types, list):
            rejections.append(
                Rejection(
                    element=f"{element}.mission_types",
                    reason="value must be a list of mission types",
                    value=raw_types,
                )
            )
            continue
        missions: list[FlightType] = []
        for position, raw in enumerate(raw_types[:MAX_ORDERS_PER_LIST]):
            mission = _MISSION_BY_VALUE.get(raw) if isinstance(raw, str) else None
            if mission is None:
                rejections.append(
                    Rejection(
                        element=f"{element}.mission_types[{position}]",
                        reason="mission type is not one the commander may assign",
                        value=raw,
                    )
                )
                continue
            if raw not in capable:
                rejections.append(
                    Rejection(
                        element=f"{element}.mission_types[{position}]",
                        reason="this squadron's airframe cannot fly this mission type",
                        value=raw,
                    )
                )
                continue
            if mission not in missions:
                missions.append(mission)
        if not missions:
            rejections.append(
                Rejection(
                    element=f"{element}.mission_types",
                    reason="no legal mission type remained, so the squadron's existing "
                    "tasking was left alone",
                    value=raw_types,
                )
            )
            continue
        if squadron_id in seen:
            rejections.append(
                Rejection(
                    element=f"{element}.squadron_id",
                    reason="duplicate identifier",
                    value=squadron_id,
                )
            )
            continue
        seen.add(squadron_id)
        orders.append(
            SquadronTaskingOrder(squadron_id=squadron_id, mission_types=tuple(missions))
        )
    return tuple(orders)


def _validate_ground_transfers(
    payload: dict[str, Any],
    brief: OperationsBrief,
    capabilities: CapabilityIndex,
    rejections: list[Rejection],
) -> tuple[GroundTransferOrder, ...]:
    known_units = frozenset(capabilities.ground_unit_ids)
    orders: list[GroundTransferOrder] = []
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(_entries(payload, "ground_transfers", rejections)):
        element = f"ground_transfers[{index}]"
        if not isinstance(entry, dict):
            rejections.append(
                Rejection(
                    element=element, reason="entry must be an object", value=entry
                )
            )
            continue
        origin = _known_id(
            f"{element}.origin_base_id",
            entry.get("origin_base_id"),
            brief.base_ids,
            "base identifier",
            rejections,
        )
        destination = _known_id(
            f"{element}.destination_base_id",
            entry.get("destination_base_id"),
            brief.base_ids,
            "base identifier",
            rejections,
        )
        if origin is None or destination is None:
            continue
        if origin == destination:
            rejections.append(
                Rejection(
                    element=f"{element}.destination_base_id",
                    reason="transfer origin and destination are the same base",
                    value=destination,
                )
            )
            continue
        raw_units = entry.get("units")
        if not isinstance(raw_units, list):
            rejections.append(
                Rejection(
                    element=f"{element}.units",
                    reason="value must be a list of unit orders",
                    value=raw_units,
                )
            )
            continue
        units: list[tuple[str, int]] = []
        for position, raw in enumerate(raw_units[:MAX_UNIT_TYPES_PER_TRANSFER]):
            unit_element = f"{element}.units[{position}]"
            if not isinstance(raw, dict):
                rejections.append(
                    Rejection(
                        element=unit_element,
                        reason="entry must be an object",
                        value=raw,
                    )
                )
                continue
            unit_id = raw.get("unit_id")
            if not isinstance(unit_id, str) or unit_id not in known_units:
                rejections.append(
                    Rejection(
                        element=f"{unit_element}.unit_id",
                        reason="unit is not a ground unit this faction operates",
                        value=unit_id,
                    )
                )
                continue
            quantity = _quantity(
                f"{unit_element}.quantity",
                raw.get("quantity"),
                MAX_QUANTITY_PER_ORDER,
                rejections,
            )
            if quantity is None:
                continue
            units.append((unit_id, quantity))
        if not units:
            rejections.append(
                Rejection(
                    element=f"{element}.units",
                    reason="no legal unit remained, so the transfer was dropped",
                    value=raw_units,
                )
            )
            continue
        key = (origin, destination)
        if key in seen:
            rejections.append(
                Rejection(
                    element=f"{element}.destination_base_id",
                    reason="duplicate identifier",
                    value=destination,
                )
            )
            continue
        seen.add(key)
        orders.append(
            GroundTransferOrder(
                origin_base_id=origin,
                destination_base_id=destination,
                units=tuple(units),
            )
        )
    return tuple(orders)


# ---------------------------------------------------------------------------
# Air tasking validation
# ---------------------------------------------------------------------------

_AIR_TASKING_KEYS = (
    "schema_version",
    "turn_id",
    "campaign_revision",
    "intent",
    "packages",
)


def validate_air_tasking_plan(
    payload: Any, brief: OperationsBrief, capabilities: CapabilityIndex
) -> tuple[Optional[AirTaskingPlan], list[Rejection]]:
    """Structurally validates an air tasking stage payload."""

    fatal = _check_envelope(payload, AIR_TASKING_SCHEMA_VERSION, brief)
    if fatal:
        return None, fatal
    assert isinstance(payload, dict)

    rejections: list[Rejection] = []
    _unexpected_keys(payload, _AIR_TASKING_KEYS, rejections)

    raw_packages = _entries(payload, "packages", rejections)
    if len(raw_packages) > MAX_PACKAGES_PER_TURN:
        rejections.append(
            Rejection(
                element="packages",
                reason=f"more than {MAX_PACKAGES_PER_TURN} packages were requested; "
                "the extra packages were dropped",
                value=len(raw_packages),
            )
        )
        raw_packages = raw_packages[:MAX_PACKAGES_PER_TURN]

    known_airframes = frozenset(capabilities.aircraft_ids)
    packages: list[ProposedPackageOrder] = []
    seen_targets: set[str] = set()
    for index, entry in enumerate(raw_packages):
        element = f"packages[{index}]"
        if not isinstance(entry, dict):
            rejections.append(
                Rejection(
                    element=element, reason="entry must be an object", value=entry
                )
            )
            continue
        target_id = _known_id(
            f"{element}.target_id",
            entry.get("target_id"),
            brief.target_ids,
            "target identifier",
            rejections,
        )
        if target_id is None:
            continue
        target = brief.target(target_id)
        if target is None:  # pragma: no cover - guarded by _known_id
            continue
        if target_id in seen_targets:
            rejections.append(
                Rejection(
                    element=f"{element}.target_id",
                    reason="duplicate identifier",
                    value=target_id,
                )
            )
            continue

        flights = _validate_flights(
            entry,
            element,
            target.legal_missions,
            known_airframes,
            capabilities,
            rejections,
        )
        if not flights:
            continue
        if flights[0].mission_type in ESCORT_MISSION_TYPES:
            rejections.append(
                Rejection(
                    element=f"{element}.flights[0].mission_type",
                    reason="an escort cannot be the primary flight of a package",
                    value=flights[0].mission_type.value,
                )
            )
            continue

        priority = entry.get("priority")
        if not isinstance(priority, int) or isinstance(priority, bool) or priority < 1:
            priority = index + 1
        asap = bool(entry.get("asap", False))

        seen_targets.add(target_id)
        packages.append(
            ProposedPackageOrder(
                target_id=target_id,
                priority=priority,
                flights=tuple(flights),
                asap=asap,
            )
        )

    packages.sort(key=lambda package: (package.priority, package.target_id))
    plan = AirTaskingPlan(
        schema_version=AIR_TASKING_SCHEMA_VERSION,
        turn_id=brief.turn_id,
        campaign_revision=brief.campaign_revision,
        intent=_intent_of(payload),
        packages=tuple(packages),
    )
    return plan, rejections


def _validate_flights(
    entry: dict[str, Any],
    element: str,
    legal_missions: Sequence[str],
    known_airframes: frozenset[str],
    capabilities: CapabilityIndex,
    rejections: list[Rejection],
) -> list[ProposedFlightOrder]:
    raw_flights = entry.get("flights")
    if not isinstance(raw_flights, list) or not raw_flights:
        rejections.append(
            Rejection(
                element=f"{element}.flights",
                reason="a package needs at least one flight",
                value=raw_flights,
            )
        )
        return []
    if len(raw_flights) > MAX_FLIGHTS_PER_PACKAGE:
        rejections.append(
            Rejection(
                element=f"{element}.flights",
                reason=f"more than {MAX_FLIGHTS_PER_PACKAGE} flights were requested; "
                "the extra flights were dropped",
                value=len(raw_flights),
            )
        )
        raw_flights = raw_flights[:MAX_FLIGHTS_PER_PACKAGE]

    allowed = set(legal_missions) | {t.value for t in ESCORT_MISSION_TYPES}
    flights: list[ProposedFlightOrder] = []
    for position, raw in enumerate(raw_flights):
        flight_element = f"{element}.flights[{position}]"
        if not isinstance(raw, dict):
            rejections.append(
                Rejection(
                    element=flight_element,
                    reason="entry must be an object",
                    value=raw,
                )
            )
            continue
        raw_mission = raw.get("mission_type")
        mission = (
            _MISSION_BY_VALUE.get(raw_mission) if isinstance(raw_mission, str) else None
        )
        if mission is None:
            rejections.append(
                Rejection(
                    element=f"{flight_element}.mission_type",
                    reason="mission type is not one the commander may plan",
                    value=raw_mission,
                )
            )
            continue
        if mission.value not in allowed:
            rejections.append(
                Rejection(
                    element=f"{flight_element}.mission_type",
                    reason="this mission type cannot be flown against this objective; "
                    f"legal values were {','.join(sorted(allowed))}",
                    value=mission.value,
                )
            )
            continue
        count = _quantity(
            f"{flight_element}.aircraft_count",
            raw.get("aircraft_count"),
            MAX_AIRCRAFT_PER_FLIGHT,
            rejections,
        )
        if count is None:
            continue
        aircraft_id = raw.get("aircraft_id")
        if aircraft_id is not None:
            if not isinstance(aircraft_id, str) or aircraft_id not in known_airframes:
                rejections.append(
                    Rejection(
                        element=f"{flight_element}.aircraft_id",
                        reason="airframe is not one this faction operates",
                        value=aircraft_id,
                    )
                )
                continue
            capability = capabilities.aircraft_for(aircraft_id)
            if capability is not None and mission.value not in capability.role_names():
                rejections.append(
                    Rejection(
                        element=f"{flight_element}.aircraft_id",
                        reason="this airframe cannot fly the requested mission type",
                        value=aircraft_id,
                    )
                )
                continue
            if capability is not None and count > capability.max_flight_size:
                rejections.append(
                    Rejection(
                        element=f"{flight_element}.aircraft_count",
                        reason="requested flight is larger than this airframe's "
                        f"maximum group size of {capability.max_flight_size}",
                        value=count,
                    )
                )
                continue
        flights.append(
            ProposedFlightOrder(
                mission_type=mission,
                aircraft_count=count,
                aircraft_id=aircraft_id if isinstance(aircraft_id, str) else None,
            )
        )
    return flights


# ---------------------------------------------------------------------------
# Schemas and worked examples for the prompt
# ---------------------------------------------------------------------------


def logistics_json_schema(
    brief: OperationsBrief, capabilities: CapabilityIndex
) -> dict[str, Any]:
    """A JSON schema tight enough to use with structured-output enforcement."""

    base_ids = sorted(brief.base_ids)
    squadron_ids = sorted(brief.squadron_ids)
    purchasable = sorted(capabilities.purchasable_ground_unit_ids)
    all_ground = sorted(capabilities.ground_unit_ids)
    missions = [t.value for t in PLANNABLE_MISSION_TYPES]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "turn_id", "campaign_revision"],
        "properties": {
            "schema_version": {"const": LOGISTICS_SCHEMA_VERSION},
            "turn_id": {"const": brief.turn_id},
            "campaign_revision": {"const": brief.campaign_revision},
            "intent": {"type": "string", "maxLength": MAX_INTENT_CHARACTERS},
            "aircraft_orders": {
                "type": "array",
                "maxItems": MAX_ORDERS_PER_LIST,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["squadron_id", "quantity"],
                    "properties": {
                        "squadron_id": {"enum": squadron_ids},
                        "quantity": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_QUANTITY_PER_ORDER,
                        },
                    },
                },
            },
            "ground_orders": {
                "type": "array",
                "maxItems": MAX_ORDERS_PER_LIST,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["base_id", "unit_id", "quantity"],
                    "properties": {
                        "base_id": {"enum": base_ids},
                        "unit_id": {"enum": purchasable},
                        "quantity": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_QUANTITY_PER_ORDER,
                        },
                    },
                },
            },
            "runway_repairs": {
                "type": "array",
                "maxItems": MAX_ORDERS_PER_LIST,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["base_id"],
                    "properties": {"base_id": {"enum": base_ids}},
                },
            },
            "squadron_relocations": {
                "type": "array",
                "maxItems": MAX_ORDERS_PER_LIST,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["squadron_id"],
                    "properties": {
                        "squadron_id": {"enum": squadron_ids},
                        "base_id": {"enum": base_ids},
                    },
                },
            },
            "squadron_tasking": {
                "type": "array",
                "maxItems": MAX_ORDERS_PER_LIST,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["squadron_id", "mission_types"],
                    "properties": {
                        "squadron_id": {"enum": squadron_ids},
                        "mission_types": {
                            "type": "array",
                            "items": {"enum": missions},
                        },
                    },
                },
            },
            "ground_transfers": {
                "type": "array",
                "maxItems": MAX_ORDERS_PER_LIST,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["origin_base_id", "destination_base_id", "units"],
                    "properties": {
                        "origin_base_id": {"enum": base_ids},
                        "destination_base_id": {"enum": base_ids},
                        "units": {
                            "type": "array",
                            "maxItems": MAX_UNIT_TYPES_PER_TRANSFER,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["unit_id", "quantity"],
                                "properties": {
                                    "unit_id": {"enum": all_ground},
                                    "quantity": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": MAX_QUANTITY_PER_ORDER,
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def air_tasking_json_schema(
    brief: OperationsBrief, capabilities: CapabilityIndex
) -> dict[str, Any]:
    target_ids = sorted(brief.target_ids)
    missions = [t.value for t in PLANNABLE_MISSION_TYPES]
    airframes = sorted(capabilities.aircraft_ids)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "turn_id", "campaign_revision", "packages"],
        "properties": {
            "schema_version": {"const": AIR_TASKING_SCHEMA_VERSION},
            "turn_id": {"const": brief.turn_id},
            "campaign_revision": {"const": brief.campaign_revision},
            "intent": {"type": "string", "maxLength": MAX_INTENT_CHARACTERS},
            "packages": {
                "type": "array",
                "maxItems": MAX_PACKAGES_PER_TURN,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["target_id", "flights"],
                    "properties": {
                        "target_id": {"enum": target_ids},
                        "priority": {"type": "integer", "minimum": 1},
                        "asap": {"type": "boolean"},
                        "flights": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": MAX_FLIGHTS_PER_PACKAGE,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["mission_type", "aircraft_count"],
                                "properties": {
                                    "mission_type": {"enum": missions},
                                    "aircraft_count": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": MAX_AIRCRAFT_PER_FLIGHT,
                                    },
                                    "aircraft_id": {"enum": airframes},
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def example_logistics_json(
    brief: OperationsBrief, capabilities: CapabilityIndex
) -> dict[str, Any]:
    """A worked example built from this campaign's own identifiers."""

    example: dict[str, Any] = {
        "schema_version": LOGISTICS_SCHEMA_VERSION,
        "turn_id": brief.turn_id,
        "campaign_revision": brief.campaign_revision,
        "intent": "reinforce the contested front and get the damaged runway working",
    }
    squadrons = sorted(brief.squadron_ids)
    if squadrons:
        example["aircraft_orders"] = [{"squadron_id": squadrons[0], "quantity": 2}]
    purchasable = sorted(capabilities.purchasable_ground_unit_ids)
    bases = sorted(brief.base_ids)
    if purchasable and bases:
        example["ground_orders"] = [
            {"base_id": bases[0], "unit_id": purchasable[0], "quantity": 4}
        ]
    repairable = [base.id for base in brief.bases if base.runway_repairable]
    if repairable:
        example["runway_repairs"] = [{"base_id": repairable[0]}]
    return example


def example_air_tasking_json(
    brief: OperationsBrief, capabilities: CapabilityIndex
) -> dict[str, Any]:
    example: dict[str, Any] = {
        "schema_version": AIR_TASKING_SCHEMA_VERSION,
        "turn_id": brief.turn_id,
        "campaign_revision": brief.campaign_revision,
        "intent": "roll back the air defences covering the front, then hit the motorpool",
        "packages": [],
    }
    packages: list[dict[str, Any]] = []
    for target in brief.targets[:2]:
        if not target.legal_missions:
            continue
        packages.append(
            {
                "target_id": target.id,
                "priority": len(packages) + 1,
                "flights": [
                    {
                        "mission_type": target.legal_missions[0],
                        "aircraft_count": 2,
                    },
                    {"mission_type": FlightType.ESCORT.value, "aircraft_count": 2},
                ],
            }
        )
    example["packages"] = packages
    return example
