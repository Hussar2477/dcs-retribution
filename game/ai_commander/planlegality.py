"""Live-state legality checking for ACTIVE mode's logistics and air tasking plans.

:mod:`game.ai_commander.plan` proves a plan is *internally* consistent with the
briefing it was produced for. This module proves each individual order is
*actually* possible right now, and binds it to the live game objects that will
carry it out.

The design rule throughout is the same one phase 1 used for postures and
procurement categories: ask the game's own predicate, never re-implement it.

* Aircraft purchases go through :class:`~game.purchaseadapter.AircraftPurchaseAdapter`
  -- so parking, squadron size limits and price all come from the code the human
  purchase screen uses.
* Ground purchases go through :class:`~game.purchaseadapter.GroundUnitPurchaseAdapter`,
  including its ``has_ground_unit_source`` requirement.
* Runway repair uses ``ControlPoint.runway_can_be_repaired`` and
  :data:`game.config.RUNWAY_REPAIR_COST`.
* Squadron relocation re-checks the exact two conditions
  :meth:`~game.squadrons.squadron.Squadron.plan_relocation` raises on, so a legal
  order can never turn into an exception during execution.
* Squadron tasking is filtered by ``Squadron.capable_of``.
* Ground transfers require the units to actually be present at the origin base.
* Packages require the target to still be in the briefing and the air wing to be
  able to plan the mission at all.

Money is checked **cumulatively**: a plan whose orders are individually
affordable but collectively are not gets truncated at the point the budget runs
out, with an explicit rejection, rather than overdrawing the coalition. Parking
and unit availability accumulate the same way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from game.ai_commander.capabilities import CapabilityIndex
from game.ai_commander.decision import Rejection
from game.ai_commander.operations import OperationsBrief, OperationsResolver
from game.ai_commander.plan import (
    AirTaskingPlan,
    LogisticsPlan,
    ProposedFlightOrder,
    ProposedPackageOrder,
)
from game.ato.flighttype import FlightType

if TYPE_CHECKING:
    from game.dcs.groundunittype import GroundUnitType
    from game.game import Game
    from game.squadrons.squadron import Squadron
    from game.theater import ControlPoint


STATE_CHANGED = "campaign state changed between briefing and application"


# ---------------------------------------------------------------------------
# Bound, executable orders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundAircraftPurchase:
    squadron: Squadron
    control_point: ControlPoint
    quantity: int
    unit_price: int
    squadron_id: str

    @property
    def total_cost(self) -> int:
        return self.unit_price * self.quantity

    def describe(self) -> str:
        return (
            f"buy {self.quantity}x {self.squadron.aircraft.variant_id} for "
            f"{self.squadron_id} ({self.total_cost}M)"
        )


@dataclass(frozen=True)
class BoundGroundPurchase:
    control_point: ControlPoint
    unit_type: GroundUnitType
    quantity: int
    unit_price: int
    base_id: str

    @property
    def total_cost(self) -> int:
        return self.unit_price * self.quantity

    def describe(self) -> str:
        return (
            f"buy {self.quantity}x {self.unit_type.variant_id} at {self.base_id} "
            f"({self.total_cost}M)"
        )


@dataclass(frozen=True)
class BoundRunwayRepair:
    control_point: ControlPoint
    cost: int
    base_id: str

    def describe(self) -> str:
        return f"repair runway at {self.base_id} ({self.cost}M)"


@dataclass(frozen=True)
class BoundRelocation:
    squadron: Squadron
    destination: Optional[ControlPoint]
    squadron_id: str
    base_id: Optional[str]

    def describe(self) -> str:
        if self.destination is None:
            return f"cancel relocation of {self.squadron_id}"
        return f"relocate {self.squadron_id} to {self.base_id}"


@dataclass(frozen=True)
class BoundTasking:
    squadron: Squadron
    mission_types: tuple[FlightType, ...]
    squadron_id: str

    def describe(self) -> str:
        tasks = ",".join(t.value for t in self.mission_types) or "none"
        return f"limit {self.squadron_id} to {tasks}"


@dataclass(frozen=True)
class BoundTransfer:
    origin: ControlPoint
    destination: ControlPoint
    units: tuple[tuple[GroundUnitType, int], ...]
    origin_base_id: str
    destination_base_id: str

    @property
    def size(self) -> int:
        return sum(count for _, count in self.units)

    def describe(self) -> str:
        units = ",".join(f"{count}x{unit.variant_id}" for unit, count in self.units)
        return (
            f"transfer {units} from {self.origin_base_id} to "
            f"{self.destination_base_id}"
        )


@dataclass(frozen=True)
class BoundFlight:
    mission_type: FlightType
    aircraft_count: int
    aircraft_id: Optional[str]


@dataclass(frozen=True)
class BoundPackage:
    target: Any
    target_id: str
    priority: int
    flights: tuple[BoundFlight, ...]
    asap: bool

    def describe(self) -> str:
        flights = ", ".join(
            f"{f.aircraft_count}x {f.mission_type.value}" for f in self.flights
        )
        return f"#{self.priority} {self.target_id}: {flights}"


@dataclass
class ExecutableLogistics:
    """Everything from a logistics plan that is legal against live state."""

    intent: str = ""
    aircraft_purchases: list[BoundAircraftPurchase] = field(default_factory=list)
    ground_purchases: list[BoundGroundPurchase] = field(default_factory=list)
    runway_repairs: list[BoundRunwayRepair] = field(default_factory=list)
    relocations: list[BoundRelocation] = field(default_factory=list)
    tasking: list[BoundTasking] = field(default_factory=list)
    transfers: list[BoundTransfer] = field(default_factory=list)

    @property
    def order_count(self) -> int:
        return (
            len(self.aircraft_purchases)
            + len(self.ground_purchases)
            + len(self.runway_repairs)
            + len(self.relocations)
            + len(self.tasking)
            + len(self.transfers)
        )

    @property
    def has_content(self) -> bool:
        return self.order_count > 0

    @property
    def committed_budget(self) -> int:
        return (
            sum(o.total_cost for o in self.aircraft_purchases)
            + sum(o.total_cost for o in self.ground_purchases)
            + sum(o.cost for o in self.runway_repairs)
        )

    def describe(self) -> list[str]:
        return [
            order.describe()
            for group in (
                self.aircraft_purchases,
                self.ground_purchases,
                self.runway_repairs,
                self.relocations,
                self.tasking,
                self.transfers,
            )
            for order in group
        ]


@dataclass
class ExecutableAirTasking:
    """Everything from an air tasking plan that is legal against live state."""

    intent: str = ""
    packages: list[BoundPackage] = field(default_factory=list)

    @property
    def order_count(self) -> int:
        return len(self.packages)

    @property
    def has_content(self) -> bool:
        return bool(self.packages)

    def describe(self) -> list[str]:
        return [package.describe() for package in self.packages]


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class PlanLegalityChecker:
    """Checks a structurally valid plan against live RED state."""

    def __init__(
        self,
        game: Game,
        brief: OperationsBrief,
        resolver: OperationsResolver,
        capabilities: CapabilityIndex,
    ) -> None:
        from game.theater.player import Player

        self.game = game
        self.brief = brief
        self.resolver = resolver
        self.capabilities = capabilities
        self.player = Player.RED
        self.coalition = game.coalition_for(self.player)

    # -- shared helpers ---------------------------------------------------

    @property
    def budget(self) -> float:
        try:
            return float(self.coalition.budget)
        except Exception:  # pragma: no cover - defensive
            return 0.0

    def _live_revision(self) -> Optional[str]:
        from game.ai_commander.intel import IntelProjector

        try:
            return IntelProjector(
                self.game, self.brief.intel_policy
            ).campaign_revision()
        except Exception:  # pragma: no cover - defensive
            logging.debug("Could not recompute campaign revision", exc_info=True)
            return None

    def _revision_rejection(self, revision: str) -> Optional[Rejection]:
        live = self._live_revision()
        if live is not None and live != revision:
            return Rejection("campaign_revision", STATE_CHANGED, revision)
        return None

    def _ground_unit_named(self, unit_id: str) -> Optional[GroundUnitType]:
        """Resolve a capability-index ground unit id to the faction's own object.

        Deliberately scoped to the RED faction's own lists rather than the global
        pydcs registry, so an id the faction does not field cannot be conjured into
        existence even if it names a real DCS unit.
        """

        faction = self.coalition.faction
        for group in (
            faction.ground_units,
            faction.frontline_units,
            faction.artillery_units,
            faction.infantry_units,
            faction.logistics_units,
            faction.air_defense_units,
            faction.missiles,
        ):
            for unit in group:
                if str(getattr(unit, "variant_id", "")) == unit_id:
                    return unit
        return None

    @staticmethod
    def _units_at(control_point: ControlPoint) -> dict[Any, int]:
        try:
            return dict(control_point.base.armor)
        except Exception:  # pragma: no cover - defensive
            return {}

    def _parking_free(self, control_point: ControlPoint, squadron: Squadron) -> int:
        from game.theater import ParkingType

        try:
            return int(
                control_point.unclaimed_parking(ParkingType().from_squadron(squadron))
            )
        except Exception:  # pragma: no cover - defensive
            logging.debug("Parking query failed", exc_info=True)
            return 0

    # -- logistics --------------------------------------------------------

    def check_logistics(
        self, plan: LogisticsPlan
    ) -> tuple[Optional[ExecutableLogistics], list[Rejection]]:
        rejections: list[Rejection] = []

        stale = self._revision_rejection(plan.campaign_revision)
        if stale is not None:
            return None, [stale]

        result = ExecutableLogistics(intent=plan.intent)
        # One running budget for the whole stage. Aircraft first, then ground units,
        # then runway repair -- the order the schema lists them in, which is also the
        # order the prompt tells the model its money will be committed in.
        remaining = self.budget

        remaining = self._check_aircraft_orders(plan, result, rejections, remaining)
        remaining = self._check_ground_orders(plan, result, rejections, remaining)
        remaining = self._check_runway_repairs(plan, result, rejections, remaining)
        self._check_relocations(plan, result, rejections)
        self._check_tasking(plan, result, rejections)
        self._check_transfers(plan, result, rejections)

        if not result.has_content:
            return None, rejections
        return result, rejections

    def _check_aircraft_orders(
        self,
        plan: LogisticsPlan,
        result: ExecutableLogistics,
        rejections: list[Rejection],
        remaining: float,
    ) -> float:
        from game.purchaseadapter import AircraftPurchaseAdapter

        # Parking is a shared resource, so track what earlier orders in this same
        # plan already claimed at each base.
        claimed: dict[int, int] = {}

        for index, order in enumerate(plan.aircraft_orders):
            element = f"aircraft_orders[{index}]"
            squadron = self.resolver.squadron(order.squadron_id)
            if squadron is None:
                rejections.append(
                    Rejection(element, STATE_CHANGED, order.squadron_id),
                )
                continue

            control_point = getattr(squadron, "location", None)
            if control_point is None or control_point.captured is not self.player:
                rejections.append(
                    Rejection(
                        element,
                        "squadron is no longer based at a RED control point",
                        order.squadron_id,
                    )
                )
                continue

            adapter = AircraftPurchaseAdapter(control_point)
            try:
                unit_price = int(adapter.price_of(squadron))
            except Exception:  # pragma: no cover - defensive
                rejections.append(
                    Rejection(
                        element, "squadron has no priceable airframe", order.squadron_id
                    )
                )
                continue

            key = id(control_point)
            already = claimed.get(key, 0)
            parking = self._parking_free(control_point, squadron) - already
            if parking <= 0:
                rejections.append(
                    Rejection(
                        element,
                        f"{control_point.name} has no free parking for more airframes",
                        order.squadron_id,
                    )
                )
                continue

            capacity = order.quantity
            checker = getattr(squadron, "has_aircraft_capacity_for", None)
            if callable(checker):
                while capacity > 0:
                    try:
                        if bool(checker(capacity)):
                            break
                    except Exception:  # pragma: no cover - defensive
                        break
                    capacity -= 1
            if capacity <= 0:
                rejections.append(
                    Rejection(
                        element,
                        "squadron is already at its aircraft limit",
                        order.squadron_id,
                    )
                )
                continue

            affordable = int(remaining // unit_price) if unit_price > 0 else capacity
            quantity = min(order.quantity, parking, capacity, affordable)
            if quantity <= 0:
                rejections.append(
                    Rejection(
                        element,
                        f"one {squadron.aircraft.variant_id} costs {unit_price} but only "
                        f"{remaining:.0f} of the budget is uncommitted",
                        order.squadron_id,
                    )
                )
                continue
            if quantity < order.quantity:
                rejections.append(
                    Rejection(
                        element,
                        f"reduced from {order.quantity} to {quantity} by budget, "
                        f"parking or squadron capacity",
                        order.squadron_id,
                    )
                )

            if not adapter.can_buy(squadron):
                rejections.append(
                    Rejection(
                        element,
                        "the game's own purchase check refused this squadron",
                        order.squadron_id,
                    )
                )
                continue

            claimed[key] = already + quantity
            remaining -= unit_price * quantity
            result.aircraft_purchases.append(
                BoundAircraftPurchase(
                    squadron=squadron,
                    control_point=control_point,
                    quantity=quantity,
                    unit_price=unit_price,
                    squadron_id=order.squadron_id,
                )
            )
        return remaining

    def _check_ground_orders(
        self,
        plan: LogisticsPlan,
        result: ExecutableLogistics,
        rejections: list[Rejection],
        remaining: float,
    ) -> float:
        from game.purchaseadapter import GroundUnitPurchaseAdapter

        for index, order in enumerate(plan.ground_orders):
            element = f"ground_orders[{index}]"
            control_point = self.resolver.base(order.base_id)
            if control_point is None or control_point.captured is not self.player:
                rejections.append(Rejection(element, STATE_CHANGED, order.base_id))
                continue

            unit_type = self._ground_unit_named(order.unit_id)
            if unit_type is None:
                rejections.append(
                    Rejection(
                        element,
                        "RED does not field this ground unit type",
                        order.unit_id,
                    )
                )
                continue

            adapter = GroundUnitPurchaseAdapter(
                control_point, self.coalition, self.game
            )
            if not control_point.has_ground_unit_source(self.game):
                rejections.append(
                    Rejection(
                        element,
                        f"{control_point.name} has no ground unit source, so ground "
                        f"units cannot be bought there",
                        order.base_id,
                    )
                )
                continue

            unit_price = int(adapter.price_of(unit_type))
            affordable = (
                int(remaining // unit_price) if unit_price > 0 else order.quantity
            )
            quantity = min(order.quantity, affordable)
            if quantity <= 0:
                rejections.append(
                    Rejection(
                        element,
                        f"one {order.unit_id} costs {unit_price} but only "
                        f"{remaining:.0f} of the budget is uncommitted",
                        order.unit_id,
                    )
                )
                continue
            if quantity < order.quantity:
                rejections.append(
                    Rejection(
                        element,
                        f"reduced from {order.quantity} to {quantity} by the "
                        f"remaining budget",
                        order.unit_id,
                    )
                )

            remaining -= unit_price * quantity
            result.ground_purchases.append(
                BoundGroundPurchase(
                    control_point=control_point,
                    unit_type=unit_type,
                    quantity=quantity,
                    unit_price=unit_price,
                    base_id=order.base_id,
                )
            )
        return remaining

    def _check_runway_repairs(
        self,
        plan: LogisticsPlan,
        result: ExecutableLogistics,
        rejections: list[Rejection],
        remaining: float,
    ) -> float:
        from game.config import RUNWAY_REPAIR_COST

        cost = int(RUNWAY_REPAIR_COST)
        for index, order in enumerate(plan.runway_repairs):
            element = f"runway_repairs[{index}]"
            control_point = self.resolver.base(order.base_id)
            if control_point is None or control_point.captured is not self.player:
                rejections.append(Rejection(element, STATE_CHANGED, order.base_id))
                continue
            if not getattr(control_point, "runway_can_be_repaired", False):
                rejections.append(
                    Rejection(
                        element,
                        f"{control_point.name} does not have a repairable runway",
                        order.base_id,
                    )
                )
                continue
            if remaining < cost:
                rejections.append(
                    Rejection(
                        element,
                        f"runway repair costs {cost} but only {remaining:.0f} of the "
                        f"budget is uncommitted",
                        order.base_id,
                    )
                )
                continue
            remaining -= cost
            result.runway_repairs.append(
                BoundRunwayRepair(
                    control_point=control_point, cost=cost, base_id=order.base_id
                )
            )
        return remaining

    def _check_relocations(
        self,
        plan: LogisticsPlan,
        result: ExecutableLogistics,
        rejections: list[Rejection],
    ) -> None:
        for index, order in enumerate(plan.squadron_relocations):
            element = f"squadron_relocations[{index}]"
            squadron = self.resolver.squadron(order.squadron_id)
            if squadron is None:
                rejections.append(Rejection(element, STATE_CHANGED, order.squadron_id))
                continue

            if order.base_id is None:
                if getattr(squadron, "destination", None) is None:
                    rejections.append(
                        Rejection(
                            element,
                            "squadron has no pending relocation to cancel",
                            order.squadron_id,
                        )
                    )
                    continue
                result.relocations.append(
                    BoundRelocation(
                        squadron=squadron,
                        destination=None,
                        squadron_id=order.squadron_id,
                        base_id=None,
                    )
                )
                continue

            destination = self.resolver.base(order.base_id)
            if destination is None or destination.captured is not self.player:
                rejections.append(Rejection(element, STATE_CHANGED, order.base_id))
                continue
            reason = self._relocation_rejection(squadron, destination)
            if reason is not None:
                rejections.append(Rejection(element, reason, order.base_id))
                continue
            result.relocations.append(
                BoundRelocation(
                    squadron=squadron,
                    destination=destination,
                    squadron_id=order.squadron_id,
                    base_id=order.base_id,
                )
            )

    def _relocation_rejection(
        self, squadron: Squadron, destination: ControlPoint
    ) -> Optional[str]:
        """The exact conditions ``Squadron.plan_relocation`` raises or warns on.

        Checking them here means execution can never turn a "legal" order into an
        exception, and the model gets a specific reason instead of a generic failure.
        """

        if destination is getattr(squadron, "location", None):
            return "squadron is already based there"
        if destination is getattr(squadron, "destination", None):
            return "squadron is already relocating there"
        can_operate = getattr(destination, "can_operate", None)
        if callable(can_operate):
            try:
                if not can_operate(squadron.aircraft):
                    return (
                        f"{destination.name} cannot operate "
                        f"{squadron.aircraft.variant_id}"
                    )
            except Exception:  # pragma: no cover - defensive
                logging.debug("can_operate check failed", exc_info=True)
        expected = getattr(squadron, "expected_size_next_turn", None)
        if isinstance(expected, int):
            parking = self._parking_free(destination, squadron)
            if expected > parking:
                return (
                    f"{destination.name} has {parking} free parking spaces but the "
                    f"squadron needs {expected}"
                )
        return None

    def _check_tasking(
        self,
        plan: LogisticsPlan,
        result: ExecutableLogistics,
        rejections: list[Rejection],
    ) -> None:
        for index, order in enumerate(plan.squadron_tasking):
            element = f"squadron_tasking[{index}]"
            squadron = self.resolver.squadron(order.squadron_id)
            if squadron is None:
                rejections.append(Rejection(element, STATE_CHANGED, order.squadron_id))
                continue
            capable: list[FlightType] = []
            for task in order.mission_types:
                checker = getattr(squadron, "capable_of", None)
                allowed = False
                if callable(checker):
                    try:
                        allowed = bool(checker(task))
                    except Exception:  # pragma: no cover - defensive
                        allowed = False
                if allowed:
                    capable.append(task)
                else:
                    rejections.append(
                        Rejection(
                            element,
                            f"squadron cannot fly {task.value}",
                            order.squadron_id,
                        )
                    )
            if not capable:
                rejections.append(
                    Rejection(
                        element,
                        "none of the requested mission types are flyable by this "
                        "squadron, so its tasking is left unchanged",
                        order.squadron_id,
                    )
                )
                continue
            result.tasking.append(
                BoundTasking(
                    squadron=squadron,
                    mission_types=tuple(capable),
                    squadron_id=order.squadron_id,
                )
            )

    def _check_transfers(
        self,
        plan: LogisticsPlan,
        result: ExecutableLogistics,
        rejections: list[Rejection],
    ) -> None:
        # Units already promised to an earlier transfer in this same plan are gone.
        spent: dict[tuple[int, str], int] = {}

        for index, order in enumerate(plan.ground_transfers):
            element = f"ground_transfers[{index}]"
            origin = self.resolver.base(order.origin_base_id)
            destination = self.resolver.base(order.destination_base_id)
            if origin is None or origin.captured is not self.player:
                rejections.append(
                    Rejection(element, STATE_CHANGED, order.origin_base_id)
                )
                continue
            if destination is None or destination.captured is not self.player:
                rejections.append(
                    Rejection(element, STATE_CHANGED, order.destination_base_id)
                )
                continue

            route = self._transit_rejection(origin, destination)
            if route is not None:
                rejections.append(Rejection(element, route, order.destination_base_id))
                continue

            present = self._units_at(origin)
            by_id = {
                str(getattr(t, "variant_id", "")): (t, c) for t, c in present.items()
            }
            bound: list[tuple[GroundUnitType, int]] = []
            for unit_id, requested in order.units:
                entry = by_id.get(unit_id)
                if entry is None:
                    rejections.append(
                        Rejection(
                            element,
                            f"{origin.name} has no {unit_id} to move",
                            unit_id,
                        )
                    )
                    continue
                unit_type, available = entry
                key = (id(origin), unit_id)
                available -= spent.get(key, 0)
                quantity = min(requested, available)
                if quantity <= 0:
                    rejections.append(
                        Rejection(
                            element,
                            f"{origin.name} has no uncommitted {unit_id} left to move",
                            unit_id,
                        )
                    )
                    continue
                if quantity < requested:
                    rejections.append(
                        Rejection(
                            element,
                            f"reduced from {requested} to {quantity} by what is "
                            f"actually present at {origin.name}",
                            unit_id,
                        )
                    )
                spent[key] = spent.get(key, 0) + quantity
                bound.append((unit_type, quantity))

            if not bound:
                continue

            result.transfers.append(
                BoundTransfer(
                    origin=origin,
                    destination=destination,
                    units=tuple(bound),
                    origin_base_id=order.origin_base_id,
                    destination_base_id=order.destination_base_id,
                )
            )

    def _transit_rejection(
        self, origin: ControlPoint, destination: ControlPoint
    ) -> Optional[str]:
        """``None`` if the transfer network can actually move units between these.

        :meth:`game.transfers.PendingTransfers.arrange_transport` walks the transit
        network and raises if there is no path, so checking here keeps an
        unreachable destination from turning into an exception mid-execution.
        """

        if origin is destination:
            return "origin and destination are the same base"
        try:
            network = self.coalition.transfers.network_for(origin)
            network.shortest_path_between(origin, destination)
        except Exception:
            return (
                f"the transit network has no route from {origin.name} to "
                f"{destination.name}"
            )
        return None

    # -- air tasking ------------------------------------------------------

    def check_air_tasking(
        self, plan: AirTaskingPlan
    ) -> tuple[Optional[ExecutableAirTasking], list[Rejection]]:
        rejections: list[Rejection] = []

        stale = self._revision_rejection(plan.campaign_revision)
        if stale is not None:
            return None, [stale]

        result = ExecutableAirTasking(intent=plan.intent)
        for index, package in enumerate(plan.packages):
            element = f"packages[{index}]"
            bound = self._check_package(element, package, rejections)
            if bound is not None:
                result.packages.append(bound)

        if not result.has_content:
            return None, rejections
        return result, rejections

    def _check_package(
        self,
        element: str,
        package: ProposedPackageOrder,
        rejections: list[Rejection],
    ) -> Optional[BoundPackage]:
        target = self.resolver.target(package.target_id)
        if target is None:
            rejections.append(Rejection(element, STATE_CHANGED, package.target_id))
            return None

        flights: list[BoundFlight] = []
        for flight_index, flight in enumerate(package.flights):
            flight_element = f"{element}.flights[{flight_index}]"
            bound = self._check_flight(flight_element, flight, rejections)
            if bound is not None:
                flights.append(bound)

        if not flights:
            rejections.append(
                Rejection(
                    element,
                    "no flight in this package could be crewed, so the package is "
                    "not planned",
                    package.target_id,
                )
            )
            return None

        return BoundPackage(
            target=target,
            target_id=package.target_id,
            priority=package.priority,
            flights=tuple(flights),
            asap=package.asap,
        )

    def _check_flight(
        self,
        element: str,
        flight: ProposedFlightOrder,
        rejections: list[Rejection],
    ) -> Optional[BoundFlight]:
        air_wing = self.coalition.air_wing
        can_plan = getattr(air_wing, "can_auto_plan", None)
        if callable(can_plan):
            try:
                planable = bool(can_plan(flight.mission_type))
            except Exception:  # pragma: no cover - defensive
                planable = True
        else:  # pragma: no cover - defensive
            planable = True
        if not planable:
            rejections.append(
                Rejection(
                    element,
                    f"RED has no squadron available to fly {flight.mission_type.value} "
                    f"this turn",
                    flight.mission_type.value,
                )
            )
            return None

        if flight.aircraft_id is not None:
            entry = self.capabilities.aircraft_for(flight.aircraft_id)
            if entry is None:
                rejections.append(
                    Rejection(
                        element,
                        "RED does not have this airframe",
                        flight.aircraft_id,
                    )
                )
                return None
            if not entry.is_fielded:
                rejections.append(
                    Rejection(
                        element,
                        "RED owns no airframes of this type yet, so it cannot be "
                        "tasked this turn",
                        flight.aircraft_id,
                    )
                )
                return None

        return BoundFlight(
            mission_type=flight.mission_type,
            aircraft_count=flight.aircraft_count,
            aircraft_id=flight.aircraft_id,
        )
