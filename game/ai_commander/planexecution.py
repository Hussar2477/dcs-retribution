"""Execution of legality-checked ACTIVE plans through Retribution's own systems.

Nothing here manipulates campaign state directly. Every order is handed to the
same code path the human player's UI uses:

===========================  ====================================================
Order                        Retribution API used
===========================  ====================================================
aircraft purchase            :class:`game.purchaseadapter.AircraftPurchaseAdapter`
ground unit purchase         :class:`game.purchaseadapter.GroundUnitPurchaseAdapter`
runway repair                ``ControlPoint.begin_runway_repair`` + ``adjust_budget``
squadron relocation          ``Squadron.plan_relocation`` / ``cancel_relocation``
squadron tasking             ``Squadron.set_auto_assignable_mission_types``
ground transfer              ``TransferOrder`` + ``PendingTransfers.new_transfer``
air tasking package          :class:`game.commander.packagefulfiller.PackageFulfiller`
===========================  ====================================================

That matters for fairness as much as for correctness: the commander is bound by
whatever those systems refuse, so it cannot buy what it cannot afford, base
aircraft where they do not fit, or fly a mission its squadrons cannot fly.

Each order is executed inside its own ``try``, and the outcome of every one is
recorded. A single failing order therefore never aborts the turn and never
leaves the turn half-applied at a coarser granularity than one order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from game.ai_commander.planlegality import (
    BoundPackage,
    ExecutableAirTasking,
    ExecutableLogistics,
)
from game.ato.flighttype import FlightType
from game.utils import meters

if TYPE_CHECKING:
    from game.coalition import Coalition
    from game.dcs.aircrafttype import AircraftType
    from game.game import Game


#: Escort mission types and the threat they are pruned against. Matches the mapping
#: :mod:`game.commander.tasks.packageplanningtask` subclasses use.
_ESCORT_TYPES: dict[FlightType, str] = {
    FlightType.ESCORT: "AirToAir",
    FlightType.SEAD_ESCORT: "Sead",
    FlightType.REFUELING: "Refuel",
}

#: A RED base is treated as "forward" for attack helicopters when it lies within
#: this many nautical miles of an active front. Rear attack-heli squadrons are
#: relocated to the nearest forward base that has room; squadrons already this
#: close are left where they are.
_HELI_FORWARD_NM: float = 50.0

#: Mission types an attack helicopter is committed to at the front, in preference
#: order. CAS supports the ground battle directly; BAI is the fallback.
_HELI_FRONT_TASKS: tuple[FlightType, ...] = (FlightType.CAS, FlightType.BAI)

#: Cap on the number of helicopters force-committed per squadron per turn, so a
#: single squadron does not empty itself into one oversized package.
_HELI_FLIGHT_SIZE: int = 2


@dataclass
class OrderOutcome:
    """What actually happened to one order."""

    kind: str
    description: str
    applied: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "order": self.description,
            "applied": self.applied,
            "detail": self.detail,
        }


@dataclass
class ExecutionReport:
    """The audit trail of one ACTIVE turn's execution."""

    outcomes: list[OrderOutcome] = field(default_factory=list)
    budget_before: float = 0.0
    budget_after: float = 0.0
    packages_added: int = 0

    @property
    def applied_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.applied)

    @property
    def failed_count(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.applied)

    @property
    def spent(self) -> float:
        return self.budget_before - self.budget_after

    def record(
        self, kind: str, description: str, applied: bool, detail: str = ""
    ) -> None:
        self.outcomes.append(
            OrderOutcome(
                kind=kind, description=description, applied=applied, detail=detail
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied_count,
            "failed": self.failed_count,
            "packages_added": self.packages_added,
            "budget_before": self.budget_before,
            "budget_after": self.budget_after,
            "spent": self.spent,
            "orders": [outcome.to_dict() for outcome in self.outcomes],
        }

    def render_summary(self) -> str:
        lines = [
            f"applied {self.applied_count} orders, {self.failed_count} failed, "
            f"{self.packages_added} packages added, spent {self.spent:.0f}M"
        ]
        for outcome in self.outcomes:
            mark = "+" if outcome.applied else "!"
            detail = f" -- {outcome.detail}" if outcome.detail else ""
            lines.append(f"  {mark} {outcome.description}{detail}")
        return "\n".join(lines)


class PlanExecutor:
    """Applies validated ACTIVE plans to the campaign."""

    def __init__(self, game: Game, coalition: Coalition) -> None:
        self.game = game
        self.coalition = coalition
        self.report = ExecutionReport()

    # -- helpers ----------------------------------------------------------

    @property
    def now(self) -> datetime:
        return self.game.conditions.start_time

    def _budget(self) -> float:
        try:
            return float(self.coalition.budget)
        except Exception:  # pragma: no cover - defensive
            return 0.0

    # -- logistics --------------------------------------------------------

    def execute_logistics(self, plan: ExecutableLogistics) -> None:
        self.report.budget_before = self._budget()
        self._buy_aircraft(plan)
        self._buy_ground_units(plan)
        self._repair_runways(plan)
        self._replace_ground_groups(plan)
        self._relocate_squadrons(plan)
        self._retask_squadrons(plan)
        self._transfer_ground_units(plan)
        self.report.budget_after = self._budget()

    def _buy_aircraft(self, plan: ExecutableLogistics) -> None:
        from game.purchaseadapter import AircraftPurchaseAdapter

        for order in plan.aircraft_purchases:
            try:
                AircraftPurchaseAdapter(order.control_point).buy(
                    order.squadron, order.quantity
                )
            except Exception as exc:
                logging.warning(
                    "RED commander aircraft purchase failed: %s", order.describe()
                )
                self.report.record("aircraft", order.describe(), False, str(exc))
                continue
            self.report.record("aircraft", order.describe(), True)

    def _buy_ground_units(self, plan: ExecutableLogistics) -> None:
        from game.purchaseadapter import GroundUnitPurchaseAdapter

        for order in plan.ground_purchases:
            adapter = GroundUnitPurchaseAdapter(
                order.control_point, self.coalition, self.game
            )
            try:
                adapter.buy(order.unit_type, order.quantity)
            except Exception as exc:
                logging.warning(
                    "RED commander ground purchase failed: %s", order.describe()
                )
                self.report.record("ground_units", order.describe(), False, str(exc))
                continue
            self.report.record("ground_units", order.describe(), True)

    def _repair_runways(self, plan: ExecutableLogistics) -> None:
        for order in plan.runway_repairs:
            try:
                order.control_point.begin_runway_repair()
                # begin_runway_repair does not charge for itself; both
                # ProcurementAi.repair_runways and the base menu deduct separately.
                self.coalition.adjust_budget(-float(order.cost))
            except Exception as exc:
                logging.warning(
                    "RED commander runway repair failed: %s", order.describe()
                )
                self.report.record("runway_repair", order.describe(), False, str(exc))
                continue
            self.report.record("runway_repair", order.describe(), True)

    def _replace_ground_groups(self, plan: ExecutableLogistics) -> None:
        """Rebuild RED static sites through the human buy path, RED money only."""

        for order in plan.ground_group_replacements:
            try:
                ground_object = order.ground_object
                # Head the rebuilt site toward the front, exactly as the buy menu.
                ground_object.heading = (
                    self.game.theater.heading_to_conflict_from(ground_object.position)
                    or ground_object.heading
                )
                # Clear whatever wreck remains and rebuild from the priced recipe.
                ground_object.groups = []
                for group_name, unit_group, dcs_type, count in order.group_plan:
                    order.force_group.create_theater_group_for_tgo(
                        ground_object,
                        unit_group,
                        f"{ground_object.name} ({group_name})",
                        self.game,
                        dcs_type,
                        count,
                    )
                self.coalition.adjust_budget(-float(order.cost))
            except Exception as exc:
                logging.warning(
                    "RED commander site rebuild failed: %s", order.describe()
                )
                self.report.record(
                    "ground_group_replacement", order.describe(), False, str(exc)
                )
                continue
            self.report.record("ground_group_replacement", order.describe(), True)

    def _relocate_squadrons(self, plan: ExecutableLogistics) -> None:
        for order in plan.relocations:
            try:
                if order.destination is None:
                    order.squadron.cancel_relocation()
                else:
                    order.squadron.plan_relocation(order.destination, self.now)
            except Exception as exc:
                logging.warning("RED commander relocation failed: %s", order.describe())
                self.report.record("relocation", order.describe(), False, str(exc))
                continue
            self.report.record("relocation", order.describe(), True)

    def _retask_squadrons(self, plan: ExecutableLogistics) -> None:
        for order in plan.tasking:
            try:
                order.squadron.set_auto_assignable_mission_types(
                    set(order.mission_types)
                )
            except Exception as exc:
                logging.warning("RED commander tasking failed: %s", order.describe())
                self.report.record("tasking", order.describe(), False, str(exc))
                continue
            self.report.record("tasking", order.describe(), True)

    def _transfer_ground_units(self, plan: ExecutableLogistics) -> None:
        from game.transfers import TransferOrder

        for order in plan.transfers:
            try:
                transfer = TransferOrder(
                    origin=order.origin,
                    destination=order.destination,
                    units={unit: count for unit, count in order.units},
                )
                self.coalition.transfers.new_transfer(transfer, self.now)
            except Exception as exc:
                logging.warning("RED commander transfer failed: %s", order.describe())
                self.report.record("transfer", order.describe(), False, str(exc))
                continue
            self.report.record("transfer", order.describe(), True)

    # -- air tasking ------------------------------------------------------

    def execute_air_tasking(self, plan: ExecutableAirTasking) -> None:
        """Plan the commander's packages before Retribution plans the rest.

        Running first matters: the built-in planner allocates from whatever
        airframes and pilots are still untasked, so the commander's packages get
        first call on the air wing and the automation fills in around them.
        """

        from game.commander.packagefulfiller import PackageFulfiller
        from game.profiling import MultiEventTracer

        fulfiller = PackageFulfiller(
            self.coalition,
            self.game.theater,
            self.game.db.flights,
            self.game.settings,
        )

        with MultiEventTracer() as tracer:
            with tracer.trace("RED commander package planning"):
                for order in sorted(plan.packages, key=lambda p: p.priority):
                    self._plan_package(fulfiller, order, tracer)

    def _plan_package(self, fulfiller: Any, order: BoundPackage, tracer: Any) -> None:
        from game.commander.missionproposals import (
            EscortType,
            ProposedFlight,
            ProposedMission,
        )

        flights = []
        for flight in order.flights:
            escort_name = _ESCORT_TYPES.get(flight.mission_type)
            escort_type = (
                getattr(EscortType, escort_name) if escort_name is not None else None
            )
            flights.append(
                ProposedFlight(
                    task=flight.mission_type,
                    num_aircraft=flight.aircraft_count,
                    escort_type=escort_type,
                    preferred_type=self._aircraft_named(flight.aircraft_id),
                    ingress_band=flight.ingress,
                )
            )

        try:
            package = fulfiller.plan_mission(
                ProposedMission(
                    location=order.target, flights=flights, asap=order.asap
                ),
                1,
                self.now,
                tracer,
            )
        except Exception as exc:
            logging.warning("RED commander package planning failed: %s", exc)
            self.report.record("package", order.describe(), False, str(exc))
            return

        if package is None:
            self.report.record(
                "package",
                order.describe(),
                False,
                "the mission planner could not crew or route this package",
            )
            return

        try:
            self.coalition.ato.add_package(package)
        except Exception as exc:  # pragma: no cover - defensive
            self.report.record("package", order.describe(), False, str(exc))
            return
        self.report.packages_added += 1
        self.report.record("package", order.describe(), True)

    def _aircraft_named(self, aircraft_id: Optional[str]) -> Optional[AircraftType]:
        """Resolve an airframe id against RED's own squadrons only.

        Restricting the lookup to the coalition's air wing rather than the global
        pydcs registry means a hallucinated or BLUE-only airframe resolves to
        ``None`` -- which simply lets Retribution pick the aircraft, exactly as if
        no preference had been expressed.
        """

        if aircraft_id is None:
            return None
        try:
            for squadron in self.coalition.air_wing.iter_squadrons():
                aircraft = getattr(squadron, "aircraft", None)
                if aircraft is not None and str(aircraft.variant_id) == aircraft_id:
                    return aircraft
        except Exception:  # pragma: no cover - defensive
            logging.debug("Airframe resolution failed", exc_info=True)
        return None

    # -- deterministic attack-helicopter commitment -----------------------

    def auto_commit_attack_helicopters(self) -> None:
        """Make RED actually use its attack helicopters, without the LLM.

        The air-tasking stage frequently leaves rotary-wing squadrons idle --
        their CAS suitability scores below fixed-wing airframes, so the planner
        prefers jets and the helicopters sit at their bases. Worse, when the
        stage times out no packages are planned at all. This runs regardless of
        that stage and, for RED only:

        * enables CAS/BAI on every capable attack-helicopter squadron;
        * forward-relocates rear squadrons to a RED base near the front that has
          room for them;
        * force-commits idle, in-range squadrons into CAS/BAI packages through
          the same :class:`PackageFulfiller` path the air-tasking stage uses.

        It is idempotent against the air-tasking stage: force-commitment draws
        only on aircraft still untasked after that stage ran, so squadrons the
        LLM already committed are left alone.
        """

        try:
            if not self.coalition.player.is_red:
                return
        except Exception:  # pragma: no cover - defensive
            return

        fronts = self._active_fronts()
        if not fronts:
            return
        squadrons = self._attack_heli_squadrons()
        if not squadrons:
            return

        for squadron in squadrons:
            self._enable_heli_cas_bai(squadron)
        for squadron in squadrons:
            self._relocate_heli_forward(squadron, fronts)
        self._commit_idle_helis(squadrons, fronts)

    def _active_fronts(self) -> list[Any]:
        """RED's own active front lines. Front geometry between RED's own bases
        and the enemy is not hidden information, so this reads only RED-owned
        control points and never indexes enemy strength."""

        fronts: list[Any] = []
        seen: set[int] = set()
        try:
            for control_point in self.game.theater.control_points_for(
                self.coalition.player
            ):
                for front in getattr(control_point, "front_lines", {}).values():
                    if id(front) not in seen:
                        seen.add(id(front))
                        fronts.append(front)
        except Exception:  # pragma: no cover - defensive
            logging.debug("RED front enumeration failed", exc_info=True)
        return fronts

    def _attack_heli_squadrons(self) -> list[Any]:
        squadrons: list[Any] = []
        try:
            for squadron in self.coalition.air_wing.iter_squadrons():
                aircraft = getattr(squadron, "aircraft", None)
                if aircraft is None or not getattr(aircraft, "helicopter", False):
                    continue
                if any(squadron.capable_of(task) for task in _HELI_FRONT_TASKS):
                    squadrons.append(squadron)
        except Exception:  # pragma: no cover - defensive
            logging.debug("Attack-heli enumeration failed", exc_info=True)
        return squadrons

    def _enable_heli_cas_bai(self, squadron: Any) -> None:
        try:
            desired = set(squadron.auto_assignable_mission_types)
            missing = {
                task
                for task in _HELI_FRONT_TASKS
                if squadron.capable_of(task) and task not in desired
            }
            if not missing:
                return
            squadron.set_auto_assignable_mission_types(desired | missing)
        except Exception as exc:  # pragma: no cover - defensive
            self.report.record("heli_tasking", str(squadron), False, str(exc))
            return
        self.report.record(
            "heli_tasking",
            f"enable CAS/BAI on {squadron}",
            True,
            "attack helicopters made auto-assignable at the front",
        )

    def _nm_to_nearest_front(self, position: Any, fronts: list[Any]) -> Optional[float]:
        best: Optional[float] = None
        for front in fronts:
            front_position = getattr(front, "position", None)
            if front_position is None:
                continue
            try:
                nm = meters(position.distance_to_point(front_position)).nautical_miles
            except Exception:  # pragma: no cover - defensive
                continue
            if best is None or nm < best:
                best = nm
        return best

    def _relocate_heli_forward(self, squadron: Any, fronts: list[Any]) -> None:
        if getattr(squadron, "destination", None) is not None:
            return
        if getattr(squadron, "owned_aircraft", 0) <= 0:
            return
        location = getattr(squadron, "location", None)
        if location is None:
            return
        current_nm = self._nm_to_nearest_front(location.position, fronts)
        if current_nm is None or current_nm <= _HELI_FORWARD_NM:
            return  # already at the front, or no usable front position

        from game.theater import ParkingType

        try:
            parking_type = ParkingType().from_squadron(squadron)
            need = squadron.expected_size_next_turn
        except Exception:  # pragma: no cover - defensive
            return

        best_cp: Any = None
        best_nm = current_nm
        try:
            candidates = self.game.theater.control_points_for(self.coalition.player)
        except Exception:  # pragma: no cover - defensive
            return
        for control_point in candidates:
            if control_point is location:
                continue
            cp_nm = self._nm_to_nearest_front(control_point.position, fronts)
            if cp_nm is None or cp_nm > _HELI_FORWARD_NM or cp_nm >= best_nm:
                continue
            try:
                if control_point.unclaimed_parking(parking_type) < need:
                    continue
                if not control_point.can_operate(squadron.aircraft):
                    continue
            except Exception:  # pragma: no cover - defensive
                continue
            best_cp = control_point
            best_nm = cp_nm

        if best_cp is None:
            return
        try:
            squadron.plan_relocation(best_cp, self.now)
        except Exception as exc:
            self.report.record(
                "heli_relocation", f"{squadron} -> {best_cp}", False, str(exc)
            )
            return
        self.report.record(
            "heli_relocation",
            f"relocate {squadron} to {best_cp}",
            True,
            f"moved attack helicopters within {best_nm:.0f}nm of the front",
        )

    def _commit_idle_helis(self, squadrons: list[Any], fronts: list[Any]) -> None:
        from game.commander.packagefulfiller import PackageFulfiller
        from game.profiling import MultiEventTracer

        fulfiller = PackageFulfiller(
            self.coalition,
            self.game.theater,
            self.game.db.flights,
            self.game.settings,
        )
        with MultiEventTracer() as tracer:
            with tracer.trace("RED attack-helicopter auto-commit"):
                for squadron in squadrons:
                    self._commit_one_heli_squadron(fulfiller, squadron, fronts, tracer)

    def _commit_one_heli_squadron(
        self, fulfiller: Any, squadron: Any, fronts: list[Any], tracer: Any
    ) -> None:
        from game.commander.missionproposals import ProposedFlight, ProposedMission

        if getattr(squadron, "destination", None) is not None:
            return  # relocating; not available at the front this turn
        untasked = getattr(squadron, "untasked_aircraft", 0)
        if untasked <= 0:
            return  # already committed elsewhere -- keeps this idempotent

        size = min(untasked, _HELI_FLIGHT_SIZE)
        task: Optional[FlightType] = None
        target: Any = None
        for candidate in _HELI_FRONT_TASKS:
            if not squadron.capable_of(candidate):
                continue
            for front in sorted(
                fronts,
                key=lambda f: self._nm_to_nearest_front(squadron.location.position, [f])
                or float("inf"),
            ):
                try:
                    if squadron.can_auto_assign_mission(
                        front, candidate, size, True, True
                    ):
                        task = candidate
                        target = front
                        break
                except Exception:  # pragma: no cover - defensive
                    continue
            if task is not None:
                break
        if task is None or target is None:
            return

        flight = ProposedFlight(
            task=task,
            num_aircraft=size,
            escort_type=None,
            preferred_type=squadron.aircraft,
            ingress_band=None,
        )
        describe = f"{task.value} from {squadron} against the front"
        try:
            package = fulfiller.plan_mission(
                ProposedMission(location=target, flights=[flight], asap=True),
                1,
                self.now,
                tracer,
            )
        except Exception as exc:
            self.report.record("heli_package", describe, False, str(exc))
            return
        if package is None:
            self.report.record(
                "heli_package",
                describe,
                False,
                "the mission planner could not crew or route this package",
            )
            return
        try:
            self.coalition.ato.add_package(package)
        except Exception as exc:  # pragma: no cover - defensive
            self.report.record("heli_package", describe, False, str(exc))
            return
        self.report.packages_added += 1
        self.report.record("heli_package", describe, True)
