"""Execution of legality-checked ACTIVE plans (:mod:`game.ai_commander.planexecution`).

:class:`PlanExecutor` is the last stage: it takes orders that have already been
proven legal and applies them, but *only* through Retribution's own systems --
the purchase adapters, ``Squadron`` relocation/tasking, ``PendingTransfers`` and
``PackageFulfiller``. These tests prove three things:

* every order class reaches the correct game API with the right arguments, so an
  accepted order genuinely changes campaign state;
* a single failing order is recorded as failed and never aborts the turn or
  half-applies it -- the other orders still go through;
* the execution report is an accurate, serialisable audit trail.

The purchase adapters and the package fulfiller are faked because the real ones
need a fully-built theater, but every fake records exactly what it was asked to
do, so the assertions are about the executor driving the right API, not about
the fake.
"""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from game.ai_commander.capabilities import CAPABILITY_CACHE
from game.ai_commander.enums import IntelPolicy
from game.ai_commander.intel import IntelProjector
from game.ai_commander.operations import OperationsProjector
from game.ai_commander.plan import (
    AircraftOrder,
    GroundTransferOrder,
    GroundUnitOrder,
    LogisticsPlan,
    RunwayRepairOrder,
    SquadronRelocationOrder,
    SquadronTaskingOrder,
)
from game.ai_commander.planlegality import (
    BoundAircraftPurchase,
    BoundFlight,
    BoundPackage,
    ExecutableAirTasking,
    ExecutableLogistics,
    PlanLegalityChecker,
)
from game.ai_commander.planexecution import PlanExecutor
from game.ato.flighttype import FlightType
from tests.ai_commander import fakes

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _objective_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes.patch_objective_finder(monkeypatch)


@pytest.fixture(autouse=True)
def _fresh_capability_cache() -> Iterator[None]:
    CAPABILITY_CACHE.clear()
    yield
    CAPABILITY_CACHE.clear()


class _RecordingAircraftAdapter:
    calls: list[tuple[Any, Any, int]] = []
    fail: bool = False

    def __init__(self, control_point: Any) -> None:
        self.control_point = control_point

    def price_of(self, squadron: Any) -> int:
        return int(squadron.aircraft.price)

    def can_buy(self, squadron: Any) -> bool:
        return True

    def buy(self, squadron: Any, quantity: int) -> None:
        if _RecordingAircraftAdapter.fail:
            raise RuntimeError("purchase refused by the game")
        _RecordingAircraftAdapter.calls.append((self.control_point, squadron, quantity))


class _RecordingGroundAdapter:
    calls: list[tuple[Any, int]] = []

    def __init__(self, control_point: Any, coalition: Any, game: Any) -> None:
        self.control_point = control_point

    def price_of(self, unit_type: Any) -> int:
        return int(unit_type.price)

    def buy(self, unit_type: Any, quantity: int) -> None:
        _RecordingGroundAdapter.calls.append((unit_type, quantity))


class _FakePackage:
    def __init__(self, mission: Any) -> None:
        self.mission = mission


class _FakeFulfiller:
    """Stands in for ``PackageFulfiller``; records missions, returns a package.

    ``refuse`` makes ``plan_mission`` return ``None`` -- the "could not crew or
    route" outcome the executor must record as a failure without raising.
    """

    missions: list[Any] = []
    refuse: bool = False
    raise_on_plan: bool = False

    def __init__(
        self, coalition: Any, theater: Any, flights: Any, settings: Any
    ) -> None:
        self.coalition = coalition

    def plan_mission(self, mission: Any, count: int, now: Any, tracer: Any) -> Any:
        _FakeFulfiller.missions.append(mission)
        if _FakeFulfiller.raise_on_plan:
            raise RuntimeError("planner exploded")
        if _FakeFulfiller.refuse:
            return None
        return _FakePackage(mission)


@pytest.fixture(autouse=True)
def _patch_execution_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingAircraftAdapter.calls.clear()
    _RecordingAircraftAdapter.fail = False
    _RecordingGroundAdapter.calls.clear()
    _FakeFulfiller.missions.clear()
    _FakeFulfiller.refuse = False
    _FakeFulfiller.raise_on_plan = False
    monkeypatch.setattr(
        "game.purchaseadapter.AircraftPurchaseAdapter", _RecordingAircraftAdapter
    )
    monkeypatch.setattr(
        "game.purchaseadapter.GroundUnitPurchaseAdapter", _RecordingGroundAdapter
    )
    monkeypatch.setattr(
        "game.commander.packagefulfiller.PackageFulfiller", _FakeFulfiller
    )


class _Fixture:
    def __init__(self) -> None:
        self.campaign, self.game = fakes.synthetic_game()
        projector = OperationsProjector(self.game, IntelPolicy.REALISTIC)
        self.brief = projector.project("hash", "rev-1")
        self.resolver = projector.resolver
        from game.ai_commander.capabilities import capability_index_for

        self.capabilities = capability_index_for(self.campaign.red)
        self.revision = IntelProjector(
            self.game, IntelPolicy.REALISTIC
        ).campaign_revision()

    def logistics_plan(self, **orders: Any) -> LogisticsPlan:
        return LogisticsPlan(
            schema_version="red-commander-logistics/1",
            turn_id=self.brief.turn_id,
            campaign_revision=self.revision,
            intent="test",
            **orders,
        )

    def check(self, plan: LogisticsPlan) -> ExecutableLogistics:
        checker = PlanLegalityChecker(
            self.game, self.brief, self.resolver, self.capabilities
        )
        result, _ = checker.check_logistics(plan)
        assert result is not None
        return result

    @property
    def executor(self) -> PlanExecutor:
        return PlanExecutor(self.game, self.campaign.red)


# ---------------------------------------------------------------------------
# Each order class reaches the right game API
# ---------------------------------------------------------------------------


class TestOrdersReachTheGame:
    def test_an_aircraft_purchase_calls_the_adapter(self) -> None:
        fx = _Fixture()
        executable = fx.check(
            fx.logistics_plan(
                aircraft_orders=(AircraftOrder(squadron_id="SQN-1", quantity=2),)
            )
        )
        executor = fx.executor
        executor.execute_logistics(executable)
        assert len(_RecordingAircraftAdapter.calls) == 1
        _, squadron, quantity = _RecordingAircraftAdapter.calls[0]
        assert quantity == 2
        assert executor.report.applied_count == 1

    def test_a_ground_purchase_calls_the_adapter(self) -> None:
        fx = _Fixture()
        executable = fx.check(
            fx.logistics_plan(
                ground_orders=(
                    GroundUnitOrder(base_id="BASE-1", unit_id="RED-TANK", quantity=3),
                )
            )
        )
        fx.executor.execute_logistics(executable)
        assert len(_RecordingGroundAdapter.calls) == 1
        assert _RecordingGroundAdapter.calls[0][1] == 3

    def test_a_runway_repair_calls_begin_and_debits_the_budget(self) -> None:
        fx = _Fixture()
        base = fx.resolver.base("BASE-2")
        assert base is not None
        executable = fx.check(
            fx.logistics_plan(runway_repairs=(RunwayRepairOrder(base_id="BASE-2"),))
        )
        fx.executor.execute_logistics(executable)
        base.begin_runway_repair.assert_called_once()
        assert -100.0 in fx.campaign.red.budget_adjustments

    def test_a_relocation_reaches_the_squadron(self) -> None:
        fx = _Fixture()
        squadron = fx.resolver.squadron("SQN-1")
        dest = fx.resolver.base("BASE-2")
        assert squadron is not None
        executable = fx.check(
            fx.logistics_plan(
                squadron_relocations=(
                    SquadronRelocationOrder(squadron_id="SQN-1", base_id="BASE-2"),
                )
            )
        )
        fx.executor.execute_logistics(executable)
        assert squadron.relocations == [dest]

    def test_a_tasking_order_reaches_the_squadron(self) -> None:
        fx = _Fixture()
        squadron = fx.resolver.squadron("SQN-1")
        assert squadron is not None
        executable = fx.check(
            fx.logistics_plan(
                squadron_tasking=(
                    SquadronTaskingOrder(
                        squadron_id="SQN-1", mission_types=(FlightType.SEAD,)
                    ),
                )
            )
        )
        fx.executor.execute_logistics(executable)
        assert squadron.tasking_calls == [frozenset({FlightType.SEAD})]

    def test_a_transfer_reaches_pending_transfers(self) -> None:
        fx = _Fixture()
        executable = fx.check(
            fx.logistics_plan(
                ground_transfers=(
                    GroundTransferOrder(
                        origin_base_id="BASE-1",
                        destination_base_id="BASE-2",
                        units=(("RED-TANK", 2),),
                    ),
                )
            )
        )
        fx.executor.execute_logistics(executable)
        orders = fx.campaign.red.transfers.orders
        assert len(orders) == 1
        assert orders[0].size == 2


# ---------------------------------------------------------------------------
# Air tasking execution
# ---------------------------------------------------------------------------


class TestAirTaskingExecution:
    def _tasking(self, *, refuse: bool = False) -> ExecutableAirTasking:
        return ExecutableAirTasking(
            intent="test",
            packages=[
                BoundPackage(
                    target=object(),
                    target_id="TGT-1",
                    priority=1,
                    flights=(
                        BoundFlight(
                            mission_type=FlightType.SEAD,
                            aircraft_count=2,
                            aircraft_id=None,
                        ),
                    ),
                    asap=False,
                )
            ],
        )

    def test_a_package_is_added_to_the_ato(self) -> None:
        fx = _Fixture()
        executor = fx.executor
        executor.execute_air_tasking(self._tasking())
        assert executor.report.packages_added == 1
        assert len(fx.campaign.red.ato.packages) == 1

    def test_a_package_the_planner_cannot_crew_is_recorded_failed(self) -> None:
        _FakeFulfiller.refuse = True
        fx = _Fixture()
        executor = fx.executor
        executor.execute_air_tasking(self._tasking())
        assert executor.report.packages_added == 0
        assert executor.report.failed_count == 1
        assert not fx.campaign.red.ato.packages

    def test_a_planner_exception_never_aborts_the_turn(self) -> None:
        _FakeFulfiller.raise_on_plan = True
        fx = _Fixture()
        executor = fx.executor
        # Must not raise.
        executor.execute_air_tasking(self._tasking())
        assert executor.report.failed_count == 1


# ---------------------------------------------------------------------------
# A failing order never aborts or half-applies the turn
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    def test_a_failed_purchase_does_not_stop_later_orders(self) -> None:
        fx = _Fixture()
        # Two aircraft purchases; make the adapter raise for both, but a tasking
        # order in the same plan must still be applied.
        _RecordingAircraftAdapter.fail = True
        squadron = fx.resolver.squadron("SQN-2")
        sqn1 = fx.resolver.squadron("SQN-1")
        cp1 = fx.resolver.base("BASE-1")
        assert squadron is not None
        assert sqn1 is not None
        assert cp1 is not None
        executable = ExecutableLogistics(
            intent="mixed",
            aircraft_purchases=[
                BoundAircraftPurchase(
                    squadron=sqn1,
                    control_point=cp1,
                    quantity=1,
                    unit_price=22,
                    squadron_id="SQN-1",
                )
            ],
        )
        from game.ai_commander.planlegality import BoundTasking

        executable.tasking.append(
            BoundTasking(
                squadron=squadron,
                mission_types=(FlightType.DEAD,),
                squadron_id="SQN-2",
            )
        )
        executor = fx.executor
        executor.execute_logistics(executable)
        # The purchase failed, the tasking applied: the turn is neither aborted
        # nor half-applied at a coarser grain than one order.
        assert executor.report.failed_count == 1
        assert executor.report.applied_count == 1
        assert squadron.tasking_calls == [frozenset({FlightType.DEAD})]


# ---------------------------------------------------------------------------
# The execution report is an accurate, serialisable audit trail
# ---------------------------------------------------------------------------


class TestExecutionReport:
    def test_the_report_serialises_with_the_documented_keys(self) -> None:
        fx = _Fixture()
        executable = fx.check(
            fx.logistics_plan(
                aircraft_orders=(AircraftOrder(squadron_id="SQN-1", quantity=1),)
            )
        )
        executor = fx.executor
        executor.execute_logistics(executable)
        data = executor.report.to_dict()
        assert set(data) == {
            "applied",
            "failed",
            "packages_added",
            "budget_before",
            "budget_after",
            "spent",
            "orders",
        }
        assert data["applied"] == 1
        assert isinstance(data["orders"], list)
        assert data["orders"][0]["applied"] is True
