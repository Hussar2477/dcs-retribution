"""Live-state legality checking for ACTIVE mode (:mod:`game.ai_commander.planlegality`).

Where :mod:`tests.ai_commander.test_active_plan` proves a plan is *internally*
consistent with the briefing it was produced for, this module proves each order
is *actually possible against live campaign state* -- and, crucially, that every
player-equivalent control the model can drive rejects an illegal order with a
specific, machine-readable reason instead of overdrawing the budget, exceeding
parking, tasking an airframe the faction has never fielded, or moving units that
are not there.

The design rule under test is "ask the game's own predicate, never
re-implement it": aircraft/ground purchases go through the real purchase
adapters, parking through ``ControlPoint.unclaimed_parking``, relocation through
the exact conditions ``Squadron.plan_relocation`` raises on, and so on. The
adapters themselves are the one seam we fake, because the real ones need a
fully-built theater; the fakes price and permit exactly as the synthetic faction
was built to expect, so an assertion here is about the checker's logic, not the
adapter's.

Every rejection carries the RED-facing reason string verbatim from the checker,
so these tests also pin the operator-visible audit text.
"""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from game.ai_commander.capabilities import (
    CAPABILITY_CACHE,
    CapabilityIndex,
    capability_index_for,
)
from game.ai_commander.enums import IntelPolicy
from game.ai_commander.intel import IntelProjector
from game.ai_commander.operations import (
    OperationsBrief,
    OperationsProjector,
    OperationsResolver,
)
from game.ai_commander.plan import (
    AircraftOrder,
    AirTaskingPlan,
    GroundTransferOrder,
    GroundUnitOrder,
    LogisticsPlan,
    ProposedFlightOrder,
    ProposedPackageOrder,
    RunwayRepairOrder,
    SquadronRelocationOrder,
    SquadronTaskingOrder,
)
from game.ai_commander.planlegality import PlanLegalityChecker
from game.ato.flighttype import FlightType
from tests.ai_commander import fakes

# ---------------------------------------------------------------------------
# Fixtures and shared plumbing
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _objective_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes.patch_objective_finder(monkeypatch)


@pytest.fixture(autouse=True)
def _fresh_capability_cache() -> Iterator[None]:
    CAPABILITY_CACHE.clear()
    yield
    CAPABILITY_CACHE.clear()


class _FakeAircraftAdapter:
    """Stands in for :class:`game.purchaseadapter.AircraftPurchaseAdapter`.

    Prices from the squadron's own airframe and permits the purchase; parking and
    capacity are still enforced by the real ``ControlPoint``/``Squadron`` surfaces,
    so the legality logic -- not the adapter -- is what the tests measure.
    """

    instances: list["_FakeAircraftAdapter"] = []

    def __init__(self, control_point: Any) -> None:
        self.control_point = control_point
        self.bought: list[tuple[Any, int]] = []
        _FakeAircraftAdapter.instances.append(self)

    def price_of(self, squadron: Any) -> int:
        return int(squadron.aircraft.price)

    def can_buy(self, squadron: Any) -> bool:
        return True

    def buy(self, squadron: Any, quantity: int) -> None:
        self.bought.append((squadron, quantity))


class _FakeGroundAdapter:
    instances: list["_FakeGroundAdapter"] = []

    def __init__(self, control_point: Any, coalition: Any, game: Any) -> None:
        self.control_point = control_point
        self.coalition = coalition
        self.game = game
        self.bought: list[tuple[Any, int]] = []
        _FakeGroundAdapter.instances.append(self)

    def price_of(self, unit_type: Any) -> int:
        return int(unit_type.price)

    def buy(self, unit_type: Any, quantity: int) -> None:
        self.bought.append((unit_type, quantity))


@pytest.fixture(autouse=True)
def _patch_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeAircraftAdapter.instances.clear()
    _FakeGroundAdapter.instances.clear()
    monkeypatch.setattr(
        "game.purchaseadapter.AircraftPurchaseAdapter", _FakeAircraftAdapter
    )
    monkeypatch.setattr(
        "game.purchaseadapter.GroundUnitPurchaseAdapter", _FakeGroundAdapter
    )


class _Context:
    """Bundles the objects a legality check is run against for one game."""

    def __init__(self, campaign: Any, game: Any, policy: IntelPolicy) -> None:
        self.campaign = campaign
        self.game = game
        self.policy = policy
        projector = OperationsProjector(game, policy)
        self.brief: OperationsBrief = projector.project("hash", "rev-1")
        self.resolver: OperationsResolver = projector.resolver
        self.capabilities: CapabilityIndex = capability_index_for(campaign.red)
        # The checker recomputes the revision from live state and compares it to
        # the plan's, so a plan built for "now" must carry the live digest.
        self.revision: str = IntelProjector(game, policy).campaign_revision()

    @property
    def checker(self) -> PlanLegalityChecker:
        return PlanLegalityChecker(
            self.game, self.brief, self.resolver, self.capabilities
        )


def _context(
    *,
    red_budget: float | None = None,
    transit_reachable: bool = True,
    policy: IntelPolicy = IntelPolicy.REALISTIC,
) -> _Context:
    campaign, game = fakes.synthetic_game(
        red_budget=red_budget, transit_reachable=transit_reachable
    )
    return _Context(campaign, game, policy)


def _reasons(rejections: list[Any]) -> str:
    return " || ".join(r.reason for r in rejections)


def _logistics(ctx: _Context, **orders: Any) -> LogisticsPlan:
    return LogisticsPlan(
        schema_version="red-commander-logistics/1",
        turn_id=ctx.brief.turn_id,
        campaign_revision=ctx.revision,
        intent="test",
        **orders,
    )


def _air_tasking(ctx: _Context, *packages: ProposedPackageOrder) -> AirTaskingPlan:
    return AirTaskingPlan(
        schema_version="red-commander-air-tasking/1",
        turn_id=ctx.brief.turn_id,
        campaign_revision=ctx.revision,
        intent="test",
        packages=tuple(packages),
    )


# ---------------------------------------------------------------------------
# The revision guard fires before anything else
# ---------------------------------------------------------------------------


class TestRevisionGuard:
    def test_a_stale_revision_rejects_the_whole_logistics_plan(self) -> None:
        ctx = _context()
        plan = LogisticsPlan(
            schema_version="red-commander-logistics/1",
            turn_id=ctx.brief.turn_id,
            campaign_revision="stale-revision",
            intent="test",
            aircraft_orders=(AircraftOrder(squadron_id="SQN-1", quantity=1),),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert [r.element for r in rejections] == ["campaign_revision"]

    def test_a_stale_revision_rejects_the_whole_air_tasking_plan(self) -> None:
        ctx = _context()
        plan = AirTaskingPlan(
            schema_version="red-commander-air-tasking/1",
            turn_id=ctx.brief.turn_id,
            campaign_revision="stale-revision",
            intent="test",
            packages=(
                ProposedPackageOrder(
                    target_id="TGT-1",
                    priority=1,
                    flights=(
                        ProposedFlightOrder(
                            mission_type=FlightType.SEAD, aircraft_count=2
                        ),
                    ),
                ),
            ),
        )
        result, rejections = ctx.checker.check_air_tasking(plan)
        assert result is None
        assert [r.element for r in rejections] == ["campaign_revision"]

    def test_expected_revision_override_re_baselines_air_tasking(self) -> None:
        """The commander's OWN applied logistics must not trip the guard.

        In ACTIVE mode the air tasking plan echoes the turn-start revision, but
        the live revision has already moved because the commander's own
        logistics spent money earlier this turn. Passing the refreshed live
        revision as ``expected_revision`` re-baselines the guard against the
        commander's own applied orders, so the stale echoed revision is no
        longer treated as external tampering.
        """

        ctx = _context()
        package = ProposedPackageOrder(
            target_id="TGT-1",
            priority=1,
            flights=(
                ProposedFlightOrder(mission_type=FlightType.SEAD, aircraft_count=2),
            ),
        )
        plan = AirTaskingPlan(
            schema_version="red-commander-air-tasking/1",
            turn_id=ctx.brief.turn_id,
            # The plan echoes the turn-start revision, which differs from live
            # because the commander's own logistics were already applied.
            campaign_revision="turn-start-revision",
            intent="test",
            packages=(package,),
        )
        # ctx.revision is the live revision (the refreshed baseline the
        # controller would compute after its own logistics applied).
        checker = PlanLegalityChecker(
            ctx.game,
            ctx.brief,
            ctx.resolver,
            ctx.capabilities,
            expected_revision=ctx.revision,
        )
        result, rejections = checker.check_air_tasking(plan)
        assert rejections == []
        assert result is not None
        assert len(result.packages) == 1

    def test_expected_revision_override_still_catches_external_change(self) -> None:
        """Anti-tamper is preserved: a genuine external change is still caught.

        If the live revision differs from the refreshed baseline for a reason
        NOT explained by the commander's own applied orders, the guard must
        still reject the whole air tasking stage as stale.
        """

        ctx = _context()
        package = ProposedPackageOrder(
            target_id="TGT-1",
            priority=1,
            flights=(
                ProposedFlightOrder(mission_type=FlightType.SEAD, aircraft_count=2),
            ),
        )
        plan = AirTaskingPlan(
            schema_version="red-commander-air-tasking/1",
            turn_id=ctx.brief.turn_id,
            campaign_revision="turn-start-revision",
            intent="test",
            packages=(package,),
        )
        # A baseline that matches neither the live revision nor the echoed one
        # simulates an external mutation between the refreshed baseline and
        # application: live != baseline, so the stage is rejected.
        checker = PlanLegalityChecker(
            ctx.game,
            ctx.brief,
            ctx.resolver,
            ctx.capabilities,
            expected_revision="externally-tampered-revision",
        )
        result, rejections = checker.check_air_tasking(plan)
        assert result is None
        assert [r.element for r in rejections] == ["campaign_revision"]
        # The rejection still reports the plan's echoed revision, so audit
        # wording is unchanged.
        assert rejections[0].value == "turn-start-revision"


# ---------------------------------------------------------------------------
# Happy paths: a legal order of each class binds to live objects
# ---------------------------------------------------------------------------


class TestLegalOrdersAreBound:
    def test_a_legal_aircraft_purchase_binds(self) -> None:
        ctx = _context()
        plan = _logistics(
            ctx, aircraft_orders=(AircraftOrder(squadron_id="SQN-1", quantity=2),)
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert rejections == []
        assert result is not None
        assert len(result.aircraft_purchases) == 1
        purchase = result.aircraft_purchases[0]
        assert purchase.squadron_id == "SQN-1"
        assert purchase.quantity == 2
        assert purchase.unit_price == 22  # RED-JET

    def test_a_legal_ground_purchase_binds(self) -> None:
        ctx = _context()
        plan = _logistics(
            ctx,
            ground_orders=(
                GroundUnitOrder(base_id="BASE-1", unit_id="RED-TANK", quantity=3),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert rejections == []
        assert result is not None
        assert len(result.ground_purchases) == 1
        assert result.ground_purchases[0].unit_price == 12  # RED-TANK

    def test_a_legal_runway_repair_binds(self) -> None:
        ctx = _context()
        # BASE-2 (rear) has a broken but repairable runway.
        plan = _logistics(ctx, runway_repairs=(RunwayRepairOrder(base_id="BASE-2"),))
        result, rejections = ctx.checker.check_logistics(plan)
        assert rejections == []
        assert result is not None
        assert len(result.runway_repairs) == 1
        assert result.runway_repairs[0].cost == 100

    def test_a_legal_tasking_order_binds(self) -> None:
        ctx = _context()
        # SQN-1 can fly SEAD and CAS.
        plan = _logistics(
            ctx,
            squadron_tasking=(
                SquadronTaskingOrder(
                    squadron_id="SQN-1",
                    mission_types=(FlightType.SEAD, FlightType.CAS),
                ),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert rejections == []
        assert result is not None
        assert len(result.tasking) == 1
        assert set(result.tasking[0].mission_types) == {
            FlightType.SEAD,
            FlightType.CAS,
        }

    def test_a_legal_transfer_binds(self) -> None:
        ctx = _context()
        # BASE-1 holds RED-TANK armour; BASE-2 is reachable.
        plan = _logistics(
            ctx,
            ground_transfers=(
                GroundTransferOrder(
                    origin_base_id="BASE-1",
                    destination_base_id="BASE-2",
                    units=(("RED-TANK", 2),),
                ),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert rejections == []
        assert result is not None
        assert len(result.transfers) == 1
        assert result.transfers[0].size == 2

    def test_a_legal_air_tasking_package_binds(self) -> None:
        ctx = _context()
        package = ProposedPackageOrder(
            target_id="TGT-1",
            priority=1,
            flights=(
                ProposedFlightOrder(mission_type=FlightType.SEAD, aircraft_count=2),
            ),
        )
        result, rejections = ctx.checker.check_air_tasking(_air_tasking(ctx, package))
        assert rejections == []
        assert result is not None
        assert len(result.packages) == 1
        assert result.packages[0].target_id == "TGT-1"


# ---------------------------------------------------------------------------
# Overspend: money is committed cumulatively and never overdrawn
# ---------------------------------------------------------------------------


class TestBudgetIsEnforced:
    def test_an_unaffordable_aircraft_order_is_rejected(self) -> None:
        # One RED-JET costs 22; a 10M budget cannot afford even one.
        ctx = _context(red_budget=10.0)
        plan = _logistics(
            ctx, aircraft_orders=(AircraftOrder(squadron_id="SQN-1", quantity=2),)
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "of the budget is uncommitted" in _reasons(rejections)

    def test_an_order_is_clamped_down_to_what_the_budget_allows(self) -> None:
        # 50M affords two RED-JETs (44M) but not the four requested.
        ctx = _context(red_budget=50.0)
        plan = _logistics(
            ctx, aircraft_orders=(AircraftOrder(squadron_id="SQN-1", quantity=4),)
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is not None
        assert result.aircraft_purchases[0].quantity == 2
        assert "reduced from 4 to 2" in _reasons(rejections)

    def test_the_budget_is_shared_across_orders_in_one_plan(self) -> None:
        # 30M: the first RED-JET order takes 22M, leaving 8M -- too little for a
        # second jet, so the second order is rejected rather than overdrawing.
        ctx = _context(red_budget=30.0)
        plan = _logistics(
            ctx,
            aircraft_orders=(
                AircraftOrder(squadron_id="SQN-1", quantity=1),
                AircraftOrder(squadron_id="SQN-1", quantity=1),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is not None
        assert len(result.aircraft_purchases) == 1
        assert "of the budget is uncommitted" in _reasons(rejections)


# ---------------------------------------------------------------------------
# Parking and squadron capacity
# ---------------------------------------------------------------------------


class TestParkingAndCapacity:
    def test_no_free_parking_rejects_the_purchase(self) -> None:
        ctx = _context()
        # Rebind SQN-1 onto a base with zero free parking.
        no_parking = fakes.make_control_point(
            cp_id=99,
            name="RED-PACKED-BASE",
            captured=fakes.Player.RED,
            position=fakes.point(0.0, 0.0),
            parking_free=0,
        )
        squadron = ctx.resolver.squadron("SQN-1")
        assert squadron is not None
        squadron.location = no_parking
        plan = _logistics(
            ctx, aircraft_orders=(AircraftOrder(squadron_id="SQN-1", quantity=1),)
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "no free parking" in _reasons(rejections)

    def test_a_squadron_at_its_aircraft_limit_is_rejected(self) -> None:
        ctx = _context()
        squadron = ctx.resolver.squadron("SQN-1")
        assert squadron is not None
        squadron.owned_aircraft = squadron.max_size  # no capacity left
        plan = _logistics(
            ctx, aircraft_orders=(AircraftOrder(squadron_id="SQN-1", quantity=1),)
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "already at its aircraft limit" in _reasons(rejections)

    def test_a_squadron_no_longer_on_a_red_base_is_rejected(self) -> None:
        ctx = _context()
        captured_base = fakes.make_control_point(
            cp_id=98,
            name="OVERRUN-BASE",
            captured=fakes.Player.BLUE,
            position=fakes.point(0.0, 0.0),
            parking_free=10,
        )
        squadron = ctx.resolver.squadron("SQN-1")
        assert squadron is not None
        squadron.location = captured_base
        plan = _logistics(
            ctx, aircraft_orders=(AircraftOrder(squadron_id="SQN-1", quantity=1),)
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "no longer based at a RED control point" in _reasons(rejections)


# ---------------------------------------------------------------------------
# Ground purchases against live state
# ---------------------------------------------------------------------------


class TestGroundPurchaseLegality:
    def test_a_base_that_is_not_red_is_rejected(self) -> None:
        ctx = _context()
        # Inject a captured base under a RED-looking id into the resolver.
        overrun = fakes.make_control_point(
            cp_id=97,
            name="OVERRUN-DEPOT",
            captured=fakes.Player.BLUE,
            position=fakes.point(0.0, 0.0),
        )
        ctx.resolver.bases["BASE-1"] = overrun
        plan = _logistics(
            ctx,
            ground_orders=(
                GroundUnitOrder(base_id="BASE-1", unit_id="RED-TANK", quantity=1),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert rejections[0].reason  # STATE_CHANGED
        assert "campaign state changed" in _reasons(rejections)

    def test_a_base_with_no_ground_unit_source_is_rejected(self) -> None:
        ctx = _context()
        base = ctx.resolver.base("BASE-1")
        assert base is not None
        base.has_ground_unit_source = lambda game: False
        plan = _logistics(
            ctx,
            ground_orders=(
                GroundUnitOrder(base_id="BASE-1", unit_id="RED-TANK", quantity=1),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "no ground unit source" in _reasons(rejections)

    def test_a_ground_unit_the_faction_does_not_field_is_rejected(self) -> None:
        ctx = _context()
        plan = _logistics(
            ctx,
            ground_orders=(
                GroundUnitOrder(base_id="BASE-1", unit_id="BLUE-TANK", quantity=1),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "does not field this ground unit type" in _reasons(rejections)


# ---------------------------------------------------------------------------
# Runway repair against live state
# ---------------------------------------------------------------------------


class TestRunwayRepairLegality:
    def test_a_base_whose_runway_cannot_be_repaired_is_rejected(self) -> None:
        ctx = _context()
        # BASE-1 (front) has a working runway, so it is not repairable.
        plan = _logistics(ctx, runway_repairs=(RunwayRepairOrder(base_id="BASE-1"),))
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "does not have a repairable runway" in _reasons(rejections)

    def test_an_unaffordable_repair_is_rejected(self) -> None:
        ctx = _context(red_budget=50.0)  # repair costs 100
        plan = _logistics(ctx, runway_repairs=(RunwayRepairOrder(base_id="BASE-2"),))
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "runway repair costs 100" in _reasons(rejections)


# ---------------------------------------------------------------------------
# Relocation against the exact conditions plan_relocation raises on
# ---------------------------------------------------------------------------


class TestRelocationLegality:
    def test_relocating_to_the_current_base_is_rejected(self) -> None:
        ctx = _context()
        # SQN-1 is already at BASE-1.
        plan = _logistics(
            ctx,
            squadron_relocations=(
                SquadronRelocationOrder(squadron_id="SQN-1", base_id="BASE-1"),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "already based there" in _reasons(rejections)

    def test_cancelling_a_relocation_that_does_not_exist_is_rejected(self) -> None:
        ctx = _context()
        plan = _logistics(
            ctx,
            squadron_relocations=(
                SquadronRelocationOrder(squadron_id="SQN-1", base_id=None),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "no pending relocation to cancel" in _reasons(rejections)

    def test_relocating_to_a_base_that_cannot_operate_the_airframe(self) -> None:
        ctx = _context()
        dest = ctx.resolver.base("BASE-2")
        assert dest is not None
        dest.can_operate = lambda aircraft: False
        plan = _logistics(
            ctx,
            squadron_relocations=(
                SquadronRelocationOrder(squadron_id="SQN-1", base_id="BASE-2"),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "cannot operate" in _reasons(rejections)

    def test_a_legal_relocation_binds(self) -> None:
        ctx = _context()
        # SQN-1 -> BASE-2, which can operate it and has ample parking.
        plan = _logistics(
            ctx,
            squadron_relocations=(
                SquadronRelocationOrder(squadron_id="SQN-1", base_id="BASE-2"),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert rejections == []
        assert result is not None
        assert result.relocations[0].base_id == "BASE-2"


# ---------------------------------------------------------------------------
# Tasking against Squadron.capable_of
# ---------------------------------------------------------------------------


class TestTaskingLegality:
    def test_a_mission_the_squadron_cannot_fly_is_rejected(self) -> None:
        ctx = _context()
        # SQN-1 (fighters) cannot fly Strike.
        plan = _logistics(
            ctx,
            squadron_tasking=(
                SquadronTaskingOrder(
                    squadron_id="SQN-1", mission_types=(FlightType.STRIKE,)
                ),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "squadron cannot fly Strike" in _reasons(rejections)

    def test_a_capable_task_survives_alongside_an_incapable_one(self) -> None:
        ctx = _context()
        plan = _logistics(
            ctx,
            squadron_tasking=(
                SquadronTaskingOrder(
                    squadron_id="SQN-1",
                    mission_types=(FlightType.SEAD, FlightType.STRIKE),
                ),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is not None
        assert result.tasking[0].mission_types == (FlightType.SEAD,)
        assert "squadron cannot fly Strike" in _reasons(rejections)


# ---------------------------------------------------------------------------
# Transfers against the transit network and unit availability
# ---------------------------------------------------------------------------


class TestTransferLegality:
    def test_a_transfer_between_the_same_base_is_rejected(self) -> None:
        ctx = _context()
        plan = _logistics(
            ctx,
            ground_transfers=(
                GroundTransferOrder(
                    origin_base_id="BASE-1",
                    destination_base_id="BASE-1",
                    units=(("RED-TANK", 1),),
                ),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "origin and destination are the same base" in _reasons(rejections)

    def test_a_transfer_with_no_route_is_rejected(self) -> None:
        ctx = _context(transit_reachable=False)
        plan = _logistics(
            ctx,
            ground_transfers=(
                GroundTransferOrder(
                    origin_base_id="BASE-1",
                    destination_base_id="BASE-2",
                    units=(("RED-TANK", 1),),
                ),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "no route from" in _reasons(rejections)

    def test_moving_a_unit_that_is_not_at_the_origin_is_rejected(self) -> None:
        ctx = _context()
        # BASE-1 holds RED-TANK and RED-ARTY, but no RED-TRUCK.
        plan = _logistics(
            ctx,
            ground_transfers=(
                GroundTransferOrder(
                    origin_base_id="BASE-1",
                    destination_base_id="BASE-2",
                    units=(("RED-TRUCK", 1),),
                ),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is None
        assert "has no RED-TRUCK to move" in _reasons(rejections)

    def test_a_transfer_is_clamped_to_what_is_present(self) -> None:
        ctx = _context()
        # BASE-1 holds 9 RED-TANK; asking for 20 is clamped to 9.
        plan = _logistics(
            ctx,
            ground_transfers=(
                GroundTransferOrder(
                    origin_base_id="BASE-1",
                    destination_base_id="BASE-2",
                    units=(("RED-TANK", 20),),
                ),
            ),
        )
        result, rejections = ctx.checker.check_logistics(plan)
        assert result is not None
        assert result.transfers[0].size == 9
        assert "reduced from 20 to 9" in _reasons(rejections)


# ---------------------------------------------------------------------------
# Air tasking against live state
# ---------------------------------------------------------------------------


class TestAirTaskingLegality:
    def test_a_target_no_longer_in_the_brief_is_rejected(self) -> None:
        ctx = _context()
        package = ProposedPackageOrder(
            target_id="TGT-GONE",
            priority=1,
            flights=(
                ProposedFlightOrder(mission_type=FlightType.SEAD, aircraft_count=2),
            ),
        )
        result, rejections = ctx.checker.check_air_tasking(_air_tasking(ctx, package))
        assert result is None
        assert "campaign state changed" in _reasons(rejections)

    def test_tasking_an_unfielded_airframe_is_rejected(self) -> None:
        ctx = _context()
        # RED-INTERCEPTOR is a faction airframe with no squadron, so it cannot be
        # tasked this turn even though it is a legal capability-index id.
        package = ProposedPackageOrder(
            target_id="TGT-1",
            priority=1,
            flights=(
                ProposedFlightOrder(
                    mission_type=FlightType.SEAD,
                    aircraft_count=1,
                    aircraft_id="RED-INTERCEPTOR",
                ),
            ),
        )
        result, rejections = ctx.checker.check_air_tasking(_air_tasking(ctx, package))
        assert result is None
        assert "owns no airframes of this type yet" in _reasons(rejections)

    def test_a_mission_no_squadron_can_auto_plan_is_rejected(self) -> None:
        ctx = _context()
        # Force the air wing to refuse auto-planning the requested mission type.
        ctx.campaign.red.air_wing.auto_plannable = frozenset({FlightType.BARCAP})
        package = ProposedPackageOrder(
            target_id="TGT-1",
            priority=1,
            flights=(
                ProposedFlightOrder(mission_type=FlightType.SEAD, aircraft_count=2),
            ),
        )
        result, rejections = ctx.checker.check_air_tasking(_air_tasking(ctx, package))
        assert result is None
        assert "no squadron available to fly SEAD" in _reasons(rejections)

    def test_a_package_where_every_flight_dies_is_dropped(self) -> None:
        ctx = _context()
        ctx.campaign.red.air_wing.auto_plannable = frozenset({FlightType.BARCAP})
        package = ProposedPackageOrder(
            target_id="TGT-1",
            priority=1,
            flights=(
                ProposedFlightOrder(mission_type=FlightType.SEAD, aircraft_count=2),
                ProposedFlightOrder(mission_type=FlightType.DEAD, aircraft_count=2),
            ),
        )
        result, rejections = ctx.checker.check_air_tasking(_air_tasking(ctx, package))
        assert result is None
        assert "no flight in this package could be crewed" in _reasons(rejections)
