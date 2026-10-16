"""Legality of a schema-clean decision against live campaign state.

A decision that validates against the brief can still be illegal: the brief is
a snapshot, and the game moved on. More importantly, a model can ask for
something the *rules* forbid -- spending money RED does not have, pushing a
front whose force balance does not support it, or buying aircraft into an air
wing that has no squadron to receive them.

Every check here goes through :class:`LegalityChecker`, which delegates to the
game's own predicates rather than re-implementing them. The invariant under
test is always the same: an illegal element is **refused with a recorded
reason** and never applied, and the legal remainder of the same decision still
goes through.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from game.ai_commander.decision import (
    DECISION_SCHEMA_VERSION,
    RankedId,
    RedCommanderDecision,
    FrontPostureRequest,
    TargetSetPriority,
    validate_decision,
)
from game.ai_commander.directive import CommanderDirective, build_directive
from game.ai_commander.enums import (
    FrontPosture,
    IntelPolicy,
    MissionPurpose,
    ProcurementCategory,
    RedStrategy,
    ReservePolicy,
    TargetSetCategory,
)
from game.ai_commander.intel import IntelProjector, RedCommanderBrief
from game.ai_commander.legality import LegalityChecker
from game.config import RUNWAY_REPAIR_COST
from tests.ai_commander import fakes


@pytest.fixture(autouse=True)
def _objective_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes.patch_objective_finder(monkeypatch)


def build(
    red_budget: Optional[float] = None,
    red_deployable: Optional[int] = None,
) -> tuple[fakes.SyntheticCampaign, Any, RedCommanderBrief]:
    campaign, game = fakes.synthetic_game(
        red_budget=red_budget, red_deployable=red_deployable
    )
    brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
    return campaign, game, brief


def decision_for(
    brief: RedCommanderBrief,
    *,
    strategy: RedStrategy = RedStrategy.GROUND_OFFENSIVE,
    reserve_policy: ReservePolicy = ReservePolicy.BALANCED,
    fronts: tuple[str, ...] = ("FRONT-1",),
    postures: tuple[tuple[str, FrontPosture], ...] = (),
    spending: tuple[str, ...] = (),
    target_sets: tuple[str, ...] = (),
    revision: Optional[str] = None,
) -> RedCommanderDecision:
    """A decision built directly, bypassing schema validation.

    Schema validation is covered in ``test_decision_schema``; here the point is
    to hand the legality checker things that are *schema-legal* so that only
    live-state legality is under test.
    """

    return RedCommanderDecision(
        schema_version=DECISION_SCHEMA_VERSION,
        campaign_revision=brief.campaign_revision if revision is None else revision,
        turn_id=brief.turn_id,
        strategy=strategy,
        front_priorities=tuple(
            RankedId(id=front, rank=index + 1) for index, front in enumerate(fronts)
        ),
        push_postures=tuple(
            FrontPostureRequest(front_id=front, posture=posture)
            for front, posture in postures
        ),
        spending_priorities=tuple(
            RankedId(id=category, rank=index + 1)
            for index, category in enumerate(spending)
        ),
        target_set_priorities=tuple(
            TargetSetPriority(
                target_set_id=target, rank=index + 1, purpose=MissionPurpose.ATTRITION
            )
            for index, target in enumerate(target_sets)
        ),
        reserve_policy=reserve_policy,
    )


def reasons(rejections: list[Any], element: str) -> list[str]:
    return [r.reason for r in rejections if r.element == element]


class TestABudgetTheCommanderDoesNotHave:
    """RED cannot outspend its live budget, whatever the model asks for."""

    def test_aircraft_are_refused_when_the_cheapest_airframe_is_unaffordable(
        self,
    ) -> None:
        _, game, brief = build(red_budget=5.0)
        checker = LegalityChecker(game, brief)
        reason = checker.procurement_rejection(ProcurementCategory.AIRCRAFT)
        assert reason is not None
        assert "cheapest available airframe costs 22" in reason
        assert "only 5 is available" in reason

    def test_ground_units_are_refused_when_unaffordable(self) -> None:
        _, game, brief = build(red_budget=3.0)
        checker = LegalityChecker(game, brief)
        for category in (
            ProcurementCategory.GROUND_COMBAT_UNITS,
            ProcurementCategory.FRONT_LINE_RESERVES,
        ):
            reason = checker.procurement_rejection(category)
            assert reason is not None
            assert "cheapest ground unit costs 12" in reason

    def test_runway_repair_is_refused_when_unaffordable(self) -> None:
        _, game, brief = build(red_budget=float(RUNWAY_REPAIR_COST) - 1.0)
        checker = LegalityChecker(game, brief)
        reason = checker.procurement_rejection(ProcurementCategory.RUNWAY_REPAIR)
        assert reason is not None
        assert f"runway repair costs {RUNWAY_REPAIR_COST}" in reason

    def test_an_unaffordable_spending_plan_is_rejected_not_applied(self) -> None:
        _, game, brief = build(red_budget=1.0)
        decision = decision_for(brief, spending=("PROC-1", "PROC-2", "PROC-4"))
        directive, rejections = LegalityChecker(game, brief).check(decision)

        assert directive is not None
        assert directive.procurement_order == ()
        assert len(rejections) == 3
        assert all(rejection.reason for rejection in rejections)

    def test_the_directive_carries_no_amounts_for_the_engine_to_honour(self) -> None:
        """The commander orders *priority*, never a sum of money.

        This is the structural reason an overspend is impossible rather than
        merely checked: there is nowhere in the directive to put an amount.
        """

        _, game, brief = build()
        decision = decision_for(brief, spending=("PROC-1",))
        directive, _ = LegalityChecker(game, brief).check(decision)
        assert directive is not None
        payload = directive.to_dict()
        assert "budget" not in payload
        assert set(payload["procurement_order"]) == {"aircraft"}
        assert all(isinstance(entry, str) for entry in payload["procurement_order"])

    def test_an_affordable_plan_is_accepted(self) -> None:
        _, game, brief = build()
        decision = decision_for(brief, spending=("PROC-1", "PROC-2"))
        directive, rejections = LegalityChecker(game, brief).check(decision)
        assert rejections == []
        assert directive is not None
        assert directive.procurement_order == (
            ProcurementCategory.AIRCRAFT,
            ProcurementCategory.GROUND_COMBAT_UNITS,
        )


class TestAircraftMustHaveSomewhereToGo:
    def test_aircraft_are_refused_when_red_has_no_squadrons(self) -> None:
        campaign, game, brief = build()
        campaign.red.air_wing = fakes.FakeAirWing([])
        reason = LegalityChecker(game, brief).procurement_rejection(
            ProcurementCategory.AIRCRAFT
        )
        assert reason == "RED has no squadrons that could receive aircraft"

    def test_base_capacity_and_airframe_choice_stay_with_the_engine(self) -> None:
        """The commander cannot name an airframe, a base or a quantity.

        Parking limits, pilot availability and unit-type choice are enforced by
        Retribution's own procurement code, which the commander only reorders.
        A decision has no field that could express any of them, so there is
        nothing for a validator to have to catch.
        """

        _, game, brief = build()
        decision = decision_for(brief, spending=("PROC-1",))
        directive, _ = LegalityChecker(game, brief).check(decision)
        assert directive is not None
        payload = directive.to_dict()
        forbidden = {
            "aircraft_type",
            "airframe",
            "base",
            "control_point",
            "quantity",
            "count",
            "squadron",
            "unit_type",
        }
        assert forbidden.isdisjoint(payload)
        # And the brief never offered a squadron or airframe identifier either.
        assert brief.procurement_ids == {"PROC-1", "PROC-2", "PROC-3", "PROC-4"}


class TestGroundProcurementNeedsALiveSource:
    def test_front_line_units_need_a_base_with_a_supply_source(self) -> None:
        campaign, game, brief = build()
        campaign.red_front_base.has_ground_unit_source.return_value = False  # type: ignore[attr-defined]
        reason = LegalityChecker(game, brief).procurement_rejection(
            ProcurementCategory.GROUND_COMBAT_UNITS
        )
        assert reason is not None
        assert "has a ground unit source" in reason

    def test_reserves_need_a_base_that_can_recruit(self) -> None:
        campaign, game, brief = build()
        for cp in (campaign.red_front_base, campaign.red_rear_base):
            cp.can_recruit_ground_units.return_value = False  # type: ignore[attr-defined]
        reason = LegalityChecker(game, brief).procurement_rejection(
            ProcurementCategory.FRONT_LINE_RESERVES
        )
        assert reason == "no RED base can currently recruit reserve ground units"

    def test_runway_repair_needs_a_repairable_runway(self) -> None:
        campaign, game, brief = build()
        campaign.red_rear_base.runway_can_be_repaired = False  # type: ignore[misc]
        reason = LegalityChecker(game, brief).procurement_rejection(
            ProcurementCategory.RUNWAY_REPAIR
        )
        assert reason == "no RED runway is currently repairable"

    def test_faction_without_ground_units_cannot_buy_them(self) -> None:
        campaign, game, brief = build()
        campaign.red.faction = fakes.FakeFaction("Air Only", frontline_units=[])
        reason = LegalityChecker(game, brief).procurement_rejection(
            ProcurementCategory.GROUND_COMBAT_UNITS
        )
        assert reason == "RED faction has no ground units available to purchase"


class TestFrontsRedCannotAct:
    """A front RED does not hold, or cannot reach, is not actionable."""

    def test_posture_on_a_front_that_no_longer_exists_is_refused(self) -> None:
        campaign, game, brief = build()
        # The front line disappears -- the base was captured, or the units are
        # gone -- after the brief was produced.
        campaign.theater.fronts = []
        checker = LegalityChecker(game, brief)
        reason = checker.posture_rejection(
            ("RED-FRONT-BASE", "BLUE-FRONT-BASE"), FrontPosture.PUSH
        )
        assert reason is not None
        assert "no longer exists" in reason

    def test_front_priority_for_an_inactive_front_is_rejected(self) -> None:
        campaign, game, brief = build()
        campaign.theater.fronts = []
        decision = decision_for(brief, fronts=("FRONT-1",), spending=("PROC-1",))
        directive, rejections = LegalityChecker(game, brief).check(decision)

        assert directive is not None
        assert directive.front_order == ()
        assert reasons(rejections, "front_priorities[0]") == [
            "front is no longer active"
        ]
        # The legal remainder of the same decision still applies.
        assert directive.procurement_order == (ProcurementCategory.AIRCRAFT,)

    def test_a_front_between_bases_red_does_not_own_is_not_in_the_brief(self) -> None:
        """RED can only be given fronts it is actually standing on.

        The brief enumerates fronts from RED's own perspective, so a front
        between two BLUE bases -- or one RED cannot reach -- has no identifier
        for a decision to reference.
        """

        campaign, _, brief = build()
        for view in brief.fronts:
            assert view.own_base == campaign.red_front_base.name
            assert view.enemy_base == campaign.blue_front_base.name

    def test_posture_requiring_more_force_than_red_has_is_refused(self) -> None:
        """``BREAKTHROUGH`` needs a 2:1 balance; at 40:1157 it does not hold."""

        campaign, game, brief = build(red_deployable=40)
        (front,) = brief.fronts
        assert FrontPosture.BREAKTHROUGH not in front.legal_postures
        assert FrontPosture.PUSH not in front.legal_postures

        checker = LegalityChecker(game, brief)
        reason = checker.posture_rejection(
            ("RED-FRONT-BASE", "BLUE-FRONT-BASE"), FrontPosture.BREAKTHROUGH
        )
        assert reason is not None
        assert "precondition" in reason

    def test_a_posture_that_became_illegal_after_briefing_is_refused(self) -> None:
        """The brief said ``PUSH`` was legal; live state now says otherwise."""

        campaign, game, brief = build()
        (front,) = brief.fronts
        assert FrontPosture.PUSH in front.legal_postures

        campaign.red_front_base.deployable_front_line_units = 1  # type: ignore[misc]
        decision = decision_for(
            brief,
            postures=(("FRONT-1", FrontPosture.PUSH),),
            spending=("PROC-1",),
            revision=brief.campaign_revision,
        )
        checker = LegalityChecker(game, brief)
        # The revision guard fires first, which is itself the correct answer:
        # nothing derived from a stale brief is applied.
        directive, rejections = checker.check(decision)
        assert directive is None
        assert reasons(rejections, "campaign_revision") == [
            "campaign state changed between briefing and application"
        ]

    def test_posture_alone_is_refused_without_touching_the_stance(self) -> None:
        campaign, game, brief = build(red_deployable=40)
        before = dict(campaign.red_front_base.stances)
        decision = decision_for(
            brief,
            fronts=(),
            postures=(("FRONT-1", FrontPosture.BREAKTHROUGH),),
            spending=("PROC-1",),
        )
        directive, rejections = LegalityChecker(game, brief).check(decision)

        assert directive is not None
        assert directive.front_postures == {}
        assert reasons(rejections, "push_postures[0]")
        assert campaign.red_front_base.stances == before

    def test_a_legal_posture_is_accepted(self) -> None:
        _, game, brief = build()
        decision = decision_for(brief, postures=(("FRONT-1", FrontPosture.PUSH),))
        directive, rejections = LegalityChecker(game, brief).check(decision)
        assert rejections == []
        assert directive is not None
        assert (
            directive.posture_for(("RED-FRONT-BASE", "BLUE-FRONT-BASE"))
            is FrontPosture.PUSH
        )


class TestTargetSetsMustStillExist:
    def test_a_category_with_nothing_known_is_refused(self) -> None:
        campaign, game, brief = build()
        checker = LegalityChecker(game, brief)
        # Shipping was never in the brief, because nothing of the sort is known.
        assert TargetSetCategory.ENEMY_SHIPPING not in {
            view.category for view in brief.known_target_sets
        }
        assert (
            checker.target_set_rejection(TargetSetCategory.ENEMY_SHIPPING)
            == "this class of objective was not in the briefing"
        )

    def test_known_categories_are_accepted(self) -> None:
        _, game, brief = build()
        checker = LegalityChecker(game, brief)
        for view in brief.known_target_sets:
            assert checker.target_set_rejection(view.category) is None


class TestNothingLegalMeansNoDirective:
    def test_a_wholly_illegal_decision_produces_no_directive(self) -> None:
        campaign, game, brief = build(red_budget=0.0, red_deployable=1)
        campaign.theater.fronts = []
        campaign.red.air_wing = fakes.FakeAirWing([])
        decision = decision_for(
            brief,
            fronts=("FRONT-1",),
            postures=(("FRONT-1", FrontPosture.BREAKTHROUGH),),
            spending=("PROC-1", "PROC-2", "PROC-3", "PROC-4"),
            target_sets=(),
            revision=IntelProjector(game, IntelPolicy.REALISTIC).campaign_revision(),
        )
        directive, rejections = LegalityChecker(game, brief).check(decision)

        assert directive is None
        assert reasons(rejections, "<directive>") == [
            "nothing in the decision was legal against live state, so the "
            "built-in RED automation keeps control of this turn"
        ]

    def test_a_strategy_with_no_orderings_changes_nothing_and_falls_back(self) -> None:
        """A bare strategy word is not an instruction the planner can act on.

        ``strategy`` is recorded for the audit trail, but every behavioural
        lever is an ordering or a posture. A decision that sets only a strategy
        would leave the turn identical to the built-in automation, so the
        controller declines to claim the AI decided anything.
        """

        _, game, brief = build()
        decision = decision_for(brief, fronts=(), strategy=RedStrategy.REBUILD)
        directive, rejections = LegalityChecker(game, brief).check(decision)
        assert directive is None
        assert reasons(rejections, "<directive>")

    def test_a_single_legal_ordering_is_enough_content(self) -> None:
        _, game, brief = build()
        decision = decision_for(
            brief, fronts=(), strategy=RedStrategy.REBUILD, spending=("PROC-4",)
        )
        directive, rejections = LegalityChecker(game, brief).check(decision)
        assert rejections == []
        assert directive is not None
        assert directive.strategy is RedStrategy.REBUILD
        assert directive.procurement_order == (ProcurementCategory.RUNWAY_REPAIR,)


class TestStaleStateIsRefusedWholesale:
    def test_state_changing_after_briefing_voids_the_whole_decision(self) -> None:
        campaign, game, brief = build()
        decision = decision_for(brief, spending=("PROC-1",))
        # Something happened between the brief and the reply arriving.
        campaign.red.budget = 42.0
        directive, rejections = LegalityChecker(game, brief).check(decision)

        assert directive is None
        assert reasons(rejections, "campaign_revision") == [
            "campaign state changed between briefing and application"
        ]

    def test_carry_forward_rechecks_every_front(self) -> None:
        campaign, game, brief = build(red_deployable=40)
        previous = build_directive(
            turn_id=brief.turn_id - 1,
            campaign_revision="stale-revision",
            strategy=RedStrategy.ATTRIT,
            reserve_policy=ReservePolicy.BUILD_RESERVES,
            fronts=(("RED-FRONT-BASE", "BLUE-FRONT-BASE"),),
            postures=(
                (("RED-FRONT-BASE", "BLUE-FRONT-BASE"), FrontPosture.BREAKTHROUGH),
            ),
            procurement=(ProcurementCategory.AIRCRAFT,),
        )
        directive, rejections = LegalityChecker(game, brief).carry_forward(previous)

        assert directive is not None
        # The strategy carries; the posture that no longer holds does not.
        assert directive.strategy is RedStrategy.ATTRIT
        assert directive.front_postures == {}
        assert directive.turn_id == brief.turn_id
        assert directive.campaign_revision == brief.campaign_revision
        assert reasons(rejections, "carry_forward.front_postures")

    def test_carry_forward_drops_fronts_that_disappeared(self) -> None:
        campaign, game, brief = build()
        campaign.theater.fronts = []
        previous = build_directive(
            turn_id=brief.turn_id - 1,
            campaign_revision="stale-revision",
            strategy=RedStrategy.DEFEND,
            reserve_policy=ReservePolicy.BALANCED,
            fronts=(("RED-FRONT-BASE", "BLUE-FRONT-BASE"),),
            procurement=(ProcurementCategory.AIRCRAFT,),
        )
        directive, rejections = LegalityChecker(game, brief).carry_forward(previous)
        assert directive is not None
        assert directive.front_order == ()
        assert reasons(rejections, "carry_forward.front_order") == [
            "front from the previous directive is no longer active"
        ]

    def test_carry_forward_with_nothing_legal_returns_none(self) -> None:
        campaign, game, brief = build()
        campaign.theater.fronts = []
        previous = CommanderDirective(
            turn_id=brief.turn_id - 1,
            campaign_revision="stale-revision",
            strategy=RedStrategy.DEFEND,
            reserve_policy=ReservePolicy.BALANCED,
            front_order=(("RED-FRONT-BASE", "BLUE-FRONT-BASE"),),
        )
        directive, rejections = LegalityChecker(game, brief).carry_forward(previous)
        assert directive is None
        assert reasons(rejections, "<carry_forward>")


class TestSchemaAndLegalityAgree:
    def test_a_posture_illegal_in_the_brief_never_reaches_legality(self) -> None:
        """Defence in depth: the schema layer refuses it first."""

        _, game, brief = build(red_deployable=40)
        payload = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "campaign_revision": brief.campaign_revision,
            "turn_id": brief.turn_id,
            "strategy": RedStrategy.GROUND_OFFENSIVE.value,
            "push_postures": [
                {"front_id": "FRONT-1", "posture": FrontPosture.BREAKTHROUGH.value}
            ],
            "reserve_policy": ReservePolicy.BALANCED.value,
        }
        outcome = validate_decision(payload, brief)
        assert outcome.decision is not None
        assert outcome.decision.push_postures == ()
        (rejection,) = [
            r for r in outcome.rejections if r.element == "push_postures[0].posture"
        ]
        assert "not legal on this front" in rejection.reason
