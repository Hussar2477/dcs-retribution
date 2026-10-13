"""The operations brief is ACTIVE mode's map, and it must not cheat.

ACTIVE mode plans from an *operations brief*: the concrete list of RED bases,
squadrons and observable enemy objectives its stage 2 and 3 prompts are built
from. It is a strictly wider surface than COMMANDER mode's strategic brief, so
it gets the same blunt sentinel scan plus checks that the ``REALISTIC`` intel
policy actually withholds what it claims to.

Retribution has no campaign-layer fog of war of its own, so ``REALISTIC`` is a
new restriction this feature adds: objectives out of observation range of RED's
own forces are dropped entirely, and no target ever carries coordinates. The
accuracy spot-checks confirm the projection still describes the campaign RED
really has, so the "no leaks" assertions are not passing by saying nothing.
"""

from __future__ import annotations

from typing import cast

import pytest

from game.ai_commander.enums import IntelPolicy, TargetSetCategory
from game.theater.player import Player
from game.ai_commander.operations import (
    MAX_ENUMERATED_GROUND_TYPES,
    OPERATIONS_SCHEMA_VERSION,
    BaseView,
    OperationsBrief,
    OperationsProjector,
    SquadronView,
    TargetView,
)
from tests.ai_commander import fakes


class _FakeObjective:
    def __init__(self, category: str) -> None:
        self.category = category


class TestEnemyTargetIncomeTag:
    """Income-generating enemy buildings carry their generic per-turn value so a
    deep strike on a derrick/oil rig reads as high value, not filler."""

    def _target(self, category: TargetSetCategory, income: float | None) -> TargetView:
        return TargetView(
            id="TGT-1",
            category=category,
            label="derrick",
            near="BASE-1",
            threatens_own_forces=False,
            legal_missions=("Strike",),
            income_value=income,
        )

    def test_income_building_target_is_tagged(self) -> None:
        rendered = self._target(TargetSetCategory.ENEMY_INFRASTRUCTURE, 10).render()
        assert "income~10/turn" in rendered

    def test_non_income_target_has_no_tag(self) -> None:
        rendered = self._target(TargetSetCategory.ENEMY_MOTORPOOLS, None).render()
        assert "income~" not in rendered

    def test_zero_income_has_no_tag(self) -> None:
        rendered = self._target(TargetSetCategory.ENEMY_INFRASTRUCTURE, 0).render()
        assert "income~" not in rendered

    def test_income_derived_from_rewards_by_category(self) -> None:
        # Only categories in the REWARDS table earn a value; others get None.
        assert OperationsProjector._target_income(_FakeObjective("derrick")) == 8.0
        assert OperationsProjector._target_income(_FakeObjective("oil")) == 10.0
        assert OperationsProjector._target_income(_FakeObjective("factory")) == 2.5
        assert OperationsProjector._target_income(_FakeObjective("ammo")) == 2.0
        assert OperationsProjector._target_income(_FakeObjective("motorpool")) is None
        assert OperationsProjector._target_income(_FakeObjective("")) is None


@pytest.fixture(autouse=True)
def _objective_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes.patch_objective_finder(monkeypatch)


def _brief(
    policy: IntelPolicy = IntelPolicy.REALISTIC,
) -> tuple[fakes.SyntheticCampaign, OperationsBrief]:
    campaign, game = fakes.synthetic_game()
    brief = OperationsProjector(game, policy).project("campaign-hash", "rev-1")
    return campaign, brief


def _blob(brief: OperationsBrief) -> str:
    return fakes.serialise_everything(brief.to_dict(), brief.render_compact())


# ---------------------------------------------------------------------------
# Anti-cheat
# ---------------------------------------------------------------------------


class TestNoBlueLeaks:
    def test_no_blue_sentinel_appears_in_the_realistic_brief(self) -> None:
        _, brief = _brief(IntelPolicy.REALISTIC)
        assert fakes.blue_leaks_in(_blob(brief)) == []

    def test_full_parity_never_leaks_economy_or_air_wing_internals(self) -> None:
        # FULL_PARITY widens RED's view of the *map* (see the withholding tests
        # below), but the enemy's money, squadrons, pilots and planned ATO are
        # engine internals no policy is allowed to expose.
        _, brief = _brief(IntelPolicy.FULL_PARITY)
        leaked = set(fakes.blue_leaks_in(_blob(brief)))
        forbidden = {
            "blue_budget",
            "blue_income_per_turn",
            "blue_squadron_name",
            "blue_aircraft_name",
            "blue_pilot_count",
            "blue_planned_package",
            "blue_undetected_tgo",
        }
        assert leaked & forbidden == set()

    def test_red_budget_and_base_survive_projection(self) -> None:
        """The ops brief legitimately carries RED's money and base name."""

        _, brief = _brief(IntelPolicy.REALISTIC)
        blob = _blob(brief)
        assert str(int(cast(float, fakes.RED_SENTINELS["red_budget"]))) in blob
        assert cast(str, fakes.RED_SENTINELS["red_base_name"]) in blob


class TestRealisticPolicyWithholding:
    def test_realistic_withholds_the_documented_fields(self) -> None:
        _, brief = _brief(IntelPolicy.REALISTIC)
        assert brief.withheld_fields == (
            "enemy_unit_coordinates",
            "enemy_planned_flights",
            "unobserved_enemy_objectives",
        )

    def test_full_parity_still_withholds_coordinates_and_flight_plans(self) -> None:
        _, brief = _brief(IntelPolicy.FULL_PARITY)
        # Full parity is not omniscience: coordinates and the enemy ATO are never
        # shared, only the "unobserved objectives" restriction is lifted.
        assert brief.withheld_fields == (
            "enemy_unit_coordinates",
            "enemy_planned_flights",
        )

    def test_unobserved_enemy_objectives_are_dropped_under_realistic(self) -> None:
        """The far BLUE airbase and distant IADS are out of RED's sight."""

        _, realistic = _brief(IntelPolicy.REALISTIC)
        blob = _blob(realistic)
        # The base RED actually faces across the front is public map data and is
        # still named; the hidden rear base is not observable and is dropped.
        assert fakes.PUBLIC_BLUE_FRONT_BASE in blob
        assert cast(str, fakes.BLUE_SENTINELS["blue_hidden_base"]) not in blob
        # Only the three observable objectives survive under REALISTIC.
        assert realistic.target_ids == frozenset({"TGT-1", "TGT-2", "TGT-3"})

    def test_full_parity_reveals_more_objectives_than_realistic(self) -> None:
        """The deliberate contrast: parity lifts the observation restriction."""

        _, realistic = _brief(IntelPolicy.REALISTIC)
        _, parity = _brief(IntelPolicy.FULL_PARITY)
        assert len(parity.target_ids) > len(realistic.target_ids)

    def test_no_target_carries_coordinates(self) -> None:
        _, brief = _brief(IntelPolicy.REALISTIC)
        for target in brief.targets:
            payload = target.to_dict()
            assert "position" not in payload
            assert "x" not in payload and "y" not in payload
            assert "lat" not in payload and "lon" not in payload


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


class TestBriefIsAccurate:
    def test_schema_version_and_policy_are_stamped(self) -> None:
        _, brief = _brief(IntelPolicy.REALISTIC)
        assert brief.schema_version == OPERATIONS_SCHEMA_VERSION
        assert brief.intel_policy is IntelPolicy.REALISTIC

    def test_it_lists_reds_own_bases_and_squadrons(self) -> None:
        _, brief = _brief()
        assert brief.base_ids == frozenset({"BASE-1", "BASE-2"})
        assert brief.squadron_ids == frozenset({"SQN-1", "SQN-2"})

    def test_runway_repairability_matches_the_campaign(self) -> None:
        _, brief = _brief()
        front = brief.base("BASE-1")
        rear = brief.base("BASE-2")
        assert front is not None and rear is not None
        # The front base's runway works and is not repairable; the rear base's
        # is down and can be repaired. Legality relies on this being right.
        assert front.runway_operational and not front.runway_repairable
        assert not rear.runway_operational and rear.runway_repairable

    def test_squadron_home_bases_and_types_match(self) -> None:
        _, brief = _brief()
        sqn1 = brief.squadron("SQN-1")
        sqn2 = brief.squadron("SQN-2")
        assert sqn1 is not None and sqn1.base_id == "BASE-1"
        assert sqn1.aircraft_id == "RED-JET"
        assert sqn2 is not None and sqn2.base_id == "BASE-2"
        assert sqn2.aircraft_id == "RED-BOMBER"

    def test_targets_expose_only_legal_missions(self) -> None:
        _, brief = _brief()
        air_defence = brief.target("TGT-1")
        airbase = brief.target("TGT-3")
        assert air_defence is not None
        assert set(air_defence.legal_missions) == {"DEAD", "SEAD", "SEAD Sweep"}
        assert air_defence.threatens_own_forces
        assert airbase is not None
        assert set(airbase.legal_missions) == {"OCA/Aircraft", "OCA/Runway"}

    def test_plannable_mission_types_are_advertised(self) -> None:
        _, brief = _brief()
        assert "SEAD" in brief.plannable_mission_types
        assert "Strike" in brief.plannable_mission_types

    def test_content_hash_is_stable(self) -> None:
        _, first = _brief()
        _, second = _brief()
        assert first.content_hash() == second.content_hash()


class TestSquadronReplacementHeadroom:
    """``can_add=N`` tells the model how many airframes a squadron can really buy at
    its home base right now: ``min(free parking, squadron capacity headroom)``,
    clamped to ``>= 0``. It is RED's own data, so surfacing it is not cheating.
    """

    def test_can_add_is_rendered_and_reflects_parking(self) -> None:
        _, brief = _brief()
        sqn1 = brief.squadron("SQN-1")
        sqn2 = brief.squadron("SQN-2")
        assert sqn1 is not None and sqn2 is not None
        # SQN-1 sits at the front base (6 free parking) with ample capacity left
        # (max_size 24, 8 on hand + 1 on order), so parking is the binding limit.
        assert sqn1.can_add == 6
        assert "can_add=6" in sqn1.render()
        # SQN-2 sits at the rear base (20 free parking) with 20 of capacity left.
        assert sqn2.can_add == 20
        assert "can_add=20" in sqn2.render()

    def test_squadron_capacity_is_the_binding_limit_when_smaller(self) -> None:
        campaign, game = fakes.synthetic_game()
        projector = OperationsProjector(game, IntelPolicy.REALISTIC)
        squadron = campaign.red_squadron_1
        # Plenty of parking, but the squadron is nearly at its own max size:
        # 10 - 8 owned - 1 on order = 1 slot of capacity left.
        squadron.location.unclaimed_parking.return_value = 12
        squadron.max_size = 10
        assert projector._squadron_can_add(squadron) == 1

    def test_can_add_is_zero_when_the_home_base_is_full(self) -> None:
        campaign, game = fakes.synthetic_game()
        projector = OperationsProjector(game, IntelPolicy.REALISTIC)
        squadron = campaign.red_squadron_2
        squadron.location.unclaimed_parking.return_value = 0
        assert projector._squadron_can_add(squadron) == 0

    def test_a_full_base_renders_can_add_zero(self) -> None:
        view = SquadronView(
            id="SQN-1",
            name="1st Fighter",
            aircraft_id="MiG-29",
            base_id="BASE-1",
            base_name="Anapa",
            aircraft_on_hand=8,
            aircraft_untasked=2,
            aircraft_on_order=0,
            pilots_available=4,
            pilot_limit_enabled=False,
            max_fulfillable_aircraft=2,
            price_per_aircraft=10,
            relocating_to_base_id=None,
            capable_tasks=("CAP",),
            auto_assignable_tasks=("CAP",),
            can_add=0,
        )
        assert "can_add=0" in view.render()


class TestGroundInventoryIsListedInFull:
    """The per-base ground breakdown must show the commander every type a base
    really holds. Listing only the top few taught the model to invent inventory
    and guess source bases, so the enumeration cap is deliberately generous;
    these are RED's OWN units at RED's OWN bases, so full disclosure is no leak.
    """

    def _base(
        self, ground_by_type: tuple[tuple[str, int], ...], truncated: int
    ) -> BaseView:
        return BaseView(
            id="BASE-1",
            name="Forward Depot",
            kind="airbase",
            is_front_line_base=True,
            runway_operational=True,
            runway_repairable=False,
            runway_repair_turns_remaining=None,
            aircraft_present=0,
            aircraft_on_order=0,
            parking_free=None,
            ground_units_present=sum(count for _, count in ground_by_type),
            ground_units_on_order=0,
            ground_units_by_type=ground_by_type,
            ground_types_truncated=truncated,
            can_recruit_ground_units=True,
            has_ground_unit_source=True,
            squadron_ids=(),
        )

    def _project(
        self, ranked: list[tuple[str, int]]
    ) -> tuple[tuple[tuple[str, int], ...], int]:
        # Mirror the projector's slice/summary so the test tracks real behaviour.
        listed = tuple(ranked[:MAX_ENUMERATED_GROUND_TYPES])
        truncated = max(0, len(ranked) - MAX_ENUMERATED_GROUND_TYPES)
        return listed, truncated

    def test_cap_is_generous_enough_for_a_realistic_armoury(self) -> None:
        # A regression that dropped the cap back to a handful of types would
        # reintroduce the invented-inventory behaviour this fix removes.
        assert MAX_ENUMERATED_GROUND_TYPES >= 16

    def test_every_type_of_a_ten_type_base_is_listed(self) -> None:
        ranked = [(f"RED-UNIT-{i:02d}", 20 - i) for i in range(10)]
        listed, truncated = self._project(ranked)
        base = self._base(listed, truncated)
        rendered = base.render()
        # All ten distinct types are named in full, with no summarised remainder.
        for name, count in ranked:
            assert f"{name} x{count}" in rendered
        assert "further types" not in rendered

    def test_a_pathological_base_still_summarises_the_overflow(self) -> None:
        # A base with more variety than the cap keeps the "(+N further types)"
        # guard so a single freak base cannot bloat the prompt without bound.
        ranked = [
            (f"RED-UNIT-{i:02d}", 100 - i)
            for i in range(MAX_ENUMERATED_GROUND_TYPES + 3)
        ]
        listed, truncated = self._project(ranked)
        base = self._base(listed, truncated)
        rendered = base.render()
        assert truncated == 3
        assert "(+3 further types)" in rendered


class TestFlyableCapacity:
    """``render_flyable_capacity`` tells the air tasking stage how many airframes
    of each type RED can actually crew and launch this turn, so it sizes packages
    the mission planner can build. It aggregates RED's own squadrons only."""

    def test_flyable_is_untasked_airframes_per_type(self) -> None:
        _, brief = _brief()
        rendered = brief.render_flyable_capacity()
        # SQN-1 has 6 untasked RED-JET, SQN-2 has 4 untasked RED-BOMBER, pilot
        # limits off in the synthetic campaign so untasked airframes are the cap.
        assert "RED-JET: 6" in rendered
        assert "RED-BOMBER: 4" in rendered
        assert "total flyable: 10" in rendered
        assert "[FLYABLE THIS TURN]" in rendered

    def test_pilot_limit_caps_flyable_at_available_pilots(self) -> None:
        # When squadron pilot limits are enabled the flyable count is the smaller
        # of untasked airframes and available pilots -- the real crew limit.
        view = SquadronView(
            id="SQN-1",
            name="1st Fighter",
            aircraft_id="MiG-29",
            base_id="BASE-1",
            base_name="Anapa",
            aircraft_on_hand=10,
            aircraft_untasked=8,
            aircraft_on_order=0,
            pilots_available=3,
            pilot_limit_enabled=True,
            max_fulfillable_aircraft=3,
            price_per_aircraft=10,
            relocating_to_base_id=None,
            capable_tasks=("CAP",),
            auto_assignable_tasks=("CAP",),
        )
        brief = OperationsBrief(
            schema_version=OPERATIONS_SCHEMA_VERSION,
            campaign_id_hash="h",
            campaign_revision="r",
            turn_id=1,
            intel_policy=IntelPolicy.REALISTIC,
            budget_available=0.0,
            runway_repair_cost=0,
            bases=(),
            squadrons=(view,),
            targets=(),
            plannable_mission_types=(),
        )
        rendered = brief.render_flyable_capacity()
        assert "MiG-29: 3" in rendered
        assert "total flyable: 3" in rendered

    def test_no_blue_leak_in_capacity_render(self) -> None:
        _, brief = _brief()
        assert fakes.blue_leaks_in(brief.render_flyable_capacity()) == []


class TestTargetReachability:
    """A target is strikeable only where some RED squadron can both fly one of
    its legal missions and reach it within mission range -- the same crew-and-
    route test the planner enforces. ``reachable_by`` and the reachability render
    expose that, using RED's own aircraft and ranges only."""

    def test_in_range_capable_targets_are_marked_reachable(self) -> None:
        _, brief = _brief()
        # The two near air-defence targets can be struck by the fighter (SEAD)
        # and the bomber (DEAD), both well within range.
        tgt1 = brief.target("TGT-1")
        tgt2 = brief.target("TGT-2")
        assert tgt1 is not None and tgt1.reachable_by == ("RED-BOMBER", "RED-JET")
        assert tgt2 is not None and tgt2.is_reachable
        # The enemy airbase needs OCA, which no RED squadron can fly, so it is
        # not strikeable this turn.
        tgt3 = brief.target("TGT-3")
        assert tgt3 is not None and tgt3.reachable_by == ()
        assert not tgt3.is_reachable

    def test_reachability_render_lists_reachable_and_unreachable(self) -> None:
        _, brief = _brief()
        rendered = brief.render_target_reachability()
        assert "[TARGET REACHABILITY]" in rendered
        assert "TGT-1(RED-BOMBER,RED-JET)" in rendered
        assert "TGT-3" in rendered
        assert "not strikeable" in rendered

    def test_out_of_range_squadron_is_excluded(self) -> None:
        # Under FULL_PARITY the distant IADS becomes visible. It is within the
        # bomber's 900nm reach but outside the fighter's 400nm, so only the
        # bomber may be offered -- proof the range gate actually filters.
        _, parity = _brief(IntelPolicy.FULL_PARITY)
        air_defence = [
            target
            for target in parity.targets
            if target.category is TargetSetCategory.ENEMY_AIR_DEFENCES
        ]
        reach_sets = {target.reachable_by for target in air_defence}
        assert ("RED-BOMBER",) in reach_sets  # the far site: bomber only
        assert ("RED-BOMBER", "RED-JET") in reach_sets  # the near sites: both

    def test_empty_when_no_targets(self) -> None:
        brief = OperationsBrief(
            schema_version=OPERATIONS_SCHEMA_VERSION,
            campaign_id_hash="h",
            campaign_revision="r",
            turn_id=1,
            intel_policy=IntelPolicy.REALISTIC,
            budget_available=0.0,
            runway_repair_cost=0,
            bases=(),
            squadrons=(),
            targets=(),
            plannable_mission_types=(),
        )
        assert brief.render_target_reachability() == ""

    def test_no_blue_leak_in_reachability(self) -> None:
        _, brief = _brief(IntelPolicy.FULL_PARITY)
        assert fakes.blue_leaks_in(brief.render_target_reachability()) == []


class TestShipTargetsOfferOnlyAntiShip:
    """A carrier or amphibious ship is an enemy control point, so the objective
    finder lists it among the enemy airbases. Offering it OCA/Aircraft the way a
    land airbase is offered gets the package refused at execution ("... is not
    valid for OCA/Aircraft missions"), so a ship control point must be projected
    as shipping, attackable only with Anti-ship.
    """

    def _carrier_brief(
        self, monkeypatch: pytest.MonkeyPatch, **ship_flags: bool
    ) -> tuple[fakes.SyntheticCampaign, OperationsBrief]:
        fakes.patch_objective_finder(monkeypatch)
        campaign, game = fakes.synthetic_game()
        carrier = fakes.make_control_point(
            cp_id=99,
            name="CVN-75 Harry S. Truman",
            captured=Player.BLUE,
            position=fakes.point(35_000.0, 0.0),
        )
        for flag, value in ship_flags.items():
            setattr(carrier, flag, value)
        game.theater.controlpoints.append(carrier)
        brief = OperationsProjector(game, IntelPolicy.FULL_PARITY).project(
            "campaign-hash", "rev-1"
        )
        return campaign, brief

    def _carrier_target(self, brief: OperationsBrief) -> TargetView:
        for target_id in sorted(brief.target_ids):
            target = brief.target(target_id)
            if target is not None and "CVN-75" in target.label:
                return target
        raise AssertionError("the carrier was not projected as a target")

    def test_a_carrier_is_projected_as_shipping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, brief = self._carrier_brief(monkeypatch, is_carrier=True)
        carrier = self._carrier_target(brief)
        assert carrier.category is TargetSetCategory.ENEMY_SHIPPING
        assert set(carrier.legal_missions) == {"Anti-ship"}
        assert "OCA/Aircraft" not in carrier.legal_missions

    def test_an_lha_is_also_projected_as_shipping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, brief = self._carrier_brief(monkeypatch, is_lha=True)
        assert self._carrier_target(brief).category is (
            TargetSetCategory.ENEMY_SHIPPING
        )

    def test_a_land_airbase_still_offers_oca(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The reclassification must not touch ordinary land airbases: an enemy
        # control point that is not a ship keeps its OCA missions.
        _, brief = self._carrier_brief(monkeypatch)
        carrier = self._carrier_target(brief)
        assert carrier.category is TargetSetCategory.ENEMY_AIRBASES
        assert set(carrier.legal_missions) == {"OCA/Aircraft", "OCA/Runway"}
