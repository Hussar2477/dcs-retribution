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

from game.ai_commander.enums import IntelPolicy
from game.ai_commander.operations import (
    OPERATIONS_SCHEMA_VERSION,
    OperationsBrief,
    OperationsProjector,
)
from tests.ai_commander import fakes


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
