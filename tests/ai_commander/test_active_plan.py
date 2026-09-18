"""Structural validation of ACTIVE mode's staged plans.

These tests exercise :func:`validate_logistics_plan` and
:func:`validate_air_tasking_plan` -- the first gate every stage-2 and stage-3
response passes through, *before* anything is checked against live campaign
state. Validation's job is narrow but critical: reject anything that is not
shaped like a plan, that references an identifier the brief never offered, or
that names an airframe/mission the faction cannot fly. It is also where the
model's most direct cheat attempts die -- an "attack" on a target that was
never in the brief, or an order for a BLUE airframe -- so those get explicit
coverage here.

Validation is deliberately lenient about *partly* bad plans: one malformed
order is dropped with a rejection and the rest are kept, so a single
hallucinated line never costs the whole stage. The partial-acceptance tests
pin that behaviour down.
"""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from game.ai_commander.capabilities import (
    CAPABILITY_CACHE,
    CapabilityIndex,
    capability_index_for,
)
from game.ato.flighttype import FlightType
from game.ai_commander.enums import IntelPolicy, TargetSetCategory
from game.ai_commander.operations import (
    OperationsBrief,
    OperationsProjector,
    TargetView,
)
from game.ai_commander.plan import (
    MAX_QUANTITY_PER_ORDER,
    ProposedFlightOrder,
    _is_time_critical,
    example_air_tasking_json,
    example_logistics_json,
    validate_air_tasking_plan,
    validate_logistics_plan,
)
from tests.ai_commander import fakes


@pytest.fixture(autouse=True)
def _objective_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes.patch_objective_finder(monkeypatch)


@pytest.fixture(autouse=True)
def _fresh_capability_cache() -> Iterator[None]:
    CAPABILITY_CACHE.clear()
    yield
    CAPABILITY_CACHE.clear()


def _context() -> tuple[OperationsBrief, CapabilityIndex]:
    campaign, game = fakes.synthetic_game()
    brief = OperationsProjector(game, IntelPolicy.REALISTIC).project("hash", "rev-1")
    return brief, capability_index_for(campaign.red)


def _reasons(rejections: list[Any]) -> str:
    return " || ".join(r.reason for r in rejections)


# ---------------------------------------------------------------------------
# The worked examples are, by construction, valid
# ---------------------------------------------------------------------------


class TestExamplesValidate:
    def test_the_example_logistics_plan_is_accepted(self) -> None:
        brief, caps = _context()
        plan, rejections = validate_logistics_plan(
            example_logistics_json(brief, caps), brief, caps
        )
        assert plan is not None
        assert rejections == []
        assert plan.has_content

    def test_the_example_air_tasking_plan_is_accepted(self) -> None:
        brief, caps = _context()
        plan, rejections = validate_air_tasking_plan(
            example_air_tasking_json(brief, caps), brief, caps
        )
        assert plan is not None
        assert rejections == []
        assert plan.has_content


# ---------------------------------------------------------------------------
# Envelope: schema, turn and revision have to match the brief
# ---------------------------------------------------------------------------


class TestEnvelopeIsFatal:
    def test_a_non_object_payload_is_rejected_whole(self) -> None:
        brief, caps = _context()
        plan, rejections = validate_logistics_plan([1, 2, 3], brief, caps)
        assert plan is None
        assert "not a JSON object" in _reasons(rejections)

    def test_a_wrong_schema_version_is_fatal(self) -> None:
        brief, caps = _context()
        payload = example_logistics_json(brief, caps)
        payload["schema_version"] = "red-commander-air-tasking/1"
        plan, rejections = validate_logistics_plan(payload, brief, caps)
        assert plan is None
        assert "expected" in _reasons(rejections)

    def test_a_stale_campaign_revision_is_fatal(self) -> None:
        brief, caps = _context()
        payload = example_logistics_json(brief, caps)
        payload["campaign_revision"] = "some-other-revision"
        plan, rejections = validate_logistics_plan(payload, brief, caps)
        assert plan is None
        assert any(r.element == "campaign_revision" for r in rejections)

    def test_a_wrong_turn_id_is_fatal(self) -> None:
        brief, caps = _context()
        payload = example_logistics_json(brief, caps)
        payload["turn_id"] = brief.turn_id + 5
        plan, rejections = validate_logistics_plan(payload, brief, caps)
        assert plan is None
        assert "turn" in _reasons(rejections).lower()


# ---------------------------------------------------------------------------
# Logistics: unknown ids, quantities, unpurchasable units
# ---------------------------------------------------------------------------


class TestLogisticsOrderValidation:
    def _payload(self, brief: OperationsBrief, **orders: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "red-commander-logistics/1",
            "turn_id": brief.turn_id,
            "campaign_revision": brief.campaign_revision,
            "intent": "test",
        }
        payload.update(orders)
        return payload

    def test_an_unknown_squadron_is_dropped(self) -> None:
        brief, caps = _context()
        payload = self._payload(
            brief,
            aircraft_orders=[{"squadron_id": "SQN-GHOST", "quantity": 2}],
        )
        plan, rejections = validate_logistics_plan(payload, brief, caps)
        # The bad order is dropped, not fatal: plan survives with no content.
        assert "is not in the brief" in _reasons(rejections)
        assert plan is not None
        assert not plan.aircraft_orders

    def test_a_zero_quantity_is_rejected(self) -> None:
        brief, caps = _context()
        squadron = sorted(brief.squadron_ids)[0]
        payload = self._payload(
            brief,
            aircraft_orders=[{"squadron_id": squadron, "quantity": 0}],
        )
        plan, rejections = validate_logistics_plan(payload, brief, caps)
        assert "at least 1" in _reasons(rejections)

    def test_a_quantity_over_the_per_order_limit_is_rejected(self) -> None:
        brief, caps = _context()
        squadron = sorted(brief.squadron_ids)[0]
        payload = self._payload(
            brief,
            aircraft_orders=[
                {"squadron_id": squadron, "quantity": MAX_QUANTITY_PER_ORDER + 1}
            ],
        )
        plan, rejections = validate_logistics_plan(payload, brief, caps)
        assert "per-order limit" in _reasons(rejections)

    def test_an_unpurchasable_ground_unit_is_rejected(self) -> None:
        """RED-SAM is fielded but air-defence-only, so it is not in the buy list."""

        brief, caps = _context()
        base = sorted(brief.base_ids)[0]
        payload = self._payload(
            brief,
            ground_orders=[{"base_id": base, "unit_id": "RED-SAM", "quantity": 2}],
        )
        plan, rejections = validate_logistics_plan(payload, brief, caps)
        assert "can purchase" in _reasons(rejections)
        assert plan is not None
        assert not plan.ground_orders

    def test_a_good_order_survives_a_bad_one_in_the_same_plan(self) -> None:
        brief, caps = _context()
        squadron = sorted(brief.squadron_ids)[0]
        payload = self._payload(
            brief,
            aircraft_orders=[
                {"squadron_id": squadron, "quantity": 2},
                {"squadron_id": "SQN-GHOST", "quantity": 2},
            ],
        )
        plan, rejections = validate_logistics_plan(payload, brief, caps)
        assert plan is not None
        assert len(plan.aircraft_orders) == 1
        assert plan.aircraft_orders[0].squadron_id == squadron
        assert _reasons(rejections)  # the ghost was reported


# ---------------------------------------------------------------------------
# Air tasking: the model's most direct cheat attempts
# ---------------------------------------------------------------------------


class TestAirTaskingCheatAttempts:
    def _package_payload(
        self, brief: OperationsBrief, package: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "schema_version": "red-commander-air-tasking/1",
            "turn_id": brief.turn_id,
            "campaign_revision": brief.campaign_revision,
            "intent": "test",
            "packages": [package],
        }

    def test_a_target_not_in_the_brief_is_rejected(self) -> None:
        """The headline cheat: attacking something RED was never shown."""

        brief, caps = _context()
        payload = self._package_payload(
            brief,
            {
                "target_id": "TGT-BLUE-CANARY",
                "priority": 1,
                "flights": [{"mission_type": "SEAD", "aircraft_count": 2}],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert "is not in the brief" in _reasons(rejections)
        assert plan is not None
        assert not plan.packages

    def test_an_airframe_the_faction_does_not_operate_is_rejected(self) -> None:
        """Ordering a BLUE jet by id must not slip through validation."""

        brief, caps = _context()
        payload = self._package_payload(
            brief,
            {
                "target_id": "TGT-1",
                "priority": 1,
                "flights": [
                    {
                        "mission_type": "SEAD",
                        "aircraft_count": 2,
                        "aircraft_id": "F-BLUELEAK-99",
                    }
                ],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert "faction operates" in _reasons(rejections)
        assert plan is not None
        assert not plan.packages

    def test_an_illegal_striker_mission_is_repaired_not_rejected(self) -> None:
        """A striker pointed at the wrong mission is remapped, not thrown away.

        TGT-1 is an air-defence site (legal missions DEAD/SEAD/SEAD Sweep); a
        Strike is not one of them. Rather than reject the package -- a mistake
        the model makes repeatedly -- the flight is silently remapped to the
        target's primary legal striking mission (DEAD), so the correctly-intended
        order is kept. No rejection is recorded, mirroring the ground-transfer
        auto-merge: the accepted plan is the audit trail.
        """

        brief, caps = _context()
        payload = self._package_payload(
            brief,
            {
                "target_id": "TGT-1",
                "priority": 1,
                "flights": [{"mission_type": "Strike", "aircraft_count": 2}],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert not rejections
        assert plan is not None
        assert len(plan.packages) == 1
        flights = plan.packages[0].flights
        assert len(flights) == 1
        # Remapped to a mission that is actually legal for the objective.
        assert flights[0].mission_type.value in ("DEAD", "SEAD", "SEAD Sweep")

    def test_an_escort_led_package_with_an_illegal_striker_is_repaired(self) -> None:
        """The two repairs combine: reorder the lead *and* remap the striker.

        The model lists an escort first and gives its striker the wrong mission
        for the target. Auto-repair promotes the striker to lead the package and
        remaps its illegal mission to a legal one, keeping the escort as a
        supporting flight. Nothing is rejected.
        """

        brief, caps = _context()
        payload = self._package_payload(
            brief,
            {
                "target_id": "TGT-1",
                "priority": 1,
                "flights": [
                    {"mission_type": "Escort", "aircraft_count": 2},
                    {"mission_type": "Strike", "aircraft_count": 2},
                ],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert not rejections
        assert plan is not None
        assert len(plan.packages) == 1
        flights = plan.packages[0].flights
        assert len(flights) == 2
        # The striker now leads with a legal mission; the escort follows.
        assert flights[0].mission_type.value in ("DEAD", "SEAD", "SEAD Sweep")
        assert flights[1].mission_type.value == "Escort"

    def test_an_airbase_strike_on_a_carrier_is_remapped_to_anti_ship(self) -> None:
        """A carrier is projected as shipping, so OCA is repaired to Anti-ship.

        The model treats a carrier like a land airbase and orders OCA/Aircraft
        against it. Because the carrier is projected as a shipping target whose
        only legal mission is Anti-ship, the illegal-mission repair silently
        remaps the flight to Anti-ship rather than letting the package die at
        execution ("... is not valid for OCA/Aircraft missions").
        """

        campaign, game = fakes.synthetic_game()
        carrier = fakes.make_control_point(
            cp_id=99,
            name="CVN-75 Harry S. Truman",
            captured=fakes.Player.BLUE,
            position=fakes.point(35_000.0, 0.0),
        )
        setattr(carrier, "is_carrier", True)
        game.theater.controlpoints.append(carrier)
        brief = OperationsProjector(game, IntelPolicy.FULL_PARITY).project(
            "hash", "rev-1"
        )
        caps = capability_index_for(campaign.red)
        carrier_id = next(
            target_id
            for target_id in sorted(brief.target_ids)
            if (target := brief.target(target_id)) is not None
            and "CVN-75" in target.label
        )
        payload = self._package_payload(
            brief,
            {
                "target_id": carrier_id,
                "priority": 1,
                "flights": [{"mission_type": "OCA/Aircraft", "aircraft_count": 2}],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert not rejections
        assert plan is not None
        assert len(plan.packages) == 1
        flights = plan.packages[0].flights
        assert len(flights) == 1
        assert flights[0].mission_type.value == "Anti-ship"

    def test_a_package_of_only_escorts_is_rejected(self) -> None:
        """Auto-repair cannot invent a striker; a package of only escorts dies.

        This is the hard limit of the reorder repair: with no strike or attack
        flight to lead the package, there is nothing to escort, so the package is
        genuinely rejected rather than silently repaired.
        """

        brief, caps = _context()
        payload = self._package_payload(
            brief,
            {
                "target_id": "TGT-1",
                "priority": 1,
                "flights": [{"mission_type": "Escort", "aircraft_count": 2}],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert "only escorts were provided" in _reasons(rejections)
        assert plan is not None
        assert not plan.packages

    def test_an_area_mission_as_a_package_flight_is_rejected(self) -> None:
        """Area missions patrol a slice of sky; they can never join a package.

        Unlike a mistargeted striker, an area mission (BARCAP/TARCAP/Fighter
        sweep) is not repaired into a strike -- it is rejected outright, because
        it is not a strike against the objective at all.
        """

        brief, caps = _context()
        payload = self._package_payload(
            brief,
            {
                "target_id": "TGT-1",
                "priority": 1,
                "flights": [{"mission_type": "BARCAP", "aircraft_count": 2}],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert "cannot be flown against this objective" in _reasons(rejections)
        assert plan is not None
        assert not plan.packages

    def test_a_flight_larger_than_the_airframe_maximum_is_rejected(self) -> None:
        brief, caps = _context()
        payload = self._package_payload(
            brief,
            {
                "target_id": "TGT-1",
                "priority": 1,
                "flights": [
                    {
                        "mission_type": "DEAD",
                        "aircraft_count": 4,
                        "aircraft_id": "RED-BOMBER",
                    }
                ],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        # The bomber flies DEAD, but its maximum group size is 2; a flight of 4
        # is impossible even for a mission it is qualified for.
        assert "maximum group size" in _reasons(rejections)

    def test_a_package_with_no_flights_is_rejected(self) -> None:
        brief, caps = _context()
        payload = self._package_payload(
            brief, {"target_id": "TGT-1", "priority": 1, "flights": []}
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert "at least one flight" in _reasons(rejections)

    def test_a_valid_package_survives_a_cheating_one(self) -> None:
        brief, caps = _context()
        payload = {
            "schema_version": "red-commander-air-tasking/1",
            "turn_id": brief.turn_id,
            "campaign_revision": brief.campaign_revision,
            "intent": "test",
            "packages": [
                {
                    "target_id": "TGT-1",
                    "priority": 1,
                    "flights": [{"mission_type": "SEAD", "aircraft_count": 2}],
                },
                {
                    "target_id": "TGT-BLUE-CANARY",
                    "priority": 2,
                    "flights": [{"mission_type": "SEAD", "aircraft_count": 2}],
                },
            ],
        }
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert plan is not None
        assert len(plan.packages) == 1
        assert plan.packages[0].target_id == "TGT-1"
        assert "is not in the brief" in _reasons(rejections)


class TestTimeCriticalPackagesLaunchAsap:
    """Front-line and defensive packages must launch before the ground battle is
    decided. The model repeatedly leaves such packages un-flagged, so ``asap`` is
    forced on for them in the validator regardless of what the model wrote.
    """

    def _payload(
        self, brief: OperationsBrief, package: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "schema_version": "red-commander-air-tasking/1",
            "turn_id": brief.turn_id,
            "campaign_revision": brief.campaign_revision,
            "intent": "test",
            "packages": [package],
        }

    def test_a_defensive_mission_is_forced_asap(self) -> None:
        # TGT-1 is an air-defence site; a DEAD package against it is defensive,
        # so it is launched as early as possible even though asap was left false.
        brief, caps = _context()
        payload = self._payload(
            brief,
            {
                "target_id": "TGT-1",
                "priority": 1,
                "asap": False,
                "flights": [{"mission_type": "DEAD", "aircraft_count": 2}],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert not rejections
        assert plan is not None
        assert plan.packages[0].asap is True

    def test_a_non_time_critical_package_keeps_its_flag(self) -> None:
        # TGT-3 is a land airbase; an OCA strike on it is not front-line or
        # defensive, so the model's own asap:false choice is respected.
        brief, caps = _context()
        payload = self._payload(
            brief,
            {
                "target_id": "TGT-3",
                "priority": 1,
                "asap": False,
                "flights": [{"mission_type": "OCA/Aircraft", "aircraft_count": 2}],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert not rejections
        assert plan is not None
        assert plan.packages[0].asap is False

    def test_an_explicit_asap_is_never_downgraded(self) -> None:
        # A model that does flag an offensive package keeps its asap:true.
        brief, caps = _context()
        payload = self._payload(
            brief,
            {
                "target_id": "TGT-3",
                "priority": 1,
                "asap": True,
                "flights": [{"mission_type": "OCA/Aircraft", "aircraft_count": 2}],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert plan is not None
        assert plan.packages[0].asap is True

    def test_a_front_target_is_time_critical_for_any_mission(self) -> None:
        # A strike against reinforcements is time-critical whatever mission flies
        # it: the convoy is only worth hitting before it reaches the front.
        target = TargetView(
            id="TGT-9",
            category=TargetSetCategory.ENEMY_REINFORCEMENTS,
            label="road convoy",
            near="BASE-1",
            threatens_own_forces=True,
            legal_missions=("BAI",),
        )
        strike = ProposedFlightOrder(mission_type=FlightType.STRIKE, aircraft_count=2)
        assert _is_time_critical(target, [strike]) is True

    def test_a_rear_strike_is_not_time_critical(self) -> None:
        target = TargetView(
            id="TGT-9",
            category=TargetSetCategory.ENEMY_INFRASTRUCTURE,
            label="oil depot",
            near="BASE-1",
            threatens_own_forces=False,
            legal_missions=("Strike",),
        )
        strike = ProposedFlightOrder(mission_type=FlightType.STRIKE, aircraft_count=2)
        assert _is_time_critical(target, [strike]) is False


class TestIngressAltitudeBands:
    """A flight may carry an optional ingress band biasing its run altitude.

    The band is the one altitude lever the commander is given. A recognised
    value is threaded through to the flight order; an unrecognised one is the
    most benign kind of mistake -- it only ever affected altitude within the
    doctrine clamp -- so it is silently dropped to the default profile rather
    than costing the flight a rejection.
    """

    def _payload(
        self, brief: OperationsBrief, package: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "schema_version": "red-commander-air-tasking/1",
            "turn_id": brief.turn_id,
            "campaign_revision": brief.campaign_revision,
            "intent": "test",
            "packages": [package],
        }

    def test_a_recognised_ingress_band_is_carried_through(self) -> None:
        brief, caps = _context()
        payload = self._payload(
            brief,
            {
                "target_id": "TGT-1",
                "priority": 1,
                "flights": [
                    {"mission_type": "DEAD", "aircraft_count": 2, "ingress": "low"}
                ],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert not rejections
        assert plan is not None
        assert plan.packages[0].flights[0].ingress == "low"

    def test_an_unrecognised_ingress_band_is_dropped_silently(self) -> None:
        brief, caps = _context()
        payload = self._payload(
            brief,
            {
                "target_id": "TGT-1",
                "priority": 1,
                "flights": [
                    {
                        "mission_type": "DEAD",
                        "aircraft_count": 2,
                        "ingress": "stratospheric",
                    }
                ],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        # No rejection: the good flight survives with the default (None) profile.
        assert not rejections
        assert plan is not None
        assert plan.packages[0].flights[0].ingress is None

    def test_a_missing_ingress_band_defaults_to_none(self) -> None:
        brief, caps = _context()
        payload = self._payload(
            brief,
            {
                "target_id": "TGT-1",
                "priority": 1,
                "flights": [{"mission_type": "DEAD", "aircraft_count": 2}],
            },
        )
        plan, rejections = validate_air_tasking_plan(payload, brief, caps)
        assert plan is not None
        assert plan.packages[0].flights[0].ingress is None

    def test_the_ingress_band_appears_in_the_schema(self) -> None:
        from game.ai_commander.plan import INGRESS_BANDS, air_tasking_json_schema

        brief, caps = _context()
        schema = air_tasking_json_schema(brief, caps)
        flight_schema = schema["properties"]["packages"]["items"]["properties"][
            "flights"
        ]["items"]["properties"]
        assert flight_schema["ingress"]["enum"] == list(INGRESS_BANDS)
