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
from game.ai_commander.enums import IntelPolicy
from game.ai_commander.operations import OperationsBrief, OperationsProjector
from game.ai_commander.plan import (
    MAX_QUANTITY_PER_ORDER,
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

    def test_an_illegal_mission_for_the_objective_is_rejected(self) -> None:
        """TGT-1 is an air-defence site; a strike is not one of its legal missions."""

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
