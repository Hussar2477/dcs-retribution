"""Duplicate-destination ground transfers are merged, not rejected.

A live turn (turn 19) showed the model emitting ~31 separate transfer objects,
nearly all bound for the same destination base. The old validator kept a set of
(origin, destination) pairs and rejected every repeat with "duplicate
identifier", which refused 30 transfers and starved the front. The validator now
MERGES transfer objects that share an (origin, destination): quantities for the
same unit are summed, distinct unit types are concatenated, and both the
per-order quantity cap and the per-transfer distinct-type cap are respected by
capping rather than rejecting. These tests pin that behaviour down.
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


def _payload(brief: OperationsBrief, transfers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "red-commander-logistics/1",
        "turn_id": brief.turn_id,
        "campaign_revision": brief.campaign_revision,
        "intent": "reinforce the front",
        "ground_transfers": transfers,
    }


class TestGroundTransferMerge:
    def test_two_transfers_to_the_same_base_are_merged(self) -> None:
        brief, caps = _context()
        origin, destination = "BASE-1", "BASE-2"
        payload = _payload(
            brief,
            [
                {
                    "origin_base_id": origin,
                    "destination_base_id": destination,
                    "units": [{"unit_id": "RED-TANK", "quantity": 5}],
                },
                {
                    "origin_base_id": origin,
                    "destination_base_id": destination,
                    "units": [
                        {"unit_id": "RED-TANK", "quantity": 3},
                        {"unit_id": "RED-ARTY", "quantity": 4},
                    ],
                },
            ],
        )
        plan, rejections = validate_logistics_plan(payload, brief, caps)
        assert plan is not None
        # No "duplicate identifier" rejection is ever emitted any more.
        assert "duplicate identifier" not in _reasons(rejections)
        # The two objects collapse into a single order for that destination.
        assert len(plan.ground_transfers) == 1
        order = plan.ground_transfers[0]
        assert order.origin_base_id == origin
        assert order.destination_base_id == destination
        # RED-TANK quantities are summed (5 + 3); RED-ARTY is concatenated.
        assert dict(order.units) == {"RED-TANK": 8, "RED-ARTY": 4}
        # First-seen unit order is preserved.
        assert [unit for unit, _ in order.units] == ["RED-TANK", "RED-ARTY"]

    def test_no_duplicate_identifier_rejection_for_transfers(self) -> None:
        brief, caps = _context()
        payload = _payload(
            brief,
            [
                {
                    "origin_base_id": "BASE-1",
                    "destination_base_id": "BASE-2",
                    "units": [{"unit_id": "RED-TANK", "quantity": 2}],
                }
            ]
            * 4,
        )
        plan, rejections = validate_logistics_plan(payload, brief, caps)
        assert plan is not None
        assert rejections == []
        assert len(plan.ground_transfers) == 1
        # Four identical objects of 2 tanks each sum to 8.
        assert dict(plan.ground_transfers[0].units) == {"RED-TANK": 8}

    def test_summed_quantity_is_capped_not_rejected(self) -> None:
        brief, caps = _context()
        payload = _payload(
            brief,
            [
                {
                    "origin_base_id": "BASE-1",
                    "destination_base_id": "BASE-2",
                    "units": [{"unit_id": "RED-TANK", "quantity": 20}],
                },
                {
                    "origin_base_id": "BASE-1",
                    "destination_base_id": "BASE-2",
                    "units": [{"unit_id": "RED-TANK", "quantity": 20}],
                },
            ],
        )
        plan, rejections = validate_logistics_plan(payload, brief, caps)
        assert plan is not None
        assert "duplicate identifier" not in _reasons(rejections)
        assert len(plan.ground_transfers) == 1
        # 20 + 20 would be 40; the per-order cap holds it at MAX_QUANTITY_PER_ORDER.
        assert dict(plan.ground_transfers[0].units) == {
            "RED-TANK": MAX_QUANTITY_PER_ORDER
        }

    def test_different_destinations_stay_separate_and_ordered(self) -> None:
        brief, caps = _context()
        # First-seen destination order (BASE-2 then BASE-1) must be preserved.
        payload = _payload(
            brief,
            [
                {
                    "origin_base_id": "BASE-1",
                    "destination_base_id": "BASE-2",
                    "units": [{"unit_id": "RED-TANK", "quantity": 2}],
                },
                {
                    "origin_base_id": "BASE-2",
                    "destination_base_id": "BASE-1",
                    "units": [{"unit_id": "RED-ARTY", "quantity": 2}],
                },
            ],
        )
        plan, rejections = validate_logistics_plan(payload, brief, caps)
        assert plan is not None
        assert rejections == []
        assert [o.destination_base_id for o in plan.ground_transfers] == [
            "BASE-2",
            "BASE-1",
        ]

    def test_same_origin_is_still_rejected_hard(self) -> None:
        brief, caps = _context()
        payload = _payload(
            brief,
            [
                {
                    "origin_base_id": "BASE-1",
                    "destination_base_id": "BASE-1",
                    "units": [{"unit_id": "RED-TANK", "quantity": 2}],
                }
            ],
        )
        plan, rejections = validate_logistics_plan(payload, brief, caps)
        assert plan is not None
        assert not plan.ground_transfers
        assert "origin and destination are the same" in _reasons(rejections)
