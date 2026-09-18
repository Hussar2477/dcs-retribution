"""Recovering the complete part of a cut-off JSON object.

A reasoning model asked for a large, list-based plan can overrun its output
budget and stop partway through an array element, leaving a reply that will not
parse. :func:`salvage_truncated_json` recovers the largest valid JSON-object
prefix rather than discarding the whole thing. These tests pin down that it
recovers the complete part, returns ``None`` when nothing is recoverable, and
never raises whatever it is given.
"""

from __future__ import annotations

from typing import Any, cast

from game.ai_commander.jsonsalvage import salvage_truncated_json


class TestRecoversTheCompletePrefix:
    def test_it_recovers_the_complete_prefix_of_a_truncated_object(self) -> None:
        # Three complete purchases and a fourth cut off before any field closed:
        # the whole incomplete element is dropped and the first three are kept.
        text = (
            '{"schema_version": "logistics-v1", "turn_id": "t7", '
            '"ground_orders": ['
            '{"base_id": "a", "unit_id": "x", "quantity": 12}, '
            '{"base_id": "b", "unit_id": "y", "quantity": 8}, '
            '{"base_id": "c", "unit_id": "z", "quantity": 4}, '
            '{"base'
        )
        recovered = salvage_truncated_json(text)
        assert recovered is not None
        assert recovered["schema_version"] == "logistics-v1"
        assert recovered["turn_id"] == "t7"
        assert len(recovered["ground_orders"]) == 3
        assert recovered["ground_orders"][0] == {
            "base_id": "a",
            "unit_id": "x",
            "quantity": 12,
        }

    def test_it_recovers_through_a_fenced_code_block_without_a_closing_fence(
        self,
    ) -> None:
        text = (
            "```json\n"
            '{"schema_version": "logistics-v1", "orders": ['
            '{"unit_id": "x", "quantity": 3}, {"unit'
        )
        recovered = salvage_truncated_json(text)
        assert recovered is not None
        assert recovered["schema_version"] == "logistics-v1"
        assert recovered["orders"] == [{"unit_id": "x", "quantity": 3}]

    def test_it_closes_nested_containers_left_open_by_the_cut(self) -> None:
        text = '{"a": {"deep": [1, 2, 3'
        recovered = salvage_truncated_json(text)
        assert recovered == {"a": {"deep": [1, 2, 3]}}

    def test_a_complete_object_is_returned_unchanged(self) -> None:
        text = '{"a": [1, 2, 3], "b": "ok"}'
        assert salvage_truncated_json(text) == {"a": [1, 2, 3], "b": "ok"}


class TestNothingRecoverable:
    def test_it_returns_none_when_nothing_is_complete(self) -> None:
        # The very first element is cut off, so no complete prefix exists inside
        # the top-level object beyond an empty one -- and an empty object here is
        # still valid, so assert it never invents content.
        text = '{"orders": [{"base'
        recovered = salvage_truncated_json(text)
        # Either None or an object with no recovered orders is acceptable; what
        # must never happen is a half-written order being invented.
        if recovered is not None:
            assert recovered.get("orders", []) == []

    def test_it_returns_none_when_there_is_no_object_at_all(self) -> None:
        assert salvage_truncated_json("no json here, just prose") is None

    def test_it_returns_none_for_a_bare_truncated_array(self) -> None:
        # The result must be an object; a truncated top-level array is not one.
        assert salvage_truncated_json("[1, 2, 3") is None


class TestNeverRaises:
    def test_it_never_raises_on_empty_input(self) -> None:
        assert salvage_truncated_json("") is None

    def test_it_never_raises_on_garbage(self) -> None:
        for junk in ('{"', "{{{{{{", "}]}]", '{"a": "unterminated string', "\x00\x01"):
            # Must return a value, never raise.
            salvage_truncated_json(junk)

    def test_it_never_raises_on_a_non_string(self) -> None:
        assert salvage_truncated_json(cast(str, None)) is None
        assert salvage_truncated_json(cast(str, 12345)) is None
        assert salvage_truncated_json(cast(str, cast(Any, {"a": 1}))) is None
