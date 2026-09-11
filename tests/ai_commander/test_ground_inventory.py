"""The per-base ground inventory must be broken down by unit type.

A live turn showed the model trying to transfer unit types a base did not hold
(inventory it invented), because the brief only reported a single ground total.
Surfacing the present stock by unit type -- truncated to the top few, with the
remainder summarised -- gives the model what it needs to move only real units.
These tests pin the rendering and its truncation down.
"""

from __future__ import annotations

from game.ai_commander.operations import BaseView


def _base(
    ground_units_by_type: tuple[tuple[str, int], ...],
    ground_types_truncated: int = 0,
) -> BaseView:
    return BaseView(
        id="BASE-1",
        name="Anapa",
        kind="airfield",
        is_front_line_base=False,
        runway_operational=True,
        runway_repairable=True,
        runway_repair_turns_remaining=None,
        aircraft_present=12,
        aircraft_on_order=0,
        parking_free=None,
        ground_units_present=sum(c for _, c in ground_units_by_type),
        ground_units_on_order=0,
        ground_units_by_type=ground_units_by_type,
        ground_types_truncated=ground_types_truncated,
        can_recruit_ground_units=True,
        has_ground_unit_source=True,
        squadron_ids=(),
    )


class TestGroundTypesRendering:
    def test_present_types_are_listed_with_counts(self) -> None:
        rendered = _base((("T-72B", 8), ("BMP-2", 4))).render()
        assert "ground_types: T-72B x8, BMP-2 x4" in rendered

    def test_truncation_is_summarised(self) -> None:
        rendered = _base(
            (("T-72B", 8), ("BMP-2", 4)), ground_types_truncated=3
        ).render()
        assert "(+3 further types)" in rendered

    def test_no_truncation_note_when_nothing_omitted(self) -> None:
        rendered = _base((("T-72B", 8),)).render()
        assert "(+" not in rendered

    def test_empty_inventory_omits_the_ground_types_field(self) -> None:
        rendered = _base(()).render()
        assert "ground_types" not in rendered
