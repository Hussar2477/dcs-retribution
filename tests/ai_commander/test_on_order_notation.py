"""On-order counts must never read like a turn-over-turn delta.

The reasoning model looped catastrophically when the own-force lines rendered
procurement as ``aircraft=81(+0)`` / ``ground=85(+0)``: the ``(+N)`` suffix is
the *on-order* count, but it reads exactly like a signed delta, so the model
tried to reconcile it against the after-action losses and never converged. The
fix renders the on-order count with an explicit ``on_order=N`` label and omits
it entirely when it is zero. These tests pin that rendering down for the force
summary (``intel.py``) and for the base and squadron lines (``operations.py``).
"""

from __future__ import annotations

import dataclasses

from game.ai_commander.enums import IntelPolicy
from game.ai_commander.intel import ForceSummary, IntelProjector
from game.ai_commander.operations import BaseView, SquadronView
from tests.ai_commander import fakes

# ---------------------------------------------------------------------------
# intel.py — the [OWN FORCES] summary line
# ---------------------------------------------------------------------------


def _own_forces_line(rendered: str) -> str:
    lines = rendered.splitlines()
    marker = lines.index("[OWN FORCES]")
    return lines[marker + 1]


def _brief_with_summary(**summary_overrides: int) -> str:
    _, game = fakes.synthetic_game()
    brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
    summary = dataclasses.replace(brief.red_force_summary, **summary_overrides)
    brief = dataclasses.replace(brief, red_force_summary=summary)
    return _own_forces_line(brief.render_compact())


class TestForceSummaryOnOrder:
    def test_zero_on_order_is_omitted(self) -> None:
        line = _brief_with_summary(
            aircraft_available=81,
            aircraft_on_order=0,
            ground_units_deployed=85,
            ground_units_on_order=0,
        )
        assert "aircraft=81 " in line
        assert "ground=85 " in line
        assert "on_order" not in line
        assert "(+" not in line

    def test_positive_on_order_is_labelled(self) -> None:
        line = _brief_with_summary(
            aircraft_available=81,
            aircraft_on_order=2,
            ground_units_deployed=85,
            ground_units_on_order=4,
        )
        assert "aircraft=81 on_order=2" in line
        assert "ground=85 on_order=4" in line
        assert "(+" not in line

    def test_a_single_side_on_order_only_labels_that_side(self) -> None:
        line = _brief_with_summary(
            aircraft_available=81,
            aircraft_on_order=3,
            ground_units_deployed=85,
            ground_units_on_order=0,
        )
        assert "aircraft=81 on_order=3" in line
        assert "ground=85 " in line
        assert "ground=85 on_order" not in line


# ---------------------------------------------------------------------------
# operations.py — the per-base and per-squadron lines
# ---------------------------------------------------------------------------


def _base(aircraft_on_order: int, ground_units_on_order: int) -> BaseView:
    return BaseView(
        id="BASE-1",
        name="Anapa",
        kind="airfield",
        is_front_line_base=False,
        runway_operational=True,
        runway_repairable=True,
        runway_repair_turns_remaining=None,
        aircraft_present=81,
        aircraft_on_order=aircraft_on_order,
        parking_free=None,
        ground_units_present=85,
        ground_units_on_order=ground_units_on_order,
        can_recruit_ground_units=True,
        has_ground_unit_source=True,
        squadron_ids=(),
    )


class TestBaseViewOnOrder:
    def test_zero_on_order_is_omitted(self) -> None:
        rendered = _base(0, 0).render()
        assert "aircraft=81 " in rendered
        assert "ground=85 " in rendered
        assert "on_order" not in rendered
        assert "(+" not in rendered

    def test_positive_on_order_is_labelled(self) -> None:
        rendered = _base(2, 4).render()
        assert "aircraft=81 on_order=2" in rendered
        assert "ground=85 on_order=4" in rendered
        assert "(+" not in rendered


def _squadron(aircraft_on_order: int) -> SquadronView:
    return SquadronView(
        id="SQN-1",
        name="1st Fighter",
        aircraft_id="MiG-29",
        base_id="BASE-1",
        base_name="Anapa",
        aircraft_on_hand=2,
        aircraft_untasked=2,
        aircraft_on_order=aircraft_on_order,
        pilots_available=4,
        pilot_limit_enabled=False,
        max_fulfillable_aircraft=2,
        price_per_aircraft=10,
        relocating_to_base_id=None,
        capable_tasks=("CAP",),
        auto_assignable_tasks=("CAP",),
    )


class TestSquadronViewOnOrder:
    def test_zero_on_order_is_omitted(self) -> None:
        rendered = _squadron(0).render()
        assert "onhand=2 untasked=2 " in rendered
        assert "on_order" not in rendered
        assert "(+" not in rendered

    def test_positive_on_order_is_labelled(self) -> None:
        rendered = _squadron(2).render()
        assert "onhand=2 untasked=2 on_order=2" in rendered
        assert "(+" not in rendered
