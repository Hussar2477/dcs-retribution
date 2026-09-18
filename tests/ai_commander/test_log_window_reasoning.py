"""The Decision Log's plain-English narrative must read like an after-action
report, not JSON.

``_render_reasoning`` is the human-facing digest of a turn: the headline, what
the commander wanted, what each stage did in its own words, and what actually
happened. It is a pure function of the record dict, so it is exercised here
directly -- no ``QDialog`` is constructed. PySide6 is guarded with
``importorskip`` so the test is a no-op where Qt cannot be imported.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("PySide6")

from qt_ui.windows.aicommander.QAiCommanderLogWindow import (  # noqa: E402
    QAiCommanderLogWindow,
)


def _active_record() -> dict[str, Any]:
    """A representative ACTIVE-mode record with an accepted directive, two
    stages carrying their own intent and some refusals, and an execution
    report with applied and failed orders."""

    return {
        "turn_id": 23,
        "accepted": True,
        "mode": "active",
        "accepted_directive": {
            "strategy": "rebuild",
            "reserve_policy": "balanced",
            "target_set_order": ["base_defence", "enemy_shipping"],
            "procurement_order": ["aircraft", "ground_combat_units"],
            "front_order": ["FOB Kohnehshahr", "FOB Seerik"],
            "front_postures": {"FOB Kohnehshahr": "retreat"},
            "commander_intent": (
                "Retreat on the northern front to preserve ground forces and "
                "rebuild air strength."
            ),
        },
        "stages": [
            {
                "stage": "logistics",
                "ran": True,
                "accepted": True,
                "rejections": [],
                "accepted_plan": {
                    "intent": "Reinforce the forward depot and repair the runway."
                },
            },
            {
                "stage": "air_tasking",
                "ran": True,
                "accepted": True,
                "rejections": [
                    {
                        "element": "packages[0].flights[0]",
                        "reason": "no squadron available",
                    },
                    {
                        "element": "packages[0].flights[1]",
                        "reason": "no squadron available",
                    },
                    {
                        "element": "packages[1].flights[0]",
                        "reason": "no squadron available",
                    },
                ],
                "parsed_plan": {
                    "intent": "Suppress enemy shipping and protect own airspace."
                },
            },
        ],
        "execution_report": {
            "applied": 2,
            "failed": 4,
            "packages_added": 2,
            "budget_before": 5756,
            "budget_after": 4496,
            "spent": 1260,
            "orders": [
                {
                    "kind": "package",
                    "description": "CAS package against FOB Seerik",
                    "applied": True,
                },
                {
                    "kind": "package",
                    "description": "OCA/Aircraft against CVN-75 Harry S. Truman",
                    "applied": False,
                    "detail": "the mission planner could not crew or route this package",
                },
            ],
        },
    }


class TestRenderReasoning:
    def test_it_is_a_plain_string_and_not_json(self) -> None:
        text = QAiCommanderLogWindow._render_reasoning(_active_record())
        assert isinstance(text, str)
        # A JSON dump would start with a brace; the narrative starts with prose.
        assert not text.lstrip().startswith("{")

    def test_it_names_the_turn_and_that_it_was_accepted(self) -> None:
        text = QAiCommanderLogWindow._render_reasoning(_active_record())
        assert "Turn 23" in text
        assert "accepted" in text

    def test_it_includes_the_commander_intent_verbatim(self) -> None:
        text = QAiCommanderLogWindow._render_reasoning(_active_record())
        assert (
            "Retreat on the northern front to preserve ground forces and "
            "rebuild air strength." in text
        )

    def test_it_includes_each_stage_intent(self) -> None:
        text = QAiCommanderLogWindow._render_reasoning(_active_record())
        assert "Reinforce the forward depot and repair the runway." in text
        assert "Suppress enemy shipping and protect own airspace." in text

    def test_it_reports_applied_and_failed_counts_readably(self) -> None:
        text = QAiCommanderLogWindow._render_reasoning(_active_record())
        assert "2 order(s) were applied and 4 failed" in text

    def test_it_lists_orders_in_plain_form(self) -> None:
        text = QAiCommanderLogWindow._render_reasoning(_active_record())
        assert "CAS package against FOB Seerik -- applied." in text
        assert "failed: the mission planner could not crew or route" in text

    def test_it_summarises_refusals_grouped_by_reason(self) -> None:
        text = QAiCommanderLogWindow._render_reasoning(_active_record())
        assert "3 refused" in text
        assert "3x no squadron available" in text

    def test_a_commander_mode_record_degrades_gracefully(self) -> None:
        record = {
            "turn_id": 5,
            "accepted": True,
            "mode": "commander",
            "accepted_directive": {
                "strategy": "attrition",
                "commander_intent": "Hold the line and bleed the attacker.",
            },
        }
        text = QAiCommanderLogWindow._render_reasoning(record)
        assert "Turn 5" in text
        assert "commander mode" in text
        assert "Hold the line and bleed the attacker." in text

    def test_a_fallback_record_reports_the_reason(self) -> None:
        record = {
            "turn_id": 8,
            "accepted": False,
            "fallback_reason": "malformed_response",
        }
        text = QAiCommanderLogWindow._render_reasoning(record)
        assert "Turn 8" in text
        assert "malformed_response" in text
