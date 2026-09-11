"""The air-tasking stage prompt must restate the legality rules the model broke.

A live turn showed the model adding flights whose mission_type was not among a
target's briefed ``missions=`` list and reusing a ``target_id`` across two
packages, both of which the validator then rejected. The data was already in the
brief; the fix reinforces it in the stage-3 instructions. These tests pin that
wording down.
"""

from __future__ import annotations

from game.ai_commander.activeprompt import stage_briefing_text
from game.ai_commander.plan import CommanderStage


class TestAirTaskingBriefing:
    def _text(self) -> str:
        return stage_briefing_text(CommanderStage.AIR_TASKING)

    def test_mission_type_rule_is_stated(self) -> None:
        text = self._text()
        assert "missions=" in text
        assert "mission_type" in text
        # The only permitted additions beyond the briefed list.
        assert "Escort" in text
        assert "SEAD Escort" in text

    def test_one_package_per_target_rule_is_stated(self) -> None:
        text = self._text()
        assert "target_id" in text
        assert "at most one package" in text

    def test_other_stages_do_not_carry_the_reminder(self) -> None:
        for stage in (CommanderStage.COMMAND, CommanderStage.LOGISTICS):
            assert "at most one package" not in stage_briefing_text(stage)

    def test_air_superiority_first_doctrine_is_stated(self) -> None:
        text = self._text().lower()
        # Win the air before committing strikers, escort every striker.
        assert "sweep" in text
        assert "escort" in text
        assert "only then commit strikers" in text
        assert "air superiority" in text

    def test_mission_type_purposes_are_explained(self) -> None:
        text = self._text()
        # A short gloss of what the fighter missions are actually for.
        assert "BARCAP" in text
        assert "Fighter sweep" in text
        assert "SEAD" in text


class TestLogisticsBriefing:
    def _text(self) -> str:
        return stage_briefing_text(CommanderStage.LOGISTICS)

    def test_ground_transfer_rules_are_stated(self) -> None:
        text = self._text()
        # Only transfer types a base holds, and never duplicate a destination.
        assert "ground_types" in text
        assert "destination_base_id" in text
        assert "retreat" in text


class TestCommandBriefing:
    def _text(self) -> str:
        return stage_briefing_text(CommanderStage.COMMAND)

    def test_posture_legality_reminder_is_stated(self) -> None:
        text = self._text()
        assert "legal=" in text
        assert "retreat" in text

    def test_recent_turns_repetition_reminder_is_stated(self) -> None:
        assert "RECENT TURNS" in self._text()
