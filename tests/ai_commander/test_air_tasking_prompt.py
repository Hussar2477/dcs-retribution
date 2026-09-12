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

    def test_front_line_air_support_doctrine_is_stated(self) -> None:
        text = self._text()
        # CAP over the front, CAS/BAI for the ground battle, escort the strikers.
        assert "FRONT-LINE AIR" in text
        assert "CAS" in text
        assert "BAI" in text
        assert "Escort" in text
        assert "ground lost to enemy air" in text

    def test_runway_strike_doctrine_is_stated(self) -> None:
        text = self._text()
        # Crater runways to ground enemy aircraft.
        assert "RUNWAY STRIKES" in text
        assert "OCA/Runway" in text
        assert "enemy_airbases" in text


class TestLogisticsBriefing:
    def _text(self) -> str:
        return stage_briefing_text(CommanderStage.LOGISTICS)

    def test_ground_transfer_rules_are_stated(self) -> None:
        text = self._text()
        # Only transfer types a base holds, and never duplicate a destination.
        assert "ground_types" in text
        assert "destination_base_id" in text
        assert "retreat" in text

    def test_force_mix_composition_doctrine_is_stated(self) -> None:
        text = self._text()
        # Break through with armour/anti-armour; survive enemy air with SHORAD.
        assert "FORCE MIX" in text
        assert "anti-armour" in text
        assert "SHORAD" in text

    def test_holding_and_advancing_gains_income_is_stated(self) -> None:
        text = self._text()
        assert "income buildings" in text

    def test_factory_recruit_link_is_stated(self) -> None:
        text = self._text()
        # Ground units can only be built where a factory stands.
        assert "factory" in text
        assert "recruit_ground=" in text


class TestCommandBriefing:
    def _text(self) -> str:
        return stage_briefing_text(CommanderStage.COMMAND)

    def test_posture_legality_reminder_is_stated(self) -> None:
        text = self._text()
        assert "legal=" in text
        assert "retreat" in text

    def test_recent_turns_repetition_reminder_is_stated(self) -> None:
        assert "RECENT TURNS" in self._text()

    def test_economic_doctrine_is_stated(self) -> None:
        text = self._text()
        # Money sustains the war: protect own income, strike the enemy's.
        assert "ECONOMY" in text
        assert "income buildings" in text
        assert "future income" in text or "future budget" in text

    def test_capturing_territory_gains_income_is_stated(self) -> None:
        text = self._text()
        assert "Capturing enemy territory" in text

    def test_factory_production_doctrine_is_stated(self) -> None:
        text = self._text()
        # Factories both earn income and enable ground-unit production.
        assert "PRODUCTION" in text
        assert "recruit_ground=yes" in text
        assert "factory" in text

    def test_commanders_art_toolbox_doctrine_is_stated(self) -> None:
        text = self._text()
        # A human-commander framing: combine all tools and adapt each turn.
        assert "COMMANDER'S ART" in text
        assert "air superiority" in text
        assert "CAS/BAI" in text
        assert "runways" in text
        assert "adapt" in text

    def test_other_stages_do_not_carry_commanders_art(self) -> None:
        for stage in (CommanderStage.LOGISTICS, CommanderStage.AIR_TASKING):
            assert "COMMANDER'S ART" not in stage_briefing_text(stage)
