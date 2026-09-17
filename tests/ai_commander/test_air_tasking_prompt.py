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
        assert "win the air first" in text
        assert "air superiority" in text

    def test_mission_type_purposes_are_explained(self) -> None:
        text = self._text()
        # A short gloss of what the fighter missions are actually for.
        assert "BARCAP" in text
        assert "Fighter sweep" in text
        assert "SEAD" in text

    def test_front_line_air_support_doctrine_is_stated(self) -> None:
        text = self._text()
        # CAS/BAI packaged against the front, escort the strikers; area CAP is
        # planned automatically rather than attached to a target here.
        assert "FRONT-LINE AIR" in text
        assert "CAS" in text
        assert "BAI" in text
        assert "Escort" in text
        assert "enemy_battle_positions" in text

    def test_area_missions_must_not_be_package_flights(self) -> None:
        text = self._text()
        # Fighter sweep / BARCAP / TARCAP are area missions and must never be
        # scheduled as a flight inside a target package -- the mistake that got
        # three packages rejected on turn 17.
        assert "AREA missions" in text
        assert "Fighter sweep" in text
        assert "BARCAP" in text
        assert "TARCAP" in text
        assert "packages[].flights" in text
        # And the package-legal alternative for cutting enemy air is spelt out.
        assert "OCA/Aircraft" in text
        assert "OCA/Runway" in text

    def test_runway_strike_doctrine_is_stated(self) -> None:
        text = self._text()
        # Crater runways to ground enemy aircraft.
        assert "RUNWAY STRIKES" in text
        assert "OCA/Runway" in text
        assert "enemy_airbases" in text

    def test_escort_may_not_be_primary_flight_rule_is_stated(self) -> None:
        text = self._text()
        # The 4th enforced rule: a package is led by a striker, escorts follow.
        assert "Four rules the planner enforces" in text
        assert "(4)" in text
        assert "primary" in text
        # Escort/SEAD Escort are supporting flights, not the lead.
        assert "Escort" in text
        assert "SEAD Escort" in text

    def test_mission_type_from_missions_list_is_reinforced(self) -> None:
        text = self._text()
        # Rule (1): mission_type must come from the target's missions= list --
        # the DEAD-on-a-Strike-target mistake is called out by name.
        assert "missions=" in text
        assert "DEAD" in text

    def test_anti_ship_massing_doctrine_is_stated(self) -> None:
        text = self._text()
        assert "ANTI-SHIP MASSING" in text
        assert "MASS" in text
        # Multiple anti-ship flights, escort and SEAD escort, several targets.
        assert "anti-ship" in text
        assert "Escort" in text
        assert "SEAD Escort" in text
        # We must NOT claim the commander controls launch range/standoff.
        assert "cannot set launch range" in text

    def test_asap_timing_doctrine_is_stated(self) -> None:
        text = self._text()
        assert "TIMING" in text
        assert "asap" in text
        assert "asap:true" in text
        # Priority/defensive packages launch early.
        assert "CAS/BAI" in text


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

    def test_commit_the_budget_doctrine_is_stated(self) -> None:
        text = self._text()
        # A large surplus on a holdable front must be committed, not hoarded.
        assert "COMMIT THE BUDGET" in text
        assert "surplus" in text
        assert "underspending" in text

    def test_buy_in_blocks_doctrine_is_stated(self) -> None:
        text = self._text()
        # Size each buy to the surplus; buy in blocks, not two or three at a time.
        assert "BLOCKS" in text
        assert "dozens" in text

    def test_front_line_air_defence_doctrine_is_stated(self) -> None:
        text = self._text()
        # Buy AAA and transfer owned SAM/SHORAD/MANPADS forward to the front.
        assert "FRONT-LINE AIR DEFENCE" in text
        assert "AAA" in text
        assert "SHORAD" in text
        assert "MANPADS" in text
        # Concrete, catalogue-plausible examples of each.
        assert "ZSU-23-4" in text
        assert "SA-15" in text

    def test_forward_base_helicopters_doctrine_is_stated(self) -> None:
        text = self._text()
        # Short-radius attack helicopters must be forward-based near the front.
        assert "FORWARD-BASE ATTACK HELICOPTERS" in text
        assert "squadron_relocations" in text
        assert "Mi-28N" in text


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

    def test_press_the_advantage_doctrine_is_stated(self) -> None:
        text = self._text()
        # With a force advantage on a capturable front, press rather than hold;
        # rebuilding air defence and pressing are not mutually exclusive.
        assert "PRESS THE ADVANTAGE" in text
        assert "capturable" in text
        assert "mutually exclusive" in text

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
