"""Tests for the RED after-action debrief summary.

These cover the four things that matter:

* :func:`classify_threat` maps every unit class to the right threat bucket;
* :func:`build_debrief_summary` attributes RED losses to their causes and counts
  confirmed BLUE losses, from a realistic mixture of DCS-reported and
  auto-resolved kills;
* the summary renders compactly and round-trips through JSON; and
* folding the summary into the RED brief leaks no BLUE-private information.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest

from game.ai_commander.debrief import (
    DebriefSummary,
    ThreatCategory,
    build_debrief_summary,
    classify_threat,
)
from game.ai_commander.enums import IntelPolicy
from game.ai_commander.intel import IntelProjector
from game.data.units import UnitClass
from game.debriefing import SideLossCounts
from game.theater import Player
from tests.ai_commander import fakes


# ---------------------------------------------------------------------------
# classify_threat
# ---------------------------------------------------------------------------


class TestClassifyThreat:
    @pytest.mark.parametrize(
        "unit_class, expected",
        [
            (UnitClass.PLANE, ThreatCategory.ENEMY_AIRCRAFT),
            (UnitClass.HELICOPTER, ThreatCategory.ENEMY_AIRCRAFT),
            (UnitClass.AAA, ThreatCategory.AAA),
            (UnitClass.LAUNCHER, ThreatCategory.GROUND_SAM),
            (UnitClass.TELAR, ThreatCategory.GROUND_SAM),
            (UnitClass.MANPAD, ThreatCategory.GROUND_SAM),
            (UnitClass.SHORAD, ThreatCategory.GROUND_SAM),
            (UnitClass.TRACK_RADAR, ThreatCategory.GROUND_SAM),
            (UnitClass.DESTROYER, ThreatCategory.NAVAL),
            (UnitClass.CRUISER, ThreatCategory.NAVAL),
            (UnitClass.FRIGATE, ThreatCategory.NAVAL),
            (UnitClass.AIRCRAFT_CARRIER, ThreatCategory.NAVAL),
            (UnitClass.SUBMARINE, ThreatCategory.NAVAL),
            (UnitClass.TANK, ThreatCategory.GROUND_FIRE),
            (UnitClass.IFV, ThreatCategory.GROUND_FIRE),
            (UnitClass.INFANTRY, ThreatCategory.GROUND_FIRE),
            (UnitClass.ARTILLERY, ThreatCategory.GROUND_FIRE),
            (UnitClass.LOGISTICS, ThreatCategory.UNKNOWN),
            (UnitClass.COMMAND_POST, ThreatCategory.UNKNOWN),
            (None, ThreatCategory.UNKNOWN),
        ],
    )
    def test_mapping(
        self, unit_class: Optional[UnitClass], expected: ThreatCategory
    ) -> None:
        assert classify_threat(unit_class) is expected

    def test_every_unit_class_has_a_category(self) -> None:
        # No unit class should raise; unmapped ones fall through to UNKNOWN.
        for unit_class in UnitClass:
            assert isinstance(classify_threat(unit_class), ThreatCategory)


# ---------------------------------------------------------------------------
# Fakes for build_debrief_summary
# ---------------------------------------------------------------------------


def _unit_type(unit_class: UnitClass) -> Any:
    return SimpleNamespace(unit_class=unit_class)


def _flying(side: Player, killer_class: Optional[UnitClass] = None) -> Any:
    """A FlyingUnit-like object. ``side`` is the owning coalition."""

    departure = SimpleNamespace(captured=side)
    unit_type = _unit_type(killer_class) if killer_class is not None else None
    flight = SimpleNamespace(departure=departure, unit_type=unit_type)
    return SimpleNamespace(flight=flight)


def _front_line(side: Player, unit_class: UnitClass) -> Any:
    origin = SimpleNamespace(captured=side)
    return SimpleNamespace(origin=origin, unit_type=_unit_type(unit_class))


def _theater(side: Player, unit_class: UnitClass) -> Any:
    is_blue = side.is_blue

    def is_friendly(to_player: Player) -> bool:
        return to_player.is_blue == is_blue

    ground_object = SimpleNamespace(is_friendly=is_friendly)
    theater_unit = SimpleNamespace(
        unit_type=_unit_type(unit_class), ground_object=ground_object
    )
    return SimpleNamespace(theater_unit=theater_unit)


class FakeUnitMap:
    def __init__(self) -> None:
        self.aircraft: dict[str, Any] = {}
        self.front: dict[str, Any] = {}
        self.motor: dict[str, Any] = {}
        self.convoys: dict[str, Any] = {}
        self.theater: dict[str, Any] = {}

    def flight(self, name: str) -> Any:
        return self.aircraft.get(name)

    def front_line_unit(self, name: str) -> Any:
        return self.front.get(name)

    def motorpool_unit(self, name: str) -> Any:
        return self.motor.get(name)

    def convoy_unit(self, name: str) -> Any:
        return self.convoys.get(name)

    def theater_units(self, name: str) -> Any:
        return self.theater.get(name)


class FakeAirLosses:
    def __init__(self, red_by_type: dict[str, int]) -> None:
        self._red_by_type = red_by_type

    def by_type(self, player: Player) -> dict[str, int]:
        return self._red_by_type if player.is_red else {}


class FakeDebriefing:
    def __init__(
        self,
        unit_map: FakeUnitMap,
        kill_causes: list[dict[str, str]],
        red_counts: SideLossCounts,
        blue_counts: SideLossCounts,
        red_air_by_type: dict[str, int],
    ) -> None:
        self.unit_map = unit_map
        self.state_data = SimpleNamespace(kill_causes=kill_causes)
        self._red = red_counts
        self._blue = blue_counts
        self.air_losses = FakeAirLosses(red_air_by_type)

    def loss_counts(self, player: Player) -> SideLossCounts:
        return self._red if player.is_red else self._blue


def _counts(**overrides: int) -> SideLossCounts:
    base = dict(
        aircraft=0,
        front_line=0,
        motorpool=0,
        convoy=0,
        cargo_ships=0,
        airlift_cargo=0,
        ground_objects=0,
        scenery=0,
        bases_lost=0,
        runways_destroyed=0,
    )
    base.update(overrides)
    return SideLossCounts(**base)


# ---------------------------------------------------------------------------
# build_debrief_summary
# ---------------------------------------------------------------------------


class TestBuildDebriefSummary:
    def _scenario(self) -> FakeDebriefing:
        unit_map = FakeUnitMap()
        # Victims (RED) and killers (BLUE) both live in the unit map.
        unit_map.aircraft = {
            "red-cas-1": _flying(Player.RED),
            "red-cas-2": _flying(Player.RED),
            "blue-jet-1": _flying(Player.BLUE),  # a BLUE victim, must be ignored
            "blue-f15-1": _flying(Player.BLUE, killer_class=UnitClass.PLANE),
            "blue-f16-1": _flying(Player.BLUE, killer_class=UnitClass.PLANE),
        }
        unit_map.front = {
            "red-tank-1": _front_line(Player.RED, UnitClass.TANK),
            "blue-tank-1": _front_line(Player.BLUE, UnitClass.TANK),
        }
        unit_map.theater = {
            "red-sam-site-1": _theater(Player.RED, UnitClass.LAUNCHER),
            "blue-ship-1": _theater(Player.BLUE, UnitClass.DESTROYER),
        }
        kill_causes = [
            # RED CAS downed by a BLUE fighter -> enemy aircraft
            {"target": "red-cas-1", "by": "blue-f15-1", "by_type": "F-15C"},
            # RED CAS downed by a BLUE warship -> naval / ship-launched SAM
            {"target": "red-cas-2", "by": "blue-ship-1", "by_type": "USS_Arleigh"},
            # RED tank destroyed by a BLUE tank -> ground fire
            {"target": "red-tank-1", "by": "blue-tank-1", "by_type": "M-1"},
            # RED SAM site destroyed by a BLUE jet -> enemy aircraft (ground victim)
            {"target": "red-sam-site-1", "by": "blue-f16-1", "by_type": "F-16C"},
            # A BLUE victim RED killed; must not be attributed to RED's losses.
            {"target": "blue-jet-1", "by": "red-mig-1", "by_type": "MiG-29"},
        ]
        red_counts = _counts(
            aircraft=3,  # 2 attributed above + 1 auto-resolved (UNKNOWN)
            front_line=2,  # 1 attributed (tank) + 1 auto-resolved (UNKNOWN)
            ground_objects=1,  # the SAM site above, attributed to enemy aircraft
            runways_destroyed=1,
        )
        blue_counts = _counts(
            aircraft=1,
            front_line=1,
            ground_objects=2,
            cargo_ships=1,
            bases_lost=1,  # RED captured one BLUE base
        )
        return FakeDebriefing(
            unit_map,
            kill_causes,
            red_counts,
            blue_counts,
            red_air_by_type={"Su-25": 2, "MiG-29": 1},
        )

    def _summary(self) -> DebriefSummary:
        debriefing = self._scenario()
        game = SimpleNamespace(turn=12)
        return build_debrief_summary(debriefing, game)

    def test_red_air_losses_attributed_by_cause(self) -> None:
        summary = self._summary()
        assert summary.red_aircraft_lost == 3
        assert summary.red_aircraft_lost_by_cause == {
            "enemy_aircraft": 1,
            "naval": 1,
            "unknown": 1,  # the auto-resolved loss with no killer
        }

    def test_red_air_losses_by_airframe(self) -> None:
        summary = self._summary()
        assert summary.red_aircraft_lost_by_type == {"MiG-29": 1, "Su-25": 2}

    def test_red_ground_losses_attributed_by_cause(self) -> None:
        summary = self._summary()
        # front_line(2) + ground_objects(1) = 3 ground victims total.
        assert summary.red_ground_units_lost == 3
        assert summary.red_ground_units_lost_by_cause == {
            "enemy_aircraft": 1,  # the SAM site killed by a jet
            "ground_fire": 1,  # the tank
            "unknown": 1,  # the auto-resolved front-line loss
        }

    def test_by_cause_totals_match_loss_totals(self) -> None:
        summary = self._summary()
        assert sum(summary.red_aircraft_lost_by_cause.values()) == (
            summary.red_aircraft_lost
        )
        assert sum(summary.red_ground_units_lost_by_cause.values()) == (
            summary.red_ground_units_lost
        )

    def test_confirmed_enemy_losses_counted(self) -> None:
        summary = self._summary()
        assert summary.blue_aircraft_killed == 1
        assert summary.blue_ground_units_killed == 1
        assert summary.blue_static_defenses_killed == 2
        assert summary.blue_ships_killed == 1
        assert summary.blue_bases_captured == 1

    def test_red_incidental_losses(self) -> None:
        summary = self._summary()
        assert summary.red_static_defenses_lost == 1
        assert summary.red_runways_damaged == 1
        assert summary.red_bases_lost == 0

    def test_turn_is_recorded(self) -> None:
        assert self._summary().turn == 12

    def test_a_quiet_turn_is_empty(self) -> None:
        unit_map = FakeUnitMap()
        debriefing = FakeDebriefing(
            unit_map,
            kill_causes=[],
            red_counts=_counts(),
            blue_counts=_counts(),
            red_air_by_type={},
        )
        summary = build_debrief_summary(debriefing, SimpleNamespace(turn=1))
        assert summary.is_empty
        assert summary.render_compact() == ""

    def test_losses_with_no_kill_causes_all_fall_to_unknown(self) -> None:
        """Auto-resolved combat carries no killer, so everything is UNKNOWN."""

        unit_map = FakeUnitMap()
        debriefing = FakeDebriefing(
            unit_map,
            kill_causes=[],
            red_counts=_counts(aircraft=4, front_line=2),
            blue_counts=_counts(),
            red_air_by_type={},
        )
        summary = build_debrief_summary(debriefing, SimpleNamespace(turn=3))
        assert summary.red_aircraft_lost_by_cause == {"unknown": 4}
        assert summary.red_ground_units_lost_by_cause == {"unknown": 2}


# ---------------------------------------------------------------------------
# Rendering and serialisation
# ---------------------------------------------------------------------------


class TestRenderingAndSerialisation:
    def _summary(self) -> DebriefSummary:
        return DebriefSummary(
            turn=9,
            red_aircraft_lost=2,
            red_aircraft_lost_by_cause={"naval": 1, "enemy_aircraft": 1},
            red_aircraft_lost_by_type={"Su-25": 2},
            red_ground_units_lost=1,
            red_ground_units_lost_by_cause={"ground_fire": 1},
            red_runways_damaged=1,
            blue_aircraft_killed=3,
            blue_bases_captured=1,
        )

    def test_render_is_human_readable(self) -> None:
        rendered = self._summary().render_compact()
        assert "[AFTER-ACTION turn=9]" in rendered
        assert "red_aircraft_lost=2" in rendered
        assert "ship / ship-launched SAM=1" in rendered
        assert "enemy aircraft=1" in rendered
        assert "Su-25=2" in rendered
        assert "confirmed_enemy_losses" in rendered
        assert "bases_captured=1" in rendered

    def test_render_is_compact(self) -> None:
        # A whole after-action block should stay small (token budget guard).
        assert len(self._summary().render_compact()) < 600

    def test_round_trips_through_dict(self) -> None:
        summary = self._summary()
        assert DebriefSummary.from_dict(summary.to_dict()) == summary

    def test_from_dict_tolerates_garbage(self) -> None:
        summary = DebriefSummary.from_dict(
            {
                "turn": "not-an-int",
                "red_aircraft_lost": None,
                "red_aircraft_lost_by_cause": {"naval": "x", "unknown": 2},
                "unexpected": "ignored",
            }
        )
        assert summary.turn == 0
        assert summary.red_aircraft_lost == 0
        # The un-parseable value is dropped; the valid one survives.
        assert summary.red_aircraft_lost_by_cause == {"unknown": 2}


# ---------------------------------------------------------------------------
# Integration with the RED brief (fairness boundary preserved)
# ---------------------------------------------------------------------------


class TestAfterActionInTheBrief:
    def test_absent_when_nothing_stored(self) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        assert brief.after_action is None
        assert "[AFTER-ACTION LAST MISSION]" not in brief.render_compact()

    def test_present_and_rendered_when_stored(self) -> None:
        campaign, game = fakes.synthetic_game()
        summary = DebriefSummary(
            turn=6,
            red_aircraft_lost=2,
            red_aircraft_lost_by_cause={"naval": 2},
            red_aircraft_lost_by_type={"Su-25": 2},
        )
        campaign.red.last_after_action = summary.to_dict()

        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        assert brief.after_action == summary
        rendered = brief.render_compact()
        assert "[AFTER-ACTION LAST MISSION]" in rendered
        assert "ship / ship-launched SAM=2" in rendered

    def test_empty_summary_is_not_rendered(self) -> None:
        campaign, game = fakes.synthetic_game()
        campaign.red.last_after_action = DebriefSummary(turn=6).to_dict()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        assert "[AFTER-ACTION LAST MISSION]" not in brief.render_compact()

    def test_malformed_stored_summary_is_ignored(self) -> None:
        campaign, game = fakes.synthetic_game()
        campaign.red.last_after_action = {"turn": object()}  # not serialisable/parseable
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        # from_dict coerces what it can; a truly broken store must not crash.
        assert isinstance(brief.render_compact(), str)

    def test_after_action_leaks_no_blue_information(self) -> None:
        campaign, game = fakes.synthetic_game()
        # A summary that names RED's own losses only.
        summary = DebriefSummary(
            turn=6,
            red_aircraft_lost=1,
            red_aircraft_lost_by_cause={"enemy_aircraft": 1},
            red_aircraft_lost_by_type={"Su-25": 1},
            blue_aircraft_killed=2,
        )
        campaign.red.last_after_action = summary.to_dict()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        blob = fakes.serialise_everything(brief.to_dict(), brief.render_compact())
        assert fakes.blue_leaks_in(blob) == []
