"""Tests for the observed-enemy-types intel channel.

RED is given a de-duplicated list of the distinct enemy unit and aircraft TYPES
it has actually observed (enemy objectives within its own sensor/threat coverage)
or met in combat (from the after-action debrief). These cover:

* :class:`ObservedEnemyForces` reports emptiness and serialises as types only;
* :meth:`IntelProjector._distinct_unit_types` de-duplicates, sorts, drops dead
  units, carries no counts, and is safe against objects that expose no units
  (including test doubles), so observation can never crash or leak;
* the aircraft types flow from the stored after-action into the brief and render;
* folding the channel into the RED brief leaks no BLUE-private information.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from game.ai_commander.debrief import DebriefSummary
from game.ai_commander.enums import IntelPolicy
from game.ai_commander.intel import IntelProjector, ObservedEnemyForces
from game.theater.theatergroundobject import IadsGroundObject
from tests.ai_commander import fakes

# ---------------------------------------------------------------------------
# ObservedEnemyForces
# ---------------------------------------------------------------------------


class TestObservedEnemyForces:
    def test_default_is_empty(self) -> None:
        assert ObservedEnemyForces().is_empty

    def test_any_populated_tuple_is_not_empty(self) -> None:
        assert not ObservedEnemyForces(aircraft_types=("F-15C",)).is_empty
        assert not ObservedEnemyForces(air_defence_types=("SA-11 Buk LN",)).is_empty
        assert not ObservedEnemyForces(ground_types=("M-1 Abrams",)).is_empty
        assert not ObservedEnemyForces(naval_types=("Ticonderoga",)).is_empty

    def test_to_dict_is_types_only(self) -> None:
        forces = ObservedEnemyForces(
            aircraft_types=("F-15C", "F-16C"),
            air_defence_types=("SA-11 Buk LN",),
            ground_types=("M-1 Abrams",),
            naval_types=("Ticonderoga",),
        )
        data = forces.to_dict()
        # jsonable renders tuples as lists of plain strings -- no counts anywhere.
        assert data == {
            "aircraft_types": ["F-15C", "F-16C"],
            "air_defence_types": ["SA-11 Buk LN"],
            "ground_types": ["M-1 Abrams"],
            "naval_types": ["Ticonderoga"],
        }


# ---------------------------------------------------------------------------
# _distinct_unit_types
# ---------------------------------------------------------------------------


def _unit(type_id: str, alive: bool = True) -> Any:
    return SimpleNamespace(alive=alive, type=SimpleNamespace(id=type_id))


def _tgo(*units: Any) -> Any:
    """A ground-object-like double whose ``units`` is directly iterable."""

    return SimpleNamespace(units=list(units))


class TestDistinctUnitTypes:
    def test_dedupes_sorts_and_ignores_dead(self) -> None:
        objects = [
            _tgo(_unit("SA-11 Buk LN"), _unit("SA-11 Buk LN"), _unit("Dog Ear radar")),
            _tgo(_unit("SA-6 Kub LN"), _unit("SA-15 Tor", alive=False)),
        ]
        result = IntelProjector._distinct_unit_types(objects)
        assert result == ("Dog Ear radar", "SA-11 Buk LN", "SA-6 Kub LN")
        # A type present only on a dead unit is not reported.
        assert "SA-15 Tor" not in result

    def test_returns_only_strings_never_counts(self) -> None:
        objects = [_tgo(_unit("M-1 Abrams"), _unit("M-1 Abrams"))]
        result = IntelProjector._distinct_unit_types(objects)
        # De-duplicated to a single type name, with no multiplicity recorded.
        assert result == ("M-1 Abrams",)
        assert all(isinstance(name, str) for name in result)

    def test_is_capped(self) -> None:
        objects = [_tgo(*(_unit(f"TYPE-{i:03d}") for i in range(50)))]
        assert len(IntelProjector._distinct_unit_types(objects)) == 24

    def test_safe_against_iads_test_double(self) -> None:
        # An observed IADS in the fake campaign is a MagicMock(spec=...); reading
        # its ``units`` must yield nothing rather than raising or leaking.
        tgo = MagicMock(spec=IadsGroundObject)
        assert IntelProjector._distinct_unit_types([tgo]) == ()

    def test_safe_against_objects_without_units(self) -> None:
        assert IntelProjector._distinct_unit_types([object()]) == ()

    def test_empty_input(self) -> None:
        assert IntelProjector._distinct_unit_types([]) == ()


# ---------------------------------------------------------------------------
# Integration with the RED brief (fairness boundary preserved)
# ---------------------------------------------------------------------------


class TestObservedEnemyTypesInTheBrief:
    def test_absent_when_nothing_observed(self) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        assert brief.observed_enemy_forces.is_empty
        assert "[OBSERVED ENEMY TYPES]" not in brief.render_compact()

    def test_aircraft_types_flow_from_after_action(self) -> None:
        campaign, game = fakes.synthetic_game()
        summary = DebriefSummary(turn=6, enemy_aircraft_types_seen=("F-15C", "F-16C"))
        campaign.red.last_after_action = summary.to_dict()

        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        assert brief.observed_enemy_forces.aircraft_types == ("F-15C", "F-16C")
        rendered = brief.render_compact()
        assert "[OBSERVED ENEMY TYPES]" in rendered
        assert "aircraft: F-15C, F-16C" in rendered
        # The section states plainly that it is types only, not a count.
        assert "not a count, roster or measure" in rendered

    def test_observed_types_leak_no_blue_information(self) -> None:
        campaign, game = fakes.synthetic_game()
        summary = DebriefSummary(
            turn=6,
            red_aircraft_lost=1,
            red_aircraft_lost_by_cause={"enemy_aircraft": 1},
            killed_by_platform_types=("F-15C",),
            killed_by_weapon_types=("AIM-120C",),
            enemy_aircraft_types_seen=("F-15C", "F-16C"),
        )
        campaign.red.last_after_action = summary.to_dict()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        blob = fakes.serialise_everything(brief.to_dict(), brief.render_compact())
        assert fakes.blue_leaks_in(blob) == []
