"""RED after-action reporting for the LLM commander.

After a mission is flown, Retribution produces a :class:`~game.debriefing.Debriefing`
describing every unit that died. This module distils that into a compact,
RED-perspective after-action summary that the model can read on its next turn:

* what RED *lost*, and — where DCS reported it — *what destroyed each loss*
  (enemy aircraft, a ground SAM, a ship / ship-launched SAM, AAA, ground fire);
* what RED *confirmed killed* (BLUE losses observed once combat resolved).

The summary is deliberately small (a handful of integers and short by-cause
maps) so that folding it into the prompt costs a negligible number of tokens and
never threatens the per-turn cost cap.

Fairness boundary
-----------------
Everything here is derived only from RED's *own* losses and from BLUE losses that
became observable *after* the engagement (a destroyed BLUE unit is not secret).
No BLUE pre-combat internals — plans, rosters, budgets, coordinates — are read.
Killer attribution uses the DCS ``S_EVENT_KILL`` initiator, which is the unit
that actually did the killing in the mission RED just fought; it is not
privileged foreknowledge.

Attribution is best-effort. DCS only emits an initiator for kills scored in the
flown mission; auto-resolved combat carries no killer, so those losses fall into
the ``UNKNOWN`` bucket rather than being dropped.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

from game.data.units import UnitClass

if TYPE_CHECKING:
    from game.debriefing import Debriefing
    from game.game import Game
    from game.unitmap import UnitMap


@unique
class ThreatCategory(Enum):
    """What kind of threat destroyed a RED unit."""

    ENEMY_AIRCRAFT = "enemy_aircraft"
    GROUND_SAM = "ground_sam"
    NAVAL = "naval"
    AAA = "aaa"
    GROUND_FIRE = "ground_fire"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return {
            ThreatCategory.ENEMY_AIRCRAFT: "enemy aircraft",
            ThreatCategory.GROUND_SAM: "ground SAM",
            ThreatCategory.NAVAL: "ship / ship-launched SAM",
            ThreatCategory.AAA: "AAA",
            ThreatCategory.GROUND_FIRE: "ground fire",
            ThreatCategory.UNKNOWN: "unknown / unattributed",
        }[self]


# Unit classes that count as land-based air defence (the credited killer for a
# SAM engagement is usually the launcher, TELAR or tracking radar).
_GROUND_SAM_CLASSES = frozenset(
    {
        UnitClass.LAUNCHER,
        UnitClass.TELAR,
        UnitClass.MANPAD,
        UnitClass.SHORAD,
        UnitClass.TRACK_RADAR,
        UnitClass.SEARCH_TRACK_RADAR,
        UnitClass.SEARCH_RADAR,
        UnitClass.SPECIALIZED_RADAR,
        UnitClass.EARLY_WARNING_RADAR,
        UnitClass.OPTICAL_TRACKER,
        UnitClass.SEARCH_LIGHT,
    }
)

_NAVAL_CLASSES = frozenset(
    {
        UnitClass.CRUISER,
        UnitClass.DESTROYER,
        UnitClass.FRIGATE,
        UnitClass.AIRCRAFT_CARRIER,
        UnitClass.HELICOPTER_CARRIER,
        UnitClass.SUBMARINE,
        UnitClass.BOAT,
        UnitClass.LANDING_SHIP,
    }
)

_GROUND_FIRE_CLASSES = frozenset(
    {
        UnitClass.TANK,
        UnitClass.IFV,
        UnitClass.APC,
        UnitClass.ARTILLERY,
        UnitClass.ATGM,
        UnitClass.INFANTRY,
    }
)


def classify_threat(unit_class: Optional[UnitClass]) -> ThreatCategory:
    """Map a killer's :class:`UnitClass` to a :class:`ThreatCategory`.

    Pure and side-effect free so it can be unit tested in isolation.
    """

    if unit_class is None:
        return ThreatCategory.UNKNOWN
    if unit_class in (UnitClass.PLANE, UnitClass.HELICOPTER):
        return ThreatCategory.ENEMY_AIRCRAFT
    if unit_class is UnitClass.AAA:
        return ThreatCategory.AAA
    if unit_class in _NAVAL_CLASSES:
        return ThreatCategory.NAVAL
    if unit_class in _GROUND_SAM_CLASSES:
        return ThreatCategory.GROUND_SAM
    if unit_class in _GROUND_FIRE_CLASSES:
        return ThreatCategory.GROUND_FIRE
    return ThreatCategory.UNKNOWN


def _killer_unit_class(
    killer_name: Optional[str], unit_map: UnitMap
) -> Optional[UnitClass]:
    """Resolve the killer's unit class from its DCS unit name, if we can."""

    if not killer_name:
        return None

    flight = unit_map.flight(killer_name)
    if flight is not None:
        return getattr(flight.flight.unit_type, "unit_class", None)

    theater = unit_map.theater_units(killer_name)
    if theater is not None:
        unit_type = theater.theater_unit.unit_type
        return getattr(unit_type, "unit_class", None) if unit_type else None

    for resolver in (unit_map.front_line_unit, unit_map.motorpool_unit):
        ground = resolver(killer_name)
        if ground is not None:
            return getattr(ground.unit_type, "unit_class", None)

    convoy = unit_map.convoy_unit(killer_name)
    if convoy is not None:
        return getattr(convoy.unit_type, "unit_class", None)

    return None


def _resolve_cause(cause: Mapping[str, str], unit_map: UnitMap) -> ThreatCategory:
    """Best-effort category for a single kill-cause record."""

    unit_class = _killer_unit_class(cause.get("by"), unit_map)
    return classify_threat(unit_class)


def _sorted_cause_map(counts: Mapping[ThreatCategory, int]) -> Dict[str, int]:
    """Deterministic, non-zero-only {category_value: count} mapping."""

    return {
        category.value: counts[category]
        for category in ThreatCategory
        if counts.get(category, 0) > 0
    }


@dataclass(frozen=True)
class DebriefSummary:
    """A compact, RED-perspective after-action report for one resolved turn."""

    turn: int

    # -- RED's own losses -------------------------------------------------
    red_aircraft_lost: int = 0
    #: {ThreatCategory.value: count}. Sums to ``red_aircraft_lost``; losses with
    #: no reported killer land in ``unknown``.
    red_aircraft_lost_by_cause: Mapping[str, int] = field(default_factory=dict)
    #: {aircraft type name: count}, for reasoning about which airframes attrit.
    red_aircraft_lost_by_type: Mapping[str, int] = field(default_factory=dict)
    red_ground_units_lost: int = 0
    #: {ThreatCategory.value: count}. Sums to ``red_ground_units_lost``.
    red_ground_units_lost_by_cause: Mapping[str, int] = field(default_factory=dict)
    red_static_defenses_lost: int = 0
    red_ships_lost: int = 0
    red_runways_damaged: int = 0
    red_bases_lost: int = 0

    # -- Confirmed BLUE losses (observed after combat) --------------------
    blue_aircraft_killed: int = 0
    blue_ground_units_killed: int = 0
    blue_static_defenses_killed: int = 0
    blue_ships_killed: int = 0
    blue_bases_captured: int = 0

    @property
    def is_empty(self) -> bool:
        """True when nothing worth reporting happened this turn."""

        return not any(
            (
                self.red_aircraft_lost,
                self.red_ground_units_lost,
                self.red_static_defenses_lost,
                self.red_ships_lost,
                self.red_runways_damaged,
                self.red_bases_lost,
                self.blue_aircraft_killed,
                self.blue_ground_units_killed,
                self.blue_static_defenses_killed,
                self.blue_ships_killed,
                self.blue_bases_captured,
            )
        )

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "red_aircraft_lost": self.red_aircraft_lost,
            "red_aircraft_lost_by_cause": dict(self.red_aircraft_lost_by_cause),
            "red_aircraft_lost_by_type": dict(self.red_aircraft_lost_by_type),
            "red_ground_units_lost": self.red_ground_units_lost,
            "red_ground_units_lost_by_cause": dict(self.red_ground_units_lost_by_cause),
            "red_static_defenses_lost": self.red_static_defenses_lost,
            "red_ships_lost": self.red_ships_lost,
            "red_runways_damaged": self.red_runways_damaged,
            "red_bases_lost": self.red_bases_lost,
            "blue_aircraft_killed": self.blue_aircraft_killed,
            "blue_ground_units_killed": self.blue_ground_units_killed,
            "blue_static_defenses_killed": self.blue_static_defenses_killed,
            "blue_ships_killed": self.blue_ships_killed,
            "blue_bases_captured": self.blue_bases_captured,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DebriefSummary:
        def as_int(key: str) -> int:
            try:
                return int(data.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0

        def as_int_map(key: str) -> Dict[str, int]:
            raw = data.get(key, {}) or {}
            result: Dict[str, int] = {}
            if isinstance(raw, Mapping):
                for name, count in raw.items():
                    try:
                        result[str(name)] = int(count)
                    except (TypeError, ValueError):
                        continue
            return result

        return cls(
            turn=as_int("turn"),
            red_aircraft_lost=as_int("red_aircraft_lost"),
            red_aircraft_lost_by_cause=as_int_map("red_aircraft_lost_by_cause"),
            red_aircraft_lost_by_type=as_int_map("red_aircraft_lost_by_type"),
            red_ground_units_lost=as_int("red_ground_units_lost"),
            red_ground_units_lost_by_cause=as_int_map("red_ground_units_lost_by_cause"),
            red_static_defenses_lost=as_int("red_static_defenses_lost"),
            red_ships_lost=as_int("red_ships_lost"),
            red_runways_damaged=as_int("red_runways_damaged"),
            red_bases_lost=as_int("red_bases_lost"),
            blue_aircraft_killed=as_int("blue_aircraft_killed"),
            blue_ground_units_killed=as_int("blue_ground_units_killed"),
            blue_static_defenses_killed=as_int("blue_static_defenses_killed"),
            blue_ships_killed=as_int("blue_ships_killed"),
            blue_bases_captured=as_int("blue_bases_captured"),
        )

    # -- rendering --------------------------------------------------------

    @staticmethod
    def _render_causes(by_cause: Mapping[str, int]) -> str:
        parts = []
        for category in ThreatCategory:
            count = by_cause.get(category.value, 0)
            if count > 0:
                parts.append(f"{category.label}={count}")
        return ", ".join(parts)

    def render_compact(self) -> str:
        """A short plain-text block for the prompt. Empty string if nothing to say."""

        if self.is_empty:
            return ""

        lines = [
            f"[AFTER-ACTION turn={self.turn}]",
            "note: losses below already happened last turn and are reflected in "
            "the current force counts above; do not reconcile or infer earlier "
            "totals.",
        ]

        if self.red_aircraft_lost:
            line = f"red_aircraft_lost={self.red_aircraft_lost}"
            causes = self._render_causes(self.red_aircraft_lost_by_cause)
            if causes:
                line += f" by_cause: {causes}"
            lines.append(line)
            if self.red_aircraft_lost_by_type:
                by_type = ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(self.red_aircraft_lost_by_type.items())
                )
                lines.append(f"  airframes_lost: {by_type}")

        if self.red_ground_units_lost:
            line = f"red_ground_units_lost={self.red_ground_units_lost}"
            causes = self._render_causes(self.red_ground_units_lost_by_cause)
            if causes:
                line += f" by_cause: {causes}"
            lines.append(line)

        other_red = []
        if self.red_static_defenses_lost:
            other_red.append(f"static_defenses={self.red_static_defenses_lost}")
        if self.red_ships_lost:
            other_red.append(f"ships={self.red_ships_lost}")
        if self.red_runways_damaged:
            other_red.append(f"runways_damaged={self.red_runways_damaged}")
        if self.red_bases_lost:
            other_red.append(f"bases_lost={self.red_bases_lost}")
        if other_red:
            lines.append("red_other_losses: " + ", ".join(other_red))

        kills = []
        if self.blue_aircraft_killed:
            kills.append(f"aircraft={self.blue_aircraft_killed}")
        if self.blue_ground_units_killed:
            kills.append(f"ground_units={self.blue_ground_units_killed}")
        if self.blue_static_defenses_killed:
            kills.append(f"static_defenses={self.blue_static_defenses_killed}")
        if self.blue_ships_killed:
            kills.append(f"ships={self.blue_ships_killed}")
        if self.blue_bases_captured:
            kills.append(f"bases_captured={self.blue_bases_captured}")
        if kills:
            lines.append("confirmed_enemy_losses: " + ", ".join(kills))

        return "\n".join(lines)


def build_debrief_summary(debriefing: Debriefing, game: Game) -> DebriefSummary:
    """Distil a :class:`Debriefing` into a RED-perspective after-action summary.

    Degrades gracefully: any resolution that fails simply leaves the affected
    loss in the ``UNKNOWN`` cause bucket instead of raising.
    """

    from game.theater import Player

    unit_map = debriefing.unit_map
    red_counts = debriefing.loss_counts(Player.RED)
    blue_counts = debriefing.loss_counts(Player.BLUE)

    # RED air losses by cause -------------------------------------------------
    air_by_cause: Dict[ThreatCategory, int] = defaultdict(int)
    ground_by_cause: Dict[ThreatCategory, int] = defaultdict(int)
    attributed_air = 0
    attributed_ground = 0
    seen_targets: set[str] = set()

    for cause in debriefing.state_data.kill_causes:
        target = cause.get("target")
        if not target or target in seen_targets:
            continue
        seen_targets.add(target)

        flying = unit_map.flight(target)
        if flying is not None:
            if not flying.flight.departure.captured.is_blue:  # RED aircraft
                air_by_cause[_resolve_cause(cause, unit_map)] += 1
                attributed_air += 1
            continue

        if _ground_victim_is_red(target, unit_map):
            ground_by_cause[_resolve_cause(cause, unit_map)] += 1
            attributed_ground += 1

    red_air_total = red_counts.aircraft
    if red_air_total > attributed_air:
        air_by_cause[ThreatCategory.UNKNOWN] += red_air_total - attributed_air

    red_ground_total = (
        red_counts.front_line
        + red_counts.motorpool
        + red_counts.convoy
        + red_counts.ground_objects
    )
    if red_ground_total > attributed_ground:
        ground_by_cause[ThreatCategory.UNKNOWN] += red_ground_total - attributed_ground

    red_air_by_type = {
        str(aircraft_type): count
        for aircraft_type, count in debriefing.air_losses.by_type(Player.RED).items()
    }

    return DebriefSummary(
        turn=int(game.turn),
        red_aircraft_lost=red_air_total,
        red_aircraft_lost_by_cause=_sorted_cause_map(air_by_cause),
        red_aircraft_lost_by_type=dict(sorted(red_air_by_type.items())),
        red_ground_units_lost=red_ground_total,
        red_ground_units_lost_by_cause=_sorted_cause_map(ground_by_cause),
        red_static_defenses_lost=red_counts.ground_objects,
        red_ships_lost=red_counts.cargo_ships,
        red_runways_damaged=red_counts.runways_destroyed,
        red_bases_lost=red_counts.bases_lost,
        blue_aircraft_killed=blue_counts.aircraft,
        blue_ground_units_killed=(
            blue_counts.front_line + blue_counts.motorpool + blue_counts.convoy
        ),
        blue_static_defenses_killed=blue_counts.ground_objects,
        blue_ships_killed=blue_counts.cargo_ships,
        blue_bases_captured=blue_counts.bases_lost,
    )


def _ground_victim_is_red(name: str, unit_map: UnitMap) -> bool:
    """True if a killed non-air unit belongs to RED.

    Mirrors the side assignment in ``Debriefing.dead_ground_units`` so that
    cause attribution counts the same population as the loss totals.
    """

    from game.theater import Player

    front_line = unit_map.front_line_unit(name)
    if front_line is not None:
        return not front_line.origin.captured.is_blue

    motorpool = unit_map.motorpool_unit(name)
    if motorpool is not None:
        return not motorpool.origin.captured.is_blue

    convoy = unit_map.convoy_unit(name)
    if convoy is not None:
        return not convoy.convoy.player_owned.is_blue

    theater = unit_map.theater_units(name)
    if theater is not None:
        return not theater.theater_unit.ground_object.is_friendly(to_player=Player.BLUE)

    return False
