"""A synthetic, headless campaign state for the RED commander tests.

Retribution's real :class:`~game.game.Game` cannot be built without DCS
installed, so the tests drive :mod:`game.ai_commander` against a hand-built
stand-in. Two rules keep that honest:

* Everything the code under test actually *reads* is present with a realistic
  value, so no assertion passes because an attribute was missing.
* Every BLUE-side value that must never reach the model is a **sentinel**: a
  value that appears nowhere else in the campaign. The intel-leak tests
  serialise the whole brief (and the whole audit record) and assert that no
  sentinel appears anywhere in it, rather than checking a hand-written list of
  field names that a future refactor could quietly grow past.

The RED side gets sentinels too. Without them a filter that returned an empty
brief would pass every leak test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence, cast
from unittest.mock import MagicMock

from dcs.mapping import Point
from dcs.terrain import Terrain

from game.ai_commander.config import AiCommanderConfig
from game.ai_commander.enums import CommanderPersonality, IntelPolicy
from game.ai_commander.llmclient import (
    ChatCompletionClient,
    LlmError,
    LlmResponse,
    TokenUsage,
)
from game.ai_commander.serialization import canonical_json
from game.ground_forces.combat_stance import CombatStance
from game.theater.controlpoint import ControlPoint
from game.theater.player import Player
from game.theater.theatergroundobject import IadsGroundObject

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

#: BLUE facts that a REALISTIC brief must never contain, in any form. Values are
#: chosen so that they cannot plausibly be produced by any other computation in the
#: projection: no count, price or capacity in the synthetic campaign collides
#: with them.
BLUE_SENTINELS: dict[str, object] = {
    # Economy.
    "blue_budget": 987654321.0,
    "blue_income_per_turn": 876543219,
    # Air wing.
    "blue_squadron_name": "SQN-BLUE-LEAK-CANARY",
    "blue_aircraft_name": "F-BLUELEAK-99",
    "blue_pilot_count": 424242,
    # Planned ATO.
    "blue_planned_package": "PKG-BLUE-LEAK-CANARY",
    # Ground forces and base internals.
    "blue_deployable_units": 1157,
    "blue_unit_capacity": 3130,
    # Things RED has no way of observing.
    "blue_undetected_tgo": "TGO-BLUE-UNDETECTED-CANARY",
    "blue_hidden_base": "BLUE-SECRET-BASE-CANARY",
}

#: RED facts the commander is entitled to see. At least one of each must survive
#: projection, otherwise the "no leaks" assertions would be vacuous.
RED_SENTINELS: dict[str, object] = {
    "red_budget": 1357911.0,
    "red_income_per_turn": 2468013,
    "red_deployable_units": 2460,
    "red_unit_capacity": 8100,
    "red_aircraft_present": 137,
    "red_aircraft_ordered": 29,
    "red_base_name": "RED-FRONT-BASE",
}

#: A BLUE base that faces RED across a front is *not* secret: base ownership and
#: front-line geography are drawn on the campaign map for both sides.
PUBLIC_BLUE_FRONT_BASE = "BLUE-FRONT-BASE"


def _sentinel_patterns(sentinels: dict[str, object]) -> dict[str, re.Pattern[str]]:
    """One matcher per sentinel.

    Numbers are matched on word boundaries so a four-digit sentinel cannot be
    "found" inside a hexadecimal campaign hash; strings are matched literally.
    """

    patterns: dict[str, re.Pattern[str]] = {}
    for name, value in sentinels.items():
        if isinstance(value, bool):  # pragma: no cover - no boolean sentinels
            raise AssertionError("boolean sentinels are not distinguishable")
        if isinstance(value, float):
            digits = str(int(value)) if value.is_integer() else str(value)
            patterns[name] = re.compile(rf"\b{re.escape(digits)}\b")
        elif isinstance(value, int):
            patterns[name] = re.compile(rf"\b{value}\b")
        else:
            patterns[name] = re.compile(re.escape(str(value)))
    return patterns


BLUE_SENTINEL_PATTERNS = _sentinel_patterns(BLUE_SENTINELS)
RED_SENTINEL_PATTERNS = _sentinel_patterns(RED_SENTINELS)


def serialise_everything(*values: Any) -> str:
    """Flatten anything into one searchable blob.

    Mappings, dataclasses, enums and tuples all go through
    :func:`~game.ai_commander.serialization.canonical_json`, which is the same
    function that writes the audit log, so the text scanned here is the text
    that would actually be shipped to a provider or written to disk.
    """

    parts: list[str] = []
    for value in values:
        parts.append(value if isinstance(value, str) else canonical_json(value))
    return "\n".join(parts)


def blue_leaks_in(blob: str) -> list[str]:
    """Names of every BLUE sentinel that appears in ``blob``."""

    return sorted(
        name for name, pattern in BLUE_SENTINEL_PATTERNS.items() if pattern.search(blob)
    )


def red_facts_missing_from(blob: str) -> list[str]:
    """Names of RED sentinels that did *not* survive projection."""

    return sorted(
        name
        for name, pattern in RED_SENTINEL_PATTERNS.items()
        if not pattern.search(blob)
    )


# ---------------------------------------------------------------------------
# Small stand-ins
# ---------------------------------------------------------------------------


def point(x: float, y: float) -> Point:
    return Point(x, y, cast(Terrain, MagicMock(spec=Terrain)))


@dataclass
class FakeAllocations:
    """Stand-in for the aircraft/ground-unit allocation summaries."""

    total_present: int = 0
    total_ordered: int = 0

    @property
    def total(self) -> int:
        return self.total_present + self.total_ordered


@dataclass
class FakeRunwayStatus:
    needs_repair: bool = False


@dataclass
class FakeThreatZone:
    """Threat coverage. ``False`` everywhere keeps observability purely spatial."""

    covers: bool = False

    def threatened(self, position: Any) -> bool:
        return self.covers


@dataclass
class FakeFrontLine:
    """Only the two accessors the commander code uses."""

    friendly: Any
    hostile: Any

    def control_point_friendly_to(self, player: Player) -> Any:
        return self.friendly if player.is_red else self.hostile

    def control_point_hostile_to(self, player: Player) -> Any:
        return self.hostile if player.is_red else self.friendly


def make_control_point(
    *,
    cp_id: int,
    name: str,
    captured: Player,
    position: Point,
    deployable: int = 0,
    capacity: int = 0,
    aircraft_present: int = 0,
    aircraft_ordered: int = 0,
    ground_present: int = 0,
    ground_ordered: int = 0,
    income_per_turn: int = 0,
    runway_operational: bool = True,
    runway_needs_repair: bool = False,
    runway_repairable: bool = False,
    ground_unit_source: bool = True,
    can_recruit: bool = True,
    active_frontline: bool = False,
    ground_objects: Optional[Sequence[Any]] = None,
    stances: Optional[dict[int, CombatStance]] = None,
) -> ControlPoint:
    """A control point exposing every attribute the commander code reads."""

    cp = MagicMock(spec=ControlPoint)
    cp.id = cp_id
    cp.name = name
    cp.captured = captured
    cp.position = position
    cp.deployable_front_line_units = deployable
    cp.frontline_unit_count_limit = capacity
    cp.income_per_turn = income_per_turn
    cp.ground_objects = list(ground_objects or [])
    cp.stances = dict(stances or {})
    cp.is_global = False
    cp.has_active_frontline = active_frontline
    cp.runway_status = FakeRunwayStatus(runway_needs_repair)
    cp.runway_can_be_repaired = runway_repairable
    cp.runway_is_operational = MagicMock(return_value=runway_operational)
    cp.has_ground_unit_source = MagicMock(return_value=ground_unit_source)
    cp.can_recruit_ground_units = MagicMock(return_value=can_recruit)
    cp.allocated_aircraft = MagicMock(
        return_value=FakeAllocations(aircraft_present, aircraft_ordered)
    )
    cp.allocated_ground_units = MagicMock(
        return_value=FakeAllocations(ground_present, ground_ordered)
    )
    cp.is_friendly = lambda player: bool(captured is player)
    return cast(ControlPoint, cp)


def make_iads(name: str, position: Point) -> Any:
    """A BLUE SAM site. ``isinstance`` still reports ``IadsGroundObject``."""

    tgo = MagicMock(spec=IadsGroundObject)
    tgo.name = name
    tgo.position = position
    tgo.category = "aa"
    tgo.is_dead = False
    return tgo


@dataclass
class FakeUnitType:
    name: str
    price: int


@dataclass
class FakeSquadron:
    name: str
    aircraft: FakeUnitType
    pilot_count: int = 0


@dataclass
class FakeAirWing:
    squadrons: list[FakeSquadron] = field(default_factory=list)

    def iter_squadrons(self) -> Iterator[FakeSquadron]:
        return iter(self.squadrons)


@dataclass
class FakeFaction:
    name: str
    frontline_units: list[FakeUnitType] = field(default_factory=list)
    artillery_units: list[FakeUnitType] = field(default_factory=list)


class _EmptyMovementPool:
    def travelling_to(self, control_point: Any) -> list[Any]:
        return []


class FakeTransfers:
    """Enough of ``PendingTransfers`` for the projector and procurement."""

    def __init__(self) -> None:
        self.convoys = _EmptyMovementPool()
        self.cargo_ships = _EmptyMovementPool()


class FakeCoalition:
    def __init__(
        self,
        player: Player,
        faction: FakeFaction,
        budget: float,
        air_wing: FakeAirWing,
        packages: Optional[list[str]] = None,
    ) -> None:
        self.player = player
        self.faction = faction
        self.budget = budget
        self.air_wing = air_wing
        self.transfers = FakeTransfers()
        self.packages = list(packages or [])
        self.game: Any = None
        self._opponent: Optional[FakeCoalition] = None

    @property
    def opponent(self) -> FakeCoalition:
        assert self._opponent is not None
        return self._opponent


# ---------------------------------------------------------------------------
# The synthetic campaign
# ---------------------------------------------------------------------------


class FakeTheater:
    def __init__(
        self, terrain_name: str, controlpoints: list[ControlPoint], fronts: list[Any]
    ) -> None:
        self.terrain_name = terrain_name
        self.controlpoints = controlpoints
        self.fronts = fronts

    def conflicts(self) -> Iterator[Any]:
        return iter(self.fronts)

    def control_points_for(self, player: Player) -> Iterator[ControlPoint]:
        return (cp for cp in self.controlpoints if cp.captured is player)


class FakeSettings:
    """Only the settings the exercised code paths read."""

    def __init__(self) -> None:
        self.enemy_income_multiplier = 1.0
        self.player_income_multiplier = 1.0
        self.frontline_reserves_factor = 100
        self.frontline_reserves_factor_red = 100
        self.airbase_threat_range = 100
        self.perf_disable_convoys = True
        self.motorpool_enabled = False
        self.motorpool_spawn_cap = 0
        self.automate_front_line_stance = True


class SyntheticCampaign:
    """A one-front toy campaign with sentinel-marked BLUE state.

    Layout, in metres, against RED's 120 km observation range:

    * ``RED-FRONT-BASE`` at (0, 0) and ``RED-REAR-BASE`` at (0, 60 000).
    * ``BLUE-FRONT-BASE`` at (40 000, 0) faces RED across the front and is well
      inside RED's observation range. Its existence is public.
    * ``BLUE-SECRET-BASE-CANARY`` at (900 000, 0) is far outside it.
    * Two BLUE SAM sites sit near the front (observable); three sit beside the
      distant base (not observable).

    RED has 2460 deployable units against BLUE's 1157, a ratio of 2.13, so every
    posture including ``BREAKTHROUGH`` passes the game's own force-balance
    precondition. That is deliberate: a campaign where nothing was legal would
    make the legality tests untrustworthy.
    """

    NEAR_IADS = 2
    FAR_IADS = 3

    def __init__(
        self,
        red_budget: Optional[float] = None,
        red_deployable: Optional[int] = None,
        turn: int = 7,
    ) -> None:
        self.red_front_base = make_control_point(
            cp_id=1,
            name=cast(str, RED_SENTINELS["red_base_name"]),
            captured=Player.RED,
            position=point(0.0, 0.0),
            deployable=(
                cast(int, RED_SENTINELS["red_deployable_units"])
                if red_deployable is None
                else red_deployable
            ),
            capacity=cast(int, RED_SENTINELS["red_unit_capacity"]),
            aircraft_present=cast(int, RED_SENTINELS["red_aircraft_present"]),
            aircraft_ordered=cast(int, RED_SENTINELS["red_aircraft_ordered"]),
            ground_present=12,
            ground_ordered=3,
            income_per_turn=cast(int, RED_SENTINELS["red_income_per_turn"]),
            active_frontline=True,
            stances={20: CombatStance.DEFENSIVE},
        )
        self.red_rear_base = make_control_point(
            cp_id=2,
            name="RED-REAR-BASE",
            captured=Player.RED,
            position=point(0.0, 60_000.0),
            deployable=40,
            capacity=100,
            aircraft_present=0,
            runway_operational=False,
            runway_needs_repair=True,
            runway_repairable=True,
        )

        near_iads = [
            make_iads(f"BLUE SAM near {index}", point(30_000.0, 1_000.0 * index))
            for index in range(self.NEAR_IADS)
        ]
        far_iads = [
            make_iads(
                f"{BLUE_SENTINELS['blue_undetected_tgo']}-{index}",
                point(900_000.0, 1_000.0 * index),
            )
            for index in range(self.FAR_IADS)
        ]

        self.blue_front_base = make_control_point(
            cp_id=20,
            name=PUBLIC_BLUE_FRONT_BASE,
            captured=Player.BLUE,
            position=point(40_000.0, 0.0),
            deployable=cast(int, BLUE_SENTINELS["blue_deployable_units"]),
            capacity=cast(int, BLUE_SENTINELS["blue_unit_capacity"]),
            aircraft_present=cast(int, BLUE_SENTINELS["blue_pilot_count"]),
            income_per_turn=cast(int, BLUE_SENTINELS["blue_income_per_turn"]),
            active_frontline=True,
            ground_objects=near_iads,
        )
        self.blue_hidden_base = make_control_point(
            cp_id=21,
            name=cast(str, BLUE_SENTINELS["blue_hidden_base"]),
            captured=Player.BLUE,
            position=point(900_000.0, 0.0),
            deployable=cast(int, BLUE_SENTINELS["blue_deployable_units"]),
            capacity=cast(int, BLUE_SENTINELS["blue_unit_capacity"]),
            income_per_turn=cast(int, BLUE_SENTINELS["blue_income_per_turn"]),
            ground_objects=far_iads,
        )

        self.control_points: list[ControlPoint] = [
            self.red_front_base,
            self.red_rear_base,
            self.blue_front_base,
            self.blue_hidden_base,
        ]
        self.front = FakeFrontLine(self.red_front_base, self.blue_front_base)
        self.theater = FakeTheater(
            "Syria", self.control_points, [cast(Any, self.front)]
        )

        self.red = FakeCoalition(
            player=Player.RED,
            faction=FakeFaction(
                "Red Sentinel Faction",
                frontline_units=[FakeUnitType("RED-TANK", 12)],
                artillery_units=[FakeUnitType("RED-ARTY", 18)],
            ),
            budget=(
                cast(float, RED_SENTINELS["red_budget"])
                if red_budget is None
                else red_budget
            ),
            air_wing=FakeAirWing(
                [
                    FakeSquadron("RED SQN 1", FakeUnitType("RED-JET", 22), 11),
                    FakeSquadron("RED SQN 2", FakeUnitType("RED-BOMBER", 34), 9),
                ]
            ),
        )
        self.blue = FakeCoalition(
            player=Player.BLUE,
            faction=FakeFaction(
                "Blue Faction", frontline_units=[FakeUnitType("BLUE-TANK", 14)]
            ),
            budget=cast(float, BLUE_SENTINELS["blue_budget"]),
            air_wing=FakeAirWing(
                [
                    FakeSquadron(
                        cast(str, BLUE_SENTINELS["blue_squadron_name"]),
                        FakeUnitType(
                            cast(str, BLUE_SENTINELS["blue_aircraft_name"]),
                            cast(int, BLUE_SENTINELS["blue_pilot_count"]),
                        ),
                        cast(int, BLUE_SENTINELS["blue_pilot_count"]),
                    )
                ]
            ),
            packages=[cast(str, BLUE_SENTINELS["blue_planned_package"])],
        )
        self.red._opponent = self.blue
        self.blue._opponent = self.red
        self.red.game = self
        self.blue.game = self

        self.settings = FakeSettings()
        self.turn = turn
        self.threat_zone = FakeThreatZone(covers=False)

    # -- the ``Game`` surface the commander uses --------------------------

    def coalition_for(self, player: Player) -> FakeCoalition:
        return self.red if player.is_red else self.blue

    def threat_zone_for(self, player: Player) -> FakeThreatZone:
        return self.threat_zone


def synthetic_game(
    red_budget: Optional[float] = None,
    red_deployable: Optional[int] = None,
    turn: int = 7,
) -> tuple[SyntheticCampaign, Any]:
    """A campaign plus the value to pass where a ``Game`` is expected."""

    campaign = SyntheticCampaign(
        red_budget=red_budget, red_deployable=red_deployable, turn=turn
    )
    return campaign, cast(Any, campaign)


# ---------------------------------------------------------------------------
# Objective enumeration
# ---------------------------------------------------------------------------


class FakeObjectiveFinder:
    """A deterministic stand-in for :class:`ObjectiveFinder`.

    The real finder needs airfield-distance caches and live theater geometry.
    Replacing it keeps the *filtering* under test: which objectives RED is
    allowed to know about is decided by :mod:`game.ai_commander.intel`, and this
    class deliberately hands it everything, observable or not.
    """

    def __init__(self, game: Any, is_player: Player) -> None:
        self.game = game
        self.is_player = is_player

    def _enemy_control_points(self) -> list[Any]:
        return [
            cp
            for cp in self.game.theater.controlpoints
            if cp.captured is not self.is_player and cp.captured is not Player.NEUTRAL
        ]

    def front_lines(self) -> Iterator[Any]:
        return self.game.theater.conflicts()

    def enemy_air_defenses(self) -> Iterator[Any]:
        for cp in self._enemy_control_points():
            for tgo in cp.ground_objects:
                if isinstance(tgo, IadsGroundObject) and not tgo.is_dead:
                    yield tgo

    def strike_targets(self) -> Iterator[Any]:
        return iter(())

    def motorpool_targets(self) -> Iterator[Any]:
        return iter(())

    def enemy_ships(self) -> Iterator[Any]:
        return iter(())

    def convoys(self) -> Iterator[Any]:
        return iter(())

    def cargo_ships(self) -> Iterator[Any]:
        return iter(())

    def prioritized_points(self) -> list[Any]:
        return self._enemy_control_points()

    def enemy_control_points(self) -> Iterator[Any]:
        return iter(self._enemy_control_points())

    def vulnerable_control_points(self) -> Iterator[Any]:
        return (
            cp
            for cp in self.game.theater.controlpoints
            if cp.captured is self.is_player and cp.has_active_frontline
        )


def patch_objective_finder(monkeypatch: Any) -> None:
    """Install :class:`FakeObjectiveFinder` everywhere the commander imports it.

    Every commander module imports ``ObjectiveFinder`` lazily from
    ``game.commander.objectivefinder``, so one patch covers intel projection,
    legality checking and posture application.
    """

    import game.commander.objectivefinder as module

    monkeypatch.setattr(module, "ObjectiveFinder", FakeObjectiveFinder)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def make_config(
    *,
    enabled: bool = True,
    model: str = "test/model",
    api_key: Optional[str] = "unit-test-key-never-real",
    base_url: str = "https://openrouter.ai/api/v1",
    intel_policy: IntelPolicy = IntelPolicy.REALISTIC,
    personality: CommanderPersonality = CommanderPersonality.BALANCED,
    cost_cap_per_turn: float = 0.5,
    max_output_tokens: int = 2000,
    log_prompts: bool = True,
    fallback_to_builtin: bool = True,
) -> AiCommanderConfig:
    """A usable configuration that never carries a real credential."""

    return AiCommanderConfig(
        enabled=enabled,
        model=model,
        base_url=base_url,
        api_key=api_key,
        personality=personality,
        intel_policy=intel_policy,
        cost_cap_per_turn=cost_cap_per_turn,
        max_output_tokens=max_output_tokens,
        log_prompts=log_prompts,
        fallback_to_builtin=fallback_to_builtin,
    )


# ---------------------------------------------------------------------------
# Transport doubles
# ---------------------------------------------------------------------------

#: A catalogue payload in OpenRouter's shape. Prices are per token, as the real
#: API reports them.
CATALOG_PAYLOAD: dict[str, Any] = {
    "data": [
        {
            "id": "test/model",
            "context_length": 128000,
            "pricing": {"prompt": "0.000001", "completion": "0.000004"},
            "supported_parameters": ["response_format", "structured_outputs"],
        }
    ]
}


class ScriptedClient(ChatCompletionClient):
    """A :class:`ChatCompletionClient` that replays canned outcomes.

    Each entry in ``script`` is either the response text to return or an
    exception instance to raise. Subclassing rather than duck typing means the
    controller is exercised through exactly the type it uses in production, and
    no HTTP request is ever attempted.
    """

    def __init__(
        self,
        script: Sequence[object],
        model: str = "test/model",
        catalog: Any = None,
        catalog_error: Optional[LlmError] = None,
        usage: Optional[TokenUsage] = None,
        had_tool_calls: bool = False,
    ) -> None:
        super().__init__(api_key="not-a-real-key", model=model)
        self.script = list(script)
        self.calls: list[list[dict[str, str]]] = []
        self.response_formats: list[Any] = []
        self.catalog = CATALOG_PAYLOAD if catalog is None else catalog
        self.catalog_error = catalog_error
        self.had_tool_calls = had_tool_calls
        self.usage = usage or TokenUsage(
            input_tokens=1000, output_tokens=200, total_tokens=1200
        )

    def fetch_model_catalog(self) -> Any:
        if self.catalog_error is not None:
            raise self.catalog_error
        return self.catalog

    def complete(
        self,
        messages: Any,
        max_output_tokens: int = 2000,
        temperature: float = 0.2,
        response_format: Any = None,
    ) -> LlmResponse:
        self.calls.append([dict(m) for m in messages])
        self.response_formats.append(response_format)
        if not self.script:
            raise AssertionError("ScriptedClient ran out of scripted responses")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return LlmResponse(
            text=str(item),
            usage=self.usage,
            model=self.model,
            finish_reason="stop",
            request_id="req-test",
            latency_seconds=0.01,
            had_tool_calls=self.had_tool_calls,
        )
