"""Deterministic attack-helicopter commitment (:mod:`game.ai_commander.planexecution`).

The air-tasking stage habitually leaves RED's rotary-wing squadrons idle, and
when it times out no packages are planned at all. ``auto_commit_attack_helicopters``
is the code-level backstop that runs regardless of that stage and, for RED only:

* enables CAS/BAI on every capable attack-helicopter squadron;
* forward-relocates rear squadrons to a RED base near the front with room;
* force-commits idle, in-range squadrons through the same ``PackageFulfiller``
  path the air-tasking stage uses.

These tests use small hand-built fakes rather than the synthetic campaign: the
behaviour under test reads a handful of squadron and control-point accessors and
the synthetic ``MagicMock`` control points carry no front-line geometry, so a
focused fake proves the logic far more directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Optional, cast
from unittest.mock import MagicMock

import pytest
from dcs.mapping import Point
from dcs.terrain import Terrain

from game.ai_commander.planexecution import PlanExecutor
from game.ato.flighttype import FlightType
from game.theater.player import Player
from game.utils import Distance, meters, nautical_miles


def _point(x: float, y: float) -> Point:
    return Point(x, y, cast(Terrain, MagicMock(spec=Terrain)))


@dataclass
class FakeAircraft:
    helicopter: bool = True
    lha_capable: bool = False
    flyable: bool = False
    name: str = "Mi-28N Havoc"


@dataclass
class FakeFront:
    position: Point


@dataclass
class FakeControlPoint:
    name: str
    position: Point
    front_lines: dict[int, FakeFront] = field(default_factory=dict)
    parking: int = 10
    operable: bool = True

    def unclaimed_parking(self, parking_type: Any) -> int:
        return self.parking

    def can_operate(self, aircraft: Any) -> bool:
        return self.operable


@dataclass
class FakeSettings:
    ground_start_ai_planes: bool = False


@dataclass
class FakeSquadron:
    name: str
    aircraft: FakeAircraft
    location: FakeControlPoint
    capable_tasks: frozenset[FlightType] = frozenset({FlightType.CAS, FlightType.BAI})
    auto_assignable_mission_types: set[FlightType] = field(default_factory=set)
    destination: Any = None
    owned_aircraft: int = 8
    untasked_aircraft: int = 8
    pending_deliveries: int = 0
    coalition: Any = None
    #: recorded calls
    relocations: list[Any] = field(default_factory=list)
    tasking_calls: list[frozenset[FlightType]] = field(default_factory=list)
    assign_queries: list[tuple[Any, FlightType, int, bool, bool]] = field(
        default_factory=list
    )
    can_assign: bool = True

    def capable_of(self, task: FlightType) -> bool:
        return task in self.capable_tasks

    @property
    def expected_size_next_turn(self) -> int:
        return self.owned_aircraft + self.pending_deliveries

    def set_auto_assignable_mission_types(self, mission_types: Any) -> None:
        allowed = frozenset(t for t in mission_types if self.capable_of(t))
        self.tasking_calls.append(allowed)
        self.auto_assignable_mission_types = set(allowed)

    def can_auto_assign_mission(
        self, location: Any, task: FlightType, size: int, heli: bool, this_turn: bool
    ) -> bool:
        self.assign_queries.append((location, task, size, heli, this_turn))
        return self.can_assign

    def plan_relocation(self, destination: Any, now: Any) -> None:
        self.relocations.append(destination)
        self.destination = destination

    def __str__(self) -> str:
        return self.name


@dataclass
class FakeAirWing:
    squadrons: list[FakeSquadron]

    def iter_squadrons(self) -> Iterator[FakeSquadron]:
        return iter(self.squadrons)


@dataclass
class FakeAto:
    packages: list[Any] = field(default_factory=list)

    def add_package(self, package: Any) -> None:
        self.packages.append(package)


@dataclass
class FakePlayer:
    is_red: bool


@dataclass
class FakeCoalition:
    player: FakePlayer
    air_wing: FakeAirWing
    ato: FakeAto
    game: Any = None


@dataclass
class FakeConditions:
    start_time: datetime = field(default_factory=lambda: datetime(2020, 6, 1, 12))


@dataclass
class FakeGameDb:
    flights: Any = None


@dataclass
class FakeTheater:
    control_points: list[FakeControlPoint]

    def control_points_for(self, player: Any) -> list[FakeControlPoint]:
        return self.control_points

    def heading_to_conflict_from(self, position: Any) -> Any:
        return None


@dataclass
class FakeGame:
    theater: FakeTheater
    settings: FakeSettings = field(default_factory=FakeSettings)
    conditions: FakeConditions = field(default_factory=FakeConditions)
    db: FakeGameDb = field(default_factory=FakeGameDb)


class _FakePackage:
    def __init__(self, mission: Any) -> None:
        self.mission = mission


class _FakeFulfiller:
    """Records the missions it is asked to plan and returns a package."""

    instances: list["_FakeFulfiller"] = []
    refuse: bool = False

    def __init__(
        self, coalition: Any, theater: Any, flights: Any, settings: Any
    ) -> None:
        self.missions: list[Any] = []
        _FakeFulfiller.instances.append(self)

    def plan_mission(self, mission: Any, count: int, now: Any, tracer: Any) -> Any:
        self.missions.append(mission)
        if _FakeFulfiller.refuse:
            return None
        return _FakePackage(mission)


@pytest.fixture(autouse=True)
def _patch_fulfiller(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeFulfiller.instances.clear()
    _FakeFulfiller.refuse = False
    monkeypatch.setattr(
        "game.commander.packagefulfiller.PackageFulfiller", _FakeFulfiller
    )


def _build(
    *,
    is_red: bool = True,
    front_at: Point | None = None,
    squadron_kwargs: dict[str, Any] | None = None,
    extra_control_points: list[FakeControlPoint] | None = None,
) -> tuple[PlanExecutor, FakeCoalition, FakeSquadron, FakeControlPoint]:
    front_position = front_at if front_at is not None else _point(0, 0)
    front = FakeFront(position=front_position)
    front_base = FakeControlPoint(
        name="Front Base", position=_point(0, 0), front_lines={id(front): front}
    )
    kwargs: dict[str, Any] = {"aircraft": FakeAircraft(), "location": front_base}
    kwargs.update(squadron_kwargs or {})
    squadron = FakeSquadron(name="Havoc SQN", **kwargs)
    control_points = [front_base]
    if extra_control_points:
        control_points.extend(extra_control_points)
    theater = FakeTheater(control_points=control_points)
    game = FakeGame(theater=theater)
    coalition = FakeCoalition(
        player=FakePlayer(is_red=is_red),
        air_wing=FakeAirWing(squadrons=[squadron]),
        ato=FakeAto(),
        game=game,
    )
    squadron.coalition = coalition
    executor = PlanExecutor(cast(Any, game), cast(Any, coalition))
    return executor, coalition, squadron, front_base


class TestTaskEnabling:
    def test_cas_and_bai_are_enabled_on_a_capable_heli_squadron(self) -> None:
        executor, _coalition, squadron, _base = _build()
        assert squadron.auto_assignable_mission_types == set()
        executor.auto_commit_attack_helicopters()
        assert FlightType.CAS in squadron.auto_assignable_mission_types
        assert FlightType.BAI in squadron.auto_assignable_mission_types

    def test_an_incapable_task_is_never_enabled(self) -> None:
        executor, _coalition, squadron, _base = _build(
            squadron_kwargs={"capable_tasks": frozenset({FlightType.CAS})}
        )
        executor.auto_commit_attack_helicopters()
        assert FlightType.BAI not in squadron.auto_assignable_mission_types


class TestRedOnlyGuard:
    def test_blue_coalition_is_left_untouched(self) -> None:
        executor, coalition, squadron, _base = _build(is_red=False)
        executor.auto_commit_attack_helicopters()
        assert squadron.tasking_calls == []
        assert squadron.relocations == []
        assert coalition.ato.packages == []

    def test_a_fixed_wing_squadron_is_ignored(self) -> None:
        executor, coalition, squadron, _base = _build(
            squadron_kwargs={"aircraft": FakeAircraft(helicopter=False)}
        )
        executor.auto_commit_attack_helicopters()
        assert squadron.tasking_calls == []
        assert coalition.ato.packages == []


class TestForwardRelocation:
    def test_a_rear_squadron_relocates_to_a_forward_base_with_room(self) -> None:
        # Squadron sits 80nm behind the front; a base on the front has room.
        rear = _point(0, nautical_miles(80).meters)
        executor, _coalition, squadron, front_base = _build()
        # Move the squadron's home to the rear; the front base stays at the front.
        squadron.location = FakeControlPoint(
            name="Rear Base", position=rear, parking=10
        )
        executor.game.theater.control_points.append(squadron.location)  # type: ignore[attr-defined]
        executor.auto_commit_attack_helicopters()
        # It relocated to the base that sits on the front (within 50nm).
        assert squadron.relocations == [front_base]

    def test_a_forward_squadron_is_not_relocated(self) -> None:
        executor, _coalition, squadron, _base = _build()
        executor.auto_commit_attack_helicopters()
        assert squadron.relocations == []

    def test_a_squadron_is_not_relocated_when_no_base_has_room(self) -> None:
        rear = _point(0, nautical_miles(80).meters)
        executor, _coalition, squadron, front_base = _build()
        front_base.parking = 0  # the only forward base has no room
        squadron.location = FakeControlPoint(
            name="Rear Base", position=rear, parking=10
        )
        executor.game.theater.control_points.append(squadron.location)  # type: ignore[attr-defined]
        executor.auto_commit_attack_helicopters()
        assert squadron.relocations == []


class TestForceCommitment:
    def test_an_idle_in_range_squadron_is_committed_into_a_package(self) -> None:
        executor, coalition, squadron, _base = _build()
        executor.auto_commit_attack_helicopters()
        assert len(coalition.ato.packages) == 1
        assert executor.report.packages_added == 1

    def test_a_relocating_squadron_is_not_committed_this_turn(self) -> None:
        rear = _point(0, nautical_miles(80).meters)
        executor, coalition, squadron, _front_base = _build()
        squadron.location = FakeControlPoint(
            name="Rear Base", position=rear, parking=10
        )
        executor.game.theater.control_points.append(squadron.location)  # type: ignore[attr-defined]
        executor.auto_commit_attack_helicopters()
        # It was relocated, so it must not also be committed to fly this turn.
        assert squadron.relocations != []
        assert coalition.ato.packages == []

    def test_a_fully_tasked_squadron_is_not_committed(self) -> None:
        executor, coalition, squadron, _base = _build(
            squadron_kwargs={"untasked_aircraft": 0}
        )
        executor.auto_commit_attack_helicopters()
        assert coalition.ato.packages == []

    def test_a_refused_mission_adds_no_package_and_records_failure(self) -> None:
        _FakeFulfiller.refuse = True
        executor, coalition, squadron, _base = _build()
        executor.auto_commit_attack_helicopters()
        assert coalition.ato.packages == []
        assert executor.report.packages_added == 0
        assert any(
            not outcome.applied and outcome.kind == "heli_package"
            for outcome in executor.report.outcomes
        )

    def test_a_squadron_the_game_will_not_auto_assign_is_not_committed(self) -> None:
        executor, coalition, squadron, _base = _build(
            squadron_kwargs={"can_assign": False}
        )
        executor.auto_commit_attack_helicopters()
        assert coalition.ato.packages == []
