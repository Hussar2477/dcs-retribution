"""RED static-site rebuild (SAM/EWR/ground) — legality pricing and execution.

RED cannot lose an air-defence or EWR site and rebuild it the way a human can
from ``QGroundObjectBuyMenu``. FIX B gives it functional parity, RED-only and on
RED's own money: the legality checker picks the neediest matching site at a RED
base, chooses the cheapest affordable armed-forces template, and prices it as
``layout price - current site value``; the executor replays that recipe through
the same ``ForceGroup.create_theater_group_for_tgo`` path the buy menu uses and
deducts the cost.

These tests drive the deterministic pricing helpers and the executor directly
with small fakes, because the synthetic campaign carries no RED sites and no
armed-forces catalogue — the behaviour under test is the pricing arithmetic and
the buy-menu-equivalent execution, which fakes prove far more directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from game.ai_commander.planexecution import ExecutableLogistics, PlanExecutor
from game.ai_commander.planlegality import (
    BoundGroundGroupReplacement,
    PlanLegalityChecker,
)

# ---------------------------------------------------------------------------
# Fakes for the deterministic template catalogue
# ---------------------------------------------------------------------------


@dataclass
class FakeUnitTypeWrapper:
    dcs_unit_type: Any
    price: int


@dataclass
class FakeUnitGroup:
    max_size: int
    optional: bool = False


@dataclass
class FakeTgoGroup:
    group_name: str
    unit_groups: list[FakeUnitGroup]


@dataclass
class FakeLayout:
    groups: list[FakeTgoGroup]


@dataclass
class FakeForceGroup:
    name: str
    layouts: list[FakeLayout]
    #: per unit_group: the dcs types available and their price
    dcs_types: dict[int, list[Any]] = field(default_factory=dict)
    prices: dict[Any, int] = field(default_factory=dict)
    #: recorded create_theater_group_for_tgo calls
    created: list[tuple[Any, Any, str, Any, int]] = field(default_factory=list)
    raise_on_create: bool = False

    def dcs_unit_types_for_group(self, unit_group: FakeUnitGroup) -> list[Any]:
        return self.dcs_types.get(id(unit_group), [])

    def unit_types_for_group(
        self, unit_group: FakeUnitGroup
    ) -> list[FakeUnitTypeWrapper]:
        return [
            FakeUnitTypeWrapper(dcs_unit_type=t, price=self.prices.get(t, 0))
            for t in self.dcs_types.get(id(unit_group), [])
        ]

    def create_theater_group_for_tgo(
        self,
        ground_object: Any,
        unit_group: Any,
        group_name: str,
        game: Any,
        unit_type: Any,
        unit_count: int,
    ) -> None:
        if self.raise_on_create:
            raise RuntimeError("could not build group")
        self.created.append(
            (ground_object, unit_group, group_name, unit_type, unit_count)
        )


@dataclass
class FakeArmedForces:
    groups: list[FakeForceGroup]

    def groups_for_tasks(self, tasks: list[Any]) -> list[FakeForceGroup]:
        return list(self.groups)


class FakeCoalition:
    def __init__(self, armed_forces: FakeArmedForces) -> None:
        self.armed_forces = armed_forces
        self.budget_adjustments: list[float] = []

    def adjust_budget(self, amount: float) -> None:
        self.budget_adjustments.append(amount)


# --- fake ground objects (sites) -------------------------------------------


class FakeSamSite:
    purchasable = True

    def __init__(self, name: str, unit_count: int, alive: int, value: int = 0) -> None:
        self.name = name
        self.unit_count = unit_count
        self.alive_unit_count = alive
        self.value = value
        self.groups: list[Any] = ["wreck"]
        self.heading = "OLD"
        self.position = object()

    @property
    def is_dead(self) -> bool:
        return self.alive_unit_count == 0


def _checker_with(coalition: FakeCoalition) -> PlanLegalityChecker:
    checker = object.__new__(PlanLegalityChecker)
    checker.coalition = cast(Any, coalition)
    return checker


def _single_group_force(
    name: str, unit_price: int, max_size: int, *, optional: bool = False
) -> FakeForceGroup:
    unit_group = FakeUnitGroup(max_size=max_size, optional=optional)
    layout = FakeLayout(groups=[FakeTgoGroup("Battery", [unit_group])])
    dcs_type = object()
    force = FakeForceGroup(name=name, layouts=[layout])
    force.dcs_types[id(unit_group)] = [dcs_type]
    force.prices[dcs_type] = unit_price
    return force


# ---------------------------------------------------------------------------
# _neediest_site
# ---------------------------------------------------------------------------


class TestNeediestSite:
    def _cp(self, sites: list[Any]) -> Any:
        return type("CP", (), {"ground_objects": sites})()

    def test_a_dead_site_is_the_top_priority(self) -> None:
        checker = _checker_with(FakeCoalition(FakeArmedForces([])))
        healthy = FakeSamSite("Healthy", unit_count=4, alive=3)
        dead = FakeSamSite("Dead", unit_count=6, alive=0)
        chosen = checker._neediest_site(self._cp([healthy, dead]), FakeSamSite)
        assert chosen is dead

    def test_the_most_degraded_site_is_chosen_when_none_are_dead(self) -> None:
        checker = _checker_with(FakeCoalition(FakeArmedForces([])))
        light = FakeSamSite("Light", unit_count=4, alive=3)  # missing 1
        heavy = FakeSamSite("Heavy", unit_count=6, alive=2)  # missing 4
        chosen = checker._neediest_site(self._cp([light, heavy]), FakeSamSite)
        assert chosen is heavy

    def test_a_full_strength_site_is_never_chosen(self) -> None:
        checker = _checker_with(FakeCoalition(FakeArmedForces([])))
        full = FakeSamSite("Full", unit_count=4, alive=4)
        assert checker._neediest_site(self._cp([full]), FakeSamSite) is None

    def test_a_non_purchasable_site_is_ignored(self) -> None:
        checker = _checker_with(FakeCoalition(FakeArmedForces([])))
        dead = FakeSamSite("Dead", unit_count=6, alive=0)
        dead.purchasable = False
        assert checker._neediest_site(self._cp([dead]), FakeSamSite) is None


# ---------------------------------------------------------------------------
# _price_layout
# ---------------------------------------------------------------------------


class TestPriceLayout:
    def test_price_is_gross_minus_current_value(self) -> None:
        checker = _checker_with(FakeCoalition(FakeArmedForces([])))
        force = _single_group_force("SA-6", unit_price=10, max_size=5)  # gross 50
        priced = checker._price_layout(force, force.layouts[0], current_value=20)
        assert priced is not None
        group_plan, cost = priced
        assert cost == 30  # 50 - 20
        assert len(group_plan) == 1
        assert group_plan[0][3] == 5  # count == max_size

    def test_price_never_goes_negative(self) -> None:
        checker = _checker_with(FakeCoalition(FakeArmedForces([])))
        force = _single_group_force("Cheap", unit_price=1, max_size=2)  # gross 2
        priced = checker._price_layout(force, force.layouts[0], current_value=100)
        assert priced is not None
        _group_plan, cost = priced
        assert cost == 0

    def test_a_required_group_with_no_dcs_type_makes_the_layout_unusable(self) -> None:
        checker = _checker_with(FakeCoalition(FakeArmedForces([])))
        unit_group = FakeUnitGroup(max_size=3, optional=False)
        layout = FakeLayout(groups=[FakeTgoGroup("Battery", [unit_group])])
        force = FakeForceGroup(name="Broken", layouts=[layout])
        # no dcs_types registered for unit_group -> required group cannot be built
        assert checker._price_layout(force, layout, current_value=0) is None

    def test_an_optional_group_with_no_dcs_type_is_skipped(self) -> None:
        checker = _checker_with(FakeCoalition(FakeArmedForces([])))
        required = FakeUnitGroup(max_size=2, optional=False)
        optional = FakeUnitGroup(max_size=2, optional=True)
        layout = FakeLayout(
            groups=[
                FakeTgoGroup("Main", [required]),
                FakeTgoGroup("Extra", [optional]),
            ]
        )
        force = FakeForceGroup(name="Mixed", layouts=[layout])
        dcs_type = object()
        force.dcs_types[id(required)] = [dcs_type]
        force.prices[dcs_type] = 7
        priced = checker._price_layout(force, layout, current_value=0)
        assert priced is not None
        group_plan, cost = priced
        assert cost == 14  # only the required group priced (7 * 2)
        assert len(group_plan) == 1


# ---------------------------------------------------------------------------
# _cheapest_template
# ---------------------------------------------------------------------------


class TestCheapestTemplate:
    def test_the_cheapest_affordable_template_is_chosen(self) -> None:
        cheap = _single_group_force("Cheap", unit_price=5, max_size=2)  # gross 10
        dear = _single_group_force("Dear", unit_price=50, max_size=2)  # gross 100
        checker = _checker_with(FakeCoalition(FakeArmedForces([dear, cheap])))
        site = FakeSamSite("Dead", unit_count=4, alive=0, value=0)
        result = checker._cheapest_template([], remaining=1000, ground_object=site)
        assert result is not None
        force_group, _plan, cost = result
        assert force_group is cheap
        assert cost == 10

    def test_nothing_is_returned_when_no_template_is_affordable(self) -> None:
        dear = _single_group_force("Dear", unit_price=500, max_size=4)  # gross 2000
        checker = _checker_with(FakeCoalition(FakeArmedForces([dear])))
        site = FakeSamSite("Dead", unit_count=4, alive=0, value=0)
        assert checker._cheapest_template([], remaining=50, ground_object=site) is None


# ---------------------------------------------------------------------------
# Execution: _replace_ground_groups
# ---------------------------------------------------------------------------


@dataclass
class FakeTheater:
    def heading_to_conflict_from(self, position: Any) -> Any:
        return "TO-FRONT"


@dataclass
class FakeConditions:
    start_time: datetime = field(default_factory=lambda: datetime(2020, 6, 1, 12))


@dataclass
class FakeGame:
    theater: FakeTheater = field(default_factory=FakeTheater)
    conditions: FakeConditions = field(default_factory=FakeConditions)


def _order(
    force: FakeForceGroup, site: FakeSamSite, cost: int
) -> BoundGroundGroupReplacement:
    unit_group = force.layouts[0].groups[0].unit_groups[0]
    dcs_type = force.dcs_types[id(unit_group)][0]
    return BoundGroundGroupReplacement(
        control_point=object(),  # type: ignore[arg-type]
        ground_object=site,
        force_group=force,
        group_plan=(("Battery", unit_group, dcs_type, unit_group.max_size),),
        cost=cost,
        base_id="RED-1",
        category="air_defence",
        site_name=site.name,
    )


class TestReplaceGroundGroups:
    def _executor(self, coalition: FakeCoalition) -> PlanExecutor:
        return PlanExecutor(FakeGame(), coalition)  # type: ignore[arg-type]

    def test_a_site_is_rebuilt_through_the_force_group_and_budget_is_deducted(
        self,
    ) -> None:
        force = _single_group_force("SA-6", unit_price=10, max_size=4)
        site = FakeSamSite("SAM North", unit_count=4, alive=0)
        coalition = FakeCoalition(FakeArmedForces([force]))
        executor = self._executor(coalition)
        plan = ExecutableLogistics()
        plan.ground_group_replacements.append(_order(force, site, cost=40))
        executor._replace_ground_groups(plan)
        # Wreck cleared, then rebuilt from the recipe.
        assert site.groups == []
        assert len(force.created) == 1
        created_go, _ug, group_name, _type, count = force.created[0]
        assert created_go is site
        assert group_name == "SAM North (Battery)"
        assert count == 4
        # Site headed toward the front, exactly as the buy menu does.
        assert site.heading == "TO-FRONT"
        # RED's own budget charged the priced cost.
        assert coalition.budget_adjustments == [-40.0]
        assert executor.report.applied_count == 1

    def test_a_failed_rebuild_is_recorded_and_charges_nothing(self) -> None:
        force = _single_group_force("SA-6", unit_price=10, max_size=4)
        force.raise_on_create = True
        site = FakeSamSite("SAM North", unit_count=4, alive=0)
        coalition = FakeCoalition(FakeArmedForces([force]))
        executor = self._executor(coalition)
        plan = ExecutableLogistics()
        plan.ground_group_replacements.append(_order(force, site, cost=40))
        executor._replace_ground_groups(plan)
        assert coalition.budget_adjustments == []
        assert executor.report.failed_count == 1
