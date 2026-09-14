"""Applying a directive through Retribution's own deterministic code.

Nothing in this module invents a game mechanic. Every effect is produced by
calling existing Retribution code:

* Mission priorities are applied by re-ordering the methods
  :class:`~game.commander.tasks.compound.nextaction.PlanNextAction` already
  yields, so the same HTN planner, the same preconditions and the same package
  builders run -- only the order in which options are offered changes.
* Spending priorities are applied by a :class:`~game.procurement.ProcurementAi`
  subclass that re-orders and re-weights the existing purchase steps. Prices,
  affordability, parking, pilot limits and unit availability are all still
  decided by the base class.
* Front postures are applied by executing the game's own
  :class:`~game.commander.tasks.frontlinestancetask.FrontLineStanceTask`, after
  re-checking its own precondition.

If the directive says nothing about a dimension, the stock behaviour for that
dimension is used unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from game.ai_commander.directive import CommanderDirective, front_key
from game.ai_commander.enums import (
    ProcurementCategory,
    ReservePolicy,
    TargetSetCategory,
)
from game.ai_commander.postures import posture_rejection_for, stance_task_for
from game.commander.tasks.compound.attackairinfrastructure import (
    AttackAirInfrastructure,
)
from game.commander.tasks.compound.attackbattlepositions import AttackBattlePositions
from game.commander.tasks.compound.attackbuildings import AttackBuildings
from game.commander.tasks.compound.attackmotorpools import AttackMotorpools
from game.commander.tasks.compound.attackships import AttackShips
from game.commander.tasks.compound.capturebases import CaptureBases
from game.commander.tasks.compound.defendbases import DefendBases
from game.commander.tasks.compound.degradeiads import DegradeIads
from game.commander.tasks.compound.interdictreinforcements import (
    InterdictReinforcements,
)
from game.commander.tasks.compound.nextaction import PlanNextAction
from game.commander.tasks.compound.protectairspace import ProtectAirSpace
from game.commander.tasks.compound.recoverysupport import RecoverySupport
from game.commander.tasks.compound.theatersupport import TheaterSupport
from game.commander.theatercommander import TheaterCommander
from game.commander.theaterstate import TheaterState
from game.htn import Method
from game.procurement import ProcurementAi

if TYPE_CHECKING:
    from game.coalition import Coalition
    from game.factions.faction import Faction
    from game.game import Game
    from game.theater import ControlPoint
    from game.theater.player import Player


#: Which compound task each rankable target-set category corresponds to. This is
#: the whole translation from "commander priority" to "planner behaviour".
_CATEGORY_TASKS: dict[TargetSetCategory, str] = {
    TargetSetCategory.AIR_SUPERIORITY: "ProtectAirSpace",
    TargetSetCategory.BASE_DEFENCE: "DefendBases",
    TargetSetCategory.ENEMY_REINFORCEMENTS: "InterdictReinforcements",
    TargetSetCategory.ENEMY_BATTLE_POSITIONS: "AttackBattlePositions",
    TargetSetCategory.BASE_CAPTURE: "CaptureBases",
    TargetSetCategory.ENEMY_AIRBASES: "AttackAirInfrastructure",
    TargetSetCategory.ENEMY_INFRASTRUCTURE: "AttackBuildings",
    TargetSetCategory.ENEMY_MOTORPOOLS: "AttackMotorpools",
    TargetSetCategory.ENEMY_SHIPPING: "AttackShips",
    TargetSetCategory.ENEMY_AIR_DEFENCES: "DegradeIads",
}

#: The stock order, used for everything the commander did not rank.
_STOCK_ORDER: tuple[str, ...] = (
    "ProtectAirSpace",
    "DefendBases",
    "InterdictReinforcements",
    "AttackBattlePositions",
    "CaptureBases",
    "AttackAirInfrastructure",
    "AttackBuildings",
    "AttackMotorpools",
    "AttackShips",
    "DegradeIads",
)


def task_order_for(directive: Optional[CommanderDirective]) -> tuple[str, ...]:
    """Task names in the order the planner should be offered them.

    Ranked categories come first in the commander's order; everything else keeps
    its stock relative position afterwards. ``TheaterSupport`` and
    ``RecoverySupport`` are not in the list: they are support tasks (tankers,
    AEW&C, recovery) rather than target sets, and stay pinned first and last so
    the commander cannot strand its own aircraft.
    """

    if directive is None:
        return _STOCK_ORDER
    ordered: list[str] = []
    for category in directive.target_set_order:
        name = _CATEGORY_TASKS.get(category)
        if name is not None and name not in ordered:
            ordered.append(name)
    for name in _STOCK_ORDER:
        if name not in ordered:
            ordered.append(name)
    return tuple(ordered)


@dataclass(frozen=True)
class DirectedPlanNextAction(PlanNextAction):
    """:class:`PlanNextAction` with the commander's ordering applied.

    Subclassing keeps every method body identical to stock; only the sequence in
    which the methods are yielded differs. The HTN planner picks the first
    yielded method whose subtasks' preconditions hold, so re-ordering is exactly
    "consider this kind of objective first", and nothing more.
    """

    task_order: tuple[str, ...] = _STOCK_ORDER

    def each_valid_method(self, state: TheaterState) -> Iterator[Method[TheaterState]]:
        builders: dict[str, Callable[[], Method[TheaterState]]] = {
            "ProtectAirSpace": lambda: [ProtectAirSpace()],
            "DefendBases": lambda: [DefendBases()],
            "InterdictReinforcements": lambda: [InterdictReinforcements()],
            "AttackBattlePositions": lambda: [AttackBattlePositions()],
            "CaptureBases": lambda: [CaptureBases()],
            "AttackAirInfrastructure": lambda: [
                AttackAirInfrastructure(self.aircraft_cold_start)
            ],
            "AttackBuildings": lambda: [AttackBuildings()],
            "AttackMotorpools": lambda: [AttackMotorpools()],
            "AttackShips": lambda: [AttackShips()],
            "DegradeIads": lambda: [DegradeIads()],
        }
        # Support tasks are never reprioritised.
        yield [TheaterSupport()]
        for name in self.task_order:
            builder = builders.get(name)
            if builder is None:  # pragma: no cover - guarded by task_order_for
                continue
            yield builder()
        yield [RecoverySupport()]  # for recovery tankers


class DirectedTheaterCommander(TheaterCommander):
    """A theater commander whose root task carries the commander's ordering."""

    def __init__(
        self, game: Game, player: Player, directive: Optional[CommanderDirective]
    ) -> None:
        super().__init__(game, player)
        from game.ato.starttype import StartType

        self.directive = directive
        # Planner.__init__ stores the root task in ``main_task``; replacing it is
        # the entire mechanism by which the directive influences planning.
        self.main_task = DirectedPlanNextAction(
            aircraft_cold_start=game.settings.default_start_type is StartType.COLD,
            task_order=task_order_for(directive),
        )


# ---------------------------------------------------------------------------
# Procurement
# ---------------------------------------------------------------------------

#: How much of the ground budget is withheld for rear-area reserves.
_RESERVE_SHARES: dict[ReservePolicy, float] = {
    ReservePolicy.COMMIT_EVERYTHING: 0.0,
    ReservePolicy.BALANCED: 0.25,
    ReservePolicy.BUILD_RESERVES: 0.6,
}

#: Multiplier applied to the stock air/ground split when the commander ranks one
#: side above the other. Deliberately modest: the commander nudges the split, it
#: does not override the game's investment-proportional logic.
_SHARE_NUDGE = 0.35


class DirectedProcurementAi(ProcurementAi):
    """Procurement with commander priorities applied.

    Every purchase still goes through the base class: unit selection,
    affordability, doctrine ratios, parking, pilot limits and squadron choice are
    untouched. What the directive changes is:

    * whether runway repair happens before or after combat purchases,
    * how the budget is split between aircraft and ground forces,
    * how the ground budget is split between active fronts and rear reserves,
    * which front is reinforced first.
    """

    def __init__(
        self,
        game: Game,
        owner: Player,
        faction: Faction,
        manage_runways: bool,
        manage_front_line: bool,
        manage_aircraft: bool,
        directive: Optional[CommanderDirective] = None,
    ) -> None:
        super().__init__(
            game, owner, faction, manage_runways, manage_front_line, manage_aircraft
        )
        self.directive = directive
        self.applied_notes: list[str] = []

    # -- ordering ---------------------------------------------------------

    def _rank_of(self, category: ProcurementCategory) -> Optional[int]:
        if self.directive is None:
            return None
        try:
            return self.directive.procurement_order.index(category)
        except ValueError:
            return None

    def _repair_first(self) -> bool:
        """Runway repair runs first unless the commander ranked it lower."""

        repair = self._rank_of(ProcurementCategory.RUNWAY_REPAIR)
        if repair is None:
            return True
        others = [
            rank
            for rank in (
                self._rank_of(ProcurementCategory.AIRCRAFT),
                self._rank_of(ProcurementCategory.GROUND_COMBAT_UNITS),
                self._rank_of(ProcurementCategory.FRONT_LINE_RESERVES),
            )
            if rank is not None
        ]
        return not others or repair < min(others)

    def calculate_ground_unit_budget_share(self) -> float:
        share = super().calculate_ground_unit_budget_share()
        if self.directive is None:
            return share
        air_rank = self._rank_of(ProcurementCategory.AIRCRAFT)
        ground_ranks = [
            rank
            for rank in (
                self._rank_of(ProcurementCategory.GROUND_COMBAT_UNITS),
                self._rank_of(ProcurementCategory.FRONT_LINE_RESERVES),
            )
            if rank is not None
        ]
        if air_rank is None and not ground_ranks:
            return share
        ground_rank = min(ground_ranks) if ground_ranks else None
        if ground_rank is None:
            nudged = share * (1.0 - _SHARE_NUDGE)
        elif air_rank is None or ground_rank < air_rank:
            nudged = share + (1.0 - share) * _SHARE_NUDGE
        elif air_rank < ground_rank:
            nudged = share * (1.0 - _SHARE_NUDGE)
        else:  # pragma: no cover - ranks are unique
            nudged = share
        clamped = max(0.0, min(1.0, nudged))
        self.applied_notes.append(f"ground budget share {share:.3f} -> {clamped:.3f}")
        return clamped

    # -- reinforcement candidate selection --------------------------------

    def _front_priority(self, control_point: ControlPoint) -> int:
        """Lower sorts first. Unranked control points keep stock ordering."""

        if self.directive is None or not self.directive.front_order:
            return 0
        best = len(self.directive.front_order) + 1
        for index, key in enumerate(self.directive.front_order):
            if key[0] == str(control_point.name):
                best = min(best, index)
        return best

    def ground_reinforcement_candidate(self) -> Optional[ControlPoint]:
        """Stock candidate selection, biased by the commander's front order.

        The set of *eligible* control points is unchanged -- it comes straight
        from the base class -- so the commander cannot buy units at a base that
        the game would not allow. Only which eligible base is picked first
        changes, and only when the commander ranked fronts.
        """

        candidate = super().ground_reinforcement_candidate()
        if candidate is None or self.directive is None:
            return candidate
        if not self.directive.front_order:
            return candidate

        preferred = self._preferred_frontline_candidate()
        return preferred if preferred is not None else candidate

    def _preferred_frontline_candidate(self) -> Optional[ControlPoint]:
        """The highest-priority front-line base that stock rules would allow.

        Mirrors the base class's first loop exactly (active front line, has a
        ground unit source, not already at its purchase target) and then sorts by
        the commander's front priority instead of by current supply.
        """

        eligible: list[ControlPoint] = []
        reserves_factor = (
            self.game.settings.frontline_reserves_factor
            if self.is_player.is_blue
            else self.game.settings.frontline_reserves_factor_red
        )
        transfers = self.game.coalition_for(self.is_player).transfers
        for cp in self.owned_points:
            if not cp.has_active_frontline:
                continue
            if not cp.has_ground_unit_source(self.game):
                continue
            purchase_target = cp.frontline_unit_count_limit * (reserves_factor / 100.0)
            allocated = cp.allocated_ground_units(transfers)
            if allocated.total >= purchase_target:
                continue
            eligible.append(cp)
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda cp: (
                self._front_priority(cp),
                cp.allocated_ground_units(transfers).total,
            ),
        )

    # -- budget flow ------------------------------------------------------

    def spend_budget(self, budget: float) -> float:
        """Stock spending steps, in the commander's order.

        Mirrors :meth:`ProcurementAi.spend_budget` step for step; the only
        differences are the position of runway repair and the reserve split.
        """

        if self.directive is None:
            return super().spend_budget(budget)

        repair_first = self._repair_first()
        if self.manage_runways and repair_first:
            budget = self.repair_runways(budget)

        if self.manage_front_line:
            armor_budget = budget * self.calculate_ground_unit_budget_share()
            budget -= armor_budget
            budget += self._reinforce_with_reserve_policy(armor_budget)

        if self.manage_aircraft:
            budget = self.purchase_aircraft(budget)

        if self.manage_runways and not repair_first:
            budget = self.repair_runways(budget)
            self.applied_notes.append("runway repair deferred behind combat purchases")
        return budget

    def _reinforce_with_reserve_policy(self, armor_budget: float) -> float:
        """Split the ground budget between fronts and reserves.

        Both halves go through the stock :meth:`reinforce_front_line`, which
        keeps its own eligibility rules. The reserve half is simply spent in a
        second pass after the front-line half, which is how the base class
        already reaches reserve bases (its candidate search falls through to
        reserves once every front-line base is at its target).
        """

        if self.directive is None:
            return self.reinforce_front_line(armor_budget)

        share = _RESERVE_SHARES.get(self.directive.reserve_policy, 0.25)
        reserve_budget = armor_budget * share
        front_budget = armor_budget - reserve_budget
        self.applied_notes.append(
            f"ground budget split front={front_budget:.0f} reserves={reserve_budget:.0f}"
        )
        leftover = self.reinforce_front_line(front_budget)
        return self.reinforce_front_line(reserve_budget + leftover)


# ---------------------------------------------------------------------------
# Front postures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostureApplication:
    """What happened when a posture was applied."""

    front: str
    posture: str
    applied: bool
    reason: str = ""


def apply_front_postures(
    coalition: Coalition, directive: Optional[CommanderDirective]
) -> list[PostureApplication]:
    """Apply the directive's postures using the game's own stance tasks.

    Each posture is re-checked against live state immediately before it is
    written, and is written by calling ``FrontLineStanceTask.execute`` -- the same
    call the built-in planner makes. A posture that fails its precondition is
    reported as not applied and the front keeps whatever the built-in planner
    chose.
    """

    from game.commander.objectivefinder import ObjectiveFinder

    results: list[PostureApplication] = []
    if directive is None or not directive.front_postures:
        return results

    player = coalition.player
    try:
        fronts = {
            front_key(front, player): front
            for front in ObjectiveFinder(coalition.game, player).front_lines()
        }
    except Exception:  # pragma: no cover - defensive
        logging.warning("Could not enumerate fronts to apply postures", exc_info=True)
        return results

    for key, front in fronts.items():
        posture = directive.posture_for(key)
        if posture is None:
            continue
        encoded = CommanderDirective.encode_front(key)
        reason = posture_rejection_for(posture, front, player)
        if reason is not None:
            logging.info(
                "RED commander posture %s on %s not applied: %s",
                posture.value,
                encoded,
                reason,
            )
            results.append(PostureApplication(encoded, posture.value, False, reason))
            continue
        try:
            stance_task_for(posture, front, player).execute(coalition)
        except Exception as err:  # pragma: no cover - defensive
            logging.warning(
                "Failed to apply posture %s on %s",
                posture.value,
                encoded,
                exc_info=True,
            )
            results.append(PostureApplication(encoded, posture.value, False, str(err)))
            continue
        results.append(PostureApplication(encoded, posture.value, True))
    return results
