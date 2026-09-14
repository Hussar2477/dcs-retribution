"""Mapping between LLM-facing front postures and Retribution combat stances.

The LLM commander is never allowed to write a :class:`CombatStance` directly.
It picks a :class:`FrontPosture`, which this module translates into the exact
same :class:`~game.commander.tasks.frontlinestancetask.FrontLineStanceTask`
objects the built-in HTN planner uses.  The legality of a posture is therefore
decided by the game's own ``have_sufficient_front_line_advantage`` predicates,
not by anything the model says.

``AMBUSH`` is intentionally unreachable: the stance exists in
:class:`CombatStance` but Retribution ships no ``AmbushStance`` task, so the
built-in planner can never select it either.  Exposing it to the LLM would give
the AI a capability the deterministic planner does not have.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping, Optional

from game.ai_commander.enums import FrontPosture
from game.commander.tasks.frontlinestancetask import FrontLineStanceTask
from game.commander.tasks.primitive.aggressiveattack import AggressiveAttack
from game.commander.tasks.primitive.breakthroughattack import BreakthroughAttack
from game.commander.tasks.primitive.defensivestance import DefensiveStance
from game.commander.tasks.primitive.eliminationattack import EliminationAttack
from game.commander.tasks.primitive.retreatstance import RetreatStance
from game.ground_forces.combat_stance import CombatStance

if TYPE_CHECKING:
    from game.game import Game
    from game.theater import FrontLine
    from game.theater.player import Player


#: Ordered least-committal first.  The order is used when a posture has to be
#: down-graded to the strongest legal alternative.
POSTURE_ORDER: tuple[FrontPosture, ...] = (
    FrontPosture.RETREAT,
    FrontPosture.HOLD,
    FrontPosture.PROBE,
    FrontPosture.PUSH,
    FrontPosture.BREAKTHROUGH,
)

#: The one-to-one posture -> stance mapping.
POSTURE_STANCES: Mapping[FrontPosture, CombatStance] = {
    FrontPosture.RETREAT: CombatStance.RETREAT,
    FrontPosture.HOLD: CombatStance.DEFENSIVE,
    FrontPosture.PROBE: CombatStance.AGGRESSIVE,
    FrontPosture.PUSH: CombatStance.ELIMINATION,
    FrontPosture.BREAKTHROUGH: CombatStance.BREAKTHROUGH,
}

_STANCE_POSTURES: Mapping[CombatStance, FrontPosture] = {
    stance: posture for posture, stance in POSTURE_STANCES.items()
}

_POSTURE_TASKS: Mapping[FrontPosture, type[FrontLineStanceTask]] = {
    FrontPosture.RETREAT: RetreatStance,
    FrontPosture.HOLD: DefensiveStance,
    FrontPosture.PROBE: AggressiveAttack,
    FrontPosture.PUSH: EliminationAttack,
    FrontPosture.BREAKTHROUGH: BreakthroughAttack,
}


def stance_for(posture: FrontPosture) -> CombatStance:
    """The :class:`CombatStance` a posture resolves to."""

    return POSTURE_STANCES[posture]


def posture_of_stance(stance: CombatStance) -> Optional[FrontPosture]:
    """The posture matching ``stance``, or ``None`` for un-modelled stances.

    ``CombatStance.AMBUSH`` has no posture; a front already set to it simply
    reports ``None`` as its current posture.
    """

    return _STANCE_POSTURES.get(stance)


def stance_task_for(
    posture: FrontPosture, front_line: FrontLine, player: Player
) -> FrontLineStanceTask:
    """Instantiate the game's own stance task for ``posture``."""

    return _POSTURE_TASKS[posture](front_line, player)


def posture_is_legal(
    posture: FrontPosture, front_line: FrontLine, player: Player
) -> bool:
    """Whether the game's own rules permit ``posture`` on ``front_line``.

    This deliberately re-uses ``FrontLineStanceTask`` rather than
    re-implementing the force-balance thresholds, so the AI can never be
    granted an advantage the built-in planner would not have.

    ``BREAKTHROUGH`` additionally requires the opposing battle positions to be
    eliminated.  That check needs a
    :class:`~game.commander.theaterstate.TheaterState`, which is only built
    inside the planner, so it is *not* evaluated here.  Breakthrough is
    therefore advertised as legal on force balance alone and re-checked by the
    HTN planner at execution time -- meaning the AI can request it and be
    refused, but can never bypass the requirement.
    """

    try:
        task = stance_task_for(posture, front_line, player)
    except KeyError:  # pragma: no cover - guarded by the enum
        return False
    try:
        if task.friendly_cp.deployable_front_line_units == 0:
            return False
        return bool(task.have_sufficient_front_line_advantage)
    except Exception:  # pragma: no cover - defensive against partial states
        logging.debug(
            "Could not evaluate posture %s legality", posture.value, exc_info=True
        )
        return False


def breakthrough_rejection(front_line: FrontLine, player: Player) -> Optional[str]:
    """The extra requirement ``BREAKTHROUGH`` carries, or ``None``.

    ``BreakthroughAttack`` also demands that the opposing battle positions no
    longer block a capture.  The planner evaluates that from a
    :class:`~game.commander.theaterstate.TheaterState`; outside the planner the
    same answer is available from
    :meth:`~game.commander.battlepositions.BattlePositions.for_control_point`,
    which is what the state itself is built from.
    """

    from game.commander.battlepositions import BattlePositions

    enemy_cp = front_line.control_point_hostile_to(player)
    try:
        positions = BattlePositions.for_control_point(enemy_cp)
    except Exception:  # pragma: no cover - defensive
        logging.debug("Could not read battle positions", exc_info=True)
        return "opposing battle positions could not be evaluated"
    if positions.blocking_capture:
        return (
            f"{len(positions.blocking_capture)} enemy battle position(s) still "
            "block a capture, which the game requires to be eliminated first"
        )
    return None


def capture_status_for(front_line: FrontLine, player: Player) -> str:
    """A compact, RED-observable summary of whether the enemy base is capturable.

    Base capture is reachable only through an aggressive posture: the enemy
    battle positions that guard the base must be eliminated (targeted through
    the ``enemy_battle_positions`` set) and then a ``breakthrough`` pressed on
    that front while a force advantage holds. This states exactly where the
    front stands against that chain, in the same terms RED could observe from
    the front line itself.

    Deliberately leaks nothing BLUE-internal: only the *count* of blocking
    battle positions (legitimate front-line reconnaissance, already used to
    gate breakthrough) and whether RED's own force balance currently satisfies
    the breakthrough precondition. No BLUE plans, budgets, rosters or exact
    positions of unobserved units are ever exposed.
    """

    from game.commander.battlepositions import BattlePositions

    enemy_cp = front_line.control_point_hostile_to(player)
    try:
        blocking = len(BattlePositions.for_control_point(enemy_cp).blocking_capture)
    except Exception:  # pragma: no cover - defensive
        logging.debug("Could not read battle positions", exc_info=True)
        return "capture=unknown (battle positions could not be evaluated)"

    if blocking:
        return (
            f"capture=blocked ({blocking} enemy battle position(s) to eliminate "
            "first via enemy_battle_positions, then breakthrough)"
        )
    if posture_is_legal(FrontPosture.BREAKTHROUGH, front_line, player):
        return (
            "capture=available (no blocking positions; breakthrough can take the base)"
        )
    return (
        "capture=needs force advantage (no blocking positions, but force balance "
        "does not yet permit breakthrough)"
    )


def posture_rejection_for(
    posture: FrontPosture, front_line: FrontLine, player: Player
) -> Optional[str]:
    """``None`` when the game's own rules permit ``posture``, else the reason.

    Single source of truth for posture legality, shared by the pre-flight
    validator and by the code that actually writes the stance, so a posture can
    never be accepted by one and refused by the other for different reasons.
    """

    if not posture_is_legal(posture, front_line, player):
        return (
            "force balance on this front no longer satisfies the game's own "
            "precondition for this stance"
        )
    if posture is FrontPosture.BREAKTHROUGH:
        return breakthrough_rejection(front_line, player)
    return None


def legal_postures_for(
    game: Game, front_line: FrontLine, player: Player
) -> tuple[FrontPosture, ...]:
    """Every posture ``player`` may legally adopt on ``front_line``.

    ``game`` is accepted for symmetry with the rest of the intel projection and
    for forwards compatibility; legality currently depends only on the front
    line itself.
    """

    del game  # Legality is a property of the front line's force balance.
    return tuple(
        posture
        for posture in POSTURE_ORDER
        if posture_is_legal(posture, front_line, player)
    )


def strongest_legal_posture_at_or_below(
    posture: FrontPosture, front_line: FrontLine, player: Player
) -> Optional[FrontPosture]:
    """The most aggressive legal posture no stronger than ``posture``.

    Used to *down-grade* an over-ambitious request instead of silently
    accepting it.  Returns ``None`` when nothing is legal, in which case the
    request is rejected outright and the built-in planner keeps control of the
    front.
    """

    ceiling = POSTURE_ORDER.index(posture)
    for candidate in reversed(POSTURE_ORDER[: ceiling + 1]):
        if posture_is_legal(candidate, front_line, player):
            return candidate
    return None
