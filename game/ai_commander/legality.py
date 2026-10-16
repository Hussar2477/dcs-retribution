"""Re-validation of a schema-clean decision against live campaign state.

:mod:`game.ai_commander.decision` proves a decision is *internally* consistent
with the brief it was produced for. This module proves it is *actually* legal
right now, using the game's own predicates rather than any re-implementation:

* Postures are checked with :class:`FrontLineStanceTask`'s own force-balance
  predicate, plus :class:`BattlePositions` for the breakthrough requirement.
* Procurement categories are checked against the same control-point queries
  :class:`~game.procurement.ProcurementAi` uses, plus affordability against the
  live coalition budget.
* Anything not currently possible is rejected and recorded, never silently
  dropped and never "helpfully" reinterpreted into something stronger.

The output is a :class:`~game.ai_commander.directive.CommanderDirective` plus the
list of rejections. If nothing legal survives, the caller falls back to
Retribution's built-in RED automation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from game.ai_commander.decision import Rejection, RedCommanderDecision
from game.ai_commander.directive import (
    CommanderDirective,
    FrontKey,
    build_directive,
    front_key,
)
from game.ai_commander.enums import (
    FrontPosture,
    MissionPurpose,
    ProcurementCategory,
    TargetSetCategory,
)
from game.ai_commander.intel import RedCommanderBrief
from game.ai_commander.postures import posture_rejection_for

if TYPE_CHECKING:
    from game.game import Game
    from game.theater import ControlPoint, FrontLine


class LegalityChecker:
    """Checks a validated decision against live state for RED."""

    def __init__(self, game: Game, brief: RedCommanderBrief) -> None:
        from game.theater.player import Player

        self.game = game
        self.brief = brief
        self.player = Player.RED
        self.coalition = game.coalition_for(self.player)
        self._fronts: Optional[dict[FrontKey, FrontLine]] = None

    # -- helpers ----------------------------------------------------------

    @property
    def fronts(self) -> dict[FrontKey, FrontLine]:
        if self._fronts is None:
            from game.commander.objectivefinder import ObjectiveFinder

            fronts: dict[FrontKey, FrontLine] = {}
            try:
                for front in ObjectiveFinder(self.game, self.player).front_lines():
                    fronts[front_key(front, self.player)] = front
            except Exception:  # pragma: no cover - defensive
                logging.warning("Could not enumerate RED fronts", exc_info=True)
            self._fronts = fronts
        return self._fronts

    def _own_control_points(self) -> list[ControlPoint]:
        return [
            cp
            for cp in getattr(self.game.theater, "controlpoints", [])
            if cp.captured is self.player
        ]

    def _front_key_for_brief_id(self, front_id: str) -> Optional[FrontKey]:
        view = self.brief.front(front_id)
        if view is None:
            return None
        return (view.own_base, view.enemy_base)

    # -- posture legality -------------------------------------------------

    def posture_rejection(self, key: FrontKey, posture: FrontPosture) -> Optional[str]:
        """``None`` if legal, otherwise the reason it is not.

        Delegates to :func:`~game.ai_commander.postures.posture_rejection_for`,
        which is the same predicate the code that writes the stance uses.
        """

        front = self.fronts.get(key)
        if front is None:
            return (
                "front no longer exists or is no longer active in live campaign "
                "state"
            )
        return posture_rejection_for(posture, front, self.player)

    # -- procurement legality --------------------------------------------

    def procurement_rejection(self, category: ProcurementCategory) -> Optional[str]:
        from game.config import RUNWAY_REPAIR_COST

        budget = float(self.coalition.budget)
        own = self._own_control_points()

        if category is ProcurementCategory.RUNWAY_REPAIR:
            if not any(cp.runway_can_be_repaired for cp in own):
                return "no RED runway is currently repairable"
            if budget < float(RUNWAY_REPAIR_COST):
                return (
                    f"runway repair costs {RUNWAY_REPAIR_COST} but only "
                    f"{budget:.0f} is available"
                )
            return None

        if category is ProcurementCategory.AIRCRAFT:
            prices = [
                float(squadron.aircraft.price)
                for squadron in self.coalition.air_wing.iter_squadrons()
                if getattr(squadron, "aircraft", None) is not None
            ]
            if not prices:
                return "RED has no squadrons that could receive aircraft"
            if budget < min(prices):
                return (
                    f"cheapest available airframe costs {min(prices):.0f} but only "
                    f"{budget:.0f} is available"
                )
            return None

        ground_units = list(
            getattr(self.coalition.faction, "frontline_units", [])
        ) + list(getattr(self.coalition.faction, "artillery_units", []))
        if not ground_units:
            return "RED faction has no ground units available to purchase"
        cheapest = min(float(u.price) for u in ground_units)
        if budget < cheapest:
            return (
                f"cheapest ground unit costs {cheapest:.0f} but only "
                f"{budget:.0f} is available"
            )

        if category is ProcurementCategory.GROUND_COMBAT_UNITS:
            if not any(
                cp.has_active_frontline and cp.has_ground_unit_source(self.game)
                for cp in own
            ):
                return (
                    "no RED base with an active front line has a ground unit "
                    "source, so front line units cannot be bought"
                )
            return None

        # Only FRONT_LINE_RESERVES remains.
        if not any(
            not cp.is_global and cp.can_recruit_ground_units(self.game) for cp in own
        ):
            return "no RED base can currently recruit reserve ground units"
        return None

    # -- target set legality ---------------------------------------------

    def target_set_rejection(self, category: TargetSetCategory) -> Optional[str]:
        """Target sets are advisory orderings, so only emptiness is checked.

        The commander only re-orders which *classes* of objective the planner
        considers first; the planner then re-derives the actual targets from live
        state. A category that no longer has any known objective is dropped so it
        cannot displace a category that does.
        """

        for view in self.brief.known_target_sets:
            if view.category is category:
                if view.known_count <= 0:
                    return "no objectives of this class are currently known"
                return None
        return "this class of objective was not in the briefing"

    # -- entry point ------------------------------------------------------

    def check(
        self, decision: RedCommanderDecision
    ) -> tuple[Optional[CommanderDirective], list[Rejection]]:
        rejections: list[Rejection] = []

        live_revision = self._live_revision()
        if live_revision is not None and live_revision != decision.campaign_revision:
            rejections.append(
                Rejection(
                    "campaign_revision",
                    "campaign state changed between briefing and application",
                    decision.campaign_revision,
                )
            )
            return None, rejections

        # Fronts, in the commander's requested order.
        ordered_fronts: list[FrontKey] = []
        for index, ranked in enumerate(decision.front_priorities):
            key = self._front_key_for_brief_id(ranked.id)
            if key is None:
                rejections.append(
                    Rejection(
                        f"front_priorities[{index}]",
                        "front identifier could not be resolved",
                        ranked.id,
                    )
                )
                continue
            if key not in self.fronts:
                rejections.append(
                    Rejection(
                        f"front_priorities[{index}]",
                        "front is no longer active",
                        CommanderDirective.encode_front(key),
                    )
                )
                continue
            ordered_fronts.append(key)

        # Postures.
        postures: list[tuple[FrontKey, FrontPosture]] = []
        for index, request in enumerate(decision.push_postures):
            key = self._front_key_for_brief_id(request.front_id)
            if key is None:
                rejections.append(
                    Rejection(
                        f"push_postures[{index}]",
                        "front identifier could not be resolved",
                        request.front_id,
                    )
                )
                continue
            reason = self.posture_rejection(key, request.posture)
            if reason is not None:
                rejections.append(
                    Rejection(
                        f"push_postures[{index}]",
                        reason,
                        {
                            "front": CommanderDirective.encode_front(key),
                            "posture": request.posture.value,
                        },
                    )
                )
                continue
            postures.append((key, request.posture))

        # Spending.
        procurement: list[ProcurementCategory] = []
        for index, ranked in enumerate(decision.spending_priorities):
            view = self.brief.procurement_category(ranked.id)
            if view is None:
                rejections.append(
                    Rejection(
                        f"spending_priorities[{index}]",
                        "spending category could not be resolved",
                        ranked.id,
                    )
                )
                continue
            reason = self.procurement_rejection(view.category)
            if reason is not None:
                rejections.append(
                    Rejection(
                        f"spending_priorities[{index}]",
                        reason,
                        view.category.value,
                    )
                )
                continue
            procurement.append(view.category)

        # Target sets.
        target_sets: list[tuple[TargetSetCategory, MissionPurpose]] = []
        for index, priority in enumerate(decision.target_set_priorities):
            target = self.brief.target_set(priority.target_set_id)
            if target is None:
                rejections.append(
                    Rejection(
                        f"target_set_priorities[{index}]",
                        "target set could not be resolved",
                        priority.target_set_id,
                    )
                )
                continue
            reason = self.target_set_rejection(target.category)
            if reason is not None:
                rejections.append(
                    Rejection(
                        f"target_set_priorities[{index}]",
                        reason,
                        target.category.value,
                    )
                )
                continue
            target_sets.append((target.category, priority.purpose))

        directive = build_directive(
            turn_id=decision.turn_id,
            campaign_revision=decision.campaign_revision,
            strategy=decision.strategy,
            reserve_policy=decision.reserve_policy,
            target_sets=target_sets,
            procurement=procurement,
            fronts=ordered_fronts,
            postures=postures,
            commander_intent=decision.commander_intent,
        )
        if not directive.has_content:
            rejections.append(
                Rejection(
                    "<directive>",
                    "nothing in the decision was legal against live state, so "
                    "the built-in RED automation keeps control of this turn",
                )
            )
            return None, rejections
        return directive, rejections

    # -- carrying a previous directive forward ----------------------------

    def carry_forward(
        self, previous: CommanderDirective
    ) -> tuple[Optional[CommanderDirective], list[Rejection]]:
        """Re-stamp an earlier directive for the current turn.

        Used only when the player has turned off "fall back to the built-in
        auto-planner": rather than handing the turn to the heuristics, RED keeps
        the strategy it was last given. The theatre-wide parts of a directive
        (strategy, reserve policy, target-set and spending order) are always
        legal because they are only orderings, but fronts move, so every front
        and posture is re-checked against live state exactly as a fresh decision
        would be. Anything that no longer holds is rejected and logged.
        """

        rejections: list[Rejection] = []

        fronts: list[FrontKey] = []
        for key in previous.front_order:
            if key in self.fronts:
                fronts.append(key)
                continue
            rejections.append(
                Rejection(
                    "carry_forward.front_order",
                    "front from the previous directive is no longer active",
                    CommanderDirective.encode_front(key),
                )
            )

        postures: list[tuple[FrontKey, FrontPosture]] = []
        for encoded, raw in previous.front_postures.items():
            decoded = self._decode_front(encoded)
            try:
                posture = FrontPosture(raw)
            except ValueError:
                rejections.append(
                    Rejection(
                        "carry_forward.front_postures",
                        "posture from the previous directive is not recognised",
                        raw,
                    )
                )
                continue
            if decoded is None:
                rejections.append(
                    Rejection(
                        "carry_forward.front_postures",
                        "front from the previous directive could not be decoded",
                        encoded,
                    )
                )
                continue
            reason = self.posture_rejection(decoded, posture)
            if reason is not None:
                rejections.append(
                    Rejection(
                        "carry_forward.front_postures",
                        reason,
                        {"front": encoded, "posture": posture.value},
                    )
                )
                continue
            postures.append((decoded, posture))

        target_sets = [
            (category, previous.purpose_for(category) or MissionPurpose.ATTRITION)
            for category in previous.target_set_order
            if self.target_set_rejection(category) is None
        ]
        procurement = [
            category
            for category in previous.procurement_order
            if self.procurement_rejection(category) is None
        ]

        directive = build_directive(
            turn_id=self.brief.turn_id,
            campaign_revision=self.brief.campaign_revision,
            strategy=previous.strategy,
            reserve_policy=previous.reserve_policy,
            target_sets=target_sets,
            procurement=procurement,
            fronts=fronts,
            postures=postures,
            commander_intent=previous.commander_intent,
        )
        if not directive.has_content:
            rejections.append(
                Rejection(
                    "<carry_forward>",
                    "nothing in the previous directive is still legal, so the "
                    "built-in RED automation keeps control of this turn",
                )
            )
            return None, rejections
        return directive, rejections

    def _decode_front(self, encoded: str) -> Optional[FrontKey]:
        for key in self.fronts:
            if CommanderDirective.encode_front(key) == encoded:
                return key
        return None

    def _live_revision(self) -> Optional[str]:
        from game.ai_commander.intel import IntelProjector

        try:
            return IntelProjector(
                self.game, self.brief.intel_policy
            ).campaign_revision()
        except Exception:  # pragma: no cover - defensive
            logging.debug("Could not recompute campaign revision", exc_info=True)
            return None
