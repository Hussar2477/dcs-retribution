"""The RED intelligence view.

This module is the fairness boundary of the whole subsystem. Nothing else builds
the payload that is sent to the model, and nothing in the payload is derived from
data that the configured :class:`IntelPolicy` disallows.

Three rules are enforced here:

1. Only aggregates and enumerations that Retribution's own deterministic RED
   planner already computes are read. No save serialisation, no BLUE flight
   plans, no BLUE budget, no BLUE squadron rosters, no coordinates.
2. Every entity the model is allowed to refer to is given an opaque,
   turn-scoped identifier (``FRONT-1``, ``TS-3``, ``PROC-2``). Those identifiers
   are meaningless outside the brief they were generated for.
3. Under :attr:`IntelPolicy.REALISTIC`, BLUE strengths are reduced to coarse
   bands and BLUE objectives are filtered to those within RED's own sensor and
   threat coverage or close to an active front.

The result is deliberately serialisable in two forms: :meth:`RedCommanderBrief.to_dict`
for the audit log, and :meth:`RedCommanderBrief.render_compact` for the prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence, TYPE_CHECKING

from game.ai_commander.enums import (
    AffordabilityBand,
    Confidence,
    FrontPosture,
    IntelPolicy,
    LocationPrecision,
    ProcurementCategory,
    RedStrategy,
    ReservePolicy,
    StrengthBand,
    TargetSetCategory,
)
from game.ai_commander.serialization import jsonable, stable_hash

if TYPE_CHECKING:
    from game.game import Game
    from game.theater import ControlPoint, FrontLine

SCHEMA_VERSION = "red-commander-brief/1"

#: Objectives further than this from any RED-held control point are treated as
#: unobserved under the realistic policy unless RED's threat zones cover them.
_REALISTIC_OBSERVATION_RANGE_METERS = 120_000.0


@dataclass(frozen=True)
class ResourceView:
    """RED's own resources. RED always knows these exactly."""

    budget_available: float
    income_last_turn: float
    currency: str = "M"

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass(frozen=True)
class ForceSummary:
    """Aggregate view of RED's own forces."""

    control_points_held: int
    airbases_operational: int
    airbases_runway_damaged: int
    aircraft_available: int
    aircraft_on_order: int
    ground_units_deployed: int
    ground_units_on_order: int
    squadrons_available: int
    active_fronts: int

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass(frozen=True)
class FrontView:
    """One active front line, addressed by an opaque turn-scoped identifier."""

    id: str
    own_base: str
    enemy_base: str
    own_deployable_units: int
    own_unit_capacity: int
    enemy_strength: StrengthBand
    enemy_unit_count: Optional[int]
    legal_postures: tuple[FrontPosture, ...]
    current_posture: Optional[FrontPosture]
    reinforcement_eligible: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass(frozen=True)
class TargetSetView:
    """A class of BLUE objective that the deterministic planner can already act on."""

    id: str
    category: TargetSetCategory
    known_count: int
    confidence: Confidence
    last_observed_turn: Optional[int]
    location_precision: LocationPrecision
    threatens_own_forces: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass(frozen=True)
class ProcurementCategoryView:
    id: str
    category: ProcurementCategory
    eligible: bool
    affordability: AffordabilityBand
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass(frozen=True)
class CommanderConstraints:
    """The action space, restated inside the brief so the model can see its limits."""

    allowed_strategies: tuple[str, ...]
    allowed_postures: tuple[str, ...]
    allowed_reserve_policies: tuple[str, ...]
    max_intent_characters: int
    rules: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def default(cls) -> CommanderConstraints:
        return cls(
            allowed_strategies=tuple(s.value for s in RedStrategy),
            allowed_postures=tuple(p.value for p in FrontPosture),
            allowed_reserve_policies=tuple(r.value for r in ReservePolicy),
            max_intent_characters=600,
            rules=(
                "Rank only the identifiers listed in this brief. Invented "
                "identifiers are rejected.",
                "Ranks start at 1 and must be unique within each list.",
                "You cannot create units, set prices, choose aircraft types, "
                "pick individual targets, plan routes or loadouts, move the "
                "front line, capture bases or spend more than the listed budget.",
                "A posture that is not listed as legal for a front is rejected.",
                "Every decision is re-validated against live campaign state "
                "before anything is applied.",
            ),
        )


@dataclass(frozen=True)
class PriorTurnSummary:
    """What was decided last turn and what visibly came of it."""

    turn: Optional[int] = None
    strategy: Optional[str] = None
    reserve_policy: Optional[str] = None
    intent: Optional[str] = None
    #: Target-set categories, in the order last turn's directive ranked them.
    target_set_order: tuple[str, ...] = ()
    #: Postures that were actually written last turn, keyed by encoded front.
    front_postures: Mapping[str, str] = field(default_factory=dict)
    control_points_delta: Optional[int] = None
    budget_delta: Optional[float] = None
    rejected_element_count: int = 0
    #: A few rejected element paths, so the model can learn what not to repeat.
    rejected_elements: tuple[str, ...] = ()
    fallback_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass(frozen=True)
class RedCommanderBrief:
    """The complete, fair intelligence view handed to the model."""

    schema_version: str
    campaign_id_hash: str
    campaign_revision: str
    turn_id: int
    intel_policy: IntelPolicy
    theater: str
    red_faction: str
    red_resources: ResourceView
    red_force_summary: ForceSummary
    fronts: tuple[FrontView, ...]
    known_target_sets: tuple[TargetSetView, ...]
    procurement_categories: tuple[ProcurementCategoryView, ...]
    commander_constraints: CommanderConstraints
    prior_decision_summary: PriorTurnSummary
    prior_outcome_summary: PriorTurnSummary
    withheld_fields: tuple[str, ...] = field(default_factory=tuple)

    # -- lookup helpers ---------------------------------------------------

    @property
    def front_ids(self) -> frozenset[str]:
        return frozenset(f.id for f in self.fronts)

    @property
    def target_set_ids(self) -> frozenset[str]:
        return frozenset(t.id for t in self.known_target_sets)

    @property
    def procurement_ids(self) -> frozenset[str]:
        return frozenset(p.id for p in self.procurement_categories)

    def front(self, front_id: str) -> Optional[FrontView]:
        for view in self.fronts:
            if view.id == front_id:
                return view
        return None

    def target_set(self, target_set_id: str) -> Optional[TargetSetView]:
        for view in self.known_target_sets:
            if view.id == target_set_id:
                return view
        return None

    def procurement_category(
        self, category_id: str
    ) -> Optional[ProcurementCategoryView]:
        for view in self.procurement_categories:
            if view.id == category_id:
                return view
        return None

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    def content_hash(self) -> str:
        return stable_hash(self.to_dict())

    def render_compact(self) -> str:
        """A token-efficient plain-text rendering used for the prompt."""

        lines: list[str] = [
            f"BRIEF {self.schema_version}",
            f"campaign={self.campaign_id_hash} revision={self.campaign_revision} "
            f"turn={self.turn_id}",
            f"intel_policy={self.intel_policy.value} theater={self.theater} "
            f"faction={self.red_faction}",
            "",
            "[RESOURCES]",
            f"budget={self.red_resources.budget_available:.0f}{self.red_resources.currency} "
            f"income_last_turn={self.red_resources.income_last_turn:.0f}"
            f"{self.red_resources.currency}",
            "",
            "[OWN FORCES]",
        ]
        summary = self.red_force_summary
        lines.append(
            f"bases={summary.control_points_held} "
            f"airbases_ok={summary.airbases_operational} "
            f"runways_damaged={summary.airbases_runway_damaged} "
            f"aircraft={summary.aircraft_available}(+{summary.aircraft_on_order}) "
            f"squadrons={summary.squadrons_available} "
            f"ground={summary.ground_units_deployed}"
            f"(+{summary.ground_units_on_order}) "
            f"fronts={summary.active_fronts}"
        )

        lines += ["", "[FRONTS]"]
        if not self.fronts:
            lines.append("(none active)")
        for front in self.fronts:
            postures = ",".join(p.value for p in front.legal_postures) or "none"
            enemy = (
                f"enemy={front.enemy_unit_count}"
                if front.enemy_unit_count is not None
                else f"enemy_strength={front.enemy_strength.value}"
            )
            current = front.current_posture.value if front.current_posture else "unset"
            lines.append(
                f"{front.id} {front.own_base} -> {front.enemy_base} | "
                f"own={front.own_deployable_units}/{front.own_unit_capacity} | "
                f"{enemy} | legal={postures} | current={current} | "
                f"reinforceable={'yes' if front.reinforcement_eligible else 'no'}"
                + (f" | {front.notes}" if front.notes else "")
            )

        lines += ["", "[KNOWN TARGET SETS]"]
        if not self.known_target_sets:
            lines.append("(none observed)")
        for target in self.known_target_sets:
            observed = (
                f"seen_turn={target.last_observed_turn}"
                if target.last_observed_turn is not None
                else "seen_turn=never"
            )
            lines.append(
                f"{target.id} {target.category.value} | known={target.known_count} | "
                f"confidence={target.confidence.value} | {observed} | "
                f"precision={target.location_precision.value} | "
                f"threatens_us={'yes' if target.threatens_own_forces else 'no'}"
                + (f" | {target.notes}" if target.notes else "")
            )

        lines += ["", "[SPENDING CATEGORIES]"]
        for category in self.procurement_categories:
            lines.append(
                f"{category.id} {category.category.value} | "
                f"eligible={'yes' if category.eligible else 'no'} | "
                f"affordability={category.affordability.value}"
                + (f" | {category.notes}" if category.notes else "")
            )

        lines += ["", "[CONSTRAINTS]"]
        constraints = self.commander_constraints
        lines.append(f"strategies: {', '.join(constraints.allowed_strategies)}")
        lines.append(f"postures: {', '.join(constraints.allowed_postures)}")
        lines.append(
            f"reserve_policies: {', '.join(constraints.allowed_reserve_policies)}"
        )
        lines.append(
            f"commander_intent max {constraints.max_intent_characters} characters"
        )
        for rule in constraints.rules:
            lines.append(f"- {rule}")

        prior = self.prior_decision_summary
        if prior.turn is not None:
            lines += ["", "[LAST TURN]"]
            lines.append(
                f"turn={prior.turn} strategy={prior.strategy} "
                f"reserve_policy={prior.reserve_policy} "
                f"rejected_elements={prior.rejected_element_count} "
                f"fallback={prior.fallback_reason or 'none'}"
            )
            if prior.target_set_order:
                lines.append("priorities: " + " > ".join(prior.target_set_order))
            for encoded, posture in prior.front_postures.items():
                lines.append(f"posture {encoded}: {posture}")
            if prior.rejected_elements:
                lines.append("refused last turn: " + ", ".join(prior.rejected_elements))
            if prior.intent:
                lines.append(f"intent: {prior.intent}")

        outcome = self.prior_outcome_summary
        if outcome.turn is not None:
            lines += ["", "[OUTCOME SINCE LAST DECISION]"]
            lines.append(
                f"bases_delta={outcome.control_points_delta} "
                f"budget_delta={outcome.budget_delta}"
            )

        if self.withheld_fields:
            lines += ["", "[WITHHELD]"]
            lines.append(
                "The following are deliberately not available to you: "
                + ", ".join(self.withheld_fields)
            )

        return "\n".join(lines)


# Fields that the realistic policy never exposes. Kept as data so the leak test
# can assert on it and so the model is told what it is missing.
REALISTIC_WITHHELD_FIELDS: tuple[str, ...] = (
    "enemy_budget",
    "enemy_income",
    "enemy_squadron_rosters",
    "enemy_aircraft_inventory",
    "enemy_exact_ground_unit_counts",
    "enemy_planned_flights",
    "enemy_unit_coordinates",
    "enemy_pending_purchases",
    "campaign_save_data",
)

FULL_PARITY_WITHHELD_FIELDS: tuple[str, ...] = (
    "enemy_planned_flights",
    "enemy_unit_coordinates",
    "campaign_save_data",
)


def _band_for_ratio(own: float, enemy: float) -> StrengthBand:
    """Coarse strength band for BLUE relative to RED."""

    if enemy <= 0:
        return StrengthBand.NEGLIGIBLE
    if own <= 0:
        return StrengthBand.MUCH_STRONGER
    ratio = enemy / own
    if ratio < 0.4:
        return StrengthBand.MUCH_WEAKER
    if ratio < 0.8:
        return StrengthBand.WEAKER
    if ratio <= 1.25:
        return StrengthBand.COMPARABLE
    if ratio <= 2.5:
        return StrengthBand.STRONGER
    return StrengthBand.MUCH_STRONGER


def _affordability(budget: float, cheapest: Optional[float]) -> AffordabilityBand:
    if cheapest is None or cheapest <= 0:
        return AffordabilityBand.NONE
    if budget < cheapest:
        return AffordabilityBand.NONE
    if budget < cheapest * 4:
        return AffordabilityBand.LIMITED
    return AffordabilityBand.COMFORTABLE


class IntelProjector:
    """Builds a :class:`RedCommanderBrief` from live campaign state.

    The projector is read-only. It never mutates the game, and it is the only
    place allowed to touch BLUE-owned objects.
    """

    def __init__(self, game: Game, policy: IntelPolicy) -> None:
        from game.theater.player import Player

        self.game = game
        self.policy = policy
        self.player = Player.RED
        self.coalition = game.coalition_for(self.player)

    # -- public API -------------------------------------------------------

    def project(
        self,
        prior_decision: Optional[PriorTurnSummary] = None,
        prior_outcome: Optional[PriorTurnSummary] = None,
    ) -> RedCommanderBrief:
        fronts = self._project_fronts()
        target_sets = self._project_target_sets(fronts)
        return RedCommanderBrief(
            schema_version=SCHEMA_VERSION,
            campaign_id_hash=self.campaign_id_hash(),
            campaign_revision=self.campaign_revision(),
            turn_id=int(self.game.turn),
            intel_policy=self.policy,
            theater=str(getattr(self.game.theater, "terrain_name", "unknown")),
            red_faction=str(getattr(self.coalition.faction, "name", "unknown")),
            red_resources=self._project_resources(),
            red_force_summary=self._project_force_summary(len(fronts)),
            fronts=fronts,
            known_target_sets=target_sets,
            procurement_categories=self._project_procurement_categories(),
            commander_constraints=CommanderConstraints.default(),
            prior_decision_summary=prior_decision or PriorTurnSummary(),
            prior_outcome_summary=prior_outcome or PriorTurnSummary(),
            withheld_fields=(
                REALISTIC_WITHHELD_FIELDS
                if self.policy is IntelPolicy.REALISTIC
                else FULL_PARITY_WITHHELD_FIELDS
            ),
        )

    def campaign_id_hash(self) -> str:
        """Stable identifier for this campaign, with no path or personal data."""

        return stable_hash(
            {
                "theater": str(getattr(self.game.theater, "terrain_name", "")),
                "red": str(getattr(self.coalition.faction, "name", "")),
                "blue": str(
                    getattr(self.coalition.opponent.faction, "name", "")
                    if self._opponent() is not None
                    else ""
                ),
                "bases": sorted(str(cp.name) for cp in self._all_control_points()),
            }
        )

    def campaign_revision(self) -> str:
        """Digest of the state the decision is being made against.

        A decision produced for one revision is refused if state has changed by
        the time it is applied.
        """

        return stable_hash(
            {
                "turn": int(self.game.turn),
                "budget": round(float(self.coalition.budget), 3),
                "ownership": [
                    (str(cp.name), str(cp.captured))
                    for cp in self._all_control_points()
                ],
                "own_ground": [
                    (str(cp.name), self._deployable_units(cp))
                    for cp in self._own_control_points()
                ],
            }
        )

    # -- internals --------------------------------------------------------

    def _opponent(self) -> Optional[Any]:
        try:
            return self.coalition.opponent
        except AssertionError:
            return None

    def _all_control_points(self) -> list[ControlPoint]:
        return list(getattr(self.game.theater, "controlpoints", []))

    def _own_control_points(self) -> list[ControlPoint]:
        return [cp for cp in self._all_control_points() if cp.captured is self.player]

    @staticmethod
    def _deployable_units(control_point: ControlPoint) -> int:
        try:
            return int(control_point.deployable_front_line_units)
        except Exception:  # pragma: no cover - defensive against theater edge cases
            logging.debug("Could not read deployable units", exc_info=True)
            return 0

    def _project_resources(self) -> ResourceView:
        from game.income import Income

        try:
            income = float(Income(self.game, self.player).total)
        except Exception:  # pragma: no cover - defensive
            logging.debug("Could not compute RED income", exc_info=True)
            income = 0.0
        return ResourceView(
            budget_available=round(float(self.coalition.budget), 2),
            income_last_turn=round(income, 2),
        )

    def _project_force_summary(self, active_fronts: int) -> ForceSummary:
        from game.theater import ParkingType

        parking = ParkingType(fixed_wing=True, fixed_wing_stol=True, rotary_wing=True)
        own = self._own_control_points()
        aircraft_present = 0
        aircraft_ordered = 0
        ground_present = 0
        ground_ordered = 0
        airbases_ok = 0
        runways_damaged = 0
        for control_point in own:
            allocations = control_point.allocated_aircraft(parking)
            aircraft_present += int(allocations.total_present)
            aircraft_ordered += int(allocations.total_ordered)
            ground = control_point.allocated_ground_units(self.coalition.transfers)
            ground_present += int(ground.total_present)
            ground_ordered += int(ground.total_ordered)
            if control_point.runway_is_operational():
                airbases_ok += 1
            if control_point.runway_status is not None and getattr(
                control_point.runway_status, "needs_repair", False
            ):
                runways_damaged += 1

        squadrons = sum(1 for _ in self.coalition.air_wing.iter_squadrons())
        return ForceSummary(
            control_points_held=len(own),
            airbases_operational=airbases_ok,
            airbases_runway_damaged=runways_damaged,
            aircraft_available=aircraft_present,
            aircraft_on_order=aircraft_ordered,
            ground_units_deployed=ground_present,
            ground_units_on_order=ground_ordered,
            squadrons_available=squadrons,
            active_fronts=active_fronts,
        )

    def _active_front_lines(self) -> list[FrontLine]:
        from game.commander.objectivefinder import ObjectiveFinder

        try:
            return list(ObjectiveFinder(self.game, self.player).front_lines())
        except Exception:  # pragma: no cover - defensive
            logging.warning("Could not enumerate RED front lines", exc_info=True)
            return []

    def _project_fronts(self) -> tuple[FrontView, ...]:
        from game.ai_commander.postures import legal_postures_for, posture_of_stance

        views: list[FrontView] = []
        for index, front_line in enumerate(self._active_front_lines(), start=1):
            own_cp = front_line.control_point_friendly_to(self.player)
            enemy_cp = front_line.control_point_hostile_to(self.player)
            own_units = self._deployable_units(own_cp)
            enemy_units = self._deployable_units(enemy_cp)
            current_stance = own_cp.stances.get(enemy_cp.id)
            views.append(
                FrontView(
                    id=f"FRONT-{index}",
                    own_base=str(own_cp.name),
                    enemy_base=str(enemy_cp.name),
                    own_deployable_units=own_units,
                    own_unit_capacity=int(own_cp.frontline_unit_count_limit),
                    enemy_strength=_band_for_ratio(own_units, enemy_units),
                    enemy_unit_count=(
                        enemy_units if self.policy is IntelPolicy.FULL_PARITY else None
                    ),
                    legal_postures=legal_postures_for(
                        self.game, front_line, self.player
                    ),
                    current_posture=(
                        posture_of_stance(current_stance) if current_stance else None
                    ),
                    reinforcement_eligible=bool(
                        own_cp.has_ground_unit_source(self.game)
                    ),
                )
            )
        return tuple(views)

    def _is_observable(self, position: Any) -> bool:
        """Whether RED could plausibly know about something at ``position``."""

        if self.policy is IntelPolicy.FULL_PARITY:
            return True
        try:
            if self.game.threat_zone_for(self.player).threatened(position):
                return True
        except Exception:  # pragma: no cover - threat zones may be uninitialised
            logging.debug("RED threat zone unavailable for observation", exc_info=True)
        for control_point in self._own_control_points():
            try:
                if (
                    position.distance_to_point(control_point.position)
                    <= _REALISTIC_OBSERVATION_RANGE_METERS
                ):
                    return True
            except Exception:  # pragma: no cover - defensive
                continue
        return False

    def _observable(self, objects: Iterable[Any]) -> list[Any]:
        return [o for o in objects if self._is_observable(getattr(o, "position", None))]

    def _project_target_sets(
        self, fronts: Sequence[FrontView]
    ) -> tuple[TargetSetView, ...]:
        from game.commander.objectivefinder import ObjectiveFinder

        finder = ObjectiveFinder(self.game, self.player)
        turn = int(self.game.turn)

        def safe(call: Any) -> list[Any]:
            try:
                return list(call())
            except Exception:  # pragma: no cover - defensive
                logging.debug("Objective enumeration failed", exc_info=True)
                return []

        air_defences = self._observable(safe(finder.enemy_air_defenses))
        infrastructure = self._observable(safe(finder.strike_targets))
        motorpools = self._observable(safe(finder.motorpool_targets))
        ships = self._observable(safe(finder.enemy_ships))
        convoys = self._observable(safe(finder.convoys))
        cargo_ships = self._observable(safe(finder.cargo_ships))
        capturable = self._observable(safe(finder.prioritized_points))
        enemy_bases = self._observable(safe(finder.enemy_control_points))
        vulnerable_own = safe(finder.vulnerable_control_points)

        confidence = (
            Confidence.CONFIRMED
            if self.policy is IntelPolicy.FULL_PARITY
            else Confidence.PROBABLE
        )
        precision = (
            LocationPrecision.PRECISE
            if self.policy is IntelPolicy.FULL_PARITY
            else LocationPrecision.AREA
        )

        threatening = self._threatening_count(air_defences)

        candidates: list[tuple[TargetSetCategory, int, bool, str]] = [
            (
                TargetSetCategory.AIR_SUPERIORITY,
                len(enemy_bases),
                True,
                "enemy air activity originating from these bases",
            ),
            (
                TargetSetCategory.BASE_DEFENCE,
                len(vulnerable_own),
                True,
                "own bases assessed as exposed",
            ),
            (
                TargetSetCategory.ENEMY_AIR_DEFENCES,
                len(air_defences),
                threatening > 0,
                f"{threatening} assessed as covering our approach routes",
            ),
            (
                TargetSetCategory.ENEMY_AIRBASES,
                len(enemy_bases),
                False,
                "runways, parking and support infrastructure",
            ),
            (
                TargetSetCategory.ENEMY_BATTLE_POSITIONS,
                sum(1 for _ in fronts),
                True,
                "dug-in enemy units opposing our fronts",
            ),
            (
                TargetSetCategory.ENEMY_REINFORCEMENTS,
                len(convoys) + len(cargo_ships),
                False,
                "road and sea movement towards the fronts",
            ),
            (
                TargetSetCategory.ENEMY_INFRASTRUCTURE,
                len(infrastructure),
                False,
                "production, fuel and ammunition sites",
            ),
            (
                TargetSetCategory.ENEMY_MOTORPOOLS,
                len(motorpools),
                False,
                "rear-area vehicle concentrations",
            ),
            (
                TargetSetCategory.ENEMY_SHIPPING,
                len(ships),
                False,
                "surface groups",
            ),
            (
                TargetSetCategory.BASE_CAPTURE,
                len(capturable),
                False,
                "bases assessed as capturable",
            ),
        ]

        views: list[TargetSetView] = []
        index = 0
        for category, count, threatens, note in candidates:
            if count <= 0:
                continue
            index += 1
            views.append(
                TargetSetView(
                    id=f"TS-{index}",
                    category=category,
                    known_count=count,
                    confidence=confidence,
                    last_observed_turn=turn,
                    location_precision=precision,
                    threatens_own_forces=threatens,
                    notes=note,
                )
            )
        return tuple(views)

    def _threatening_count(self, air_defences: Sequence[Any]) -> int:
        own = self._own_control_points()
        count = 0
        for air_defence in air_defences:
            position = getattr(air_defence, "position", None)
            if position is None:
                continue
            for control_point in own:
                try:
                    if (
                        position.distance_to_point(control_point.position)
                        <= _REALISTIC_OBSERVATION_RANGE_METERS
                    ):
                        count += 1
                        break
                except Exception:  # pragma: no cover - defensive
                    continue
        return count

    def _project_procurement_categories(
        self,
    ) -> tuple[ProcurementCategoryView, ...]:
        from game.config import RUNWAY_REPAIR_COST

        budget = float(self.coalition.budget)
        faction = self.coalition.faction
        ground_units = list(getattr(faction, "frontline_units", [])) + list(
            getattr(faction, "artillery_units", [])
        )
        cheapest_ground = min((float(u.price) for u in ground_units), default=None)

        aircraft_prices: list[float] = []
        for squadron in self.coalition.air_wing.iter_squadrons():
            price = getattr(getattr(squadron, "aircraft", None), "price", None)
            if price is not None:
                aircraft_prices.append(float(price))
        cheapest_aircraft = min(aircraft_prices, default=None)

        own = self._own_control_points()
        repairable = any(cp.runway_can_be_repaired for cp in own)
        can_recruit = any(cp.has_ground_unit_source(self.game) for cp in own)
        can_reserve = any(
            not cp.is_global and cp.can_recruit_ground_units(self.game) for cp in own
        )

        entries: list[tuple[ProcurementCategory, bool, AffordabilityBand, str]] = [
            (
                ProcurementCategory.AIRCRAFT,
                bool(aircraft_prices),
                _affordability(budget, cheapest_aircraft),
                "replaces losses and fills planned missions",
            ),
            (
                ProcurementCategory.GROUND_COMBAT_UNITS,
                can_recruit,
                _affordability(budget, cheapest_ground),
                "front line reinforcement at bases with a supply source",
            ),
            (
                ProcurementCategory.FRONT_LINE_RESERVES,
                can_reserve,
                _affordability(budget, cheapest_ground),
                "units held at rear bases until a front reaches them",
            ),
            (
                ProcurementCategory.RUNWAY_REPAIR,
                repairable,
                _affordability(budget, float(RUNWAY_REPAIR_COST)),
                "restores flight operations at a damaged airbase",
            ),
        ]

        return tuple(
            ProcurementCategoryView(
                id=f"PROC-{index}",
                category=category,
                eligible=eligible,
                affordability=affordability,
                notes=note,
            )
            for index, (category, eligible, affordability, note) in enumerate(
                entries, start=1
            )
        )
