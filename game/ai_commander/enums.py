"""Enumerations shared by the LLM RED commander subsystem.

Every value the LLM is allowed to emit is defined here. The prompt is generated
from these enumerations and the validator rejects anything that is not a member,
so the model can never widen its own action space.
"""

from __future__ import annotations

from enum import Enum, unique


@unique
class IntelPolicy(Enum):
    """How much of the opposing coalition's state the RED commander may see.

    Retribution has no campaign-layer fog of war of its own (the in-game "Intel"
    window shows the human player the enemy's full economy, air wing and ground
    forces, and the built-in OPFOR planner reads authoritative state directly).
    The policy therefore has to be an explicit choice made by this subsystem
    rather than a reflection of an existing engine rule.
    """

    #: Only what RED could plausibly observe: its own state in full, plus BLUE
    #: facts that are either public (base ownership, front line geography) or
    #: covered by RED's own sensor/threat coverage. Strength figures for BLUE are
    #: reduced to coarse bands.
    REALISTIC = "realistic"

    #: Parity with what the human player can already see in Retribution's own
    #: Intel window, and with what the built-in OPFOR planner uses. Still no raw
    #: save data, no BLUE flight plans and no cheat actions.
    FULL_PARITY = "full_parity"


@unique
class RedStrategy(Enum):
    """The single overall posture the commander picks for the turn."""

    DEFEND = "defend"
    ATTRIT = "attrit"
    AIR_SUPERIORITY = "air_superiority"
    GROUND_OFFENSIVE = "ground_offensive"
    DEEP_INTERDICTION = "deep_interdiction"
    REBUILD = "rebuild"


@unique
class FrontPosture(Enum):
    """Posture requested for one front line.

    Each member maps onto exactly one of Retribution's own ``CombatStance``
    values, and is only applied if the same advantage precondition the built-in
    planner uses is satisfied.
    """

    RETREAT = "retreat"
    HOLD = "hold"
    PROBE = "probe"
    PUSH = "push"
    BREAKTHROUGH = "breakthrough"


@unique
class TargetSetCategory(Enum):
    """A class of objective the deterministic planner already knows how to plan.

    The commander ranks these categories; the deterministic planner still picks
    the individual targets, packages, flights, routes and loadouts.
    """

    AIR_SUPERIORITY = "air_superiority"
    BASE_DEFENCE = "base_defence"
    ENEMY_AIR_DEFENCES = "enemy_air_defences"
    ENEMY_AIRBASES = "enemy_airbases"
    ENEMY_BATTLE_POSITIONS = "enemy_battle_positions"
    ENEMY_REINFORCEMENTS = "enemy_reinforcements"
    ENEMY_INFRASTRUCTURE = "enemy_infrastructure"
    ENEMY_MOTORPOOLS = "enemy_motorpools"
    ENEMY_SHIPPING = "enemy_shipping"
    BASE_CAPTURE = "base_capture"


@unique
class ProcurementCategory(Enum):
    """Spending categories the commander may rank."""

    AIRCRAFT = "aircraft"
    GROUND_COMBAT_UNITS = "ground_combat_units"
    FRONT_LINE_RESERVES = "front_line_reserves"
    RUNWAY_REPAIR = "runway_repair"


@unique
class MissionPurpose(Enum):
    """Why the commander wants a target set attacked.

    Advisory only: recorded for the audit log and the UI, and used to break ties
    between equally ranked target sets. It never unlocks a capability.
    """

    ATTRITION = "attrition"
    SHAPE_FRONT = "shape_front"
    PROTECT_OWN_FORCES = "protect_own_forces"
    OPEN_CORRIDOR = "open_corridor"
    DENY_REINFORCEMENT = "deny_reinforcement"
    ECONOMIC_PRESSURE = "economic_pressure"


@unique
class ReservePolicy(Enum):
    """How much of the ground budget is held back from active fronts."""

    COMMIT_EVERYTHING = "commit_everything"
    BALANCED = "balanced"
    BUILD_RESERVES = "build_reserves"


@unique
class Confidence(Enum):
    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    POSSIBLE = "possible"


@unique
class LocationPrecision(Enum):
    PRECISE = "precise"
    AREA = "area"
    REGION = "region"


@unique
class StrengthBand(Enum):
    """Coarse relative strength, used instead of exact BLUE unit counts."""

    NEGLIGIBLE = "negligible"
    MUCH_WEAKER = "much_weaker"
    WEAKER = "weaker"
    COMPARABLE = "comparable"
    STRONGER = "stronger"
    MUCH_STRONGER = "much_stronger"
    UNKNOWN = "unknown"


@unique
class AffordabilityBand(Enum):
    NONE = "none"
    LIMITED = "limited"
    COMFORTABLE = "comfortable"


@unique
class CommanderPersonality(Enum):
    """Prompt-level personality preset. Never changes what is legal."""

    CAUTIOUS = "cautious"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ATTRITIONAL = "attritional"


@unique
class FallbackReason(Enum):
    """Why a turn fell back to Retribution's built-in RED automation."""

    DISABLED = "disabled"
    NOT_CONFIGURED = "not_configured"
    COST_CAP = "cost_cap"
    TRANSPORT_ERROR = "transport_error"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"
    STALE_RESPONSE = "stale_response"
    NO_LEGAL_CONTENT = "no_legal_content"
    UNEXPECTED_ERROR = "unexpected_error"
