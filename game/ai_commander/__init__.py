"""An LLM-driven strategic commander for the RED coalition.

Retribution's OPFOR is planned by a deterministic hierarchical task network. This
package can put a language model *in front of* that planner: the model chooses a
strategy, orders the target sets, orders the spending categories and sets a
posture per front, and Retribution's own code then carries the plan out.

The model never executes anything itself. Three separate gates stand between a
model response and the campaign:

``intel``
    Builds the only view of the campaign the model is shown. RED sees its own
    state in full and BLUE only through what RED could plausibly observe.

``decision`` / ``legality``
    Validate the response against the schema and then against live campaign
    state, using the game's own predicates. Anything unaffordable, impossible or
    invented is rejected and recorded.

``execution``
    Applies what survived by *ordering* Retribution's existing planner and
    procurement code, never by bypassing it.

Everything the model was told, asked for and was refused is written to a JSON
decision log next to the campaign save, so a suspicious result can be audited
after the fact.

Nothing here runs unless ``Settings.ai_commander_enabled`` is set; the default is
off and stock behaviour is then bit-for-bit unchanged.
"""

from game.ai_commander.audit import AiDecisionRecord, AuditLog
from game.ai_commander.config import AiCommanderConfig
from game.ai_commander.controller import (
    CommanderTurnResult,
    RedCommanderTurn,
    describe_turn_result,
    plan_red_commander_turn,
)
from game.ai_commander.decision import (
    RedCommanderDecision,
    ValidationOutcome,
    parse_decision,
    validate_decision,
)
from game.ai_commander.directive import CommanderDirective
from game.ai_commander.enums import (
    CommanderPersonality,
    FallbackReason,
    FrontPosture,
    IntelPolicy,
    ProcurementCategory,
    RedStrategy,
    ReservePolicy,
    TargetSetCategory,
)
from game.ai_commander.execution import (
    DirectedProcurementAi,
    DirectedTheaterCommander,
    PostureApplication,
    apply_front_postures,
    task_order_for,
)
from game.ai_commander.intel import IntelProjector, RedCommanderBrief
from game.ai_commander.legality import LegalityChecker
from game.ai_commander.pricing import CostCapExceeded, CostLedger, ModelCatalog
from game.ai_commander.secretstore import SecretStore

__all__ = [
    "AiCommanderConfig",
    "AiDecisionRecord",
    "AuditLog",
    "CommanderDirective",
    "CommanderPersonality",
    "CommanderTurnResult",
    "CostCapExceeded",
    "CostLedger",
    "DirectedProcurementAi",
    "DirectedTheaterCommander",
    "FallbackReason",
    "FrontPosture",
    "IntelPolicy",
    "IntelProjector",
    "LegalityChecker",
    "ModelCatalog",
    "PostureApplication",
    "ProcurementCategory",
    "RedCommanderBrief",
    "RedCommanderDecision",
    "RedCommanderTurn",
    "RedStrategy",
    "ReservePolicy",
    "SecretStore",
    "TargetSetCategory",
    "ValidationOutcome",
    "apply_front_postures",
    "describe_turn_result",
    "parse_decision",
    "plan_red_commander_turn",
    "task_order_for",
    "validate_decision",
]
