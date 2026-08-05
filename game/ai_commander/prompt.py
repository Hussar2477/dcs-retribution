"""Prompt construction for the RED commander.

The prompt is assembled from three things only:

1. The fixed role description below.
2. A personality preset, which changes *tone and risk appetite* and nothing else.
   Presets cannot widen the action space -- legality is decided by
   :mod:`game.ai_commander.decision` and :mod:`game.ai_commander.legality`.
3. :meth:`RedCommanderBrief.render_compact`, the fog-of-war-filtered brief.

Nothing else about the campaign reaches the model. In particular there is no
free-text channel from game state into the prompt other than base names and the
brief's own enumerated fields.
"""

from __future__ import annotations

from typing import Any

from game.ai_commander.decision import (
    DECISION_SCHEMA_VERSION,
    decision_json_schema,
    example_decision_json,
)
from game.ai_commander.enums import CommanderPersonality
from game.ai_commander.intel import RedCommanderBrief
from game.ai_commander.serialization import canonical_json

SYSTEM_PROMPT = """\
You are the RED coalition theater commander in a turn-based DCS Retribution \
campaign. You issue one strategic directive per turn.

You command at commander level only. You do not choose aircraft types, weapons, \
targets, routes, altitudes, timings, unit counts or prices; a deterministic \
staff (the game's own planner) does all of that from your priorities. Your job \
is to decide what matters this turn and in what order.

Rules you cannot break:
- You may only refer to the identifiers given in the briefing. Invented \
identifiers are discarded.
- You may only use the enumerated values listed in the briefing.
- You cannot see BLUE's budget, aircraft inventory, squadrons or planned \
flights, and you must not speculate about exact numbers you were not given.
- Your directive is re-validated against live campaign state. Anything illegal \
or unaffordable is rejected and logged; it does not happen.
- Ranks start at 1, where 1 is the highest priority, and must be unique within \
each list.
- Omitting a list is allowed and means "no preference"; the deterministic \
planner keeps its own ordering for that dimension.

Reply with a single JSON object and nothing else. No prose, no code fences, no \
explanation outside the commander_intent field.\
"""

_PERSONALITIES: dict[CommanderPersonality, str] = {
    CommanderPersonality.CAUTIOUS: (
        "Command style: cautious. You protect what you hold, keep reserves, "
        "repair and rebuild before attacking, and only push a front when the "
        "force balance is clearly in your favour. You would rather lose a turn "
        "of initiative than a base."
    ),
    CommanderPersonality.BALANCED: (
        "Command style: balanced. You weigh defence and offence on their merits, "
        "spend broadly, and take a front forward when the opportunity is real "
        "rather than merely available."
    ),
    CommanderPersonality.AGGRESSIVE: (
        "Command style: aggressive. You seek the initiative, contest the air, "
        "press fronts where they are legal to press, and accept losses to keep "
        "BLUE reacting. You still do not waste forces on hopeless attacks."
    ),
    CommanderPersonality.ATTRITIONAL: (
        "Command style: attritional. You trade methodically: hit reinforcements, "
        "motor pools, infrastructure and air defences to wear BLUE down, and "
        "hold ground rather than trading it for tempo."
    ),
}


def personality_text(personality: CommanderPersonality) -> str:
    return _PERSONALITIES.get(
        personality, _PERSONALITIES[CommanderPersonality.BALANCED]
    )


def build_user_prompt(
    brief: RedCommanderBrief,
    personality: CommanderPersonality = CommanderPersonality.BALANCED,
) -> str:
    """The single user message: personality, brief, schema and example."""

    sections = [
        personality_text(personality),
        "",
        "=== INTELLIGENCE BRIEFING ===",
        brief.render_compact(),
        "",
        "=== REQUIRED RESPONSE SHAPE ===",
        f'schema_version must be exactly "{DECISION_SCHEMA_VERSION}".',
        f'campaign_revision must be exactly "{brief.campaign_revision}".',
        f"turn_id must be exactly {brief.turn_id}.",
        "",
        "JSON Schema:",
        canonical_json(decision_json_schema(brief)),
        "",
        "Example of a well-formed (not necessarily wise) response:",
        example_decision_json(brief),
        "",
        "Now issue your directive for this turn as a single JSON object.",
    ]
    return "\n".join(sections)


def build_messages(
    brief: RedCommanderBrief,
    personality: CommanderPersonality = CommanderPersonality.BALANCED,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(brief, personality)},
    ]


def build_repair_messages(
    brief: RedCommanderBrief,
    personality: CommanderPersonality,
    previous_response: str,
    error_summary: str,
) -> list[dict[str, str]]:
    """The one permitted repair attempt.

    The repair request contains the validation errors and the same legal
    identifiers -- never extra state. That distinction matters: a repair prompt
    that leaked more information would be a way to obtain intel by deliberately
    failing validation.
    """

    messages = build_messages(brief, personality)
    messages.append({"role": "assistant", "content": previous_response[:4000]})
    messages.append(
        {
            "role": "user",
            "content": (
                "That response was rejected by the validator:\n"
                f"{error_summary}\n\n"
                "Send the corrected directive as a single JSON object. Use only "
                "the identifiers and enumerated values from the briefing above. "
                "No additional information is available to you."
            ),
        }
    )
    return messages


def response_format_for(
    brief: RedCommanderBrief, supports_json_schema: bool, supports_json_object: bool
) -> Any:
    """Choose the strictest structured-output mode the model supports.

    Local validation runs regardless, so a provider that silently ignores this is
    not a correctness problem -- only a token-efficiency one.
    """

    if supports_json_schema:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "red_commander_decision",
                "strict": True,
                "schema": decision_json_schema(brief),
            },
        }
    if supports_json_object:
        return {"type": "json_object"}
    return None
