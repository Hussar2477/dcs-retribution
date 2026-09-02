"""Prompt construction for ACTIVE mode's three-stage turn.

COMMANDER mode (:mod:`game.ai_commander.prompt`) asks the model for one
strategic directive and lets the deterministic staff do everything else. ACTIVE
mode instead walks the same chain of command a human player walks:

1. :attr:`CommanderStage.COMMAND` -- aggression, doctrine and priorities. This
   reuses the phase-1 decision schema verbatim so all of its validation,
   legality and audit machinery applies unchanged.
2. :attr:`CommanderStage.LOGISTICS` -- money and force structure: aircraft and
   ground-unit purchases, runway repairs, squadron relocation, squadron tasking
   and ground transfers.
3. :attr:`CommanderStage.AIR_TASKING` -- the air tasking order: which targets to
   strike, with what missions, at what size.

Every stage sees exactly the same three information sources and nothing else:

* the *capability index* (:mod:`game.ai_commander.capabilities`), derived from
  the game's own unit tables and scoped to what RED owns or can buy;
* the fog-filtered strategic brief (:mod:`game.ai_commander.intel`);
* the fog-filtered operations brief (:mod:`game.ai_commander.operations`).

The capability index is the anti-hallucination measure: the model is told what
its airframes can actually do by the campaign's data files rather than being
left to recall it. It is *also* an anti-cheat measure, because it is built from
the RED coalition alone and structurally cannot contain BLUE's order of battle.
"""

from __future__ import annotations

from typing import Any, Optional

from game.ai_commander.capabilities import CapabilityIndex
from game.ai_commander.decision import (
    DECISION_SCHEMA_VERSION,
    decision_json_schema,
    example_decision_json,
)
from game.ai_commander.enums import CommanderPersonality
from game.ai_commander.intel import RedCommanderBrief
from game.ai_commander.operations import PLANNABLE_MISSION_TYPES, OperationsBrief
from game.ai_commander.plan import (
    AIR_TASKING_SCHEMA_VERSION,
    LOGISTICS_SCHEMA_VERSION,
    CommanderStage,
    air_tasking_json_schema,
    example_air_tasking_json,
    example_logistics_json,
    logistics_json_schema,
)
from game.ai_commander.prompt import personality_text
from game.ai_commander.serialization import canonical_json
from game.ato.flighttype import FlightType

ACTIVE_SYSTEM_PROMPT = """\
You are the RED coalition commander in a turn-based DCS Retribution campaign, \
playing to win against a human BLUE player. You have the same controls a human \
player has and no others.

You act in three stages within one turn: command intent, then logistics, then \
the air tasking order. Each stage is a separate request. Earlier stages are \
summarised back to you; you cannot revisit them.

Rules you cannot break:
- You may only refer to identifiers given in the briefing for this stage. \
Invented identifiers are discarded.
- You may only use the enumerated values listed for each field.
- The capability index lists every unit type RED owns or can buy, with the \
figures the campaign's own data files hold. Use those figures. Do not rely on \
outside knowledge of these aircraft; where the index and your recollection \
disagree, the index is correct.
- You cannot see BLUE's budget, aircraft inventory, squadrons, pilots or \
planned flights. Enemy ground objects and bases appear only when RED forces \
can observe them. Do not speculate about numbers you were not given.
- You cannot spend money you do not have. Every order is re-priced against live \
campaign state; anything unaffordable is rejected and logged.
- Every order is re-validated against live campaign state before it happens. \
Illegal orders are rejected individually, with a reason, and the rest of your \
plan still runs.
- Omitting a list is allowed and means "no orders of that kind". The game's \
own automation fills whatever you leave unspent or unplanned.

Reply with a single JSON object and nothing else. No prose, no code fences, no \
explanation outside the intent field.\
"""

_STAGE_BRIEFING: dict[CommanderStage, str] = {
    CommanderStage.COMMAND: (
        "STAGE 1 of 3: COMMAND INTENT. Set this turn's strategy, the posture of "
        "each front, which target sets matter and where money should go. You are "
        "not placing individual orders yet -- you are deciding what this turn is "
        "for. Stages 2 and 3 will spend and plan against this intent."
    ),
    CommanderStage.LOGISTICS: (
        "STAGE 2 of 3: LOGISTICS. Spend the budget and shape the force: buy "
        "aircraft and ground units, repair runways, move squadrons, set what "
        "each squadron is allowed to be auto-tasked with, and transfer ground "
        "units between bases. Anything you do not spend is spent for you by the "
        "game's own procurement automation, which is competent but has no "
        "knowledge of your intent."
    ),
    CommanderStage.AIR_TASKING: (
        "STAGE 3 of 3: AIR TASKING ORDER. Plan this turn's packages. Each "
        "package attacks one target with one or more flights. The game's own "
        "mission planner turns each package into routes, altitudes, timings, "
        "loadouts and escorts, and will refuse a package it cannot crew or "
        "reach. Plan the packages that serve your intent; leftover capacity is "
        "planned for you. Two rules the planner enforces: (1) each flight's "
        "mission_type must be one of the mission types listed in that target's "
        "own missions= field in the briefing -- the only additions allowed "
        "beyond that list are Escort and SEAD Escort as supporting flights; "
        "(2) attack each target with at most one package -- never repeat a "
        "target_id."
    ),
}

_STAGE_CLOSING: dict[CommanderStage, str] = {
    CommanderStage.COMMAND: (
        "Now issue your command intent for this turn as a single JSON object."
    ),
    CommanderStage.LOGISTICS: (
        "Now issue your logistics orders for this turn as a single JSON object."
    ),
    CommanderStage.AIR_TASKING: (
        "Now issue your air tasking order for this turn as a single JSON object."
    ),
}

_SCHEMA_NAMES: dict[CommanderStage, str] = {
    CommanderStage.COMMAND: "red_commander_decision",
    CommanderStage.LOGISTICS: "red_commander_logistics",
    CommanderStage.AIR_TASKING: "red_commander_air_tasking",
}


def stage_briefing_text(stage: CommanderStage) -> str:
    return _STAGE_BRIEFING[stage]


def _capability_tasks(
    stage: CommanderStage, ops: OperationsBrief
) -> Optional[frozenset[FlightType]]:
    """Which task columns of the capability index this stage actually needs.

    Stage 1 does not choose aircraft at all, so it gets the unfiltered index
    (roles still matter for judging what the force is good at). Stages 2 and 3
    are pruned to the missions that are plannable in this campaign, which keeps
    the rendered index a few hundred tokens smaller without hiding anything the
    model is allowed to act on.
    """

    if stage is CommanderStage.COMMAND:
        return None
    plannable = set(ops.plannable_mission_types)
    return frozenset(
        task for task in PLANNABLE_MISSION_TYPES if task.value in plannable
    )


def _schema_for(
    stage: CommanderStage,
    brief: RedCommanderBrief,
    ops: OperationsBrief,
    capabilities: CapabilityIndex,
) -> dict[str, Any]:
    if stage is CommanderStage.COMMAND:
        return decision_json_schema(brief)
    if stage is CommanderStage.LOGISTICS:
        return logistics_json_schema(ops, capabilities)
    return air_tasking_json_schema(ops, capabilities)


def _example_for(
    stage: CommanderStage,
    brief: RedCommanderBrief,
    ops: OperationsBrief,
    capabilities: CapabilityIndex,
) -> str:
    if stage is CommanderStage.COMMAND:
        return example_decision_json(brief)
    if stage is CommanderStage.LOGISTICS:
        return canonical_json(example_logistics_json(ops, capabilities))
    return canonical_json(example_air_tasking_json(ops, capabilities))


def _schema_version_for(stage: CommanderStage) -> str:
    if stage is CommanderStage.COMMAND:
        return DECISION_SCHEMA_VERSION
    if stage is CommanderStage.LOGISTICS:
        return LOGISTICS_SCHEMA_VERSION
    return AIR_TASKING_SCHEMA_VERSION


def build_stage_user_prompt(
    stage: CommanderStage,
    brief: RedCommanderBrief,
    ops: OperationsBrief,
    capabilities: CapabilityIndex,
    personality: CommanderPersonality = CommanderPersonality.BALANCED,
    prior_stage_summary: Optional[str] = None,
) -> str:
    """Assemble the single user message for one stage of an ACTIVE turn."""

    sections: list[str] = [
        personality_text(personality),
        "",
        stage_briefing_text(stage),
        "",
        "=== RED CAPABILITY INDEX ===",
        "Compiled from this campaign's own unit data. RED assets only.",
        capabilities.render_compact(_capability_tasks(stage, ops)),
        "",
        "=== INTELLIGENCE BRIEFING ===",
        brief.render_compact(),
    ]

    if stage is not CommanderStage.COMMAND:
        sections += [
            "",
            "=== OPERATIONS BRIEFING ===",
            ops.render_compact(),
        ]

    if prior_stage_summary:
        sections += [
            "",
            "=== DECISIONS ALREADY MADE THIS TURN ===",
            prior_stage_summary,
        ]

    schema_version = _schema_version_for(stage)
    sections += [
        "",
        "=== REQUIRED RESPONSE SHAPE ===",
        f'schema_version must be exactly "{schema_version}".',
        f'campaign_revision must be exactly "{brief.campaign_revision}".',
        f"turn_id must be exactly {brief.turn_id}.",
        "",
        "JSON Schema:",
        canonical_json(_schema_for(stage, brief, ops, capabilities)),
        "",
        "Example of a well-formed (not necessarily wise) response:",
        _example_for(stage, brief, ops, capabilities),
        "",
        _STAGE_CLOSING[stage],
    ]
    return "\n".join(sections)


def build_stage_messages(
    stage: CommanderStage,
    brief: RedCommanderBrief,
    ops: OperationsBrief,
    capabilities: CapabilityIndex,
    personality: CommanderPersonality = CommanderPersonality.BALANCED,
    prior_stage_summary: Optional[str] = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ACTIVE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_stage_user_prompt(
                stage, brief, ops, capabilities, personality, prior_stage_summary
            ),
        },
    ]


def build_stage_repair_messages(
    stage: CommanderStage,
    brief: RedCommanderBrief,
    ops: OperationsBrief,
    capabilities: CapabilityIndex,
    personality: CommanderPersonality,
    previous_response: str,
    error_summary: str,
    prior_stage_summary: Optional[str] = None,
) -> list[dict[str, str]]:
    """The one permitted repair attempt for a stage.

    As in COMMANDER mode the repair request adds the validation errors and
    nothing else. A repair prompt that disclosed more state than the original
    would turn deliberate validation failure into an intelligence channel.
    """

    messages = build_stage_messages(
        stage, brief, ops, capabilities, personality, prior_stage_summary
    )
    messages.append({"role": "assistant", "content": previous_response[:4000]})
    messages.append(
        {
            "role": "user",
            "content": (
                "That response was rejected by the validator:\n"
                f"{error_summary}\n\n"
                "Send the corrected response for this stage as a single JSON "
                "object. Use only the identifiers and enumerated values from the "
                "briefing above. No additional information is available to you."
            ),
        }
    )
    return messages


def stage_response_format(
    stage: CommanderStage,
    brief: RedCommanderBrief,
    ops: OperationsBrief,
    capabilities: CapabilityIndex,
    supports_json_schema: bool,
    supports_json_object: bool,
) -> Any:
    """Choose the strictest structured-output mode the model supports.

    Local validation runs regardless, so a provider that silently ignores this
    is a token-efficiency problem and never a correctness one.
    """

    if supports_json_schema:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": _SCHEMA_NAMES[stage],
                "strict": True,
                "schema": _schema_for(stage, brief, ops, capabilities),
            },
        }
    if supports_json_object:
        return {"type": "json_object"}
    return None
