#!/usr/bin/env python3
"""Offline dry-run harness for the RED LLM commander.

Runs complete RED commander turns against a synthetic campaign without DCS,
without a save file and -- unless you ask for it -- without a network. It exists
to answer four questions before anyone trusts the feature in a real campaign:

1. Does a turn survive every way the model can misbehave? Each scenario feeds a
   canned response (valid, malformed, cheating, tool-calling, transport failure,
   timeout) and prints what the validator accepted, what it refused and why, and
   which fallback took over.
2. Does the briefing leak BLUE-private intelligence? The synthetic campaign puts
   unique sentinel values on every BLUE-private field, and the harness scans the
   whole assembled prompt for them.
3. What does a turn cost? The prompt and a representative response are measured,
   then priced against several candidate models and compared with the per-turn
   cost cap.
4. Does it work against the real provider? With ``OPENROUTER_API_KEY`` set, one
   real call is made. Without a key the harness still runs everything else and
   exits 0.

Usage::

    python tools/ai_commander_dryrun.py
    python -m tools.ai_commander_dryrun --list
    python tools/ai_commander_dryrun.py --scenario cheating-intel --verbose
    python tools/ai_commander_dryrun.py --json results.json

The exit code reflects the offline scenarios only: 0 when every scenario ended
the way it was expected to. A live call that fails for network reasons is
reported but does not change the exit code (use ``--strict-live`` to change it).

The synthetic campaign comes from ``tests.ai_commander.fakes`` on purpose, so the
harness and the unit tests exercise byte-for-byte the same state; a drift in one
shows up in the other.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, cast

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from game.ai_commander.audit import AuditLog
from game.ai_commander.config import AiCommanderConfig
from game.ai_commander.controller import RedCommanderTurn, describe_turn_result
from game.ai_commander.decision import (
    example_decision_json,
    decision_json_schema,
    parse_decision,
)
from game.ai_commander.enums import (
    CommanderPersonality,
    FallbackReason,
    IntelPolicy,
)
from game.ai_commander.intel import IntelProjector, RedCommanderBrief
from game.ai_commander.legality import LegalityChecker
from game.ai_commander.llmclient import (
    ChatCompletionClient,
    DEFAULT_MODEL,
    LlmError,
    LlmTimeout,
    LlmTransportError,
    LlmResponse,
    TokenUsage,
)
from game.ai_commander.pricing import (
    CostLedger,
    FALLBACK_INPUT_PRICE_PER_MILLION,
    FALLBACK_OUTPUT_PRICE_PER_MILLION,
    ModelCatalog,
    ModelPrice,
    estimate_tokens,
)
from game.ai_commander.prompt import build_messages, build_repair_messages
from game.ai_commander.secretstore import ENV_VAR, SecretStore, mask
from game.ground_forces.combat_stance import CombatStance
from game.theater.player import Player
from tests.ai_commander.fakes import (
    BLUE_SENTINELS,
    CATALOG_PAYLOAD,
    RED_SENTINELS,
    FakeFrontLine,
    FakeObjectiveFinder,
    FakeTheater,
    ScriptedClient,
    make_control_point,
    make_iads,
    point,
    blue_leaks_in,
    make_config,
    red_facts_missing_from,
    serialise_everything,
    synthetic_game,
)

# ---------------------------------------------------------------------------
# Candidate models.
#
# Prices are US dollars per million tokens as recorded in the design-inputs
# research document on 2026-08-05. They are *reference figures for this report
# only*: at runtime the controller always reads live prices from the provider's
# ``/models`` endpoint and falls back to the deliberately pessimistic
# ``pricing.FALLBACK_*`` figures if the catalogue is unavailable. Re-run with a
# key (or check the provider's pricing page) before quoting these numbers.
# ---------------------------------------------------------------------------
CANDIDATE_MODELS: tuple[tuple[str, float, float, str], ...] = (
    ("deepseek/deepseek-v4-flash-0731", 0.09, 0.18, "shipped default"),
    ("qwen/qwen3.7-flash", 0.03, 0.13, "cheapest surveyed"),
    ("openai/gpt-5.6-luna", 0.10, 0.60, "cheap frontier-family"),
    ("z-ai/glm-5.2", 0.76, 2.42, "mid-priced"),
    ("moonshotai/kimi-k3", 3.00, 15.00, "expensive; worst case"),
    (
        "<catalogue unavailable>",
        FALLBACK_INPUT_PRICE_PER_MILLION,
        FALLBACK_OUTPUT_PRICE_PER_MILLION,
        "built-in pessimistic fallback price",
    ),
)

#: The controller makes at most this many chat completions for one RED turn: the
#: initial request plus, only if the first response fails validation, exactly one
#: repair request. See ``controller.RedCommanderTurn._run``. The one ``GET
#: /models`` catalogue lookup per turn is not billed as a completion.
MAX_COMPLETIONS_PER_TURN = 2

#: Money ceiling this feature was asked to stay under, per RED turn.
COST_CEILING_PER_TURN = 0.50


# ---------------------------------------------------------------------------
# Scenario plumbing
# ---------------------------------------------------------------------------


@dataclass
class ScenarioOutcome:
    """What one scenario actually did, for printing and for the exit code."""

    name: str
    question: str
    expected: str
    accepted: bool = False
    fallback_reason: Optional[str] = None
    completions: int = 0
    rejections: list[dict[str, Any]] = field(default_factory=list)
    log_path: Optional[str] = None
    directive: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    actual_cost: float = 0.0
    matched_expectation: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "question": self.question,
            "expected": self.expected,
            "accepted": self.accepted,
            "fallback_reason": self.fallback_reason,
            "completions": self.completions,
            "rejections": self.rejections,
            "log_path": self.log_path,
            "directive": self.directive,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "actual_cost": self.actual_cost,
            "matched_expectation": self.matched_expectation,
            "notes": self.notes,
        }


class MutatingClient(ScriptedClient):
    """A client that changes campaign state while the request is "in flight".

    Stands in for anything that can move the campaign on between the briefing
    and the moment orders are applied. The decision comes back correct for a
    state that no longer exists, which must be refused rather than applied.
    """

    def __init__(self, script: Sequence[Any], mutate: Callable[[], None], **kw: Any):
        super().__init__(script, **kw)
        self._mutate = mutate

    def complete(self, *args: Any, **kwargs: Any) -> LlmResponse:
        response = super().complete(*args, **kwargs)
        self._mutate()
        return response


def install_fake_objective_finder() -> None:
    """Point the commander at the synthetic theater instead of a real one.

    Every commander module imports ``ObjectiveFinder`` lazily inside the methods
    that use it, so replacing the attribute on its defining module covers intel
    projection, legality checking and posture application at once. This is the
    non-pytest equivalent of ``fakes.patch_objective_finder`` and is applied for
    the whole process: this harness never touches a real campaign.
    """

    import game.commander.objectivefinder as module

    setattr(module, "ObjectiveFinder", FakeObjectiveFinder)


def scaled_campaign(front_pairs: int, iads_per_base: int = 3) -> tuple[Any, Any]:
    """The synthetic campaign widened to ``front_pairs`` contested fronts.

    Used to measure how the briefing -- and therefore the bill -- grows with
    campaign size, instead of assuming it. Each added pair is one RED base facing
    one BLUE base 40 km away, inside RED's observation range, with its own SAM
    sites, so every added front also adds known target objects.
    """

    campaign, game = synthetic_game()
    if front_pairs <= 1:
        return campaign, game

    control_points = list(campaign.control_points)
    fronts = list(campaign.theater.fronts)
    for index in range(1, front_pairs):
        offset = 80_000.0 * index
        blue_id = 200 + index
        blue = make_control_point(
            cp_id=blue_id,
            name=f"BLUE-FRONT-{index}",
            captured=Player.BLUE,
            position=point(40_000.0, offset),
            deployable=900,
            capacity=2_000,
            income_per_turn=50,
            active_frontline=True,
            ground_objects=[
                make_iads(
                    f"BLUE SAM {index}.{site}",
                    point(30_000.0, offset + 500.0 * site),
                )
                for site in range(iads_per_base)
            ],
        )
        red = make_control_point(
            cp_id=100 + index,
            name=f"RED-FRONT-{index}",
            captured=Player.RED,
            position=point(0.0, offset),
            deployable=1_200,
            capacity=2_400,
            aircraft_present=24,
            ground_present=8,
            income_per_turn=90,
            active_frontline=True,
            stances={blue_id: CombatStance.DEFENSIVE},
        )
        control_points += [red, blue]
        fronts.append(cast(Any, FakeFrontLine(red, blue)))

    campaign.control_points = control_points
    campaign.theater = FakeTheater(
        campaign.theater.terrain_name, control_points, fronts
    )
    return campaign, game


def _brief_for(game: Any, policy: IntelPolicy = IntelPolicy.REALISTIC) -> Any:
    """The same briefing the controller will build for itself."""

    return IntelProjector(game, policy).project()


def _measured_usage(prompt_text: str, response_text: str) -> TokenUsage:
    """Token usage a provider would plausibly report for these exact strings.

    Real providers count with their own tokeniser; this harness has none, so it
    uses the same estimator the controller uses to size its cost reservation
    (see :func:`pricing.estimate_tokens`). Scenario cost figures are therefore
    estimates of the same kind the controller itself makes -- never a claim about
    a real invoice.
    """

    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(response_text)
    return TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _prompt_text(brief: Any, personality: CommanderPersonality) -> str:
    return "\n".join(m["content"] for m in build_messages(brief, personality))


def _cheating_plan(brief: Any) -> str:
    """A plan built out of things RED was never told.

    Every identifier here is invented, and the intent quotes BLUE-private
    sentinel values. A real model could not produce these from the briefing --
    the sentinel scan in this harness proves the values are not in it -- so this
    stands in for a jailbroken or hallucinating model, and for a future build
    that accidentally widens the briefing.
    """

    payload = json.loads(example_decision_json(brief))
    payload["front_priorities"] = [{"front_id": "FRONT-BLUE-SECRET", "rank": 1}]
    payload["push_postures"] = [
        {"front_id": "FRONT-BLUE-SECRET", "posture": "breakthrough"}
    ]
    payload["target_set_priorities"] = [
        {
            "target_set_id": "TS-BLUE-SQUADRON-ROSTER",
            "rank": 1,
            "purpose": "attrition",
        },
        {"target_set_id": "TS-404", "rank": 2, "purpose": "attrition"},
    ]
    payload["spending_priorities"] = [
        {"category_id": "PROC-BLUE-PENDING-ORDERS", "rank": 1}
    ]
    payload["commander_intent"] = (
        f"Hit {BLUE_SENTINELS['blue_hidden_base']} where "
        f"{BLUE_SENTINELS['blue_squadron_name']} parks "
        f"{BLUE_SENTINELS['blue_aircraft_name']}; they hold exactly "
        f"{BLUE_SENTINELS['blue_deployable_units']} deployable units and "
        f"{BLUE_SENTINELS['blue_budget']} in the bank."
    )
    return json.dumps(payload)


def _greedy_plan(brief: Any) -> str:
    """A plan that asks to buy everything, whatever the bank says."""

    payload = json.loads(example_decision_json(brief))
    payload["strategy"] = "rebuild"
    payload["reserve_policy"] = "commit_everything"
    payload["spending_priorities"] = [
        {"category_id": procurement_id, "rank": rank}
        for rank, procurement_id in enumerate(sorted(brief.procurement_ids), start=1)
    ]
    return json.dumps(payload)


def _posture_plan(brief: Any, posture: str) -> str:
    payload = json.loads(example_decision_json(brief))
    front_id = sorted(brief.front_ids)[0]
    payload["push_postures"] = [{"front_id": front_id, "posture": posture}]
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _run_turn(
    outcome: ScenarioOutcome,
    game: Any,
    config: AiCommanderConfig,
    client: Optional[ScriptedClient],
    audit_root: Path,
    expect_accepted: Optional[bool],
    expect_reason: Optional[FallbackReason],
    expect_completions: Optional[int] = None,
) -> ScenarioOutcome:
    """Run one full turn and fill in ``outcome``."""

    # Each scenario gets its own decision-log root. Every scenario plays the same
    # turn of the same campaign, and the controller deliberately refuses to pay
    # twice for one turn: sharing a root would make later scenarios replay the
    # first scenario's stored answer instead of exercising their own path.
    result = RedCommanderTurn(
        game, config, audit_log=AuditLog(audit_root / outcome.name), client=client
    ).run()
    described = describe_turn_result(result)

    outcome.accepted = bool(described["accepted"])
    outcome.fallback_reason = described["fallback_reason"]
    outcome.completions = len(client.calls) if client is not None else 0
    # describe_turn_result() reports a count; the full list lives on the record
    # (which holds every refusal from every attempt) or on the result itself when
    # the turn stopped before a record existed.
    if result.record is not None:
        outcome.rejections = [dict(item) for item in result.record.rejections]
    else:
        outcome.rejections = [item.to_dict() for item in result.rejections]
    outcome.log_path = described["log_path"]
    outcome.prompt_tokens = int(described["prompt_tokens"])
    outcome.completion_tokens = int(described["completion_tokens"])
    outcome.actual_cost = float(described["actual_cost"])
    if result.directive is not None:
        outcome.directive = result.directive.render_summary()

    matched = True
    if expect_accepted is not None and outcome.accepted is not expect_accepted:
        matched = False
        outcome.notes.append(
            f"MISMATCH: expected accepted={expect_accepted}, got {outcome.accepted}"
        )
    expected_reason = expect_reason.value if expect_reason is not None else None
    if expect_reason is not None and outcome.fallback_reason != expected_reason:
        matched = False
        outcome.notes.append(
            f"MISMATCH: expected fallback {expected_reason}, "
            f"got {outcome.fallback_reason}"
        )
    if expect_completions is not None and outcome.completions != expect_completions:
        matched = False
        outcome.notes.append(
            f"MISMATCH: expected {expect_completions} completion(s), "
            f"got {outcome.completions}"
        )
    outcome.matched_expectation = matched
    return outcome


def scenario_valid(audit_root: Path) -> ScenarioOutcome:
    """Does a well-formed plan get applied?"""

    outcome = ScenarioOutcome(
        name="valid",
        question="Does a well-formed plan get applied?",
        expected="accepted, one completion, orders written to the decision log",
    )
    campaign, game = synthetic_game()
    brief = _brief_for(game)
    plan = example_decision_json(brief)
    client = ScriptedClient(
        [plan],
        catalog=CATALOG_PAYLOAD,
        usage=_measured_usage(_prompt_text(brief, CommanderPersonality.BALANCED), plan),
    )
    return _run_turn(
        outcome,
        game,
        make_config(),
        client,
        audit_root,
        True,
        None,
        expect_completions=1,
    )


def scenario_repair(audit_root: Path) -> ScenarioOutcome:
    """Does one bad response get a second chance, and only one?"""

    outcome = ScenarioOutcome(
        name="repair",
        question="Does one bad response get a second chance, and only one?",
        expected="accepted after exactly two completions",
    )
    campaign, game = synthetic_game()
    brief = _brief_for(game)
    client = ScriptedClient(
        ["Sure! Here is my plan: attack everywhere.", example_decision_json(brief)],
        catalog=CATALOG_PAYLOAD,
    )
    return _run_turn(
        outcome,
        game,
        make_config(),
        client,
        audit_root,
        True,
        None,
        expect_completions=2,
    )


def scenario_malformed(audit_root: Path) -> ScenarioOutcome:
    """What happens when the model never produces valid JSON?"""

    outcome = ScenarioOutcome(
        name="malformed",
        question="What happens when the model never produces valid JSON?",
        expected="refused as MALFORMED_RESPONSE after two completions, built-in "
        "automation keeps the turn",
    )
    campaign, game = synthetic_game()
    client = ScriptedClient(
        ["I cannot do that.", "<thinking>still not JSON</thinking>"],
        catalog=CATALOG_PAYLOAD,
    )
    return _run_turn(
        outcome,
        game,
        make_config(),
        client,
        audit_root,
        False,
        FallbackReason.MALFORMED_RESPONSE,
        expect_completions=2,
    )


def scenario_truncated_json(audit_root: Path) -> ScenarioOutcome:
    """What happens when the response is cut off mid-object?"""

    outcome = ScenarioOutcome(
        name="truncated-json",
        question="What happens when the response is cut off mid-object?",
        expected="refused as MALFORMED_RESPONSE, not applied half-way",
    )
    campaign, game = synthetic_game()
    brief = _brief_for(game)
    half = example_decision_json(brief)[: len(example_decision_json(brief)) // 2]
    client = ScriptedClient([half, half], catalog=CATALOG_PAYLOAD)
    return _run_turn(
        outcome,
        game,
        make_config(),
        client,
        audit_root,
        False,
        FallbackReason.MALFORMED_RESPONSE,
    )


def scenario_cheating_intel(audit_root: Path) -> ScenarioOutcome:
    """Can a plan act on BLUE-private intelligence RED was never given?"""

    outcome = ScenarioOutcome(
        name="cheating-intel",
        question="Can a plan act on BLUE-private intelligence RED was never given?",
        expected="every invented identifier refused; nothing legal remains, so "
        "NO_LEGAL_CONTENT and the built-in automation keeps the turn",
    )
    campaign, game = synthetic_game()
    brief = _brief_for(game)

    prompt = _prompt_text(brief, CommanderPersonality.BALANCED)
    leaks = blue_leaks_in(serialise_everything(brief.to_dict(), prompt))
    outcome.notes.append(
        "sentinel scan of the assembled prompt: "
        + ("NO BLUE-private values present" if not leaks else f"LEAKED {leaks}")
    )
    if leaks:
        outcome.notes.append("MISMATCH: the briefing leaked BLUE-private values")

    client = ScriptedClient([_cheating_plan(brief)] * 2, catalog=CATALOG_PAYLOAD)
    result = _run_turn(
        outcome,
        game,
        make_config(),
        client,
        audit_root,
        False,
        FallbackReason.NO_LEGAL_CONTENT,
    )
    if leaks:
        result.matched_expectation = False
    return result


def scenario_cheating_overspend(audit_root: Path) -> ScenarioOutcome:
    """Can a plan spend money RED does not have?"""

    outcome = ScenarioOutcome(
        name="cheating-overspend",
        question="Can a plan spend money RED does not have?",
        expected="accepted, but every unaffordable purchase refused with the "
        "price it could not meet",
    )
    campaign, game = synthetic_game(red_budget=5.0)
    brief = _brief_for(game)
    client = ScriptedClient([_greedy_plan(brief)], catalog=CATALOG_PAYLOAD)
    result = _run_turn(outcome, game, make_config(), client, audit_root, True, None)
    spending = [r for r in result.rejections if "spending" in r["element"]]
    result.notes.append(
        f"RED bank: {5.0:.0f}; procurement requests refused: {len(spending)} of 4"
    )
    if len(spending) != 4:
        result.matched_expectation = False
        result.notes.append("MISMATCH: expected all four purchases to be refused")
    return result


def scenario_illegal_posture(audit_root: Path) -> ScenarioOutcome:
    """Can a weak RED force order an attack the game's own rules forbid?"""

    outcome = ScenarioOutcome(
        name="illegal-posture",
        question="Can a weak RED force order an attack the game's own rules forbid?",
        expected="the posture is refused against the stance precondition; the rest "
        "of the plan still applies",
    )
    # 500 RED against 1157 BLUE: the game's own stance rules allow only a retreat.
    campaign, game = synthetic_game(red_deployable=500)
    brief = _brief_for(game)
    front = brief.front(sorted(brief.front_ids)[0])
    outcome.notes.append(
        "postures the briefing declared legal: "
        + ", ".join(p.value for p in front.legal_postures)
    )
    client = ScriptedClient(
        [_posture_plan(brief, "breakthrough")] * 2, catalog=CATALOG_PAYLOAD
    )
    result = _run_turn(outcome, game, make_config(), client, audit_root, None, None)
    posture_refusals = [r for r in result.rejections if "posture" in r["element"]]
    if not posture_refusals:
        result.matched_expectation = False
        result.notes.append("MISMATCH: the illegal posture was not refused")
    else:
        result.matched_expectation = True
    return result


def scenario_stale_state(audit_root: Path) -> ScenarioOutcome:
    """What if the campaign moves on while the model is thinking?"""

    outcome = ScenarioOutcome(
        name="stale-state",
        question="What if the campaign moves on while the model is thinking?",
        expected="refused as STALE_RESPONSE rather than applied to a state it was "
        "not planned for",
    )
    campaign, game = synthetic_game()
    brief = _brief_for(game)

    def spend_the_budget() -> None:
        campaign.red.budget -= 1000.0

    client = MutatingClient(
        [example_decision_json(brief)] * 2,
        mutate=spend_the_budget,
        catalog=CATALOG_PAYLOAD,
    )
    return _run_turn(
        outcome,
        game,
        make_config(),
        client,
        audit_root,
        False,
        FallbackReason.STALE_RESPONSE,
    )


def scenario_tool_calls(audit_root: Path) -> ScenarioOutcome:
    """What if the response tries to call tools that were never offered?"""

    outcome = ScenarioOutcome(
        name="tool-calls",
        question="What if the response tries to call tools that were never offered?",
        expected="flagged in the decision log; the plan itself is still judged on "
        "its merits",
    )
    campaign, game = synthetic_game()
    brief = _brief_for(game)
    client = ScriptedClient(
        [example_decision_json(brief)], catalog=CATALOG_PAYLOAD, had_tool_calls=True
    )
    result = _run_turn(outcome, game, make_config(), client, audit_root, True, None)
    flagged = [r for r in result.rejections if r["element"] == "<tool_calls>"]
    if not flagged:
        result.matched_expectation = False
        result.notes.append("MISMATCH: unsolicited tool calls were not flagged")
    return result


def scenario_transport_failure(audit_root: Path) -> ScenarioOutcome:
    """What happens when the provider cannot be reached?"""

    outcome = ScenarioOutcome(
        name="transport-failure",
        question="What happens when the provider cannot be reached?",
        expected="one completion attempted, TRANSPORT_ERROR logged, built-in "
        "automation keeps the turn",
    )
    campaign, game = synthetic_game()
    client = ScriptedClient(
        [LlmTransportError("connection reset by peer")], catalog=CATALOG_PAYLOAD
    )
    return _run_turn(
        outcome,
        game,
        make_config(),
        client,
        audit_root,
        False,
        FallbackReason.TRANSPORT_ERROR,
    )


def scenario_timeout(audit_root: Path) -> ScenarioOutcome:
    """What happens when the model is too slow?"""

    outcome = ScenarioOutcome(
        name="timeout",
        question="What happens when the model is too slow?",
        expected="TIMEOUT logged, built-in automation keeps the turn",
    )
    campaign, game = synthetic_game()
    client = ScriptedClient(
        [LlmTimeout("no response within 90s")], catalog=CATALOG_PAYLOAD
    )
    return _run_turn(
        outcome, game, make_config(), client, audit_root, False, FallbackReason.TIMEOUT
    )


def scenario_catalogue_unavailable(audit_root: Path) -> ScenarioOutcome:
    """What if live prices cannot be fetched?"""

    outcome = ScenarioOutcome(
        name="catalogue-unavailable",
        question="What if live prices cannot be fetched?",
        expected="the turn proceeds on the pessimistic built-in price rather than "
        "on an unpriced guess",
    )
    campaign, game = synthetic_game()
    brief = _brief_for(game)
    client = ScriptedClient(
        [example_decision_json(brief)],
        catalog_error=LlmTransportError("model catalogue unreachable"),
    )
    result = _run_turn(outcome, game, make_config(), client, audit_root, True, None)
    result.notes.append(
        "priced at the built-in fallback rate "
        f"(${FALLBACK_INPUT_PRICE_PER_MILLION:.2f} in / "
        f"${FALLBACK_OUTPUT_PRICE_PER_MILLION:.2f} out per million tokens)"
    )
    return result


def scenario_cost_cap(audit_root: Path) -> ScenarioOutcome:
    """Does a zero budget stop the call before any money is spent?"""

    outcome = ScenarioOutcome(
        name="cost-cap",
        question="Does a zero budget stop the call before any money is spent?",
        expected="COST_CAP with zero completions -- refused before the request is "
        "sent, not after",
    )
    campaign, game = synthetic_game()
    brief = _brief_for(game)
    client = ScriptedClient([example_decision_json(brief)], catalog=CATALOG_PAYLOAD)
    return _run_turn(
        outcome,
        game,
        make_config(cost_cap_per_turn=0.0),
        client,
        audit_root,
        False,
        FallbackReason.COST_CAP,
        expect_completions=0,
    )


def scenario_not_configured(audit_root: Path) -> ScenarioOutcome:
    """What happens when the feature is on but has no API key?"""

    outcome = ScenarioOutcome(
        name="not-configured",
        question="What happens when the feature is on but has no API key?",
        expected="NOT_CONFIGURED with zero completions and a stated reason",
    )
    campaign, game = synthetic_game()
    client = ScriptedClient(["never used"], catalog=CATALOG_PAYLOAD)
    return _run_turn(
        outcome,
        game,
        make_config(api_key=None),
        client,
        audit_root,
        False,
        FallbackReason.NOT_CONFIGURED,
        expect_completions=0,
    )


def scenario_disabled(audit_root: Path) -> ScenarioOutcome:
    """Is stock Retribution untouched when the feature is off?"""

    outcome = ScenarioOutcome(
        name="disabled",
        question="Is stock Retribution untouched when the feature is off?",
        expected="DISABLED, zero completions, no decision log written at all",
    )
    campaign, game = synthetic_game()
    client = ScriptedClient(["never used"], catalog=CATALOG_PAYLOAD)
    result = _run_turn(
        outcome,
        game,
        make_config(enabled=False),
        client,
        audit_root,
        False,
        FallbackReason.DISABLED,
        expect_completions=0,
    )
    if result.log_path is not None:
        result.matched_expectation = False
        result.notes.append("MISMATCH: a disabled turn wrote a decision log")
    return result


SCENARIOS: dict[str, Callable[[Path], ScenarioOutcome]] = {
    "valid": scenario_valid,
    "repair": scenario_repair,
    "malformed": scenario_malformed,
    "truncated-json": scenario_truncated_json,
    "cheating-intel": scenario_cheating_intel,
    "cheating-overspend": scenario_cheating_overspend,
    "illegal-posture": scenario_illegal_posture,
    "stale-state": scenario_stale_state,
    "tool-calls": scenario_tool_calls,
    "transport-failure": scenario_transport_failure,
    "timeout": scenario_timeout,
    "catalogue-unavailable": scenario_catalogue_unavailable,
    "cost-cap": scenario_cost_cap,
    "not-configured": scenario_not_configured,
    "disabled": scenario_disabled,
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _rule(title: str) -> None:
    print()
    print(f"== {title} " + "=" * max(0, 74 - len(title)))


def report_campaign(brief: RedCommanderBrief) -> None:
    _rule("SYNTHETIC CAMPAIGN")
    print(f"  theater              {brief.theater}")
    print(f"  turn                 {brief.turn_id}")
    print(f"  RED faction          {brief.red_faction}")
    print(f"  campaign id hash     {brief.campaign_id_hash}")
    print(f"  campaign revision    {brief.campaign_revision}")
    print(f"  intel policy         {brief.intel_policy.value}")
    print(f"  schema version       {brief.schema_version}")
    print(
        "  RED resources        "
        f"budget {brief.red_resources.budget_available:.0f}, "
        f"income {brief.red_resources.income_last_turn:.0f}"
    )
    summary = brief.red_force_summary
    print(
        "  RED forces           "
        f"{summary.control_points_held} bases "
        f"({summary.airbases_operational} operational, "
        f"{summary.airbases_runway_damaged} runway damaged), "
        f"{summary.aircraft_available} aircraft, "
        f"{summary.ground_units_deployed} ground units"
    )
    print(f"  fronts               {len(brief.fronts)}")
    for front in brief.fronts:
        print(
            f"    {front.id:<10} {front.own_base} -> {front.enemy_base}  "
            f"own {front.own_deployable_units}/{front.own_unit_capacity}, "
            f"enemy {front.enemy_strength.value}, "
            f"enemy count {front.enemy_unit_count}"
        )
    print(f"  target sets          {len(brief.known_target_sets)}")
    for target_set in brief.known_target_sets:
        print(
            f"    {target_set.id:<10} {target_set.category.value:<24} "
            f"count {target_set.known_count}, {target_set.confidence.value}, "
            f"{target_set.location_precision.value}"
        )
    print(f"  procurement options  {len(brief.procurement_categories)}")
    for option in brief.procurement_categories:
        print(
            f"    {option.id:<10} {option.category.value:<24} "
            f"eligible {str(option.eligible):<5} {option.affordability.value}"
        )


def report_intel_fairness() -> bool:
    """Scan everything RED is handed for BLUE-private sentinel values."""

    _rule("INTEL FAIRNESS (sentinel scan)")
    print(
        "  The synthetic campaign stamps a unique sentinel on every BLUE-private\n"
        "  value (budget, income, squadron and airframe names, pilot counts,\n"
        "  planned packages, exact unit counts, base capacity, and the units and\n"
        "  base beyond RED's observation range). Everything RED is handed is then\n"
        "  serialised and searched for those sentinels."
    )
    ok = True
    for policy in (IntelPolicy.REALISTIC, IntelPolicy.FULL_PARITY):
        campaign, game = synthetic_game()
        brief = _brief_for(game, policy)
        blob = serialise_everything(
            brief.to_dict(),
            brief.render_compact(),
            decision_json_schema(brief),
            example_decision_json(brief),
            *[
                build_messages(brief, personality)
                for personality in CommanderPersonality
            ],
        )
        leaks = blue_leaks_in(blob)
        missing = red_facts_missing_from(blob)
        print()
        print(f"  policy {policy.value}")
        print(f"    BLUE-private values found   {leaks if leaks else 'none'}")
        print(f"    RED's own facts present     {'all' if not missing else missing}")
        print(f"    fields withheld by policy   {len(brief.withheld_fields)}")
        for withheld in brief.withheld_fields:
            print(f"      - {withheld}")
        if policy is IntelPolicy.REALISTIC:
            if leaks:
                ok = False
                print("    VERDICT: FAIL -- BLUE-private data reached RED")
            elif missing:
                ok = False
                print(
                    "    VERDICT: FAIL -- the filter is empty; RED cannot even see "
                    "its own state"
                )
            else:
                print(
                    "    VERDICT: PASS -- no BLUE-private value reached RED, and "
                    "RED does see all of its own"
                )
        else:
            print(
                "    NOTE: full_parity is the deliberately unfair setting. Exact\n"
                "    enemy unit counts are expected here, which is why the\n"
                "    sentinel above shows up; coordinates, planned flights and "
                "save\n    data are withheld even so."
            )
    return ok


def report_scenarios(
    names: Sequence[str], audit_root: Path, verbose: bool
) -> list[ScenarioOutcome]:
    _rule("SCENARIOS")
    outcomes: list[ScenarioOutcome] = []
    for name in names:
        outcome = SCENARIOS[name](audit_root)
        outcomes.append(outcome)
        status = "OK " if outcome.matched_expectation else "BAD"
        print()
        print(f"  [{status}] {outcome.name}")
        print(f"        asks      {outcome.question}")
        print(f"        expected  {outcome.expected}")
        print(
            f"        result    accepted={outcome.accepted} "
            f"fallback={outcome.fallback_reason or 'none'} "
            f"completions={outcome.completions}"
        )
        if outcome.directive:
            for line in outcome.directive.splitlines():
                print(f"        orders    {line}")
        if outcome.prompt_tokens or outcome.completion_tokens:
            print(
                f"        tokens    {outcome.prompt_tokens} in / "
                f"{outcome.completion_tokens} out, "
                f"cost ${outcome.actual_cost:.6f}"
            )
        if outcome.rejections:
            print(f"        refused   {len(outcome.rejections)} item(s):")
            shown = outcome.rejections if verbose else outcome.rejections[:6]
            for rejection in shown:
                print(
                    f"                  - {rejection['element']}: {rejection['reason']}"
                )
            if len(shown) < len(outcome.rejections):
                print(
                    f"                  ... {len(outcome.rejections) - len(shown)} "
                    "more (use --verbose)"
                )
        for note in outcome.notes:
            print(f"        note      {note}")
        if outcome.log_path:
            print(f"        log       {outcome.log_path}")
    return outcomes


def report_audit_trail(audit_root: Path) -> None:
    """Show that a turn can be reconstructed from the files on disk."""

    _rule("AUDIT TRAIL")
    campaign, game = synthetic_game()
    brief = _brief_for(game)
    plan = example_decision_json(brief)
    client = ScriptedClient(
        [plan],
        catalog=CATALOG_PAYLOAD,
        usage=_measured_usage(_prompt_text(brief, CommanderPersonality.BALANCED), plan),
    )
    result = RedCommanderTurn(
        game, make_config(), audit_log=AuditLog(audit_root / "audit"), client=client
    ).run()
    if result.log_path is None:
        print("  no decision log was written")
        return
    payload = json.loads(result.log_path.read_text(encoding="utf-8"))
    print(f"  file  {result.log_path}")
    print("  fields a reviewer can check, straight out of that file:")
    for key in (
        "record_schema_version",
        "written_at",
        "campaign_id_hash",
        "turn_id",
        "campaign_revision",
        "intel_policy",
        "intel_hash",
        "personality",
        "accepted",
        "fallback_reason",
        "fallback_policy",
        "cost_cap_per_turn",
        "prior_cost_this_turn",
        "estimated_cost",
        "actual_cost",
        "total_prompt_tokens",
        "total_completion_tokens",
        "prompt_logging_enabled",
        "planner_task_order",
    ):
        if key in payload:
            value = payload[key]
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value) or "(empty)"
            print(f"    {key:<26} {value}")
    attempt = payload["attempts"][0] if payload.get("attempts") else {}
    print("  first LLM attempt:")
    for key in (
        "kind",
        "requested_model",
        "actual_model",
        "finish_reason",
        "prompt_hash",
        "response_hash",
        "prompt_tokens",
        "completion_tokens",
        "reserved_cost",
        "actual_cost",
        "cost_is_estimated",
        "retries",
    ):
        if key in attempt:
            print(f"    {key:<26} {attempt[key]}")
    print(
        "  the prompt body itself is in attempts[].prompt_messages when prompt\n"
        "  logging is on, and the raw reply in attempts[].response_text"
    )


def report_costs(brief: RedCommanderBrief) -> bool:
    _rule("TOKEN MEASUREMENT")
    prompts = {
        personality.value: _prompt_text(brief, personality)
        for personality in CommanderPersonality
    }
    response = example_decision_json(brief)
    repair_prompt = "\n".join(
        m["content"]
        for m in build_repair_messages(
            brief,
            CommanderPersonality.BALANCED,
            "not JSON",
            "front_priorities[0]: identifier is not in the brief",
        )
    )

    print(
        "  Method: there is no provider tokeniser in this repo and no new\n"
        "  dependency may be added, so token counts come from\n"
        "  pricing.estimate_tokens() -- ceil(characters / 4.0 * 1.15), the same\n"
        "  estimator the controller uses to size its cost reservation before a\n"
        "  request. The 1.15 factor deliberately over-counts so the reservation\n"
        "  errs high. Character counts below are exact measurements of the real\n"
        "  assembled prompt; token counts are that estimator applied to them.\n"
        "  A provider that reports real usage overrides the estimate for billing."
    )
    print()
    print(f"  {'text':<34}{'characters':>12}{'est. tokens':>14}")
    rows: list[tuple[str, str]] = [
        (f"initial prompt ({name})", text) for name, text in prompts.items()
    ]
    rows.append(("repair prompt (balanced)", repair_prompt))
    rows.append(("representative response", response))
    for label, text in rows:
        print(f"  {label:<34}{len(text):>12}{estimate_tokens(text):>14}")

    biggest_prompt = max(prompts.values(), key=len)
    prompt_tokens = estimate_tokens(biggest_prompt)
    repair_tokens = estimate_tokens(repair_prompt)
    response_tokens = estimate_tokens(response)
    max_output = 2000

    print()
    print(
        "  How the briefing grows with campaign size (measured, not assumed: the\n"
        "  synthetic theater is widened to N contested fronts and the prompt is\n"
        "  reassembled and re-measured each time):"
    )
    print()
    print(
        f"  {'fronts':>8}{'bases':>8}{'target sets':>13}"
        f"{'characters':>12}{'est. tokens':>14}"
    )
    for pairs in (1, 2, 4, 8, 16, 32):
        scaled, scaled_game = scaled_campaign(pairs)
        scaled_brief = _brief_for(scaled_game)
        text = _prompt_text(scaled_brief, CommanderPersonality.BALANCED)
        print(
            f"  {len(scaled_brief.fronts):>8}"
            f"{len(scaled.control_points):>8}"
            f"{len(scaled_brief.known_target_sets):>13}"
            f"{len(text):>12}{estimate_tokens(text):>14}"
        )
    print()
    print(
        "  The briefing is a fixed-shape summary -- one line per front, one per\n"
        "  known target class, one per procurement option -- so it scales with the\n"
        "  number of fronts, not with the number of units, aircraft or objects in\n"
        "  the campaign. A 32-front theater is far larger than any stock campaign."
    )

    _rule("COST PER RED TURN")
    print(
        f"  Call-count assumption: the controller makes at most\n"
        f"  {MAX_COMPLETIONS_PER_TURN} chat completions for one RED turn -- the\n"
        "  initial request, plus exactly one repair request if and only if the\n"
        "  first response fails validation. There is no retry loop beyond that.\n"
        "  It also makes one unbilled GET /models catalogue lookup per turn.\n"
        "  Provider-side transport retries (HTTP 429/5xx) re-send the same\n"
        "  request and are billed by the provider only if a response is served.\n"
        "  If a future build adds a second decision point per turn, double the\n"
        "  figures below."
    )
    print()
    print(f"  typical turn    = 1 call:  {prompt_tokens} in + {response_tokens} out")
    print(
        f"  worst-case turn = 2 calls: {prompt_tokens + repair_tokens} in + "
        f"{2 * max_output} out (both replies at the {max_output}-token output cap)"
    )
    print()
    header = (
        f"  {'model':<34}{'$/M in':>8}{'$/M out':>9}"
        f"{'typical':>11}{'worst':>10}  note"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    worst_costs: list[tuple[str, float]] = []
    for model_id, input_price, output_price, note in CANDIDATE_MODELS:
        price = ModelPrice(
            model_id=model_id,
            input_per_million=input_price,
            output_per_million=output_price,
        )
        typical = price.cost_for(prompt_tokens, response_tokens)
        worst = price.cost_for(prompt_tokens + repair_tokens, 2 * max_output)
        worst_costs.append((model_id, worst))
        print(
            f"  {model_id:<34}{input_price:>8.2f}{output_price:>9.2f}"
            f"{typical:>11.5f}{worst:>10.5f}  {note}"
        )

    print()
    print(f"  Ceiling asked for: ${COST_CEILING_PER_TURN:.2f} per RED turn.")
    over = [
        model_id for model_id, worst in worst_costs if worst > COST_CEILING_PER_TURN
    ]
    for model_id, worst in worst_costs:
        headroom = COST_CEILING_PER_TURN / worst if worst > 0 else float("inf")
        verdict = "OVER" if worst > COST_CEILING_PER_TURN else "under"
        print(
            f"    {model_id:<34}worst ${worst:.5f}  {verdict} the cap "
            f"({headroom:.0f}x headroom)"
        )
    if over:
        print(
            "  VERDICT: every surveyed model stays under the ceiling except\n"
            f"  {', '.join(over)}. The shipped default\n"
            f"  ({CANDIDATE_MODELS[0][0]}) is far under it."
        )
    else:
        print(
            "  VERDICT: every surveyed model's worst-case turn stays under the\n"
            f"  ceiling, including the pessimistic fallback price. The shipped\n"
            f"  default is {CANDIDATE_MODELS[0][0]}."
        )
    print(
        "  These are estimates from measured character counts and published\n"
        "  prices, not observed invoices. Prices were recorded on 2026-08-05 and\n"
        "  change without notice; the controller always prices from the live\n"
        "  /models catalogue at runtime."
    )

    print()
    print("  Cost cap enforcement, exercised against the real ledger:")
    default_price = ModelPrice(
        model_id=CANDIDATE_MODELS[0][0],
        input_per_million=CANDIDATE_MODELS[0][1],
        output_per_million=CANDIDATE_MODELS[0][2],
    )
    for cap in (COST_CEILING_PER_TURN, 0.001, 0.0):
        ledger = CostLedger(cap=cap)
        worst = ledger.worst_case_cost(default_price, prompt_tokens, max_output)
        affordable = ledger.can_afford(default_price, prompt_tokens, max_output)
        print(
            f"    cap ${cap:<8.3f} worst case ${worst:.6f} -> "
            f"{'allowed' if affordable else 'refused before sending'}"
        )
    return True


def report_live_call(model: Optional[str], strict: bool) -> bool:
    _rule("LIVE PROVIDER CALL")
    store = SecretStore()
    key = store.load()
    if not key:
        print(
            f"  No key found, so no network call was made. Set {ENV_VAR} (or store\n"
            "  a key from the settings window) to have this section make exactly\n"
            "  one real request. Everything above ran without it."
        )
        return True

    print(f"  key       {mask(key)} from {store.source}")
    campaign, game = synthetic_game()
    brief = _brief_for(game)
    requested = model or DEFAULT_MODEL
    client = ChatCompletionClient(
        api_key=key,
        model=requested,
        base_url="https://openrouter.ai/api/v1",
    )
    print(f"  endpoint  {client.describe()}")

    try:
        catalog = ModelCatalog.from_payload(client.fetch_model_catalog(), "openrouter")
    except LlmError as error:
        print(f"  catalogue unavailable: {error}")
        catalog = ModelCatalog.unavailable(str(error))
    price = catalog.price_for(requested) or ModelPrice.fallback_for(requested)
    print(
        f"  price     ${price.input_per_million:.4f}/M in, "
        f"${price.output_per_million:.4f}/M out"
        + ("  (fallback estimate)" if price.is_fallback_estimate else "  (live)")
    )

    messages = build_messages(brief, CommanderPersonality.BALANCED)
    try:
        response = client.complete(messages, max_output_tokens=2000)
    except LlmError as error:
        print(f"  the call failed: {type(error).__name__}: {error}")
        print("  (a live failure does not change the exit code unless --strict-live)")
        return not strict

    usage = response.usage
    reported = usage.cost
    computed = price.cost_for(usage.input_tokens, usage.output_tokens)
    print(
        f"  usage     {usage.input_tokens} in / {usage.output_tokens} out "
        f"({usage.total_tokens} total) in {response.latency_seconds:.1f}s"
    )
    print(
        f"  cost      ${computed:.6f} computed"
        + (f", ${reported:.6f} reported by the provider" if reported else "")
    )
    print(
        f"  estimator said {estimate_tokens(_prompt_text(brief, CommanderPersonality.BALANCED))} prompt tokens; provider said {usage.input_tokens}"
    )

    outcome = parse_decision(response.text, brief)
    if not outcome.ok:
        print(f"  the reply was refused: {outcome.error_summary()}")
        return not strict
    decision = outcome.decision
    assert decision is not None
    directive, rejections = LegalityChecker(game, brief).check(decision)
    print(f"  parsed    {len(outcome.rejections)} schema refusal(s)")
    print(f"  legality  {len(rejections)} refusal(s)")
    if directive is None:
        print("  nothing in the reply was legal; the built-in automation would run")
    else:
        for line in directive.render_summary().splitlines():
            print(f"  orders    {line}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai_commander_dryrun",
        description="Run RED LLM commander turns offline against a synthetic "
        "campaign, then report fairness, refusals, fallbacks and cost.",
    )
    parser.add_argument(
        "--list", action="store_true", help="list the scenarios and exit"
    )
    parser.add_argument(
        "--scenario",
        action="append",
        metavar="NAME",
        help="run only this scenario (repeatable); default is all of them",
    )
    parser.add_argument(
        "--audit-dir",
        metavar="PATH",
        help="write decision logs here instead of a temporary directory",
    )
    parser.add_argument(
        "--model",
        metavar="ID",
        help=f"model for the optional live call (default {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help=f"never call the provider even if {ENV_VAR} is set",
    )
    parser.add_argument(
        "--strict-live",
        action="store_true",
        help="a failed live call also fails the exit code",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print every refusal, not the first few"
    )
    parser.add_argument(
        "--json", metavar="PATH", help="also write the machine-readable results here"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list:
        print("scenarios:")
        for name, function in SCENARIOS.items():
            doc = (function.__doc__ or "").strip().splitlines()
            print(f"  {name:<24}{doc[0] if doc else ''}")
        return 0

    names = args.scenario or list(SCENARIOS)
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        print(f"unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"known: {', '.join(SCENARIOS)}", file=sys.stderr)
        return 2

    print("DCS Retribution -- RED LLM commander dry run")
    print(f"  {'python':<21} {sys.version.split()[0]}")
    print(f"  {'repository':<21} {_REPO_ROOT}")
    print(
        f"  {ENV_VAR:<21} "
        + ("set (value never printed)" if os.environ.get(ENV_VAR) else "not set")
    )
    print(f"  {'DCS':<21} not required; nothing here launches or reads DCS")

    temporary: Optional[tempfile.TemporaryDirectory[str]] = None
    if args.audit_dir:
        audit_root = Path(args.audit_dir).expanduser().resolve()
        audit_root.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="ai-commander-dryrun-")
        audit_root = Path(temporary.name)
    print(
        f"  {'decision logs':<21} {audit_root}" + ("  (temporary)" if temporary else "")
    )

    install_fake_objective_finder()

    try:
        campaign, game = synthetic_game()
        brief = _brief_for(game)
        report_campaign(brief)
        fairness_ok = report_intel_fairness()
        outcomes = report_scenarios(names, audit_root, args.verbose)
        report_audit_trail(audit_root)
        costs_ok = report_costs(brief)
        live_ok = (
            True
            if args.no_live
            else report_live_call(args.model, strict=args.strict_live)
        )

        _rule("SUMMARY")
        failed = [o.name for o in outcomes if not o.matched_expectation]
        print(f"  scenarios run          {len(outcomes)}")
        print(f"  behaved as expected    {len(outcomes) - len(failed)}")
        print(
            f"  deviated               {len(failed)}"
            + (f": {failed}" if failed else "")
        )
        print(f"  intel fairness         {'PASS' if fairness_ok else 'FAIL'}")
        print(f"  cost report            {'produced' if costs_ok else 'failed'}")
        print(
            "  not covered here       anything that needs DCS itself: mission\n"
            "                         generation, the Qt windows, save/load of a\n"
            "                         real campaign, and real provider invoices"
        )

        if args.json:
            payload = {
                "scenarios": [o.to_dict() for o in outcomes],
                "intel_fairness_ok": fairness_ok,
                "sentinels": {"blue": BLUE_SENTINELS, "red": RED_SENTINELS},
            }
            Path(args.json).write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            print(f"  machine-readable results written to {args.json}")

        ok = not failed and fairness_ok and costs_ok and live_ok
        print()
        print("RESULT: " + ("all offline checks passed" if ok else "CHECKS FAILED"))
        return 0 if ok else 1
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    sys.exit(main())
