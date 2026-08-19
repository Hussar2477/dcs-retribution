"""Orchestration of one RED commander turn.

This is the only module that touches everything at once, and it exists so the
call site in :class:`~game.coalition.Coalition` stays a single line. The order of
operations is deliberate and is what makes the feature auditable and fair:

1. Resolve configuration. Disabled means "do nothing at all", not "do nothing
   quietly" -- a misconfiguration is recorded once so the player can see why the
   AI did not run.
2. Project the intel brief. This is the *only* view of the campaign the model is
   ever shown, and it is produced before any network activity so the audit log
   records what was known independently of whether the call succeeded.
3. Check the decision log for this exact ``(turn, campaign revision)``.
   Retribution re-runs turn initialisation several times per turn, and paying
   for a second completion each time would be both expensive and inconsistent.
4. Reserve worst-case spend *before* the request, never after. A cap that is
   only checked afterwards is not a cap.
5. Validate, then re-validate against live state. Schema validity and legality
   are separate gates and both are logged.
6. Write one record per decision point, whatever the outcome.

Every failure path ends at Retribution's built-in RED automation. There is no
path in which a model failure blocks or corrupts a turn.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

from game.ai_commander.activeprompt import (
    build_stage_messages,
    build_stage_repair_messages,
    stage_response_format,
)
from game.ai_commander.audit import (
    AiDecisionRecord,
    AuditLog,
    LlmAttempt,
    StageRecord,
    prompt_digest,
)
from game.ai_commander.capabilities import CapabilityIndex, capability_index_for
from game.ai_commander.config import AiCommanderConfig
from game.ai_commander.decision import (
    DecisionValidationError,
    Rejection,
    extract_json_object,
    decision_schema_hash,
    parse_decision,
)
from game.ai_commander.directive import CommanderDirective
from game.ai_commander.enums import CommanderMode, FallbackReason
from game.ai_commander.execution import task_order_for
from game.ai_commander.intel import IntelProjector, RedCommanderBrief
from game.ai_commander.legality import LegalityChecker
from game.ai_commander.operations import (
    OperationsBrief,
    OperationsProjector,
    OperationsResolver,
)
from game.ai_commander.plan import (
    AirTaskingPlan,
    CommanderStage,
    LogisticsPlan,
    validate_air_tasking_plan,
    validate_logistics_plan,
)
from game.ai_commander.planexecution import ExecutionReport, PlanExecutor
from game.ai_commander.planlegality import (
    ExecutableAirTasking,
    ExecutableLogistics,
    PlanLegalityChecker,
)
from game.ai_commander.llmclient import (
    ChatCompletionClient,
    LlmError,
    LlmHttpError,
    LlmResponse,
    LlmTimeout,
)
from game.ai_commander.pricing import (
    CostCapExceeded,
    CostLedger,
    ModelCatalog,
    ModelPrice,
    estimate_tokens,
)
from game.ai_commander.prompt import (
    build_messages,
    build_repair_messages,
    response_format_for,
)
from game.ai_commander.serialization import stable_hash

if TYPE_CHECKING:
    from game.coalition import Coalition
    from game.game import Game


#: Human-readable descriptions of what happens after a fallback, recorded in the
#: log so an audit does not have to infer it from the settings of the day.
FALLBACK_TO_BUILTIN = "retribution built-in RED automation"
FALLBACK_TO_PREVIOUS = "previous accepted directive, re-checked against live state"

#: Sentinel distinguishing "compute the default response format" from an explicit
#: ``None``, which legitimately means "this provider supports neither mode".
_UNSET: Any = object()

#: What the built-in automation covers when an ACTIVE stage degrades. Recorded as
#: a stage note so the log says which parts of the turn RED planned itself.
_STAGE_FALLBACK_TEXT = {
    "logistics": (
        "logistics stage degraded; Retribution's procurement AI spends the "
        "remaining budget for this turn"
    ),
    "air_tasking": (
        "air tasking stage degraded; Retribution's mission planner builds RED's "
        "packages for this turn"
    ),
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class _StageOutcome:
    """Uniform view of one validated response, whatever stage produced it.

    Phase 1's :class:`~game.ai_commander.decision.ValidationOutcome` is
    specifically about a strategic decision. ACTIVE mode's later stages produce
    logistics and air tasking plans instead, so the request/accounting machinery
    is given this common wrapper and stays identical for all three.
    """

    value: Optional[Any] = None
    rejections: list[Rejection] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def rejection_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.rejections]

    def error_summary(self, limit: int = 12) -> str:
        if not self.rejections:
            return "no specific errors recorded"
        return "; ".join(str(r) for r in self.rejections[:limit])


@dataclass
class CommanderTurnResult:
    """Everything one decision point produced.

    Returned by :class:`RedCommanderTurn` so the dry-run harness and the tests
    can inspect the whole outcome; the game itself only needs
    :attr:`directive`.
    """

    directive: Optional[CommanderDirective] = None
    record: Optional[AiDecisionRecord] = None
    brief: Optional[RedCommanderBrief] = None
    log_path: Optional[Path] = None
    fallback_reason: Optional[FallbackReason] = None
    rejections: list[Rejection] = field(default_factory=list)
    #: ACTIVE mode only: the operations brief the later stages were planned from.
    operations: Optional[OperationsBrief] = None
    #: ACTIVE mode only: what the executor actually managed to apply.
    execution: Optional[ExecutionReport] = None

    @property
    def accepted(self) -> bool:
        return self.directive is not None


class RedCommanderTurn:
    """Runs the commander for one ``initialize_turn`` call.

    Dependencies are injected rather than constructed internally so tests and
    the dry-run harness can substitute a fake transport and a temporary log
    directory without monkey-patching anything.
    """

    def __init__(
        self,
        game: Game,
        config: AiCommanderConfig,
        audit_log: Optional[AuditLog] = None,
        client: Optional[ChatCompletionClient] = None,
    ) -> None:
        self.game = game
        self.config = config
        self.audit_log = audit_log
        self._client = client
        self.record = AiDecisionRecord()
        self.brief: Optional[RedCommanderBrief] = None
        self.ledger: Optional[CostLedger] = None
        self.catalog: Optional[ModelCatalog] = None
        self.price: Optional[ModelPrice] = None
        self._last_fallback_reason = FallbackReason.UNEXPECTED_ERROR
        # ACTIVE mode only. Left unset in COMMANDER mode so that mode continues
        # to touch nothing but the strategic brief.
        self.operations: Optional[OperationsBrief] = None
        self.capabilities: Optional[CapabilityIndex] = None
        self.resolver: OperationsResolver = OperationsResolver()

    # -- entry point ------------------------------------------------------

    def run(self) -> CommanderTurnResult:
        """Never raises. Any unexpected error becomes a logged fallback."""

        try:
            return self._run()
        except Exception:  # pragma: no cover - defensive belt and braces
            logging.exception("RED commander turn failed; using built-in automation")
            self.record.notes.append("unexpected error; see the Retribution log")
            return self._fail(FallbackReason.UNEXPECTED_ERROR)

    def _run(self) -> CommanderTurnResult:
        config = self.config
        if not config.enabled:
            return CommanderTurnResult(fallback_reason=FallbackReason.DISABLED)

        # 1. Brief first: it is cheap, read-only, and needed for the idempotency
        #    key even on paths that never make a request.
        self.brief = self._project_brief()
        brief = self.brief
        self._start_record(brief)

        # 2. Idempotency. initialize_turn can legitimately run several times.
        replayed = self._replay_if_already_decided(brief)
        if replayed is not None:
            return replayed

        if not config.is_usable:
            for problem in config.problems:
                self.record.rejections.append(
                    Rejection("<configuration>", problem).to_dict()
                )
            if config.requires_api_key and not config.api_key:
                self.record.rejections.append(
                    Rejection(
                        "<configuration>",
                        "no API key is configured for a remote provider",
                    ).to_dict()
                )
            return self._fail(FallbackReason.NOT_CONFIGURED)

        if not config.allows_paid_requests:
            self.record.rejections.append(
                Rejection(
                    "<cost_cap>",
                    "the per-turn cost cap is set to $0.00, so no billable "
                    "request may be made",
                ).to_dict()
            )
            return self._fail(FallbackReason.COST_CAP)

        client = self._build_client()
        self._load_catalog(client)
        price = self.price
        assert price is not None  # set by _load_catalog

        already = 0.0
        if self.audit_log is not None:
            already = self.audit_log.spent_this_turn(
                brief.campaign_id_hash, brief.turn_id
            )
        self.record.prior_cost_this_turn = already
        self.ledger = CostLedger(config.cost_cap_per_turn, already_spent=already)

        if config.mode is CommanderMode.ACTIVE:
            return self._run_active(client, price)
        return self._run_commander(client, price)

    def _run_commander(
        self, client: ChatCompletionClient, price: ModelPrice
    ) -> CommanderTurnResult:
        """COMMANDER mode: one request, then Retribution's own staff execute."""

        assert self.brief is not None
        brief = self.brief
        config = self.config

        # 3. First attempt.
        messages = build_messages(brief, config.personality)
        outcome, response = self._attempt(client, messages, price, kind="initial")
        if outcome is None:
            return self._fail(self._last_fallback_reason)

        # 4. At most one repair, with the validation errors and no new state.
        if not outcome.ok and response is not None:
            repair = build_repair_messages(
                brief,
                config.personality,
                response.text,
                outcome.error_summary(),
            )
            self.record.notes.append(
                "first response failed validation; one repair attempt made"
            )
            repaired, repair_response = self._attempt(
                client, repair, price, kind="repair"
            )
            if repaired is not None:
                outcome = repaired
                response = repair_response or response

        if outcome is None or not outcome.ok or outcome.value is None:
            self.record.rejections.extend(
                outcome.rejection_dicts if outcome is not None else []
            )
            return self._fail(FallbackReason.MALFORMED_RESPONSE)

        decision = outcome.value
        self.record.parsed_decision = decision.to_dict()
        self.record.rejections.extend(outcome.rejection_dicts)

        # 5. Legality against live state.
        checker = LegalityChecker(self.game, brief)
        directive, rejections = checker.check(decision)
        self.record.rejections.extend(r.to_dict() for r in rejections)
        if directive is None:
            stale = any(r.element == "campaign_revision" for r in rejections)
            return self._fail(
                FallbackReason.STALE_RESPONSE
                if stale
                else FallbackReason.NO_LEGAL_CONTENT
            )

        return self._accept(directive)

    # -- ACTIVE mode ------------------------------------------------------

    @property
    def coalition(self) -> Coalition:
        """RED's coalition, resolved lazily so COMMANDER mode never needs it."""

        from game.theater.player import Player

        return self.game.coalition_for(Player.RED)

    def _run_active(
        self, client: ChatCompletionClient, price: ModelPrice
    ) -> CommanderTurnResult:
        """ACTIVE mode: three requests, then RED's own orders are applied.

        The stages degrade independently and in one direction only. Stage 1 is
        the only one whose failure costs the whole turn, because without a
        directive there is nothing for the built-in planner to be steered by and
        stock automation is exactly the right answer. Once a directive exists the
        turn is committed: a failed logistics stage leaves the money to
        Retribution's procurement AI, and a failed air tasking stage leaves the
        packages to Retribution's mission planner. Neither can undo work already
        applied, and neither is allowed to abort the turn.
        """

        assert self.brief is not None
        brief = self.brief
        config = self.config

        try:
            projector = OperationsProjector(self.game, config.intel_policy)
            operations = projector.project(
                brief.campaign_id_hash, brief.campaign_revision
            )
            capabilities = capability_index_for(self.coalition)
        except Exception:
            # A theater this projector cannot read is a bug, but it must not cost
            # the player a turn: COMMANDER mode needs neither brief.
            logging.warning(
                "Could not project the RED operations brief; running the "
                "commander in COMMANDER mode for this turn",
                exc_info=True,
            )
            self.record.notes.append(
                "operations projection failed; this turn ran in COMMANDER mode"
            )
            return self._run_commander(client, price)

        self.operations = operations
        self.capabilities = capabilities
        self.resolver = projector.resolver
        self.record.set_operations(
            operations, capabilities.content_hash(), include_rendered=config.log_prompts
        )

        # -- stage 1: command intent --------------------------------------
        directive = self._active_command_stage(client, price)
        if directive is None:
            return self._fail(self._last_fallback_reason)

        decisions: list[str] = [directive.render_summary()]
        executor = PlanExecutor(self.game, self.coalition)

        # -- stage 2: logistics -------------------------------------------
        logistics = self._active_logistics_stage(client, price, "\n".join(decisions))
        if logistics is not None:
            executor.execute_logistics(logistics)
            decisions.append("")
            decisions.append("Logistics orders applied this turn:")
            decisions.extend(f"  {line}" for line in logistics.describe())
            decisions.append(f"  {executor.report.render_summary()}")
        else:
            decisions.append("")
            decisions.append(
                "No logistics orders were applied; the automated staff is "
                "handling procurement this turn."
            )

        # -- stage 3: air tasking -----------------------------------------
        tasking = self._active_air_tasking_stage(client, price, "\n".join(decisions))
        if tasking is not None:
            executor.execute_air_tasking(tasking)

        self.record.execution_report = executor.report.to_dict()
        return self._accept(directive, execution=executor.report)

    def _stage_degraded(
        self, stage: StageRecord, reason: FallbackReason, note: str = ""
    ) -> None:
        """Mark one ACTIVE stage as having produced nothing usable."""

        stage.accepted = False
        stage.fallback_reason = reason.value
        text = note or _STAGE_FALLBACK_TEXT.get(stage.stage, "")
        if text:
            stage.notes.append(text)
        self.record.notes.append(f"{stage.stage} stage: {reason.value}")

    def _run_stage(
        self,
        stage: CommanderStage,
        client: ChatCompletionClient,
        price: ModelPrice,
        parser: Callable[[str], _StageOutcome],
        prior_summary: Optional[str] = None,
    ) -> tuple[StageRecord, Optional[_StageOutcome]]:
        """Issue one stage's request plus at most one repair.

        Returns the stage's audit entry and the validated outcome, which is
        ``None`` when no usable response was obtained. The caller decides what
        that means: only stage 1 treats it as losing the turn.
        """

        assert self.brief is not None
        assert self.operations is not None
        assert self.capabilities is not None
        brief = self.brief
        operations = self.operations
        capabilities = self.capabilities
        config = self.config

        entry = self.record.stage_record(stage)
        entry.schema_version = stage.schema_version

        messages = build_stage_messages(
            stage,
            brief,
            operations,
            capabilities,
            config.personality,
            prior_summary,
        )
        response_format = stage_response_format(
            stage,
            brief,
            operations,
            capabilities,
            price.supports_json_schema,
            price.supports_response_format or not self.catalog_available,
        )
        outcome, response = self._attempt(
            client,
            messages,
            price,
            kind=f"{stage.value}:initial",
            parser=parser,
            response_format=response_format,
            stage=entry,
        )
        if outcome is None:
            return entry, None

        if not outcome.ok and response is not None:
            repair = build_stage_repair_messages(
                stage,
                brief,
                operations,
                capabilities,
                config.personality,
                response.text,
                outcome.error_summary(),
                prior_summary,
            )
            entry.notes.append("first response failed validation; one repair made")
            repaired, _ = self._attempt(
                client,
                repair,
                price,
                kind=f"{stage.value}:repair",
                parser=parser,
                response_format=response_format,
                stage=entry,
            )
            if repaired is not None:
                outcome = repaired

        entry.rejections.extend(outcome.rejection_dicts)
        self.record.rejections.extend(outcome.rejection_dicts)
        if not outcome.ok:
            return entry, None
        return entry, outcome

    def _active_command_stage(
        self, client: ChatCompletionClient, price: ModelPrice
    ) -> Optional[CommanderDirective]:
        """Stage 1. Identical action space and validation to COMMANDER mode."""

        assert self.brief is not None
        entry, outcome = self._run_stage(
            CommanderStage.COMMAND, client, price, self._parse_decision_outcome
        )
        if outcome is None or outcome.value is None:
            reason = self._stage_reason(entry)
            self._stage_degraded(
                entry,
                reason,
                "command stage degraded; the whole turn falls back to "
                "Retribution's built-in RED automation",
            )
            self._last_fallback_reason = reason
            return None

        decision = outcome.value
        entry.parsed_plan = decision.to_dict()
        self.record.parsed_decision = entry.parsed_plan

        directive, rejections = LegalityChecker(self.game, self.brief).check(decision)
        payloads = [r.to_dict() for r in rejections]
        entry.rejections.extend(payloads)
        self.record.rejections.extend(payloads)
        if directive is None:
            stale = any(r.element == "campaign_revision" for r in rejections)
            reason = (
                FallbackReason.STALE_RESPONSE
                if stale
                else FallbackReason.NO_LEGAL_CONTENT
            )
            self._stage_degraded(
                entry,
                reason,
                "command stage produced nothing legal; the whole turn falls back "
                "to Retribution's built-in RED automation",
            )
            self._last_fallback_reason = reason
            return None

        entry.accepted = True
        entry.accepted_plan = directive.to_dict()
        return directive

    def _active_logistics_stage(
        self,
        client: ChatCompletionClient,
        price: ModelPrice,
        prior_summary: str,
    ) -> Optional[ExecutableLogistics]:
        """Stage 2. Money and force structure, checked against live state."""

        assert self.operations is not None
        assert self.capabilities is not None
        operations = self.operations
        capabilities = self.capabilities

        def parse(text: str) -> _StageOutcome:
            return self._parse_plan_outcome(
                text,
                lambda payload: validate_logistics_plan(
                    payload, operations, capabilities
                ),
            )

        entry, outcome = self._run_stage(
            CommanderStage.LOGISTICS, client, price, parse, prior_summary
        )
        if outcome is None or outcome.value is None:
            self._stage_degraded(entry, self._stage_reason(entry))
            return None

        plan = outcome.value
        assert isinstance(plan, LogisticsPlan)
        entry.parsed_plan = plan.to_dict()
        if not plan.has_content:
            entry.accepted = True
            entry.notes.append("the commander ordered no logistics changes this turn")
            return None

        executable, rejections = self._plan_checker().check_logistics(plan)
        payloads = [r.to_dict() for r in rejections]
        entry.rejections.extend(payloads)
        self.record.rejections.extend(payloads)
        if executable is None:
            stale = any(r.element == "campaign_revision" for r in rejections)
            self._stage_degraded(
                entry,
                (
                    FallbackReason.STALE_RESPONSE
                    if stale
                    else FallbackReason.NO_LEGAL_CONTENT
                ),
            )
            return None

        entry.accepted = True
        entry.accepted_plan = {
            "intent": executable.intent,
            "orders": executable.describe(),
            "committed_budget": executable.committed_budget,
        }
        return executable

    def _active_air_tasking_stage(
        self,
        client: ChatCompletionClient,
        price: ModelPrice,
        prior_summary: str,
    ) -> Optional[ExecutableAirTasking]:
        """Stage 3. The air tasking order, one package at a time."""

        assert self.operations is not None
        assert self.capabilities is not None
        operations = self.operations
        capabilities = self.capabilities

        def parse(text: str) -> _StageOutcome:
            return self._parse_plan_outcome(
                text,
                lambda payload: validate_air_tasking_plan(
                    payload, operations, capabilities
                ),
            )

        entry, outcome = self._run_stage(
            CommanderStage.AIR_TASKING, client, price, parse, prior_summary
        )
        if outcome is None or outcome.value is None:
            self._stage_degraded(entry, self._stage_reason(entry))
            return None

        plan = outcome.value
        assert isinstance(plan, AirTaskingPlan)
        entry.parsed_plan = plan.to_dict()
        if not plan.has_content:
            entry.accepted = True
            entry.notes.append("the commander ordered no packages of its own")
            return None

        executable, rejections = self._plan_checker().check_air_tasking(plan)
        payloads = [r.to_dict() for r in rejections]
        entry.rejections.extend(payloads)
        self.record.rejections.extend(payloads)
        if executable is None:
            stale = any(r.element == "campaign_revision" for r in rejections)
            self._stage_degraded(
                entry,
                (
                    FallbackReason.STALE_RESPONSE
                    if stale
                    else FallbackReason.NO_LEGAL_CONTENT
                ),
            )
            return None

        entry.accepted = True
        entry.accepted_plan = {
            "intent": executable.intent,
            "packages": executable.describe(),
        }
        return executable

    def _plan_checker(self) -> PlanLegalityChecker:
        assert self.operations is not None
        assert self.capabilities is not None
        return PlanLegalityChecker(
            self.game, self.operations, self.resolver, self.capabilities
        )

    def _attempts_of(self, entry: StageRecord) -> list[LlmAttempt]:
        return [
            self.record.attempts[index]
            for index in entry.attempt_indices
            if 0 <= index < len(self.record.attempts)
        ]

    def _stage_reason(self, entry: StageRecord) -> FallbackReason:
        """Why a stage produced nothing usable.

        A stage that never issued a request lost to the cost cap or a transport
        problem, which :attr:`_last_fallback_reason` already records. A stage that
        did get answers but failed validation twice is a malformed response, and
        must not be misreported as a transport failure.
        """

        if entry.fallback_reason:
            return FallbackReason(entry.fallback_reason)
        attempts = self._attempts_of(entry)
        if not attempts:
            return self._last_fallback_reason
        if all(attempt.error for attempt in attempts):
            return self._last_fallback_reason
        return FallbackReason.MALFORMED_RESPONSE

    # -- steps ------------------------------------------------------------

    def _project_brief(self) -> RedCommanderBrief:
        projector = IntelProjector(self.game, self.config.intel_policy)
        prior = None
        if self.audit_log is not None:
            prior = self.audit_log.latest_summary(
                projector.campaign_id_hash(), int(self.game.turn)
            )
        return projector.project(prior_decision=prior)

    def _start_record(self, brief: RedCommanderBrief) -> None:
        record = self.record
        record.set_brief(brief, include_rendered=self.config.log_prompts)
        record.personality = self.config.personality.value
        record.mode = self.config.mode.value
        record.base_url = self.config.base_url
        record.configured_model = self.config.model
        record.decision_schema_hash = decision_schema_hash(brief)
        record.cost_cap_per_turn = self.config.cost_cap_per_turn
        record.prompt_logging_enabled = self.config.log_prompts
        record.fallback_policy = (
            FALLBACK_TO_BUILTIN
            if self.config.fallback_to_builtin
            else FALLBACK_TO_PREVIOUS
        )

    def _replay_if_already_decided(
        self, brief: RedCommanderBrief
    ) -> Optional[CommanderTurnResult]:
        log = self.audit_log
        if log is None:
            return None
        existing = log.accepted_directive_for(
            brief.campaign_id_hash, brief.turn_id, brief.campaign_revision
        )
        if existing is not None:
            logging.info(
                "Reusing the RED commander directive already recorded for turn %d",
                brief.turn_id,
            )
            return CommanderTurnResult(
                directive=existing, brief=brief, record=None, log_path=None
            )
        if log.has_record_for_revision(
            brief.campaign_id_hash, brief.turn_id, brief.campaign_revision
        ):
            logging.info(
                "Turn %d was already decided against this campaign state; not "
                "requesting a second RED commander decision",
                brief.turn_id,
            )
            return CommanderTurnResult(brief=brief)
        return None

    def _build_client(self) -> ChatCompletionClient:
        if self._client is not None:
            return self._client
        assert self.brief is not None
        self._client = ChatCompletionClient(
            api_key=self.config.api_key or "",
            model=self.config.model,
            base_url=self.config.base_url,
            timeout_seconds=self.config.timeout_seconds,
            session_id=f"{self.brief.campaign_id_hash}:{self.brief.turn_id}",
        )
        return self._client

    def _load_catalog(self, client: ChatCompletionClient) -> None:
        """Refresh prices, tolerating a provider that cannot supply them.

        A missing catalogue must not block the turn, but it must not be silently
        treated as free either: the conservative fallback price is used and the
        estimate is flagged in the log.
        """

        try:
            catalog = ModelCatalog.from_payload(
                client.fetch_model_catalog(), source=self.config.base_url
            )
        except LlmError as error:
            catalog = ModelCatalog.unavailable(str(error))
            self.record.catalog_notes.append(
                f"model catalogue unavailable ({error}); budgeting with the "
                "conservative fallback price"
            )
        self.catalog = catalog
        price = catalog.price_for(self.config.model)
        if price.is_fallback_estimate:
            self.record.catalog_notes.append(
                f"{self.config.model} is not in the provider catalogue; "
                "budgeting with the conservative fallback price"
            )
        self.price = price

        self.record.catalog_retrieved_at = (
            datetime.fromtimestamp(catalog.retrieved_at, timezone.utc).isoformat(
                timespec="seconds"
            )
            if catalog.retrieved_at
            else ""
        )
        self.record.catalog_input_price_per_million = price.input_per_million
        self.record.catalog_output_price_per_million = price.output_per_million
        self.record.catalog_context_length = price.context_length

    def _reject(self, rejection: Rejection, stage: Optional[StageRecord]) -> None:
        """Record a refusal on the turn and, in ACTIVE mode, on its stage.

        Duplication is deliberate: the flat ``rejections`` list is what the
        phase-1 audit viewer and ``summary_from_payload`` read, and the per-stage
        list is what makes a three-call turn diagnosable.
        """

        payload = rejection.to_dict()
        self.record.rejections.append(payload)
        if stage is not None:
            stage.rejections.append(payload)

    def _attempt(
        self,
        client: ChatCompletionClient,
        messages: Sequence[dict[str, str]],
        price: ModelPrice,
        kind: str,
        parser: Optional[Callable[[str], _StageOutcome]] = None,
        response_format: Any = _UNSET,
        stage: Optional[StageRecord] = None,
    ) -> tuple[Optional[_StageOutcome], Optional[LlmResponse]]:
        """One request/validate cycle, fully accounted for.

        Returns ``(None, None)`` when the attempt could not be made or failed at
        the transport level; :attr:`_last_fallback_reason` then says why.

        ``parser`` and ``response_format`` are injected so all three ACTIVE
        stages share this method verbatim. That matters more than it looks: the
        cost ledger, the reservation/release/settle discipline and the audit
        entry are the parts that must not diverge between stages, because a
        second copy of them is a second place for the cap to be wrong.
        """

        assert self.brief is not None
        assert self.ledger is not None
        brief = self.brief
        ledger = self.ledger
        config = self.config
        if parser is None:
            parser = self._parse_decision_outcome
        if response_format is _UNSET:
            response_format = response_format_for(
                brief,
                price.supports_json_schema,
                price.supports_response_format or not self.catalog_available,
            )

        attempt = LlmAttempt(
            attempt=len(self.record.attempts) + 1,
            kind=kind,
            started_at=_utcnow(),
            requested_model=config.model,
            prompt_hash=prompt_digest(messages),
        )
        if config.log_prompts:
            attempt.prompt_messages = [dict(m) for m in messages]
        self.record.attempts.append(attempt)
        if stage is not None:
            stage.ran = True
            stage.attempt_indices.append(len(self.record.attempts) - 1)

        prompt_tokens = sum(
            estimate_tokens(str(m.get("content", ""))) for m in messages
        )
        attempt.prompt_tokens = prompt_tokens
        self.record.estimated_cost += price.cost_for(
            prompt_tokens, config.max_output_tokens
        )

        try:
            reserved = ledger.reserve(
                price, prompt_tokens, config.max_output_tokens, label=f"{kind} request"
            )
        except CostCapExceeded as error:
            attempt.error = str(error)
            self._reject(
                Rejection("<cost_cap>", str(error), round(error.would_be_total, 4)),
                stage,
            )
            self._last_fallback_reason = FallbackReason.COST_CAP
            return None, None
        attempt.reserved_cost = reserved
        self.record.reserved_cost += reserved

        started = time.monotonic()
        try:
            response = client.complete(
                messages,
                max_output_tokens=config.max_output_tokens,
                response_format=response_format,
            )
        except LlmError as error:
            # The request was not billed, so the reservation is released rather
            # than settled. A failed call must not eat the turn's budget.
            ledger.release(reserved)
            self.record.reserved_cost -= reserved
            attempt.reserved_cost = 0.0
            attempt.error = str(error)
            attempt.latency_seconds = time.monotonic() - started
            if isinstance(error, LlmHttpError):
                attempt.http_status = error.status
                self._last_fallback_reason = FallbackReason.HTTP_ERROR
            elif isinstance(error, LlmTimeout):
                self._last_fallback_reason = FallbackReason.TIMEOUT
            else:
                self._last_fallback_reason = FallbackReason.TRANSPORT_ERROR
            return None, None

        entry = ledger.settle(
            reserved,
            price,
            response.usage.input_tokens or prompt_tokens,
            response.usage.output_tokens,
            provider_cost=response.usage.cost,
            model_id=response.model or config.model,
            label=kind,
        )
        self.record.reserved_cost -= reserved
        self.record.actual_cost += entry.cost

        attempt.latency_seconds = response.latency_seconds
        attempt.actual_model = response.model
        attempt.response_id = response.request_id or ""
        attempt.finish_reason = response.finish_reason
        attempt.prompt_tokens = response.usage.input_tokens or prompt_tokens
        attempt.completion_tokens = response.usage.output_tokens
        attempt.cached_tokens = response.usage.cached_input_tokens
        attempt.reasoning_tokens = response.usage.reasoning_tokens
        attempt.total_tokens = response.usage.total_tokens
        attempt.actual_cost = entry.cost
        attempt.cost_is_estimated = not entry.reported_by_provider
        attempt.reserved_cost = 0.0
        attempt.retries = max(0, response.attempts - 1)
        attempt.response_text = response.text if config.log_prompts else ""
        attempt.response_hash = stable_hash(response.text)

        if stage is not None:
            stage.prompt_tokens += attempt.prompt_tokens
            stage.completion_tokens += attempt.completion_tokens
            stage.actual_cost += entry.cost

        if response.had_tool_calls:
            # No tools are ever offered, so this is either a misbehaving proxy or
            # an attempt to reach outside the sanctioned action space.
            self._reject(
                Rejection(
                    "<tool_calls>",
                    "the response contained tool calls, which are never offered "
                    "and are never executed",
                ),
                stage,
            )

        if response.model and response.model != config.model:
            self.record.notes.append(
                f"provider served {response.model} for requested {config.model}"
            )

        return parser(response.text), response

    # -- per-stage parsers ------------------------------------------------

    def _parse_decision_outcome(self, text: str) -> _StageOutcome:
        """Phase 1's strategic decision, also ACTIVE mode's stage 1."""

        assert self.brief is not None
        try:
            outcome = parse_decision(text, self.brief)
        except DecisionValidationError as error:
            return _StageOutcome(
                value=None, rejections=[Rejection("<response>", str(error))]
            )
        return _StageOutcome(value=outcome.decision, rejections=outcome.rejections)

    def _parse_plan_outcome(
        self,
        text: str,
        validator: Callable[[Any], tuple[Optional[Any], list[Rejection]]],
    ) -> _StageOutcome:
        """Shared body of the logistics and air tasking parsers."""

        try:
            payload = extract_json_object(text)
        except DecisionValidationError as error:
            return _StageOutcome(
                value=None, rejections=[Rejection("<response>", str(error))]
            )
        plan, rejections = validator(payload)
        return _StageOutcome(value=plan, rejections=rejections)

    # -- outcomes ---------------------------------------------------------

    @property
    def catalog_available(self) -> bool:
        return self.catalog is not None and self.catalog.available

    def _accept(
        self,
        directive: CommanderDirective,
        execution: Optional[ExecutionReport] = None,
    ) -> CommanderTurnResult:
        self.record.accepted = True
        self.record.accepted_directive = directive.to_dict()
        self.record.planner_task_order = list(task_order_for(directive))
        if self.catalog is not None:
            self.record.catalog_notes.extend(
                note for note in ([self.catalog.error] if self.catalog.error else [])
            )
        path = self._write()
        return CommanderTurnResult(
            directive=directive,
            record=self.record,
            brief=self.brief,
            log_path=path,
            operations=self.operations,
            execution=execution,
        )

    def _fail(self, reason: FallbackReason) -> CommanderTurnResult:
        """Record the fallback and, if configured, keep the previous strategy."""

        directive: Optional[CommanderDirective] = None
        if not self.config.fallback_to_builtin and self.brief is not None:
            directive = self._carry_previous_forward(self.brief)

        self.record.set_fallback(
            reason,
            FALLBACK_TO_PREVIOUS if directive is not None else FALLBACK_TO_BUILTIN,
        )
        if directive is not None:
            self.record.accepted_directive = directive.to_dict()
            self.record.planner_task_order = list(task_order_for(directive))
            self.record.notes.append(
                "built-in fallback is disabled; the previous directive was "
                "re-checked against live state and kept"
            )
        path = self._write() if reason is not FallbackReason.DISABLED else None
        return CommanderTurnResult(
            directive=directive,
            record=self.record,
            brief=self.brief,
            log_path=path,
            fallback_reason=reason,
            operations=self.operations,
        )

    def _carry_previous_forward(
        self, brief: RedCommanderBrief
    ) -> Optional[CommanderDirective]:
        if self.audit_log is None:
            return None
        previous = self.audit_log.latest_accepted_directive(
            brief.campaign_id_hash, before_turn=brief.turn_id - 1
        )
        if previous is None:
            return None
        try:
            directive, rejections = LegalityChecker(self.game, brief).carry_forward(
                previous
            )
        except Exception:  # pragma: no cover - defensive
            logging.warning(
                "Could not carry the previous directive forward", exc_info=True
            )
            return None
        self.record.rejections.extend(r.to_dict() for r in rejections)
        return directive

    def _write(self) -> Optional[Path]:
        if self.audit_log is None:
            return None
        return self.audit_log.write(self.record)


# ---------------------------------------------------------------------------
# The single call the game makes.
# ---------------------------------------------------------------------------


def plan_red_commander_turn(
    coalition: Coalition,
    config: Optional[AiCommanderConfig] = None,
    audit_log: Optional[AuditLog] = None,
    client: Optional[ChatCompletionClient] = None,
) -> Optional[CommanderDirective]:
    """Ask the LLM commander for RED's directive for this turn.

    Returns ``None`` whenever RED should be planned exactly as it is today. That
    is the case when the feature is off, when it is misconfigured, when the model
    fails or costs too much, and when nothing it asked for was legal.
    """

    if not coalition.player.is_red:
        return None

    resolved = config
    if resolved is None:
        resolved = AiCommanderConfig.from_settings(coalition.game.settings)
    if not resolved.enabled:
        return None

    log = audit_log
    if log is None:
        log = AuditLog.for_save_directory(resolved.audit_directory)

    result = RedCommanderTurn(coalition.game, resolved, log, client).run()
    if result.fallback_reason is not None:
        logging.info(
            "RED commander fell back to %s (%s)",
            result.record.fallback_policy if result.record else FALLBACK_TO_BUILTIN,
            result.fallback_reason.value,
        )
    return result.directive


def describe_turn_result(result: CommanderTurnResult) -> dict[str, Any]:
    """Compact summary used by the dry-run harness and the tests."""

    record = result.record
    summary: dict[str, Any] = {
        "accepted": result.accepted,
        "fallback_reason": (
            result.fallback_reason.value if result.fallback_reason else None
        ),
        "prompt_tokens": record.total_prompt_tokens if record else 0,
        "completion_tokens": record.total_completion_tokens if record else 0,
        "estimated_cost": round(record.estimated_cost, 6) if record else 0.0,
        "actual_cost": round(record.actual_cost, 6) if record else 0.0,
        "rejections": len(record.rejections) if record else 0,
        "log_path": str(result.log_path) if result.log_path else None,
    }
    if record is not None and record.stages:
        summary["mode"] = record.mode
        summary["requests"] = len(record.attempts)
        summary["stages"] = {
            stage.stage: (
                "accepted"
                if stage.accepted
                else (stage.fallback_reason or ("ran" if stage.ran else "skipped"))
            )
            for stage in record.stages
        }
    if result.execution is not None:
        summary["orders_applied"] = result.execution.applied_count
        summary["orders_failed"] = result.execution.failed_count
        summary["packages_added"] = result.execution.packages_added
        summary["budget_spent"] = round(result.execution.spent, 1)
    return summary
