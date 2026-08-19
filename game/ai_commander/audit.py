"""The audit trail: one JSON file per commander decision.

Everything the AI opponent does is reconstructable from these files. Each record
answers four questions:

1. **What did the model know?** The full, serialisable intel brief -- the exact
   object the prompt was rendered from -- plus its content hash.
2. **What did it propose?** The raw response text and the parsed decision.
3. **What did validation change or refuse?** Every rejection with its reason.
4. **What actually ran?** The accepted directive, the planner task order it
   produced, and the postures that were written.

Design constraints that shaped this module:

* **No save-format change.** Records are ordinary JSON files next to the save
  directory, never attributes on the pickled ``Game``. An existing campaign can
  be loaded by a build without this feature and vice versa.
* **Idempotence.** ``Coalition.initialize_turn`` can run several times for one
  turn (cheat capture, front-line cheat, buying or selling a TGO). The first run
  of a turn pays for the LLM call; later runs replay the accepted directive from
  disk. :meth:`AuditLog.accepted_directive_for` is that lookup.
* **Per-turn cost accounting survives a crash.** ``spent_this_turn`` is derived
  from the files, so an aborted turn cannot lose track of money already spent.
* **No secrets.** The API key is never passed to this module. Prompt text is
  only stored when the user opted in; otherwise only its hash is kept.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

from game.ai_commander.directive import CommanderDirective
from game.ai_commander.enums import FallbackReason
from game.ai_commander.intel import PriorTurnSummary, RedCommanderBrief
from game.ai_commander.serialization import canonical_json, jsonable, stable_hash

#: Schema version of the record format itself, so a future reader can migrate.
#:
#: Version 2 adds the ACTIVE-mode fields (``mode``, ``capability_hash``,
#: ``operations_hash``, ``operations_brief``, ``stages`` and
#: ``execution_report``). Every one of them has a default, so a version 1 file
#: still loads: readers must treat missing keys as "COMMANDER mode, no stages".
RECORD_SCHEMA_VERSION = "red-commander-audit/2"

#: The version this module first shipped with. Kept so readers can recognise
#: pre-ACTIVE records explicitly rather than by the absence of a key.
LEGACY_RECORD_SCHEMA_VERSION = "red-commander-audit/1"

#: Sub-directory of the save directory that holds the decision log.
AUDIT_DIRECTORY_NAME = "AiDecisions"

_FILENAME_RE = re.compile(r"^turn_(\d+)_(\d+)\.json$")

#: Never write more than this many records for a single turn. Re-entrant
#: ``initialize_turn`` calls are replays, so a runaway count means a bug; the cap
#: keeps it from filling the user's disk.
MAX_RECORDS_PER_TURN = 50


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LlmAttempt:
    """One HTTP round trip to the provider."""

    #: 1 for the first call, 2 for the single permitted repair attempt.
    attempt: int
    #: ``"initial"`` or ``"repair"``.
    kind: str
    started_at: str = ""
    latency_seconds: float = 0.0
    #: Model as configured by the user.
    requested_model: str = ""
    #: Model the provider says actually answered (alias resolution is logged).
    actual_model: str = ""
    response_id: str = ""
    finish_reason: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    #: Worst-case cost reserved from the turn budget before the call.
    reserved_cost: float = 0.0
    #: Cost from the provider's own ``usage`` block when present, else estimated
    #: from catalog prices and the returned token counts.
    actual_cost: float = 0.0
    cost_is_estimated: bool = True
    #: Only populated when the user enabled raw prompt logging.
    prompt_messages: Optional[list[dict[str, str]]] = None
    prompt_hash: str = ""
    response_text: str = ""
    response_hash: str = ""
    error: str = ""
    http_status: Optional[int] = None
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass
class StageRecord:
    """One stage of an ACTIVE-mode turn.

    ACTIVE mode asks the model three questions in sequence -- commander intent,
    then logistics, then air tasking -- and each one is a separate request with
    its own prompt, its own schema and its own validation pass. A single flat
    record could not say which call produced which rejection, so each stage gets
    its own entry and the top-level ``attempts`` list stays a chronological view
    across the whole turn.

    A stage that never ran (because an earlier one failed, or the cost cap was
    reached) is still written, with ``ran`` false and ``fallback_reason`` set.
    That is what makes a partially-degraded turn readable after the fact.
    """

    #: :class:`~game.ai_commander.plan.CommanderStage` value.
    stage: str
    #: Schema version the stage asked the model to satisfy.
    schema_version: str = ""
    #: ``True`` once at least one request was issued for this stage.
    ran: bool = False
    #: ``True`` when validation produced something the executor could use.
    accepted: bool = False
    #: Why this stage produced nothing, when it did not.
    fallback_reason: Optional[str] = None
    #: Indices into :attr:`AiDecisionRecord.attempts` belonging to this stage.
    attempt_indices: list[int] = field(default_factory=list)
    #: Schema-validated payload, before legality checking against live state.
    parsed_plan: Optional[dict[str, Any]] = None
    #: What survived legality checking. For COMMAND this is the directive.
    accepted_plan: Optional[dict[str, Any]] = None
    #: Rejections raised by this stage only. Also merged into the record-level
    #: list so an existing reader still sees every refusal.
    rejections: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Cost attributed to this stage, so an expensive stage is identifiable.
    actual_cost: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass
class AiDecisionRecord:
    """Everything about one commander turn, successful or not."""

    record_schema_version: str = RECORD_SCHEMA_VERSION
    written_at: str = field(default_factory=_utcnow)

    # -- identity ---------------------------------------------------------
    campaign_id_hash: str = ""
    turn_id: int = 0
    campaign_revision: str = ""
    intel_policy: str = ""
    intel_hash: str = ""
    decision_schema_hash: str = ""
    personality: str = ""
    #: ``"commander"`` or ``"active"``. Absent in version 1 records, which were
    #: always COMMANDER mode.
    mode: str = ""

    # -- provider configuration ------------------------------------------
    base_url: str = ""
    configured_model: str = ""
    catalog_retrieved_at: str = ""
    catalog_input_price_per_million: Optional[float] = None
    catalog_output_price_per_million: Optional[float] = None
    catalog_context_length: Optional[int] = None
    catalog_notes: list[str] = field(default_factory=list)

    # -- what the model was told and what it said -------------------------
    #: The full brief. This is the "what red could legitimately know" snapshot.
    intel_brief: dict[str, Any] = field(default_factory=dict)
    #: The compact rendering that was actually placed in the prompt.
    intel_rendered: str = ""
    prompt_logging_enabled: bool = False
    attempts: list[LlmAttempt] = field(default_factory=list)
    #: The schema-validated decision, before legality checking.
    parsed_decision: Optional[dict[str, Any]] = None
    #: What survived legality checking and was handed to the planner.
    accepted_directive: Optional[dict[str, Any]] = None
    #: Everything refused, with the reason. Never silently dropped.
    rejections: list[dict[str, Any]] = field(default_factory=list)

    # -- ACTIVE mode ------------------------------------------------------
    #: Hash of the RED capability index the prompts were rendered from. Lets a
    #: reader tell whether two turns saw the same force description.
    capability_hash: str = ""
    #: Hash of the operations brief (bases, squadrons, observed targets).
    operations_hash: str = ""
    #: The full operations brief, alongside ``intel_brief``. Together these are
    #: the complete "what RED could legitimately know" snapshot for ACTIVE mode.
    operations_brief: dict[str, Any] = field(default_factory=dict)
    #: The compact operations rendering placed in the stage 2 and 3 prompts.
    operations_rendered: str = ""
    #: One entry per ACTIVE stage, in execution order. Empty in COMMANDER mode.
    stages: list[StageRecord] = field(default_factory=list)
    #: What the executor actually managed to apply, order by order.
    execution_report: Optional[dict[str, Any]] = None

    # -- what deterministic code then did ---------------------------------
    #: The order in which the HTN planner was offered its compound tasks.
    planner_task_order: list[str] = field(default_factory=list)
    #: One entry per posture the directive asked for, applied or not.
    posture_applications: list[dict[str, Any]] = field(default_factory=list)
    #: Free-form notes emitted by the procurement adapter.
    procurement_notes: list[str] = field(default_factory=list)

    # -- cost -------------------------------------------------------------
    cost_cap_per_turn: float = 0.0
    estimated_cost: float = 0.0
    reserved_cost: float = 0.0
    actual_cost: float = 0.0
    #: Cost already recorded for this turn by earlier records, so the cap is
    #: enforced across a re-entrant turn rather than per call.
    prior_cost_this_turn: float = 0.0

    # -- outcome ----------------------------------------------------------
    #: ``True`` when the planner ran under a directive.
    accepted: bool = False
    #: Set whenever the built-in RED automation kept control.
    fallback_reason: Optional[str] = None
    fallback_policy: str = ""
    #: ``True`` when this record documents a replay of an earlier decision.
    replayed_from_turn_record: bool = False
    notes: list[str] = field(default_factory=list)

    # -- derived ----------------------------------------------------------

    @property
    def total_prompt_tokens(self) -> int:
        return sum(a.prompt_tokens for a in self.attempts)

    @property
    def total_completion_tokens(self) -> int:
        return sum(a.completion_tokens for a in self.attempts)

    def to_dict(self) -> dict[str, Any]:
        payload = jsonable(self)
        payload["total_prompt_tokens"] = self.total_prompt_tokens
        payload["total_completion_tokens"] = self.total_completion_tokens
        return payload

    def set_brief(self, brief: RedCommanderBrief, include_rendered: bool) -> None:
        """Attach the intel snapshot the decision was made from."""

        self.campaign_id_hash = brief.campaign_id_hash
        self.turn_id = brief.turn_id
        self.campaign_revision = brief.campaign_revision
        self.intel_policy = brief.intel_policy.value
        self.intel_brief = brief.to_dict()
        self.intel_hash = brief.content_hash()
        if include_rendered:
            self.intel_rendered = brief.render_compact()

    def set_operations(
        self,
        brief: Any,
        capability_hash: str,
        include_rendered: bool,
    ) -> None:
        """Attach the ACTIVE-mode operations snapshot.

        Typed loosely on purpose: ``audit`` must not import
        :mod:`game.ai_commander.operations`, which reaches into the theater, or a
        COMMANDER-mode turn would pay for machinery it never uses.
        """

        self.operations_brief = brief.to_dict()
        self.operations_hash = brief.content_hash()
        self.capability_hash = capability_hash
        if include_rendered:
            self.operations_rendered = brief.render_compact()

    def stage_record(self, stage: Any) -> StageRecord:
        """Get or create the record for ``stage``, appending in first-seen order."""

        name = getattr(stage, "value", stage)
        key = str(name)
        for existing in self.stages:
            if existing.stage == key:
                return existing
        created = StageRecord(stage=key)
        self.stages.append(created)
        return created

    def set_fallback(self, reason: FallbackReason, policy: str) -> None:
        self.accepted = False
        self.accepted_directive = None
        self.fallback_reason = reason.value
        self.fallback_policy = policy

    def summarise(self) -> PriorTurnSummary:
        """Condense this record into the "last turn" block of the next brief.

        Deliberately terse: the next turn's prompt gets what the commander
        itself decided and how much of it was refused, not a replay of the
        campaign state, which the new brief already carries.
        """

        return summary_from_payload(self.to_dict())


class AuditLog:
    """Reader/writer for the on-disk decision log.

    ``root`` is the directory the ``AiDecisions`` tree lives under. It is passed
    in rather than discovered so tests and the dry-run harness never touch a real
    save directory, and so a missing DCS installation cannot break a turn.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- location ---------------------------------------------------------

    @classmethod
    def for_save_directory(cls, save_dir: Optional[Path] = None) -> Optional[AuditLog]:
        """Build a log rooted at the campaign save directory.

        Returns ``None`` when no writable location can be determined --
        ``game.persistency.save_dir`` asserts that DCS's saved-games folder has
        been configured, which is not true in a headless test or dry run. A
        missing audit location must degrade the feature, never crash a turn.
        """

        if save_dir is not None:
            return cls(Path(save_dir))
        override = os.environ.get("RETRIBUTION_AI_AUDIT_DIR")
        if override:
            return cls(Path(override))
        try:
            from game.persistency import save_dir as retribution_save_dir

            return cls(Path(retribution_save_dir()))
        except Exception:
            logging.debug(
                "No Retribution save directory available; AI decision log disabled",
                exc_info=True,
            )
            return None

    def campaign_directory(self, campaign_id_hash: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", campaign_id_hash) or "unknown"
        return self.root / AUDIT_DIRECTORY_NAME / safe

    # -- writing ----------------------------------------------------------

    def write(self, record: AiDecisionRecord) -> Optional[Path]:
        """Persist ``record``. Never raises; a failed write is logged only."""

        directory = self.campaign_directory(record.campaign_id_hash)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            logging.warning(
                "Could not create AI decision log directory %s",
                directory,
                exc_info=True,
            )
            return None

        existing = len(list(self._turn_files(directory, record.turn_id)))
        if existing >= MAX_RECORDS_PER_TURN:
            logging.warning(
                "Refusing to write more than %d AI decision records for turn %d",
                MAX_RECORDS_PER_TURN,
                record.turn_id,
            )
            return None

        path = directory / f"turn_{record.turn_id:04d}_{existing:02d}.json"
        payload = record.to_dict()
        try:
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(canonical_json(payload), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            logging.warning("Could not write %s", path, exc_info=True)
            return None
        return path

    # -- reading ----------------------------------------------------------

    def _turn_files(self, directory: Path, turn_id: int) -> Iterator[Path]:
        prefix = f"turn_{turn_id:04d}_"
        if not directory.is_dir():
            return
        for path in sorted(directory.glob(f"{prefix}*.json")):
            yield path

    def _load(self, path: Path) -> Optional[dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logging.warning("Unreadable AI decision record %s", path, exc_info=True)
            return None
        if not isinstance(data, dict):
            return None
        return data

    def records_for_turn(
        self, campaign_id_hash: str, turn_id: int
    ) -> list[dict[str, Any]]:
        directory = self.campaign_directory(campaign_id_hash)
        records: list[dict[str, Any]] = []
        for path in self._turn_files(directory, turn_id):
            data = self._load(path)
            if data is not None:
                data.setdefault("_path", str(path))
                records.append(data)
        return records

    def turns(self, campaign_id_hash: str) -> list[int]:
        directory = self.campaign_directory(campaign_id_hash)
        if not directory.is_dir():
            return []
        found: set[int] = set()
        for path in directory.glob("turn_*.json"):
            match = _FILENAME_RE.match(path.name)
            if match is not None:
                found.add(int(match.group(1)))
        return sorted(found)

    def campaigns(self) -> list[str]:
        base = self.root / AUDIT_DIRECTORY_NAME
        if not base.is_dir():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    # -- queries used while planning a turn -------------------------------

    def accepted_directive_for(
        self, campaign_id_hash: str, turn_id: int, campaign_revision: str
    ) -> Optional[CommanderDirective]:
        """The directive already accepted for this exact state, if any.

        Requiring the revision to match means a turn whose state changed (the
        player used a cheat, sold a TGO) is treated as a new decision point
        rather than silently re-applying a directive formed against stale state.
        """

        for data in reversed(self.records_for_turn(campaign_id_hash, turn_id)):
            if not data.get("accepted"):
                continue
            if data.get("campaign_revision") != campaign_revision:
                continue
            payload = data.get("accepted_directive")
            if not isinstance(payload, dict):
                continue
            try:
                return CommanderDirective.from_dict(payload)
            except (KeyError, TypeError, ValueError):
                logging.warning(
                    "Ignoring unreadable accepted directive in AI decision log",
                    exc_info=True,
                )
        return None

    def latest_accepted_directive(
        self, campaign_id_hash: str, before_turn: Optional[int] = None
    ) -> Optional[CommanderDirective]:
        """The most recent accepted directive, regardless of turn or revision.

        Only used to keep RED's previous strategy when the player has disabled
        the built-in fallback. The caller must re-check it against live state
        before applying it.
        """

        for turn_id in reversed(self.turns(campaign_id_hash)):
            if before_turn is not None and turn_id > before_turn:
                continue
            for data in reversed(self.records_for_turn(campaign_id_hash, turn_id)):
                if not data.get("accepted"):
                    continue
                payload = data.get("accepted_directive")
                if not isinstance(payload, dict):
                    continue
                try:
                    return CommanderDirective.from_dict(payload)
                except (KeyError, TypeError, ValueError):
                    logging.warning(
                        "Ignoring unreadable directive in AI decision log",
                        exc_info=True,
                    )
        return None

    def has_record_for_revision(
        self, campaign_id_hash: str, turn_id: int, campaign_revision: str
    ) -> bool:
        """Whether this turn *and state* was already decided, even if refused.

        Prevents paying for a second call after a decision that legitimately
        ended in fallback, which would otherwise be retried on every re-entrant
        ``initialize_turn``.
        """

        return any(
            data.get("campaign_revision") == campaign_revision
            for data in self.records_for_turn(campaign_id_hash, turn_id)
        )

    def spent_this_turn(self, campaign_id_hash: str, turn_id: int) -> float:
        total = 0.0
        for data in self.records_for_turn(campaign_id_hash, turn_id):
            try:
                total += float(data.get("actual_cost") or 0.0)
            except (TypeError, ValueError):
                continue
        return total

    def latest_summary(
        self, campaign_id_hash: str, before_turn: int
    ) -> Optional[PriorTurnSummary]:
        """Summary of the most recent decided turn before ``before_turn``."""

        for turn_id in reversed(self.turns(campaign_id_hash)):
            if turn_id >= before_turn:
                continue
            records = self.records_for_turn(campaign_id_hash, turn_id)
            if not records:
                continue
            return summary_from_payload(records[-1])
        return None

    # -- UI support -------------------------------------------------------

    def all_records(self, campaign_id_hash: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for turn_id in self.turns(campaign_id_hash):
            records.extend(self.records_for_turn(campaign_id_hash, turn_id))
        return records


def summary_from_payload(payload: Mapping[str, Any]) -> PriorTurnSummary:
    """Rebuild a :class:`PriorTurnSummary` from a stored record."""

    raw_directive = payload.get("accepted_directive")
    directive: Mapping[str, Any] = (
        raw_directive if isinstance(raw_directive, Mapping) else {}
    )
    rejections: Iterable[Any] = payload.get("rejections") or ()
    rejected: list[str] = []
    for item in rejections:
        if isinstance(item, Mapping) and item.get("element"):
            rejected.append(str(item["element"]))
    postures = directive.get("front_postures")
    return PriorTurnSummary(
        turn=int(payload.get("turn_id") or 0),
        strategy=str(directive.get("strategy") or "") or None,
        reserve_policy=str(directive.get("reserve_policy") or "") or None,
        intent=str(directive.get("commander_intent") or "") or None,
        target_set_order=tuple(
            str(x) for x in (directive.get("target_set_order") or ())
        ),
        front_postures=(
            {str(k): str(v) for k, v in postures.items()}
            if isinstance(postures, Mapping)
            else {}
        ),
        rejected_element_count=len(rejected),
        rejected_elements=tuple(rejected[:8]),
        fallback_reason=(
            str(payload["fallback_reason"]) if payload.get("fallback_reason") else None
        ),
    )


def prompt_digest(messages: Iterable[Mapping[str, str]]) -> str:
    """Stable hash of a message list, for logs that omit the prompt text."""

    return stable_hash([dict(m) for m in messages], length=16)
