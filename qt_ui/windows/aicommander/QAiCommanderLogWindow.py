"""Viewer for the LLM commander's decision log.

Auditability is only real if it is reachable, so the same JSON the commander
writes next to the campaign save is presented here: for each decision point, what
the model was told, what it asked for, what was refused and why, and what it
cost.

The window is a pure reader. It never triggers a request, never mutates the
campaign, and works with no API key configured -- an existing log can be reviewed
long after the key has been removed.

Two record shapes exist. A COMMANDER-mode turn is one request, so it has a single
decision and one flat list of refusals. An ACTIVE-mode turn is a sequence of
requests -- commander intent, then logistics, then air tasking -- each with its own
prompt, schema, refusals and cost, followed by an execution report listing every
order that was actually pushed through Retribution's own purchase, repair,
transfer and mission-planning code. Both shapes render here, and records written
before ACTIVE mode existed (audit schema version 1) still open: the ACTIVE tabs
simply report that there is nothing staged to show.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import qt_ui.uiconstants as CONST
from game.ai_commander.audit import AuditLog
from game.ai_commander.config import AiCommanderConfig
from game.ai_commander.intel import IntelProjector
from game.game import Game


def _monospace(text: str) -> QPlainTextEdit:
    widget = QPlainTextEdit()
    widget.setReadOnly(True)
    widget.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    widget.setPlainText(text)
    font = widget.font()
    font.setFamily("Courier New")
    widget.setFont(font)
    return widget


def _stages_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The ACTIVE-mode stage entries of ``record``, or an empty list.

    Version 1 records have no ``stages`` key at all, and a COMMANDER-mode
    version 2 record has an empty one, so both answer "not staged" here.
    """

    stages = record.get("stages")
    if not isinstance(stages, list):
        return []
    return [stage for stage in stages if isinstance(stage, dict)]


def _stage_headline(stage: dict[str, Any]) -> str:
    """A single line describing what one stage did."""

    name = str(stage.get("stage") or "?")
    if stage.get("accepted"):
        state = "accepted"
    elif not stage.get("ran"):
        state = "skipped"
    else:
        state = "no usable output"
    reason = stage.get("fallback_reason")
    suffix = f" ({reason})" if reason else ""
    tokens = (
        f"in={stage.get('prompt_tokens') or 0} "
        f"out={stage.get('completion_tokens') or 0} "
        f"${float(stage.get('actual_cost') or 0):.4f}"
    )
    refused = len(stage.get("rejections") or [])
    return f"{name:<12} {state}{suffix}  {tokens}  refused={refused}"


class QAiCommanderLogWindow(QDialog):
    """Per-turn view of the AI opponent's decisions."""

    def __init__(self, game: Game, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.game = game
        self.setWindowTitle("AI Opponent Decision Log")
        self.setWindowIcon(CONST.ICONS.get("Intel", CONST.ICONS["Generator"]))
        self.setMinimumSize(1000, 640)

        self.records: list[dict[str, Any]] = []
        self.campaign_id_hash = ""
        self.log: Optional[AuditLog] = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.header = QLabel()
        self.header.setWordWrap(True)
        self.header.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, stretch=1)

        self.turn_list = QListWidget()
        self.turn_list.setMaximumWidth(280)
        self.turn_list.currentRowChanged.connect(self._show_record)
        splitter.addWidget(self.turn_list)

        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)

        self.summary_view = _monospace("")
        self.tabs.addTab(self.summary_view, "Decision")

        self.rejection_tree = QTreeWidget()
        self.rejection_tree.setColumnCount(3)
        self.rejection_tree.setHeaderLabels(["Element", "Reason", "Value"])
        self.tabs.addTab(self.rejection_tree, "Rejected")

        self.intel_view = _monospace("")
        self.tabs.addTab(self.intel_view, "Intel given to the AI")

        self.prompt_view = _monospace("")
        self.tabs.addTab(self.prompt_view, "Prompt and response")

        self.stage_view = _monospace("")
        self.tabs.addTab(self.stage_view, "Stages")

        self.execution_view = _monospace("")
        self.tabs.addTab(self.execution_view, "Orders executed")

        self.capability_view = _monospace("")
        self.tabs.addTab(self.capability_view, "Capabilities and targets")

        self.cost_view = _monospace("")
        self.tabs.addTab(self.cost_view, "Tokens and cost")

        self.raw_view = _monospace("")
        self.tabs.addTab(self.raw_view, "Raw record")

        buttons = QHBoxLayout()
        buttons.addStretch()
        reload_button = QPushButton("Reload")
        reload_button.clicked.connect(self.reload)
        buttons.addWidget(reload_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        self.reload()

    # -- loading ----------------------------------------------------------

    def reload(self) -> None:
        config = AiCommanderConfig.from_settings(self.game.settings)
        self.log = AuditLog.for_save_directory(config.audit_directory)
        try:
            self.campaign_id_hash = IntelProjector(
                self.game, config.intel_policy
            ).campaign_id_hash()
        except Exception:
            self.campaign_id_hash = ""

        if self.log is None:
            self.header.setText(
                "<strong>No decision log location is available.</strong> "
                "Retribution could not determine your DCS saved games folder, "
                "so nothing has been written."
            )
            self.records = []
        else:
            self.records = self.log.all_records(self.campaign_id_hash)
            location = self.log.campaign_directory(self.campaign_id_hash)
            self.header.setText(
                f"<strong>{len(self.records)} recorded decision "
                f"{'point' if len(self.records) == 1 else 'points'}</strong> "
                f"for this campaign.<br />Log directory: {location}"
            )

        self.turn_list.clear()
        for record in self.records:
            self.turn_list.addItem(QListWidgetItem(self._label_for(record)))
        if self.records:
            self.turn_list.setCurrentRow(len(self.records) - 1)
        else:
            self._clear_views()

    @staticmethod
    def _label_for(record: dict[str, Any]) -> str:
        turn = record.get("turn_id", "?")
        if record.get("accepted"):
            state = "accepted"
        else:
            state = f"fallback: {record.get('fallback_reason') or 'unknown'}"
        cost = record.get("actual_cost") or 0.0
        stages = _stages_of(record)
        mode = ""
        if stages:
            accepted = sum(1 for stage in stages if stage.get("accepted"))
            mode = f" [active {accepted}/{len(stages)}]"
        return f"Turn {turn}{mode} - {state} (${float(cost):.4f})"

    def _clear_views(self) -> None:
        for view in (
            self.summary_view,
            self.stage_view,
            self.execution_view,
            self.capability_view,
            self.intel_view,
            self.prompt_view,
            self.cost_view,
            self.raw_view,
        ):
            view.setPlainText("")
        self.rejection_tree.clear()

    # -- rendering --------------------------------------------------------

    def _show_record(self, row: int) -> None:
        if row < 0 or row >= len(self.records):
            self._clear_views()
            return
        record = self.records[row]
        self.summary_view.setPlainText(self._render_decision(record))
        self._render_rejections(record)
        self.stage_view.setPlainText(self._render_stages(record))
        self.execution_view.setPlainText(self._render_execution(record))
        self.capability_view.setPlainText(self._render_capabilities(record))
        self.intel_view.setPlainText(self._render_intel(record))
        self.prompt_view.setPlainText(self._render_prompt(record))
        self.cost_view.setPlainText(self._render_cost(record))
        self.raw_view.setPlainText(json.dumps(record, indent=2, sort_keys=True))

    @staticmethod
    def _render_decision(record: dict[str, Any]) -> str:
        lines: list[str] = [
            f"Turn:              {record.get('turn_id')}",
            f"Campaign revision: {record.get('campaign_revision')}",
            f"Intel policy:      {record.get('intel_policy')}",
            f"Mode:              {record.get('mode') or 'commander'}",
            f"Personality:       {record.get('personality')}",
            f"Requested model:   {record.get('configured_model')}",
            f"Accepted:          {record.get('accepted')}",
        ]
        if record.get("fallback_reason"):
            lines.append(f"Fallback reason:   {record.get('fallback_reason')}")
            lines.append(f"Fallback policy:   {record.get('fallback_policy')}")
        lines.append("")

        directive = record.get("accepted_directive")
        if isinstance(directive, dict):
            lines.append("ACCEPTED DIRECTIVE")
            lines.append(f"  strategy:       {directive.get('strategy')}")
            lines.append(f"  reserve policy: {directive.get('reserve_policy')}")
            lines.append(
                f"  target sets:    {', '.join(directive.get('target_set_order') or []) or '(none)'}"
            )
            lines.append(
                f"  spending:       {', '.join(directive.get('procurement_order') or []) or '(none)'}"
            )
            fronts = directive.get("front_order") or []
            lines.append(
                f"  fronts:         {', '.join(str(f) for f in fronts) or '(none)'}"
            )
            postures = directive.get("front_postures") or {}
            if postures:
                lines.append("  postures:")
                for front, posture in postures.items():
                    lines.append(f"    {front}: {posture}")
            intent = directive.get("commander_intent") or ""
            if intent:
                lines.append(f"  intent:         {intent}")
        else:
            lines.append("No directive was accepted for this decision point.")

        stages = _stages_of(record)
        if stages:
            lines.append("")
            lines.append("STAGES THIS TURN")
            for stage in stages:
                lines.append(f"  {_stage_headline(stage)}")
            report = record.get("execution_report")
            if isinstance(report, dict):
                lines.append("")
                lines.append("EXECUTION")
                lines.append(
                    f"  orders applied:  {report.get('applied')}"
                    f"   failed: {report.get('failed')}"
                )
                lines.append(f"  packages added:  {report.get('packages_added')}")
                lines.append(
                    f"  budget:          {float(report.get('budget_before') or 0):.0f}M"
                    f" -> {float(report.get('budget_after') or 0):.0f}M"
                    f" (spent {float(report.get('spent') or 0):.0f}M)"
                )

        order = record.get("planner_task_order") or []
        if order:
            lines.append("")
            lines.append("PLANNER TASK ORDER APPLIED")
            for index, task in enumerate(order, start=1):
                lines.append(f"  {index:>2}. {task}")

        notes = record.get("notes") or []
        if notes:
            lines.append("")
            lines.append("NOTES")
            lines.extend(f"  - {note}" for note in notes)
        return "\n".join(lines)

    @staticmethod
    def _rejection_item(rejection: dict[str, Any]) -> QTreeWidgetItem:
        value = rejection.get("value")
        return QTreeWidgetItem(
            [
                str(rejection.get("element", "")),
                str(rejection.get("reason", "")),
                "" if value is None else json.dumps(value),
            ]
        )

    def _render_rejections(self, record: dict[str, Any]) -> None:
        self.rejection_tree.clear()
        rejections = [r for r in record.get("rejections") or [] if isinstance(r, dict)]
        stages = _stages_of(record)
        if stages:
            # An ACTIVE turn refuses things in three different conversations, and
            # "which call asked for this" is the first thing an audit needs, so
            # group by stage rather than presenting one undifferentiated list.
            for stage in stages:
                staged = [
                    r for r in stage.get("rejections") or [] if isinstance(r, dict)
                ]
                parent = QTreeWidgetItem(
                    [str(stage.get("stage") or "?"), f"{len(staged)} refused", ""]
                )
                for rejection in staged:
                    parent.addChild(self._rejection_item(rejection))
                self.rejection_tree.addTopLevelItem(parent)
                parent.setExpanded(True)
        else:
            for rejection in rejections:
                self.rejection_tree.addTopLevelItem(self._rejection_item(rejection))
        for column in range(3):
            self.rejection_tree.resizeColumnToContents(column)
        self.tabs.setTabText(1, f"Rejected ({len(rejections)})")

    @staticmethod
    def _render_stages(record: dict[str, Any]) -> str:
        stages = _stages_of(record)
        if not stages:
            return (
                "This turn was planned in commander mode: one request, one "
                "decision, no separate stages.\n\nSwitch the AI Opponent mode to "
                "Active to have the model allocate the budget and write the air "
                "tasking order itself."
            )
        lines: list[str] = []
        for stage in stages:
            lines.append(f"=== {_stage_headline(stage)}")
            lines.append(f"  schema:   {stage.get('schema_version') or '(none)'}")
            attempts = stage.get("attempt_indices") or []
            lines.append(
                "  requests: "
                + (", ".join(f"#{index}" for index in attempts) or "(none issued)")
            )
            for note in stage.get("notes") or []:
                lines.append(f"  note:     {note}")
            parsed = stage.get("parsed_plan")
            if isinstance(parsed, dict):
                lines.append("  AS ASKED FOR (schema-valid, before legality checks)")
                lines.extend(
                    f"    {line}"
                    for line in json.dumps(
                        parsed, indent=2, sort_keys=True
                    ).splitlines()
                )
            accepted = stage.get("accepted_plan")
            if isinstance(accepted, dict):
                lines.append("  AS ACCEPTED (checked against live campaign state)")
                lines.extend(
                    f"    {line}"
                    for line in json.dumps(
                        accepted, indent=2, sort_keys=True
                    ).splitlines()
                )
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_execution(record: dict[str, Any]) -> str:
        report = record.get("execution_report")
        if not isinstance(report, dict):
            return (
                "Nothing was executed directly by the AI this turn.\n\nIn "
                "commander mode the model only sets priorities and Retribution's "
                "own auto-planner does the spending and the mission planning, so "
                "there is no order-by-order trail to show."
            )
        lines = [
            f"Orders applied:  {report.get('applied')}",
            f"Orders failed:   {report.get('failed')}",
            f"Packages added:  {report.get('packages_added')}",
            f"Budget before:   {float(report.get('budget_before') or 0):.0f}M",
            f"Budget after:    {float(report.get('budget_after') or 0):.0f}M",
            f"Spent:           {float(report.get('spent') or 0):.0f}M",
            "",
            "Every line below went through the same purchase, repair, transfer "
            "and mission-planning code the player's own UI calls.",
            "",
        ]
        orders = report.get("orders")
        if not isinstance(orders, list) or not orders:
            lines.append("(no orders were issued)")
            return "\n".join(lines)
        for order in orders:
            if not isinstance(order, dict):
                continue
            mark = "+" if order.get("applied") else "!"
            detail = order.get("detail") or ""
            suffix = f"  -- {detail}" if detail else ""
            lines.append(
                f"  {mark} [{order.get('kind')}] {order.get('description')}{suffix}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_capabilities(record: dict[str, Any]) -> str:
        rendered = record.get("operations_rendered") or ""
        brief = record.get("operations_brief")
        capability_hash = record.get("capability_hash") or ""
        operations_hash = record.get("operations_hash") or ""
        if not (rendered or capability_hash or isinstance(brief, dict)):
            return (
                "No capability index or operations briefing was built for this "
                "turn.\n\nThese are active-mode inputs: the index lists the units "
                "and airframes OPFOR actually owns or can buy, taken from the "
                "game's own data files, and the operations briefing lists the "
                "bases, squadrons and observed targets it may plan against."
            )
        parts: list[str] = [
            f"Capability index hash: {capability_hash or '(not built)'}",
            f"Operations brief hash: {operations_hash or '(not built)'}",
            "",
            "The capability index is derived entirely from Retribution's own unit "
            "data files and is restricted to what OPFOR owns or can purchase, so "
            "the model cannot invent a weapon it does not have and cannot be told "
            "anything about your order of battle.",
            "",
        ]
        if rendered:
            parts.append("OPERATIONS BRIEFING AS SENT TO THE MODEL")
            parts.append(rendered)
            parts.append("")
        parts.append("FULL OPERATIONS BRIEF (serialised)")
        parts.append(
            json.dumps(brief, indent=2, sort_keys=True)
            if isinstance(brief, dict)
            else "(not recorded)"
        )
        return "\n".join(parts)

    @staticmethod
    def _render_intel(record: dict[str, Any]) -> str:
        rendered = record.get("intel_rendered") or ""
        brief = record.get("intel_brief")
        parts: list[str] = [
            f"Intel hash: {record.get('intel_hash')}",
            f"Policy:     {record.get('intel_policy')}",
            "",
        ]
        if rendered:
            parts.append("AS SENT TO THE MODEL")
            parts.append(rendered)
            parts.append("")
        parts.append("FULL BRIEF (serialised)")
        parts.append(
            json.dumps(brief, indent=2, sort_keys=True)
            if isinstance(brief, dict)
            else "(not recorded)"
        )
        return "\n".join(parts)

    @staticmethod
    def _render_prompt(record: dict[str, Any]) -> str:
        if not record.get("prompt_logging_enabled"):
            lines = [
                "Prompt logging is disabled for this campaign, so only hashes "
                "were recorded.",
                "",
            ]
        else:
            lines = []
        for attempt in record.get("attempts") or []:
            if not isinstance(attempt, dict):
                continue
            lines.append(
                f"--- attempt {attempt.get('attempt')} ({attempt.get('kind')}) ---"
            )
            lines.append(f"prompt hash:   {attempt.get('prompt_hash')}")
            lines.append(f"response hash: {attempt.get('response_hash')}")
            if attempt.get("error"):
                lines.append(f"error:         {attempt.get('error')}")
            messages = attempt.get("prompt_messages")
            if isinstance(messages, list):
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    lines.append("")
                    lines.append(f"[{message.get('role')}]")
                    lines.append(str(message.get("content", "")))
            response = attempt.get("response_text")
            if response:
                lines.append("")
                lines.append("[response]")
                lines.append(str(response))
            lines.append("")
        return "\n".join(lines) if lines else "No requests were made."

    @staticmethod
    def _render_cost(record: dict[str, Any]) -> str:
        lines = [
            f"Cost cap this turn:     ${float(record.get('cost_cap_per_turn') or 0):.2f}",
            f"Already spent on entry: ${float(record.get('prior_cost_this_turn') or 0):.4f}",
            f"Estimated worst case:   ${float(record.get('estimated_cost') or 0):.4f}",
            f"Actually charged:       ${float(record.get('actual_cost') or 0):.4f}",
            "",
            f"Catalogue retrieved:    {record.get('catalog_retrieved_at') or '(unavailable)'}",
            f"Input  $/M tokens:      {record.get('catalog_input_price_per_million')}",
            f"Output $/M tokens:      {record.get('catalog_output_price_per_million')}",
            f"Context length:         {record.get('catalog_context_length')}",
        ]
        for note in record.get("catalog_notes") or []:
            lines.append(f"  ! {note}")
        lines.append("")
        lines.append("PER ATTEMPT")
        for attempt in record.get("attempts") or []:
            if not isinstance(attempt, dict):
                continue
            lines.append(
                f"  #{attempt.get('attempt')} {attempt.get('kind'):<8} "
                f"model={attempt.get('actual_model') or attempt.get('requested_model')} "
                f"in={attempt.get('prompt_tokens')} out={attempt.get('completion_tokens')} "
                f"cached={attempt.get('cached_tokens')} "
                f"reasoning={attempt.get('reasoning_tokens')} "
                f"cost=${float(attempt.get('actual_cost') or 0):.4f}"
                f"{' (estimated)' if attempt.get('cost_is_estimated') else ''} "
                f"finish={attempt.get('finish_reason')} "
                f"retries={attempt.get('retries')} "
                f"latency={float(attempt.get('latency_seconds') or 0):.2f}s"
            )
        return "\n".join(lines)
