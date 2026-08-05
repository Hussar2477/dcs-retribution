"""Viewer for the LLM commander's decision log.

Auditability is only real if it is reachable, so the same JSON the commander
writes next to the campaign save is presented here: for each decision point, what
the model was told, what it asked for, what was refused and why, and what it
cost.

The window is a pure reader. It never triggers a request, never mutates the
campaign, and works with no API key configured -- an existing log can be reviewed
long after the key has been removed.
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
        return f"Turn {turn} - {state} (${float(cost):.4f})"

    def _clear_views(self) -> None:
        for view in (
            self.summary_view,
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

    def _render_rejections(self, record: dict[str, Any]) -> None:
        self.rejection_tree.clear()
        rejections = record.get("rejections") or []
        for rejection in rejections:
            if not isinstance(rejection, dict):
                continue
            value = rejection.get("value")
            item = QTreeWidgetItem(
                [
                    str(rejection.get("element", "")),
                    str(rejection.get("reason", "")),
                    "" if value is None else json.dumps(value),
                ]
            )
            self.rejection_tree.addTopLevelItem(item)
        for column in range(3):
            self.rejection_tree.resizeColumnToContents(column)
        self.tabs.setTabText(1, f"Rejected ({len(rejections)})")

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
