"""The one repair after a *truncated* answer gets more room; a schema error does not.

The reasoning-loop failure mode -- a reasoning model burning the whole output
budget re-deriving its own working and running out before it emits the JSON --
truncates the answer (``finish_reason == "length"``, or an empty answer with the
reasoning channel exhausted). The controller detects that and gives the single
repair an enlarged output budget so it has room to finish; a genuine schema
error keeps the ordinary budget because more room would not help. Both remain
bounded by the cost ledger, which reserves against the budget before every call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from game.ai_commander.audit import AuditLog
from game.ai_commander.controller import RedCommanderTurn
from game.ai_commander.decision import example_decision_json
from game.ai_commander.enums import IntelPolicy
from game.ai_commander.intel import IntelProjector
from game.ai_commander.llmclient import (
    MAX_OUTPUT_TOKENS_CEILING,
    LlmResponse,
    TokenUsage,
)
from tests.ai_commander import fakes


@pytest.fixture(autouse=True)
def _objective_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes.patch_objective_finder(monkeypatch)


def _truncated_by_length() -> LlmResponse:
    """A reply the provider cut off at the output budget."""

    return LlmResponse(
        text="",
        usage=TokenUsage(input_tokens=1000, output_tokens=2000, total_tokens=3000),
        model="test/model",
        finish_reason="length",
        request_id="req-truncated",
        latency_seconds=0.01,
    )


def _empty_with_exhausted_reasoning(budget: int) -> LlmResponse:
    """A reply that stopped 'normally' but spent the whole budget reasoning."""

    return LlmResponse(
        text="",
        usage=TokenUsage(
            input_tokens=1000,
            output_tokens=budget,
            total_tokens=budget + 1000,
            reasoning_tokens=budget,
        ),
        model="test/model",
        finish_reason="stop",
        request_id="req-empty-reasoning",
        latency_seconds=0.01,
    )


def _run(
    script: list[Any], tmp_path: Path, *, max_output_tokens: int = 2000
) -> tuple[Any, fakes.ScriptedClient]:
    _, game = fakes.synthetic_game()
    brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
    resolved = [
        example_decision_json(brief) if item is _VALID else item for item in script
    ]
    client = fakes.ScriptedClient(resolved)
    result = RedCommanderTurn(
        game,
        fakes.make_config(max_output_tokens=max_output_tokens),
        audit_log=AuditLog(tmp_path),
        client=client,
    ).run()
    return result, client


_VALID = object()


class TestTruncationEnlargesTheRepair:
    def test_a_length_truncation_gives_the_repair_a_larger_budget(
        self, tmp_path: Path
    ) -> None:
        result, client = _run(
            [_truncated_by_length(), _VALID], tmp_path, max_output_tokens=2000
        )
        assert result.accepted
        assert len(client.calls) == 2
        initial, repair = client.max_output_tokens_calls
        assert initial == 2000
        assert repair > initial, "the repair after a truncation must get more room"
        assert repair == min(int(2000 * 1.5), MAX_OUTPUT_TOKENS_CEILING)

    def test_an_empty_answer_with_exhausted_reasoning_is_treated_as_truncation(
        self, tmp_path: Path
    ) -> None:
        result, client = _run(
            [_empty_with_exhausted_reasoning(2000), _VALID],
            tmp_path,
            max_output_tokens=2000,
        )
        assert result.accepted
        initial, repair = client.max_output_tokens_calls
        assert repair > initial

    def test_the_enlarged_repair_is_noted_in_the_record(self, tmp_path: Path) -> None:
        result, _ = _run([_truncated_by_length(), _VALID], tmp_path)
        assert any("cut off" in note for note in result.record.notes)


class TestSchemaErrorsDoNotEnlarge:
    def test_a_plain_schema_error_keeps_the_ordinary_budget(
        self, tmp_path: Path
    ) -> None:
        # A complete, non-empty reply that simply is not JSON: finish_reason
        # "stop", so it is a schema error, not a truncation.
        result, client = _run(
            ["I will not answer in JSON.", _VALID], tmp_path, max_output_tokens=2000
        )
        assert result.accepted
        initial, repair = client.max_output_tokens_calls
        assert initial == repair == 2000, "a schema error must not enlarge the budget"

    def test_a_schema_error_note_is_not_the_truncation_note(
        self, tmp_path: Path
    ) -> None:
        result, _ = _run(["not json", _VALID], tmp_path)
        assert any("failed validation" in note for note in result.record.notes)
        assert not any("cut off" in note for note in result.record.notes)


class TestEnlargementIsBounded:
    def test_the_enlarged_budget_never_exceeds_the_ceiling(
        self, tmp_path: Path
    ) -> None:
        # A base already above two-thirds of the ceiling would scale past it;
        # the enlargement must clamp to MAX_OUTPUT_TOKENS_CEILING.
        base = 30000
        result, client = _run(
            [_truncated_by_length(), _VALID], tmp_path, max_output_tokens=base
        )
        assert result.accepted
        _, repair = client.max_output_tokens_calls
        assert repair == MAX_OUTPUT_TOKENS_CEILING
        assert repair <= MAX_OUTPUT_TOKENS_CEILING
