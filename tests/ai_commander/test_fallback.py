"""A turn must never break because the model did.

The commander is an optional advisor bolted onto a game that already knows how
to play RED. Every failure mode -- no network, a slow endpoint, an HTTP error, a
model that will not produce JSON, a reply that arrived too late to be valid --
resolves the same way: a recorded fallback reason and Retribution's built-in RED
automation keeping control of the turn.

``RedCommanderTurn.run`` is documented as never raising, so these tests also
cover the paths where an exception escapes something it calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from game.ai_commander.audit import AuditLog
from game.ai_commander.controller import (
    FALLBACK_TO_BUILTIN,
    FALLBACK_TO_PREVIOUS,
    RedCommanderTurn,
    describe_turn_result,
)
from game.ai_commander.decision import (
    DECISION_SCHEMA_VERSION,
    example_decision_json,
)
from game.ai_commander.enums import (
    FallbackReason,
    FrontPosture,
    IntelPolicy,
    ProcurementCategory,
    RedStrategy,
    ReservePolicy,
)
from game.ai_commander.directive import build_directive
from game.ai_commander.intel import IntelProjector, RedCommanderBrief
from game.ai_commander.llmclient import (
    LlmHttpError,
    LlmTimeout,
    LlmTransportError,
    TokenUsage,
)
from tests.ai_commander import fakes


@pytest.fixture(autouse=True)
def _objective_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes.patch_objective_finder(monkeypatch)


def run_turn(
    script: list[Any],
    tmp_path: Path,
    *,
    turn: int = 7,
    **config_kwargs: Any,
) -> tuple[Any, fakes.ScriptedClient, RedCommanderBrief]:
    campaign, game = fakes.synthetic_game(turn=turn)
    brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
    resolved = [
        example_decision_json(brief) if item is VALID else item for item in script
    ]
    client = fakes.ScriptedClient(resolved)
    result = RedCommanderTurn(
        game,
        fakes.make_config(**config_kwargs),
        audit_log=AuditLog(tmp_path),
        client=client,
    ).run()
    return result, client, brief


#: Placeholder replaced by a plan that is valid for the brief under test.
VALID = object()


def logged(result: Any) -> dict[str, Any]:
    assert result.log_path is not None
    payload: dict[str, Any] = json.loads(result.log_path.read_text(encoding="utf-8"))
    return payload


class TestTransportFailures:
    def test_a_connection_failure_falls_back(self, tmp_path: Path) -> None:
        result, client, _ = run_turn(
            [LlmTransportError("connection refused")], tmp_path
        )
        assert not result.accepted
        assert result.fallback_reason is FallbackReason.TRANSPORT_ERROR
        assert len(client.calls) == 1, "no repair attempt after a transport failure"

    def test_a_timeout_falls_back(self, tmp_path: Path) -> None:
        result, _, _ = run_turn([LlmTimeout("timed out after 90s")], tmp_path)
        assert result.fallback_reason is FallbackReason.TIMEOUT
        assert result.directive is None

    def test_an_http_error_falls_back_and_records_the_status(
        self, tmp_path: Path
    ) -> None:
        result, _, _ = run_turn([LlmHttpError(401, "invalid api key")], tmp_path)
        assert result.fallback_reason is FallbackReason.HTTP_ERROR
        payload = logged(result)
        (attempt,) = payload["attempts"]
        assert attempt["http_status"] == 401
        assert "invalid api key" in attempt["error"]

    def test_a_failed_call_does_not_consume_the_turn_budget(
        self, tmp_path: Path
    ) -> None:
        result, _, _ = run_turn([LlmTransportError("reset by peer")], tmp_path)
        payload = logged(result)
        assert payload["actual_cost"] == 0.0
        assert payload["reserved_cost"] == pytest.approx(0.0)

    def test_a_missing_price_catalogue_does_not_stop_the_turn(
        self, tmp_path: Path
    ) -> None:
        campaign, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        client = fakes.ScriptedClient(
            [example_decision_json(brief)],
            catalog_error=LlmTransportError("catalogue unreachable"),
        )
        turn = RedCommanderTurn(
            game, fakes.make_config(), audit_log=AuditLog(tmp_path), client=client
        )
        result = turn.run()
        assert result.accepted
        assert turn.price is not None
        assert turn.price.is_fallback_estimate
        assert any("catalogue unavailable" in n for n in turn.record.catalog_notes)


class TestMalformedOutput:
    def test_one_malformed_reply_earns_one_repair_attempt(self, tmp_path: Path) -> None:
        result, client, _ = run_turn(
            ["I am not going to answer in JSON.", VALID], tmp_path
        )
        assert result.accepted
        assert len(client.calls) == 2
        repair = client.calls[1]
        assert repair[-2]["role"] == "assistant"
        assert repair[-1]["role"] == "user"
        assert any("repair attempt" in note for note in result.record.notes)

    def test_repeatedly_malformed_output_falls_back(self, tmp_path: Path) -> None:
        result, client, _ = run_turn(["not json", "still not json"], tmp_path)
        assert not result.accepted
        assert result.fallback_reason is FallbackReason.MALFORMED_RESPONSE
        assert len(client.calls) == 2, "exactly one repair, never an unbounded loop"

    def test_the_reason_for_each_malformed_reply_is_logged(
        self, tmp_path: Path
    ) -> None:
        result, _, _ = run_turn(["not json", "still not json"], tmp_path)
        payload = logged(result)
        reasons = [r["reason"] for r in payload["rejections"]]
        assert any("did not contain a JSON object" in reason for reason in reasons)

    def test_an_empty_reply_falls_back(self, tmp_path: Path) -> None:
        result, _, _ = run_turn(["", ""], tmp_path)
        assert result.fallback_reason is FallbackReason.MALFORMED_RESPONSE

    def test_a_reply_for_the_wrong_campaign_state_falls_back(
        self, tmp_path: Path
    ) -> None:
        bad = json.dumps(
            {
                "schema_version": DECISION_SCHEMA_VERSION,
                "campaign_revision": "ffffffffffffffff",
                "turn_id": 7,
                "strategy": RedStrategy.DEFEND.value,
                "reserve_policy": ReservePolicy.BALANCED.value,
            }
        )
        result, client, _ = run_turn([bad, bad], tmp_path)
        assert result.fallback_reason is FallbackReason.MALFORMED_RESPONSE
        assert len(client.calls) == 2

    def test_a_schema_clean_reply_with_nothing_legal_falls_back(
        self, tmp_path: Path
    ) -> None:
        campaign, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        # Schema-legal, but every element names something that is not in the
        # brief, so nothing survives.
        reply = json.dumps(
            {
                "schema_version": DECISION_SCHEMA_VERSION,
                "campaign_revision": brief.campaign_revision,
                "turn_id": brief.turn_id,
                "strategy": RedStrategy.DEFEND.value,
                "reserve_policy": ReservePolicy.BALANCED.value,
                "front_priorities": [{"front_id": "FRONT-99", "rank": 1}],
                "spending_priorities": [{"category_id": "PROC-99", "rank": 1}],
            }
        )
        result = RedCommanderTurn(
            game,
            fakes.make_config(),
            audit_log=AuditLog(tmp_path),
            client=fakes.ScriptedClient([reply, reply]),
        ).run()
        assert not result.accepted
        assert result.fallback_reason is FallbackReason.NO_LEGAL_CONTENT

    def test_state_moving_under_a_valid_reply_is_reported_as_stale(
        self, tmp_path: Path
    ) -> None:
        campaign, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        reply = example_decision_json(brief)

        class MovingClient(fakes.ScriptedClient):
            def complete(self, *args: Any, **kwargs: Any) -> Any:
                response = super().complete(*args, **kwargs)
                # A concurrent purchase lands while the request is in flight.
                campaign.red.budget = 11.0
                return response

        result = RedCommanderTurn(
            game,
            fakes.make_config(),
            audit_log=AuditLog(tmp_path),
            client=MovingClient([reply]),
        ).run()
        assert not result.accepted
        assert result.fallback_reason is FallbackReason.STALE_RESPONSE

    def test_tool_calls_are_recorded_and_never_executed(self, tmp_path: Path) -> None:
        campaign, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        client = fakes.ScriptedClient(
            [example_decision_json(brief)], had_tool_calls=True
        )
        result = RedCommanderTurn(
            game, fakes.make_config(), audit_log=AuditLog(tmp_path), client=client
        ).run()
        payload = logged(result)
        assert any(r["element"] == "<tool_calls>" for r in payload["rejections"])
        # The decision itself was still usable, so the turn is not wasted.
        assert result.accepted


class TestConfigurationFailures:
    def test_a_disabled_commander_does_nothing_at_all(self, tmp_path: Path) -> None:
        _, game = fakes.synthetic_game()
        client = fakes.ScriptedClient([])
        result = RedCommanderTurn(
            game,
            fakes.make_config(enabled=False),
            audit_log=AuditLog(tmp_path),
            client=client,
        ).run()

        assert result.fallback_reason is FallbackReason.DISABLED
        assert result.directive is None
        assert result.record is None
        assert result.log_path is None
        assert client.calls == []
        assert list(tmp_path.rglob("*.json")) == [], "no log for a disabled feature"

    def test_a_missing_api_key_falls_back_without_calling_out(
        self, tmp_path: Path
    ) -> None:
        _, game = fakes.synthetic_game()
        client = fakes.ScriptedClient([])
        result = RedCommanderTurn(
            game,
            fakes.make_config(api_key=None),
            audit_log=AuditLog(tmp_path),
            client=client,
        ).run()

        assert result.fallback_reason is FallbackReason.NOT_CONFIGURED
        assert client.calls == []
        payload = logged(result)
        assert any(r["element"] == "<configuration>" for r in payload["rejections"])

    def test_a_local_endpoint_needs_no_key(self, tmp_path: Path) -> None:
        """Ollama and friends are reachable without a credential."""

        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        result = RedCommanderTurn(
            game,
            fakes.make_config(api_key=None, base_url="http://localhost:11434/v1"),
            audit_log=AuditLog(tmp_path),
            client=fakes.ScriptedClient([example_decision_json(brief)]),
        ).run()
        assert result.accepted


class TestFallbackNeverRaises:
    def test_a_broken_projection_becomes_a_logged_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, game = fakes.synthetic_game()

        def explode(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("something in the campaign is inconsistent")

        monkeypatch.setattr(IntelProjector, "project", explode)
        result = RedCommanderTurn(
            game,
            fakes.make_config(),
            audit_log=AuditLog(tmp_path),
            client=fakes.ScriptedClient([]),
        ).run()

        assert result.fallback_reason is FallbackReason.UNEXPECTED_ERROR
        assert result.directive is None

    def test_a_client_raising_an_unexpected_error_is_contained(
        self, tmp_path: Path
    ) -> None:
        result, _, _ = run_turn([ValueError("a bug in the transport")], tmp_path)
        assert result.fallback_reason is FallbackReason.UNEXPECTED_ERROR
        assert result.directive is None

    def test_an_unwritable_log_directory_does_not_break_the_turn(
        self, tmp_path: Path
    ) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("this is a file", encoding="utf-8")
        result = RedCommanderTurn(
            game,
            fakes.make_config(),
            audit_log=AuditLog(blocked),
            client=fakes.ScriptedClient([example_decision_json(brief)]),
        ).run()

        assert result.accepted, "the decision still applies even if it cannot be logged"
        assert result.log_path is None

    def test_the_turn_runs_with_no_audit_log_at_all(self) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        result = RedCommanderTurn(
            game,
            fakes.make_config(),
            audit_log=None,
            client=fakes.ScriptedClient([example_decision_json(brief)]),
        ).run()
        assert result.accepted
        assert result.log_path is None


class TestFallbackPolicy:
    def test_the_default_policy_hands_the_turn_to_the_builtin_automation(
        self, tmp_path: Path
    ) -> None:
        result, _, _ = run_turn([LlmTimeout("nope")], tmp_path)
        payload = logged(result)
        assert payload["fallback_policy"] == FALLBACK_TO_BUILTIN
        assert payload["accepted_directive"] is None

    def test_keeping_the_previous_directive_rechecks_it(self, tmp_path: Path) -> None:
        campaign, game = fakes.synthetic_game(turn=9)
        audit = AuditLog(tmp_path)
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()

        # Seed an accepted directive from an earlier turn.
        earlier = RedCommanderTurn(
            game,
            fakes.make_config(),
            audit_log=audit,
            client=fakes.ScriptedClient([example_decision_json(brief)]),
        )
        seeded = earlier.run()
        assert seeded.accepted
        assert seeded.record is not None
        seeded.record.turn_id = brief.turn_id - 2

        # Rewrite the record under an earlier turn number so it is "previous".
        path = seeded.log_path
        assert path is not None
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["turn_id"] = brief.turn_id - 2
        (path.parent / f"turn_{brief.turn_id - 2:04d}_00.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        path.unlink()

        result = RedCommanderTurn(
            game,
            fakes.make_config(fallback_to_builtin=False),
            audit_log=audit,
            client=fakes.ScriptedClient([LlmTimeout("still down")]),
        ).run()

        assert result.fallback_reason is FallbackReason.TIMEOUT
        assert result.directive is not None, "the previous strategy is kept"
        assert result.directive.turn_id == brief.turn_id
        assert result.directive.campaign_revision == brief.campaign_revision
        assert logged(result)["fallback_policy"] == FALLBACK_TO_PREVIOUS

    def test_with_no_previous_directive_it_still_falls_back_to_builtin(
        self, tmp_path: Path
    ) -> None:
        result, _, _ = run_turn(
            [LlmTimeout("down")], tmp_path, fallback_to_builtin=False
        )
        assert result.directive is None
        assert logged(result)["fallback_policy"] == FALLBACK_TO_BUILTIN

    def test_carrying_forward_drops_what_is_no_longer_legal(self) -> None:
        """Keeping the previous strategy is not the same as keeping its orders."""

        campaign, game = fakes.synthetic_game(red_deployable=40)
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        from game.ai_commander.legality import LegalityChecker

        previous = build_directive(
            turn_id=brief.turn_id - 1,
            campaign_revision="whatever-it-was",
            strategy=RedStrategy.GROUND_OFFENSIVE,
            reserve_policy=ReservePolicy.COMMIT_EVERYTHING,
            postures=(
                (("RED-FRONT-BASE", "BLUE-FRONT-BASE"), FrontPosture.BREAKTHROUGH),
            ),
            procurement=(ProcurementCategory.AIRCRAFT,),
        )
        directive, rejections = LegalityChecker(game, brief).carry_forward(previous)
        assert directive is not None
        assert directive.front_postures == {}
        assert directive.procurement_order == (ProcurementCategory.AIRCRAFT,)
        assert rejections


class TestResultReporting:
    def test_describe_turn_result_summarises_a_success(self, tmp_path: Path) -> None:
        result, _, _ = run_turn([VALID], tmp_path)
        described = describe_turn_result(result)
        assert described["accepted"] is True
        assert described["fallback_reason"] is None
        assert described["prompt_tokens"] > 0
        assert described["completion_tokens"] > 0
        assert described["rejections"] == 0

    def test_describe_turn_result_summarises_a_fallback(self, tmp_path: Path) -> None:
        result, _, _ = run_turn([LlmTransportError("no route to host")], tmp_path)
        described = describe_turn_result(result)
        assert described["accepted"] is False
        assert described["fallback_reason"] == FallbackReason.TRANSPORT_ERROR.value

    def test_provider_reported_usage_is_preferred_over_the_estimate(
        self, tmp_path: Path
    ) -> None:
        campaign, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        client = fakes.ScriptedClient(
            [example_decision_json(brief)],
            usage=TokenUsage(
                input_tokens=4321, output_tokens=123, total_tokens=4444, cost=0.00042
            ),
        )
        result = RedCommanderTurn(
            game, fakes.make_config(), audit_log=AuditLog(tmp_path), client=client
        ).run()
        assert result.record is not None
        assert result.record.total_prompt_tokens == 4321
        assert result.record.total_completion_tokens == 123
        assert result.record.actual_cost == pytest.approx(0.00042)
        payload = logged(result)
        (attempt,) = payload["attempts"]
        assert attempt["cost_is_estimated"] is False
