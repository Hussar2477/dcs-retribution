"""The per-turn spending cap, and the arithmetic underneath it.

Money is reserved at its **worst case** before a request is sent -- the prompt
as measured, plus the full ``max_output_tokens`` the model is allowed to
produce -- and only settled to the real figure afterwards. A request that could
push the turn past the cap is never sent at all, which is the only way a cap can
be honoured when the cost is not known until the reply arrives.

Refusing a call must never break the turn: the controller degrades to
Retribution's own RED automation instead of raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from game.ai_commander.audit import AuditLog
from game.ai_commander.controller import RedCommanderTurn, describe_turn_result
from game.ai_commander.decision import example_decision_json
from game.ai_commander.enums import FallbackReason, IntelPolicy
from game.ai_commander.intel import IntelProjector
from game.ai_commander.pricing import (
    CHARACTERS_PER_TOKEN,
    FALLBACK_INPUT_PRICE_PER_MILLION,
    FALLBACK_OUTPUT_PRICE_PER_MILLION,
    TOKEN_ESTIMATE_SAFETY_FACTOR,
    CostCapExceeded,
    CostLedger,
    ModelCatalog,
    ModelPrice,
    estimate_tokens,
)
from tests.ai_commander import fakes


def price(
    input_per_million: float = 1.0, output_per_million: float = 4.0
) -> ModelPrice:
    return ModelPrice(
        model_id="test/model",
        input_per_million=input_per_million,
        output_per_million=output_per_million,
    )


@pytest.fixture(autouse=True)
def _objective_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes.patch_objective_finder(monkeypatch)


class TestTokenEstimation:
    def test_estimate_is_characters_over_four_with_a_safety_margin(self) -> None:
        text = "x" * 4000
        expected = int(
            -(-len(text) / CHARACTERS_PER_TOKEN * TOKEN_ESTIMATE_SAFETY_FACTOR // 1)
        )
        assert estimate_tokens(text) == expected
        assert estimate_tokens(text) == 1150

    def test_empty_text_costs_nothing(self) -> None:
        assert estimate_tokens("") == 0

    def test_the_estimate_never_rounds_down(self) -> None:
        """Under-counting tokens would let a turn quietly exceed the cap."""

        for length in range(1, 40):
            text = "y" * length
            assert estimate_tokens(text) >= length / CHARACTERS_PER_TOKEN


class TestCostArithmetic:
    def test_cost_is_per_million_tokens(self) -> None:
        assert price(1.0, 4.0).cost_for(1_000_000, 0) == pytest.approx(1.0)
        assert price(1.0, 4.0).cost_for(0, 1_000_000) == pytest.approx(4.0)
        assert price(1.0, 4.0).cost_for(100_000, 10_000) == pytest.approx(0.14)

    def test_a_realistic_turn_on_a_cheap_model(self) -> None:
        """The default model at 100k in / 10k out, as documented."""

        deepseek = price(0.09, 0.18)
        assert deepseek.cost_for(100_000, 10_000) == pytest.approx(0.0108)

    def test_a_realistic_turn_on_an_expensive_model(self) -> None:
        frontier = price(3.00, 15.00)
        assert frontier.cost_for(100_000, 10_000) == pytest.approx(0.45)

    def test_the_fallback_price_is_the_expensive_end_of_the_market(self) -> None:
        """An unknown model is budgeted pessimistically, never as free."""

        fallback = ModelPrice.fallback_for("who/knows")
        assert fallback.is_fallback_estimate
        assert fallback.input_per_million == FALLBACK_INPUT_PRICE_PER_MILLION
        assert fallback.output_per_million == FALLBACK_OUTPUT_PRICE_PER_MILLION

    def test_catalogue_prices_are_converted_from_per_token(self) -> None:
        catalog = ModelCatalog.from_payload(fakes.CATALOG_PAYLOAD, source="test")
        assert catalog.available
        parsed = catalog.price_for("test/model")
        assert not parsed.is_fallback_estimate
        assert parsed.input_per_million == pytest.approx(1.0)
        assert parsed.output_per_million == pytest.approx(4.0)
        assert parsed.context_length == 128000

    def test_an_unavailable_catalogue_still_yields_a_price(self) -> None:
        catalog = ModelCatalog.unavailable("connection refused")
        assert not catalog.available
        assert catalog.price_for("test/model").is_fallback_estimate


class TestTheLedgerRefusesBeforeSpending:
    def test_worst_case_uses_the_full_output_allowance(self) -> None:
        ledger = CostLedger(cap=1.0)
        # 1000 input tokens at $1/M plus 2000 output tokens at $4/M.
        assert ledger.worst_case_cost(price(), 1000, 2000) == pytest.approx(0.009)

    def test_a_call_within_the_cap_is_reserved(self) -> None:
        ledger = CostLedger(cap=1.0)
        reserved = ledger.reserve(price(), 1000, 2000)
        assert reserved == pytest.approx(0.009)
        assert ledger.reserved == pytest.approx(0.009)
        assert ledger.remaining == pytest.approx(1.0 - 0.009)

    def test_a_call_whose_worst_case_exceeds_the_cap_is_refused(self) -> None:
        ledger = CostLedger(cap=0.005)
        assert not ledger.can_afford(price(), 1000, 2000)
        with pytest.raises(CostCapExceeded) as err:
            ledger.reserve(price(), 1000, 2000)
        assert err.value.cap == pytest.approx(0.005)
        assert err.value.would_be_total == pytest.approx(0.009)
        # Nothing was committed by the refused call.
        assert ledger.reserved == 0.0
        assert ledger.committed == 0.0

    def test_earlier_spending_in_the_same_turn_counts_against_the_cap(self) -> None:
        ledger = CostLedger(cap=0.01, already_spent=0.008)
        assert ledger.settled == pytest.approx(0.008)
        with pytest.raises(CostCapExceeded):
            ledger.reserve(price(), 1000, 2000)

    def test_a_second_call_is_refused_once_the_first_used_the_budget(self) -> None:
        ledger = CostLedger(cap=0.010)
        ledger.reserve(price(), 1000, 2000)
        with pytest.raises(CostCapExceeded):
            ledger.reserve(price(), 1000, 2000)

    def test_settling_frees_the_unused_reservation(self) -> None:
        """Reserving the worst case must not permanently consume the cap."""

        ledger = CostLedger(cap=0.015)
        reserved = ledger.reserve(price(), 1000, 2000)
        assert not ledger.can_afford(price(), 1000, 2000), "worst case is held"
        entry = ledger.settle(reserved, price(), 1000, 50)
        assert entry.cost == pytest.approx(0.0012)
        assert ledger.reserved == 0.0
        assert ledger.settled == pytest.approx(0.0012)
        # The repair attempt now fits, because the first reply was short.
        assert ledger.can_afford(price(), 1000, 2000)

    def test_releasing_a_failed_call_costs_nothing(self) -> None:
        ledger = CostLedger(cap=0.010)
        reserved = ledger.reserve(price(), 1000, 2000)
        ledger.release(reserved)
        assert ledger.reserved == 0.0
        assert ledger.settled == 0.0
        assert ledger.remaining == pytest.approx(0.010)

    def test_a_provider_reported_cost_overrides_the_estimate(self) -> None:
        ledger = CostLedger(cap=1.0)
        reserved = ledger.reserve(price(), 1000, 2000)
        entry = ledger.settle(reserved, price(), 1000, 200, provider_cost=0.000123)
        assert entry.reported_by_provider
        assert entry.cost == pytest.approx(0.000123)
        assert ledger.settled == pytest.approx(0.000123)

    def test_a_zero_cap_refuses_everything(self) -> None:
        ledger = CostLedger(cap=0.0)
        with pytest.raises(CostCapExceeded):
            ledger.reserve(price(), 1, 1)


class TestTheControllerFallsBackInsteadOfRaising:
    def test_a_cap_smaller_than_one_call_never_sends_the_call(
        self, tmp_path: Path
    ) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        client = fakes.ScriptedClient([example_decision_json(brief)])
        result = RedCommanderTurn(
            game,
            fakes.make_config(cost_cap_per_turn=0.000001),
            audit_log=AuditLog(tmp_path),
            client=client,
        ).run()

        assert not result.accepted
        assert result.fallback_reason is FallbackReason.COST_CAP
        assert result.directive is None
        assert client.calls == [], "the request must not have been sent"

    def test_the_refusal_is_logged_with_a_reason(self, tmp_path: Path) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        result = RedCommanderTurn(
            game,
            fakes.make_config(cost_cap_per_turn=0.000001),
            audit_log=AuditLog(tmp_path),
            client=fakes.ScriptedClient([example_decision_json(brief)]),
        ).run()

        assert result.log_path is not None
        payload = json.loads(result.log_path.read_text(encoding="utf-8"))
        assert payload["fallback_reason"] == FallbackReason.COST_CAP.value
        assert payload["accepted"] is False
        (rejection,) = [
            r for r in payload["rejections"] if r["element"] == "<cost_cap>"
        ]
        assert "cap" in rejection["reason"]

    def test_a_zero_cap_is_treated_as_do_not_call_at_all(self, tmp_path: Path) -> None:
        _, game = fakes.synthetic_game()
        client = fakes.ScriptedClient([])
        result = RedCommanderTurn(
            game,
            fakes.make_config(cost_cap_per_turn=0.0),
            audit_log=AuditLog(tmp_path),
            client=client,
        ).run()

        assert result.fallback_reason is FallbackReason.COST_CAP
        assert client.calls == []
        assert result.log_path is not None
        payload = json.loads(result.log_path.read_text(encoding="utf-8"))
        assert payload["attempts"] == []

    def test_the_same_turn_is_never_paid_for_twice(self, tmp_path: Path) -> None:
        """``initialize_turn`` can legitimately run more than once.

        A second pass over unchanged state replays the recorded directive
        instead of buying a second opinion.
        """

        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        audit = AuditLog(tmp_path)

        first = RedCommanderTurn(
            game,
            fakes.make_config(),
            audit_log=audit,
            client=fakes.ScriptedClient([example_decision_json(brief)]),
        )
        first_result = first.run()
        assert first_result.accepted
        spent = audit.spent_this_turn(brief.campaign_id_hash, brief.turn_id)
        assert spent > 0.0

        replay_client = fakes.ScriptedClient([example_decision_json(brief)])
        second = RedCommanderTurn(
            game, fakes.make_config(), audit_log=audit, client=replay_client
        )
        second_result = second.run()

        assert replay_client.calls == [], "no second request may be made"
        assert second_result.accepted
        assert second_result.directive is not None
        assert first_result.directive is not None
        assert second_result.directive.to_dict() == first_result.directive.to_dict()
        assert audit.spent_this_turn(brief.campaign_id_hash, brief.turn_id) == spent

    def test_prior_spending_is_seeded_into_the_ledger(self, tmp_path: Path) -> None:
        """A retry after a *changed* state still counts what the turn spent."""

        campaign, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        audit = AuditLog(tmp_path)
        assert (
            RedCommanderTurn(
                game,
                fakes.make_config(),
                audit_log=audit,
                client=fakes.ScriptedClient([example_decision_json(brief)]),
            )
            .run()
            .accepted
        )

        # State moves on inside the same turn, so the replay guard does not fire.
        campaign.red.budget = 4242.0
        moved = IntelProjector(game, IntelPolicy.REALISTIC).project()
        second = RedCommanderTurn(
            game,
            fakes.make_config(),
            audit_log=audit,
            client=fakes.ScriptedClient([example_decision_json(moved)]),
        )
        second.run()
        assert second.ledger is not None
        assert second.record.prior_cost_this_turn > 0.0
        assert second.ledger.settled > second.record.prior_cost_this_turn

    def test_the_default_cap_is_the_documented_ceiling(self) -> None:
        assert fakes.make_config().cost_cap_per_turn == pytest.approx(0.5)

    def test_a_normal_turn_stays_far_below_the_default_cap(
        self, tmp_path: Path
    ) -> None:
        """The measured cost of a real prompt against a real price."""

        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        result = RedCommanderTurn(
            game,
            fakes.make_config(),
            audit_log=AuditLog(tmp_path),
            client=fakes.ScriptedClient([example_decision_json(brief)]),
        ).run()

        assert result.accepted
        described = describe_turn_result(result)
        # $1/M in and $4/M out from the test catalogue, one call.
        assert described["actual_cost"] < 0.5
        assert described["estimated_cost"] < 0.5

    def test_estimated_cost_is_reserved_before_the_reply_is_known(
        self, tmp_path: Path
    ) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        turn = RedCommanderTurn(
            game,
            fakes.make_config(max_output_tokens=2000),
            audit_log=AuditLog(tmp_path),
            client=fakes.ScriptedClient([example_decision_json(brief)]),
        )
        result = turn.run()
        assert result.record is not None
        # The worst case (full output allowance) always exceeds what was used.
        assert result.record.estimated_cost > result.record.actual_cost

    def test_an_unpriced_model_is_budgeted_with_the_fallback_price(
        self, tmp_path: Path
    ) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        turn = RedCommanderTurn(
            game,
            fakes.make_config(model="unknown/model"),
            audit_log=AuditLog(tmp_path),
            client=fakes.ScriptedClient(
                [example_decision_json(brief)], model="unknown/model"
            ),
        )
        turn.run()
        assert turn.price is not None
        assert turn.price.is_fallback_estimate
        assert turn.record.catalog_notes
