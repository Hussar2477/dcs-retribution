"""End-to-end ACTIVE-mode turns through :class:`RedCommanderTurn`.

The other ACTIVE test modules prove the pieces in isolation -- the operations
brief (:mod:`test_operations_brief`), the capability index
(:mod:`test_capabilities`), staged validation (:mod:`test_active_plan`),
legality (:mod:`test_active_legality`) and execution
(:mod:`test_active_execution`). This module wires all three stages together and
drives them through the one public entry point the game uses, asserting the
behaviour the whole feature turns on:

* a clean turn issues exactly three requests, accepts every stage and applies
  RED's own orders through the real systems (faked here only where a full
  theater would be needed);
* the stages degrade **independently and in one direction only** -- a broken
  stage 2 or 3 leaves that slice of the turn to Retribution's built-in
  automation but never undoes work already applied and never aborts the turn,
  while a broken stage 1 costs the whole turn because there is no directive to
  steer the built-in planner with;
* the cost cap is a single turn-wide ledger spanning all three stages, so it can
  be exhausted *mid-hierarchy* and the remaining stages degrade cleanly rather
  than overspending or crashing;
* a stage that spends money genuinely changes campaign state, which a later
  stage's staleness guard then catches -- proving the guard is live, not a
  snapshot.

Every assertion is about the controller's orchestration. The purchase adapters
and the package fulfiller are faked because the real ones need a fully-built
theater, but they are no-ops that never touch the budget, so any campaign-state
change in these tests is one the controller genuinely drove.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import pytest

from game.ai_commander.audit import AuditLog
from game.ai_commander.capabilities import CAPABILITY_CACHE, capability_index_for
from game.ai_commander.controller import RedCommanderTurn, describe_turn_result
from game.ai_commander.decision import example_decision_json
from game.ai_commander.enums import CommanderMode, FallbackReason, IntelPolicy
from game.ai_commander.intel import IntelProjector
from game.ai_commander.operations import OperationsProjector
from game.ai_commander.plan import (
    example_air_tasking_json,
    example_logistics_json,
)
from tests.ai_commander import fakes

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _objective_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes.patch_objective_finder(monkeypatch)


@pytest.fixture(autouse=True)
def _fresh_capability_cache() -> Iterator[None]:
    CAPABILITY_CACHE.clear()
    yield
    CAPABILITY_CACHE.clear()


class _NoopAircraftAdapter:
    """A purchase adapter that never touches the budget.

    Stage 2 executes for real, so the adapters have to exist, but they must not
    change campaign state in these tests -- otherwise the staleness of stage 3
    could not be reasoned about. Only an explicit runway repair moves the budget,
    and that goes through ``coalition.adjust_budget`` directly, not an adapter.
    """

    def __init__(self, control_point: Any) -> None:
        self.control_point = control_point

    def price_of(self, squadron: Any) -> int:
        return int(squadron.aircraft.price)

    def can_buy(self, squadron: Any) -> bool:
        return True

    def buy(self, squadron: Any, quantity: int) -> None:
        pass


class _NoopGroundAdapter:
    def __init__(self, control_point: Any, coalition: Any, game: Any) -> None:
        pass

    def price_of(self, unit_type: Any) -> int:
        return int(unit_type.price)

    def buy(self, unit_type: Any, quantity: int) -> None:
        pass


class _FakePackage:
    def __init__(self, mission: Any) -> None:
        self.mission = mission


class _FakeFulfiller:
    def __init__(
        self, coalition: Any, theater: Any, flights: Any, settings: Any
    ) -> None:
        pass

    def plan_mission(self, mission: Any, count: int, now: Any, tracer: Any) -> Any:
        return _FakePackage(mission)


@pytest.fixture(autouse=True)
def _patch_execution_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "game.purchaseadapter.AircraftPurchaseAdapter", _NoopAircraftAdapter
    )
    monkeypatch.setattr(
        "game.purchaseadapter.GroundUnitPurchaseAdapter", _NoopGroundAdapter
    )
    monkeypatch.setattr(
        "game.commander.packagefulfiller.PackageFulfiller", _FakeFulfiller
    )


class _Turn:
    """Builds a synthetic campaign and the three scripted stage payloads.

    The stage JSON is built from the campaign's *own* identifiers via the
    ``example_*`` helpers, exactly the worked examples the prompts show the
    model, so a scripted "model" that echoes them is a well-behaved model.
    """

    def __init__(self, **game_kwargs: Any) -> None:
        self.campaign, self.game = fakes.synthetic_game(**game_kwargs)
        self.brief = IntelProjector(self.game, IntelPolicy.REALISTIC).project()
        self.operations = OperationsProjector(self.game, IntelPolicy.REALISTIC).project(
            self.brief.campaign_id_hash, self.brief.campaign_revision
        )
        self.capabilities = capability_index_for(self.campaign.red)

    # -- stage payloads ---------------------------------------------------

    def stage1(self) -> str:
        return example_decision_json(self.brief)

    def stage2_no_spend(self) -> str:
        """A logistics plan that changes force posture but spends no money.

        Aircraft/ground purchases go through the faked adapters (no budget
        change); dropping the runway repair keeps the budget -- and therefore
        the campaign revision -- fixed, so stage 3 stays legal.
        """

        plan = example_logistics_json(self.operations, self.capabilities)
        plan.pop("runway_repairs", None)
        plan.pop("ground_orders", None)
        return json.dumps(plan)

    def stage2_full(self) -> str:
        """A logistics plan whose runway repair debits the budget for real."""

        return json.dumps(example_logistics_json(self.operations, self.capabilities))

    def stage3(self) -> str:
        return json.dumps(example_air_tasking_json(self.operations, self.capabilities))

    # -- driving ----------------------------------------------------------

    def run(self, script: list[object], tmp_path: Any, **config_kwargs: Any) -> Any:
        client = fakes.ScriptedClient(script)
        config = fakes.make_config(mode=CommanderMode.ACTIVE, **config_kwargs)
        result = RedCommanderTurn(
            self.game, config, audit_log=AuditLog(tmp_path), client=client
        ).run()
        return result, client


def _logged(result: Any) -> dict[str, Any]:
    assert result.log_path is not None
    return json.loads(result.log_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The clean, everything-works turn
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_a_clean_turn_runs_three_stages_and_applies_orders(
        self, tmp_path: Any
    ) -> None:
        turn = _Turn()
        result, client = turn.run(
            [turn.stage1(), turn.stage2_no_spend(), turn.stage3()], tmp_path
        )
        assert result.accepted
        assert result.fallback_reason is None
        # Exactly one request per stage: no wasted repair calls.
        assert len(client.calls) == 3
        # RED's own orders were applied through the game's systems.
        assert result.execution is not None
        assert result.execution.applied_count >= 1
        assert result.execution.packages_added == 2

    def test_the_summary_reports_every_stage_accepted(self, tmp_path: Any) -> None:
        turn = _Turn()
        result, _ = turn.run(
            [turn.stage1(), turn.stage2_no_spend(), turn.stage3()], tmp_path
        )
        summary = describe_turn_result(result)
        assert summary["mode"] == "active"
        assert summary["requests"] == 3
        assert summary["stages"] == {
            "command": "accepted",
            "logistics": "accepted",
            "air_tasking": "accepted",
        }
        assert summary["orders_applied"] >= 1
        assert summary["packages_added"] == 2

    def test_the_execution_report_is_persisted_in_the_audit_log(
        self, tmp_path: Any
    ) -> None:
        turn = _Turn()
        result, _ = turn.run(
            [turn.stage1(), turn.stage2_no_spend(), turn.stage3()], tmp_path
        )
        logged = _logged(result)
        assert "execution_report" in logged
        report = logged["execution_report"]
        assert set(report) >= {"applied", "failed", "packages_added", "orders"}
        assert report["packages_added"] == 2


# ---------------------------------------------------------------------------
# Independent, one-directional degradation
# ---------------------------------------------------------------------------


class TestStageDegradation:
    def test_a_broken_logistics_stage_does_not_cost_the_turn(
        self, tmp_path: Any
    ) -> None:
        # Stage 2 is malformed twice (initial + one repair), so it degrades to
        # the built-in procurement AI -- but stage 1's directive stands and
        # stage 3 still runs and is accepted.
        turn = _Turn()
        result, client = turn.run(
            [turn.stage1(), "not json", "still not json", turn.stage3()],
            tmp_path,
        )
        assert result.accepted
        assert result.fallback_reason is None
        assert len(client.calls) == 4  # stage2 burned a repair call
        summary = describe_turn_result(result)
        assert summary["stages"]["command"] == "accepted"
        assert summary["stages"]["logistics"] == "malformed_response"
        assert summary["stages"]["air_tasking"] == "accepted"

    def test_a_broken_air_tasking_stage_leaves_packages_to_the_planner(
        self, tmp_path: Any
    ) -> None:
        turn = _Turn()
        result, client = turn.run(
            [turn.stage1(), turn.stage2_no_spend(), "not json", "still not json"],
            tmp_path,
        )
        assert result.accepted
        assert result.fallback_reason is None
        assert result.execution is not None
        # No AI packages were added; Retribution's mission planner takes over.
        assert result.execution.packages_added == 0
        summary = describe_turn_result(result)
        assert summary["stages"]["logistics"] == "accepted"
        assert summary["stages"]["air_tasking"] == "malformed_response"

    def test_a_broken_command_stage_costs_the_whole_turn(self, tmp_path: Any) -> None:
        # With no directive there is nothing to steer the built-in planner, so
        # the entire turn correctly falls back to stock automation.
        turn = _Turn()
        result, client = turn.run(["not json", "still not json"], tmp_path)
        assert not result.accepted
        assert result.fallback_reason is FallbackReason.MALFORMED_RESPONSE
        assert result.execution is None
        # Stages 2 and 3 were never even attempted.
        assert len(client.calls) == 2

    def test_a_transport_failure_in_stage_one_falls_back(self, tmp_path: Any) -> None:
        from game.ai_commander.llmclient import LlmTimeout

        turn = _Turn()
        result, client = turn.run([LlmTimeout("slow")], tmp_path)
        assert not result.accepted
        assert result.fallback_reason is FallbackReason.TIMEOUT
        assert result.execution is None


# ---------------------------------------------------------------------------
# A later stage sees the state an earlier stage changed (live staleness guard)
# ---------------------------------------------------------------------------


class TestLiveStalenessGuard:
    def test_the_commanders_own_spending_no_longer_makes_stage_three_stale(
        self, tmp_path: Any
    ) -> None:
        # The full logistics plan repairs a runway, which debits the budget and
        # therefore changes the campaign revision. Stage 3's payload still
        # carries the revision from the start of the turn, but the guard is
        # re-baselined after the commander's OWN applied logistics, so the live
        # revision matches the refreshed baseline and stage 3 is accepted --
        # the commander's own spending must not trip the staleness check.
        turn = _Turn()
        result, _ = turn.run(
            [turn.stage1(), turn.stage2_full(), turn.stage3()], tmp_path
        )
        assert result.accepted
        assert result.execution is not None
        assert result.execution.spent == pytest.approx(100.0)
        summary = describe_turn_result(result)
        assert summary["stages"]["logistics"] == "accepted"
        assert summary["stages"]["air_tasking"] == "accepted"
        assert summary["packages_added"] >= 1

    def test_an_external_change_after_logistics_still_rejects_stage_three(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Anti-tamper is preserved: if campaign state changes for a reason NOT
        # explained by the commander's own applied orders (simulated here by an
        # external budget mutation applied *after* the refreshed baseline is
        # taken but *before* the air tasking check), stage 3 is still rejected
        # as stale and its packages are left to the built-in planner.
        turn = _Turn()

        from game.ai_commander import controller as controller_module

        original = controller_module.RedCommanderTurn._refreshed_revision

        def tampering_refresh(self: Any) -> Any:
            # Take the genuine refreshed baseline, then simulate an external
            # actor mutating the campaign afterwards so the live revision the
            # air tasking check recomputes no longer matches the baseline.
            baseline = original(self)
            self.coalition.adjust_budget(12345.0)
            return baseline

        monkeypatch.setattr(
            controller_module.RedCommanderTurn,
            "_refreshed_revision",
            tampering_refresh,
        )

        result, _ = turn.run(
            [turn.stage1(), turn.stage2_full(), turn.stage3()], tmp_path
        )
        assert result.accepted  # the turn still stands on stage 1 + stage 2
        summary = describe_turn_result(result)
        assert summary["stages"]["logistics"] == "accepted"
        assert summary["stages"]["air_tasking"] == "stale_response"
        assert summary["packages_added"] == 0


# ---------------------------------------------------------------------------
# One turn-wide cost ledger across all three stages
# ---------------------------------------------------------------------------


class TestTurnWideCostLedger:
    def test_the_cost_of_all_three_stages_accumulates(self, tmp_path: Any) -> None:
        turn = _Turn()
        result, _ = turn.run(
            [turn.stage1(), turn.stage2_no_spend(), turn.stage3()], tmp_path
        )
        record = result.record
        assert record is not None
        # Three billed calls at the catalogue's per-call actual cost.
        assert record.actual_cost == pytest.approx(3 * 0.0018)
        # The worst-case reservation is always at least the actual spend.
        assert record.estimated_cost >= record.actual_cost
        # Per-stage accounting sums to the turn total.
        per_stage = sum(stage.actual_cost for stage in record.stages)
        assert per_stage == pytest.approx(record.actual_cost)

    def test_the_cap_can_be_exhausted_mid_hierarchy(self, tmp_path: Any) -> None:
        # A cap large enough for stage 1 but not for a second call: stage 1 is
        # accepted, and stages 2 and 3 each hit the cap and degrade cleanly.
        turn = _Turn()
        result, client = turn.run(
            [turn.stage1(), turn.stage2_no_spend(), turn.stage3()],
            tmp_path,
            cost_cap_per_turn=0.012,
        )
        assert result.accepted  # stage 1's directive still stands
        assert result.fallback_reason is None
        summary = describe_turn_result(result)
        assert summary["stages"]["command"] == "accepted"
        assert summary["stages"]["logistics"] == "cost_cap"
        assert summary["stages"]["air_tasking"] == "cost_cap"
        # Only stage 1 was billed; nothing was applied.
        assert result.record is not None
        assert result.record.actual_cost == pytest.approx(0.0018)
        assert result.execution is not None
        assert result.execution.applied_count == 0
        assert result.execution.packages_added == 0

    def test_the_cap_never_overspends_even_across_stages(self, tmp_path: Any) -> None:
        turn = _Turn()
        result, _ = turn.run(
            [turn.stage1(), turn.stage2_no_spend(), turn.stage3()],
            tmp_path,
            cost_cap_per_turn=0.012,
        )
        assert result.record is not None
        assert result.record.actual_cost <= 0.012
