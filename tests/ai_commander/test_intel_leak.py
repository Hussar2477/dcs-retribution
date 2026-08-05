"""The intel filter must not leak BLUE-private information to the RED model.

These are the most important tests in the package. Every other guarantee the
commander makes is a matter of the campaign staying consistent; this one is the
difference between an opponent that plays the same game the human plays and one
that cheats.

The approach is deliberately blunt. Rather than asserting on a list of field
names -- which a future field would silently escape -- the synthetic campaign
tags every BLUE-private value with a sentinel that appears nowhere else, the
brief (and the prompt, and the schema, and the audit record written to disk) is
serialised in full, and the whole blob is scanned for every sentinel.

RED's own equivalents are sentinels too, and are asserted *present*. Without
that, a filter that returned an empty brief would pass every leak test here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from game.ai_commander.audit import AuditLog
from game.ai_commander.controller import RedCommanderTurn
from game.ai_commander.decision import decision_json_schema, example_decision_json
from game.ai_commander.enums import CommanderPersonality, IntelPolicy
from game.ai_commander.intel import (
    FULL_PARITY_WITHHELD_FIELDS,
    REALISTIC_WITHHELD_FIELDS,
    IntelProjector,
    RedCommanderBrief,
)
from game.ai_commander.prompt import build_messages
from tests.ai_commander import fakes


def _brief(policy: IntelPolicy) -> tuple[fakes.SyntheticCampaign, RedCommanderBrief]:
    campaign, game = fakes.synthetic_game()
    return campaign, IntelProjector(game, policy).project()


def _everything_the_model_sees(brief: RedCommanderBrief) -> str:
    """Serialise every artefact derived from the brief that reaches the model.

    The structured brief, the compact rendering embedded in the prompt, the
    fully assembled chat messages, the response schema and the worked example
    are all built from the same projection, and any one of them leaking would
    be a leak.
    """

    return fakes.serialise_everything(
        brief.to_dict(),
        brief.render_compact(),
        build_messages(brief, CommanderPersonality.BALANCED),
        build_messages(brief, CommanderPersonality.AGGRESSIVE),
        decision_json_schema(brief),
        example_decision_json(brief),
    )


@pytest.fixture(autouse=True)
def _objective_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes.patch_objective_finder(monkeypatch)


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test in this module may pick up a developer's real API key."""

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


class TestRealisticBriefWithholdsBlueInformation:
    def test_no_blue_sentinel_appears_anywhere(self) -> None:
        _, brief = _brief(IntelPolicy.REALISTIC)
        assert fakes.blue_leaks_in(_everything_the_model_sees(brief)) == []

    def test_red_still_sees_its_own_equivalents(self) -> None:
        """Guards against a filter that passes by returning nothing useful."""

        _, brief = _brief(IntelPolicy.REALISTIC)
        assert fakes.red_facts_missing_from(_everything_the_model_sees(brief)) == []

    def test_blue_economy_is_absent_from_the_structured_brief(self) -> None:
        _, brief = _brief(IntelPolicy.REALISTIC)
        payload = json.dumps(brief.to_dict())
        assert str(fakes.BLUE_SENTINELS["blue_budget"]) not in payload
        assert str(fakes.BLUE_SENTINELS["blue_income_per_turn"]) not in payload
        # RED's own economy is reported, so the absence above is a filter and
        # not an empty projection.
        assert brief.red_resources.budget_available == pytest.approx(
            float(str(fakes.RED_SENTINELS["red_budget"]))
        )

    def test_no_blue_squadron_or_pilot_information(self) -> None:
        _, brief = _brief(IntelPolicy.REALISTIC)
        blob = _everything_the_model_sees(brief)
        assert str(fakes.BLUE_SENTINELS["blue_squadron_name"]) not in blob
        assert str(fakes.BLUE_SENTINELS["blue_aircraft_name"]) not in blob
        assert str(fakes.BLUE_SENTINELS["blue_pilot_count"]) not in blob

    def test_no_blue_planned_packages(self) -> None:
        _, brief = _brief(IntelPolicy.REALISTIC)
        assert str(
            fakes.BLUE_SENTINELS["blue_planned_package"]
        ) not in _everything_the_model_sees(brief)

    def test_no_blue_base_capacity_internals(self) -> None:
        _, brief = _brief(IntelPolicy.REALISTIC)
        blob = _everything_the_model_sees(brief)
        assert str(fakes.BLUE_SENTINELS["blue_unit_capacity"]) not in blob
        assert str(fakes.BLUE_SENTINELS["blue_deployable_units"]) not in blob

    def test_enemy_front_strength_is_a_band_not_a_count(self) -> None:
        _, brief = _brief(IntelPolicy.REALISTIC)
        (front,) = brief.fronts
        assert front.enemy_unit_count is None
        assert front.enemy_strength.value == "weaker"
        # RED's own numbers on the same front are exact.
        assert front.own_deployable_units == fakes.RED_SENTINELS["red_deployable_units"]
        assert front.own_unit_capacity == fakes.RED_SENTINELS["red_unit_capacity"]

    def test_undetected_blue_units_are_not_counted(self) -> None:
        """Only the SAM sites inside RED's observation range are reported."""

        campaign, brief = _brief(IntelPolicy.REALISTIC)
        air_defences = next(
            target
            for target in brief.known_target_sets
            if target.category.value == "enemy_air_defences"
        )
        assert air_defences.known_count == campaign.NEAR_IADS
        assert (
            air_defences.known_count
            < campaign.NEAR_IADS + campaign.FAR_IADS  # the finder offered all of them
        )

    def test_blue_bases_outside_observation_range_are_hidden(self) -> None:
        _, brief = _brief(IntelPolicy.REALISTIC)
        blob = _everything_the_model_sees(brief)
        assert str(fakes.BLUE_SENTINELS["blue_hidden_base"]) not in blob
        assert str(fakes.BLUE_SENTINELS["blue_undetected_tgo"]) not in blob
        # The base RED is actually fighting is public and must still be named,
        # otherwise the brief would be useless.
        assert fakes.PUBLIC_BLUE_FRONT_BASE in blob

    def test_target_sets_carry_no_coordinates(self) -> None:
        _, brief = _brief(IntelPolicy.REALISTIC)
        for target in brief.known_target_sets:
            assert target.location_precision.value == "area"
            assert target.confidence.value == "probable"

    def test_withheld_fields_are_declared_to_the_model(self) -> None:
        _, brief = _brief(IntelPolicy.REALISTIC)
        assert brief.withheld_fields == REALISTIC_WITHHELD_FIELDS
        rendered = brief.render_compact()
        for field_name in REALISTIC_WITHHELD_FIELDS:
            assert field_name in rendered


class TestFullParityIsTheDeliberateContrast:
    """``FULL_PARITY`` exists to make the restriction visible and auditable."""

    def test_full_parity_reports_exact_enemy_counts(self) -> None:
        _, brief = _brief(IntelPolicy.FULL_PARITY)
        (front,) = brief.fronts
        assert front.enemy_unit_count == fakes.BLUE_SENTINELS["blue_deployable_units"]

    def test_full_parity_sees_every_air_defence(self) -> None:
        campaign, brief = _brief(IntelPolicy.FULL_PARITY)
        air_defences = next(
            target
            for target in brief.known_target_sets
            if target.category.value == "enemy_air_defences"
        )
        assert air_defences.known_count == campaign.NEAR_IADS + campaign.FAR_IADS

    def test_full_parity_sees_every_enemy_base(self) -> None:
        _, realistic = _brief(IntelPolicy.REALISTIC)
        _, parity = _brief(IntelPolicy.FULL_PARITY)

        def airbases(brief: RedCommanderBrief) -> int:
            return next(
                target.known_count
                for target in brief.known_target_sets
                if target.category.value == "enemy_airbases"
            )

        assert airbases(realistic) == 1
        assert airbases(parity) == 2

    def test_full_parity_still_withholds_engine_internals(self) -> None:
        """Even at parity the model never gets flight plans or save data."""

        _, brief = _brief(IntelPolicy.FULL_PARITY)
        assert brief.withheld_fields == FULL_PARITY_WITHHELD_FIELDS
        blob = _everything_the_model_sees(brief)
        assert str(fakes.BLUE_SENTINELS["blue_planned_package"]) not in blob
        assert str(fakes.BLUE_SENTINELS["blue_budget"]) not in blob
        assert str(fakes.BLUE_SENTINELS["blue_squadron_name"]) not in blob

    def test_the_two_policies_actually_differ(self) -> None:
        """A guard against both policies collapsing to the same projection."""

        _, realistic = _brief(IntelPolicy.REALISTIC)
        _, parity = _brief(IntelPolicy.FULL_PARITY)
        assert realistic.to_dict() != parity.to_dict()
        assert fakes.blue_leaks_in(_everything_the_model_sees(parity)) == [
            "blue_deployable_units"
        ]


class TestTheAuditTrailDoesNotLeakEither:
    """The decision log is written to disk and is human-readable, so the same
    rule applies to it: it records what RED was told, not what RED could not
    see."""

    def test_written_record_contains_no_blue_sentinel(self, tmp_path: Path) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        client = fakes.ScriptedClient([example_decision_json(brief)])
        result = RedCommanderTurn(
            game,
            fakes.make_config(),
            audit_log=AuditLog(tmp_path),
            client=client,
        ).run()

        assert result.accepted
        assert result.log_path is not None
        written = result.log_path.read_text(encoding="utf-8")
        assert fakes.blue_leaks_in(written) == []
        # The record is genuinely populated -- it holds RED's own state.
        assert fakes.red_facts_missing_from(written) == []

    def test_logged_prompt_is_the_prompt_that_was_sent(self, tmp_path: Path) -> None:
        """Auditing a turn is only meaningful if the log is the real prompt."""

        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        client = fakes.ScriptedClient([example_decision_json(brief)])
        result = RedCommanderTurn(
            game,
            fakes.make_config(log_prompts=True),
            audit_log=AuditLog(tmp_path),
            client=client,
        ).run()

        assert result.log_path is not None
        payload: dict[str, Any] = json.loads(
            result.log_path.read_text(encoding="utf-8")
        )
        (attempt,) = payload["attempts"]
        assert attempt["prompt_messages"] == client.calls[0]

    def test_prompt_logging_can_be_switched_off(self, tmp_path: Path) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        result = RedCommanderTurn(
            game,
            fakes.make_config(log_prompts=False),
            audit_log=AuditLog(tmp_path),
            client=fakes.ScriptedClient([example_decision_json(brief)]),
        ).run()

        assert result.log_path is not None
        payload = json.loads(result.log_path.read_text(encoding="utf-8"))
        (attempt,) = payload["attempts"]
        assert attempt["prompt_messages"] is None
        # The hash survives, so a turn can still be shown to match a prompt.
        assert attempt["prompt_hash"]

    def test_api_key_never_reaches_the_record(self, tmp_path: Path) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        secret = "sk-do-not-log-me-4242"
        result = RedCommanderTurn(
            game,
            fakes.make_config(api_key=secret),
            audit_log=AuditLog(tmp_path),
            client=fakes.ScriptedClient([example_decision_json(brief)]),
        ).run()

        assert result.log_path is not None
        assert secret not in result.log_path.read_text(encoding="utf-8")


class TestProjectionIsStable:
    def test_same_state_projects_to_the_same_revision(self) -> None:
        _, first = _brief(IntelPolicy.REALISTIC)
        _, second = _brief(IntelPolicy.REALISTIC)
        assert first.campaign_revision == second.campaign_revision
        assert first.campaign_id_hash == second.campaign_id_hash
        assert first.content_hash() == second.content_hash()

    def test_changing_red_state_changes_the_revision(self) -> None:
        _, baseline = _brief(IntelPolicy.REALISTIC)
        _, moved = fakes.synthetic_game(red_deployable=999)
        changed = IntelProjector(moved, IntelPolicy.REALISTIC).project()
        assert changed.campaign_revision != baseline.campaign_revision
