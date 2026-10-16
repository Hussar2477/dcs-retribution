"""Validation of the model's reply.

Nothing a language model returns is trusted. The reply is parsed out of
whatever prose or fencing surrounds it, then every identifier, rank and enum is
checked against the brief that was actually sent. Two failure classes matter:

* **Fatal** -- the reply is not a decision at all (wrong schema, wrong campaign
  state, wrong turn). ``ValidationOutcome.decision`` is ``None`` and the turn
  falls back.
* **Non-fatal** -- individual elements are wrong. Those elements are dropped
  and recorded as rejections; whatever was well-formed still applies.

Both classes must produce a *recorded reason*, because an unexplained rejection
is indistinguishable from a bug in the validator.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from game.ai_commander.decision import (
    DECISION_SCHEMA_VERSION,
    MAX_INTENT_CHARACTERS,
    MAX_LIST_ENTRIES,
    DecisionValidationError,
    decision_json_schema,
    decision_schema_hash,
    example_decision_json,
    extract_json_object,
    parse_decision,
    validate_decision,
)
from game.ai_commander.enums import (
    FrontPosture,
    IntelPolicy,
    MissionPurpose,
    RedStrategy,
    ReservePolicy,
)
from game.ai_commander.intel import IntelProjector, RedCommanderBrief
from tests.ai_commander import fakes


@pytest.fixture(autouse=True)
def _objective_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes.patch_objective_finder(monkeypatch)


@pytest.fixture
def brief() -> RedCommanderBrief:
    _, game = fakes.synthetic_game()
    return IntelProjector(game, IntelPolicy.REALISTIC).project()


def valid_payload(brief: RedCommanderBrief, **overrides: Any) -> dict[str, Any]:
    """A minimal decision that the validator accepts in full."""

    payload: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "campaign_revision": brief.campaign_revision,
        "turn_id": brief.turn_id,
        "strategy": RedStrategy.GROUND_OFFENSIVE.value,
        "front_priorities": [{"front_id": "FRONT-1", "rank": 1}],
        "push_postures": [{"front_id": "FRONT-1", "posture": FrontPosture.PUSH.value}],
        "spending_priorities": [
            {"category_id": "PROC-2", "rank": 1},
            {"category_id": "PROC-1", "rank": 2},
        ],
        "target_set_priorities": [
            {
                "target_set_id": "TS-3",
                "rank": 1,
                "purpose": MissionPurpose.PROTECT_OWN_FORCES.value,
            }
        ],
        "reserve_policy": ReservePolicy.COMMIT_EVERYTHING.value,
        "commander_intent": "Break the northern front.",
    }
    payload.update(overrides)
    return payload


def reasons_for(outcome: Any, element: str) -> list[str]:
    return [r.reason for r in outcome.rejections if r.element == element]


class TestTheHappyPath:
    def test_a_well_formed_decision_is_accepted_whole(
        self, brief: RedCommanderBrief
    ) -> None:
        outcome = validate_decision(valid_payload(brief), brief)
        assert outcome.ok
        assert outcome.rejections == []
        decision = outcome.decision
        assert decision is not None
        assert decision.strategy is RedStrategy.GROUND_OFFENSIVE
        assert decision.reserve_policy is ReservePolicy.COMMIT_EVERYTHING
        assert decision.ordered_front_ids == ("FRONT-1",)
        assert decision.ordered_spending_ids == ("PROC-2", "PROC-1")
        assert decision.ordered_target_set_ids == ("TS-3",)
        assert decision.posture_for("FRONT-1") is FrontPosture.PUSH

    def test_the_generated_example_validates_against_its_own_brief(
        self, brief: RedCommanderBrief
    ) -> None:
        """The worked example shown to the model must not itself be illegal."""

        outcome = parse_decision(example_decision_json(brief), brief)
        assert outcome.ok
        assert outcome.rejections == []

    def test_schema_advertises_only_identifiers_from_the_brief(
        self, brief: RedCommanderBrief
    ) -> None:
        schema = decision_json_schema(brief)
        body = json.dumps(schema)
        assert brief.campaign_revision in body
        for identifier in (
            brief.front_ids | brief.target_set_ids | brief.procurement_ids
        ):
            assert identifier in body
        assert "FRONT-99" not in body

    def test_schema_hash_tracks_the_brief(self, brief: RedCommanderBrief) -> None:
        _, other_game = fakes.synthetic_game(red_deployable=17)
        other = IntelProjector(other_game, IntelPolicy.REALISTIC).project()
        assert decision_schema_hash(brief) == decision_schema_hash(brief)
        assert decision_schema_hash(brief) != decision_schema_hash(other)


class TestMalformedJson:
    def test_prose_around_the_object_is_tolerated(
        self, brief: RedCommanderBrief
    ) -> None:
        body = json.dumps(valid_payload(brief))
        text = f"Certainly! Here is my plan:\n{body}\nLet me know if you need more."
        assert parse_decision(text, brief).ok

    def test_fenced_code_block_is_tolerated(self, brief: RedCommanderBrief) -> None:
        body = json.dumps(valid_payload(brief))
        assert parse_decision(f"```json\n{body}\n```", brief).ok

    def test_empty_response_is_rejected(self, brief: RedCommanderBrief) -> None:
        with pytest.raises(DecisionValidationError) as err:
            parse_decision("   \n ", brief)
        assert "empty" in str(err.value)

    def test_pure_prose_is_rejected(self, brief: RedCommanderBrief) -> None:
        with pytest.raises(DecisionValidationError) as err:
            parse_decision("I would prefer not to answer in JSON.", brief)
        assert "did not contain a JSON object" in str(err.value)

    def test_truncated_json_is_rejected_rather_than_repaired(
        self, brief: RedCommanderBrief
    ) -> None:
        body = json.dumps(valid_payload(brief))[:-20]
        with pytest.raises(DecisionValidationError) as err:
            parse_decision(body, brief)
        assert "not valid JSON" in str(err.value)

    def test_json_array_is_rejected(self, brief: RedCommanderBrief) -> None:
        with pytest.raises(DecisionValidationError) as err:
            extract_json_object("[1, 2, 3]")
        assert "expected an object" in str(err.value)

    def test_non_object_payload_is_fatal(self, brief: RedCommanderBrief) -> None:
        outcome = validate_decision(["not", "a", "decision"], brief)
        assert not outcome.ok
        assert outcome.decision is None
        assert reasons_for(outcome, "<root>") == ["expected a JSON object"]


class TestMissingOrWrongRequiredFields:
    def test_missing_schema_version_is_fatal(self, brief: RedCommanderBrief) -> None:
        payload = valid_payload(brief)
        del payload["schema_version"]
        outcome = validate_decision(payload, brief)
        assert outcome.decision is None
        assert reasons_for(outcome, "schema_version") == [
            f"expected {DECISION_SCHEMA_VERSION}"
        ]

    def test_wrong_schema_version_is_fatal(self, brief: RedCommanderBrief) -> None:
        outcome = validate_decision(
            valid_payload(brief, schema_version="red-commander-decision/99"), brief
        )
        assert outcome.decision is None
        assert outcome.error_summary()

    def test_missing_strategy_is_fatal(self, brief: RedCommanderBrief) -> None:
        payload = valid_payload(brief)
        del payload["strategy"]
        outcome = validate_decision(payload, brief)
        assert outcome.decision is None
        assert reasons_for(outcome, "strategy") == ["expected a string enum value"]

    def test_unknown_strategy_is_fatal(self, brief: RedCommanderBrief) -> None:
        outcome = validate_decision(
            valid_payload(brief, strategy="nuclear_first_strike"), brief
        )
        assert outcome.decision is None
        (reason,) = reasons_for(outcome, "strategy")
        assert reason.startswith("not one of ")

    def test_missing_reserve_policy_is_fatal(self, brief: RedCommanderBrief) -> None:
        payload = valid_payload(brief)
        del payload["reserve_policy"]
        outcome = validate_decision(payload, brief)
        assert outcome.decision is None
        assert reasons_for(outcome, "reserve_policy")

    def test_enum_spelling_variants_are_normalised(
        self, brief: RedCommanderBrief
    ) -> None:
        """Case and separator noise is a formatting slip, not a cheat."""

        outcome = validate_decision(
            valid_payload(brief, strategy=" Ground-Offensive "), brief
        )
        assert outcome.decision is not None
        assert outcome.decision.strategy is RedStrategy.GROUND_OFFENSIVE

    def test_optional_lists_may_be_omitted(self, brief: RedCommanderBrief) -> None:
        payload = valid_payload(brief)
        for key in (
            "front_priorities",
            "push_postures",
            "spending_priorities",
            "target_set_priorities",
            "commander_intent",
        ):
            del payload[key]
        outcome = validate_decision(payload, brief)
        assert outcome.ok
        assert outcome.decision is not None
        assert outcome.decision.ordered_front_ids == ()

    def test_unexpected_top_level_key_is_recorded_but_not_fatal(
        self, brief: RedCommanderBrief
    ) -> None:
        outcome = validate_decision(
            valid_payload(brief, secret_orders="capture the enemy capital"), brief
        )
        assert outcome.decision is not None
        assert reasons_for(outcome, "<root>") == [
            "unexpected top-level keys were ignored"
        ]


class TestUnknownIdentifiers:
    def test_unknown_front_is_dropped_with_a_reason(
        self, brief: RedCommanderBrief
    ) -> None:
        outcome = validate_decision(
            valid_payload(
                brief,
                front_priorities=[
                    {"front_id": "FRONT-1", "rank": 1},
                    {"front_id": "FRONT-INVENTED", "rank": 2},
                ],
            ),
            brief,
        )
        assert outcome.decision is not None
        assert outcome.decision.ordered_front_ids == ("FRONT-1",)
        assert reasons_for(outcome, "front_priorities[1]") == [
            "identifier is not in the brief"
        ]

    def test_unknown_procurement_category_is_dropped(
        self, brief: RedCommanderBrief
    ) -> None:
        outcome = validate_decision(
            valid_payload(
                brief, spending_priorities=[{"category_id": "PROC-NAVY", "rank": 1}]
            ),
            brief,
        )
        assert outcome.decision is not None
        assert outcome.decision.ordered_spending_ids == ()
        assert reasons_for(outcome, "spending_priorities[0]") == [
            "identifier is not in the brief"
        ]

    def test_unknown_target_set_is_dropped(self, brief: RedCommanderBrief) -> None:
        outcome = validate_decision(
            valid_payload(
                brief,
                target_set_priorities=[
                    {
                        "target_set_id": "TS-BLUE-CARRIER",
                        "rank": 1,
                        "purpose": MissionPurpose.ATTRITION.value,
                    }
                ],
            ),
            brief,
        )
        assert outcome.decision is not None
        assert outcome.decision.ordered_target_set_ids == ()
        assert reasons_for(outcome, "target_set_priorities[0]") == [
            "identifier is not in the brief"
        ]

    def test_a_squadron_or_base_name_is_not_an_identifier(
        self, brief: RedCommanderBrief
    ) -> None:
        """The model cannot address game objects, only briefing identifiers.

        This is what stops a model that guessed a real base or squadron name
        from acting on it.
        """

        outcome = validate_decision(
            valid_payload(
                brief,
                front_priorities=[{"front_id": "RED-FRONT-BASE", "rank": 1}],
                spending_priorities=[
                    {
                        "category_id": str(fakes.BLUE_SENTINELS["blue_squadron_name"]),
                        "rank": 1,
                    }
                ],
            ),
            brief,
        )
        assert outcome.decision is not None
        assert outcome.decision.ordered_front_ids == ()
        assert outcome.decision.ordered_spending_ids == ()
        assert len(outcome.rejections) == 2

    def test_unknown_posture_front_is_dropped(self, brief: RedCommanderBrief) -> None:
        outcome = validate_decision(
            valid_payload(
                brief,
                push_postures=[
                    {"front_id": "FRONT-404", "posture": FrontPosture.HOLD.value}
                ],
            ),
            brief,
        )
        assert outcome.decision is not None
        assert outcome.decision.push_postures == ()
        assert reasons_for(outcome, "push_postures[0]") == [
            "identifier is not in the brief"
        ]


class TestDuplicateRanks:
    def test_duplicate_rank_is_rejected(self, brief: RedCommanderBrief) -> None:
        outcome = validate_decision(
            valid_payload(
                brief,
                spending_priorities=[
                    {"category_id": "PROC-1", "rank": 1},
                    {"category_id": "PROC-2", "rank": 1},
                ],
            ),
            brief,
        )
        assert outcome.decision is not None
        assert outcome.decision.ordered_spending_ids == ("PROC-1",)
        assert reasons_for(outcome, "spending_priorities[1].rank") == ["duplicate rank"]

    def test_duplicate_identifier_is_rejected(self, brief: RedCommanderBrief) -> None:
        outcome = validate_decision(
            valid_payload(
                brief,
                spending_priorities=[
                    {"category_id": "PROC-1", "rank": 1},
                    {"category_id": "PROC-1", "rank": 2},
                ],
            ),
            brief,
        )
        assert outcome.decision is not None
        assert outcome.decision.ordered_spending_ids == ("PROC-1",)
        assert reasons_for(outcome, "spending_priorities[1]") == [
            "duplicate identifier"
        ]

    def test_duplicate_target_set_rank_is_rejected(
        self, brief: RedCommanderBrief
    ) -> None:
        outcome = validate_decision(
            valid_payload(
                brief,
                target_set_priorities=[
                    {"target_set_id": "TS-1", "rank": 2, "purpose": "attrition"},
                    {"target_set_id": "TS-2", "rank": 2, "purpose": "attrition"},
                ],
            ),
            brief,
        )
        assert outcome.decision is not None
        assert outcome.decision.ordered_target_set_ids == ("TS-1",)
        assert reasons_for(outcome, "target_set_priorities[1].rank") == [
            "duplicate rank"
        ]

    def test_rank_zero_is_rejected(self, brief: RedCommanderBrief) -> None:
        outcome = validate_decision(
            valid_payload(brief, front_priorities=[{"front_id": "FRONT-1", "rank": 0}]),
            brief,
        )
        assert outcome.decision is not None
        assert outcome.decision.ordered_front_ids == ()
        assert reasons_for(outcome, "front_priorities[0].rank") == ["ranks start at 1"]

    def test_non_integer_rank_is_rejected(self, brief: RedCommanderBrief) -> None:
        outcome = validate_decision(
            valid_payload(
                brief, front_priorities=[{"front_id": "FRONT-1", "rank": "first"}]
            ),
            brief,
        )
        assert outcome.decision is not None
        assert outcome.decision.ordered_front_ids == ()
        assert reasons_for(outcome, "front_priorities[0].rank") == [
            "rank must be an integer"
        ]

    def test_entries_are_ordered_by_rank_not_by_position(
        self, brief: RedCommanderBrief
    ) -> None:
        outcome = validate_decision(
            valid_payload(
                brief,
                spending_priorities=[
                    {"category_id": "PROC-4", "rank": 3},
                    {"category_id": "PROC-1", "rank": 1},
                    {"category_id": "PROC-3", "rank": 2},
                ],
            ),
            brief,
        )
        assert outcome.decision is not None
        assert outcome.decision.ordered_spending_ids == ("PROC-1", "PROC-3", "PROC-4")


class TestStaleCampaignRevision:
    def test_revision_from_another_state_is_fatal(
        self, brief: RedCommanderBrief
    ) -> None:
        outcome = validate_decision(
            valid_payload(brief, campaign_revision="0000000000000000"), brief
        )
        assert outcome.decision is None
        assert reasons_for(outcome, "campaign_revision") == [
            "decision was produced for a different campaign state"
        ]

    def test_missing_revision_is_fatal(self, brief: RedCommanderBrief) -> None:
        payload = valid_payload(brief)
        del payload["campaign_revision"]
        assert validate_decision(payload, brief).decision is None

    def test_wrong_turn_is_fatal(self, brief: RedCommanderBrief) -> None:
        outcome = validate_decision(
            valid_payload(brief, turn_id=brief.turn_id + 1), brief
        )
        assert outcome.decision is None
        assert reasons_for(outcome, "turn_id") == [f"expected turn {brief.turn_id}"]

    def test_non_integer_turn_is_fatal(self, brief: RedCommanderBrief) -> None:
        outcome = validate_decision(valid_payload(brief, turn_id="seven"), brief)
        assert outcome.decision is None
        assert reasons_for(outcome, "turn_id") == ["must be an integer"]

    def test_a_decision_for_a_changed_campaign_cannot_be_replayed(self) -> None:
        """The revision is what makes a stale reply detectable."""

        _, before = fakes.synthetic_game(red_deployable=2460)
        _, after = fakes.synthetic_game(red_deployable=11)
        old_brief = IntelProjector(before, IntelPolicy.REALISTIC).project()
        new_brief = IntelProjector(after, IntelPolicy.REALISTIC).project()
        assert old_brief.campaign_revision != new_brief.campaign_revision
        assert validate_decision(valid_payload(old_brief), new_brief).decision is None


class TestAbuseResistance:
    def test_oversized_lists_are_truncated_with_a_reason(
        self, brief: RedCommanderBrief
    ) -> None:
        outcome = validate_decision(
            valid_payload(
                brief,
                front_priorities=[
                    {"front_id": "FRONT-1", "rank": index + 1}
                    for index in range(MAX_LIST_ENTRIES + 5)
                ],
            ),
            brief,
        )
        assert outcome.decision is not None
        assert any(
            "entry limit" in reason
            for reason in reasons_for(outcome, "front_priorities")
        )

    def test_overlong_intent_is_truncated_with_a_reason(
        self, brief: RedCommanderBrief
    ) -> None:
        limit = brief.commander_constraints.max_intent_characters
        assert limit == MAX_INTENT_CHARACTERS
        outcome = validate_decision(
            valid_payload(brief, commander_intent="A" * (limit + 500)), brief
        )
        assert outcome.decision is not None
        assert len(outcome.decision.commander_intent) == limit
        assert reasons_for(outcome, "commander_intent") == [
            f"longer than {limit} characters; truncated"
        ]

    def test_nested_object_where_a_list_was_expected(
        self, brief: RedCommanderBrief
    ) -> None:
        outcome = validate_decision(
            valid_payload(brief, front_priorities={"front_id": "FRONT-1"}), brief
        )
        assert outcome.decision is not None
        assert reasons_for(outcome, "front_priorities") == ["expected a list"]

    def test_extra_keys_inside_an_entry_are_rejected(
        self, brief: RedCommanderBrief
    ) -> None:
        """Prevents smuggling instructions through an unread field."""

        outcome = validate_decision(
            valid_payload(
                brief,
                front_priorities=[
                    {
                        "front_id": "FRONT-1",
                        "rank": 1,
                        "also_move_units_to": "BLUE-FRONT-BASE",
                    }
                ],
            ),
            brief,
        )
        assert outcome.decision is not None
        assert outcome.decision.ordered_front_ids == ()
        assert reasons_for(outcome, "front_priorities[0]") == ["unexpected keys"]

    def test_error_summary_is_bounded(self, brief: RedCommanderBrief) -> None:
        outcome = validate_decision(
            valid_payload(
                brief,
                front_priorities=[
                    {"front_id": f"FRONT-{index}", "rank": index + 1}
                    for index in range(20)
                ],
            ),
            brief,
        )
        summary = outcome.error_summary(limit=5)
        assert summary.count("\n") <= 5
