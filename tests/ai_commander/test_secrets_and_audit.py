"""The key never leaks, and a turn is auditable from the files on disk.

Two documented promises are checked here:

* the provider API key does not reach any artefact Retribution writes -- not the
  pickled ``Settings``, not the settings JSON dump, not the decision log, not
  ``repr()`` of the objects that hold it;
* everything a reviewer needs to audit a turn is present in the decision log
  file, and reading it back reproduces what the controller reported.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, cast

import pytest

from game.ai_commander.audit import (
    AUDIT_DIRECTORY_NAME,
    RECORD_SCHEMA_VERSION,
    AiDecisionRecord,
    AuditLog,
    LlmAttempt,
    prompt_digest,
    summary_from_payload,
)
from game.ai_commander.config import AiCommanderConfig
from game.ai_commander.controller import RedCommanderTurn, describe_turn_result
from game.ai_commander.decision import example_decision_json
from game.ai_commander.enums import FallbackReason, IntelPolicy
from game.ai_commander.llmclient import ChatCompletionClient, TokenUsage
from game.ai_commander.secretstore import (
    ENV_VAR,
    REDACTED,
    SECRETS_FILENAME,
    SecretStore,
    mask,
    user_data_path,
)
from game.ai_commander.serialization import canonical_json
from tests.ai_commander.fakes import (
    CATALOG_PAYLOAD,
    ScriptedClient,
    make_config,
    synthetic_game,
)

#: Not a real key. Long enough that :func:`mask` takes its partial-reveal path.
FAKE_KEY = "sk-or-v1-UNIT-TEST-NOT-A-REAL-KEY-0123456789"


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's real environment must not change these results."""

    monkeypatch.delenv(ENV_VAR, raising=False)


class TestMasking:
    def test_an_absent_key_is_described_rather_than_masked(self) -> None:
        assert mask(None) == "(not set)"
        assert mask("") == "(not set)"

    def test_a_short_value_is_fully_masked(self) -> None:
        # Partially revealing a short key could reveal most of it.
        assert mask("abcdefghij") == "*" * 10
        assert "abc" not in mask("abcdefghij")

    def test_a_long_key_reveals_only_a_fingerprint(self) -> None:
        masked = mask(FAKE_KEY)
        assert FAKE_KEY not in masked
        assert masked.startswith("sk-or-")
        assert masked.endswith(f"({len(FAKE_KEY)} chars)")
        # The middle -- the part that actually authenticates -- is gone.
        assert "NOT-A-REAL-KEY" not in masked


class TestSecretStore:
    def test_the_key_lives_outside_the_save_and_outside_the_repo(self) -> None:
        path = user_data_path() / SECRETS_FILENAME
        assert path.name == SECRETS_FILENAME
        # Same per-user directory as retribution_preferences.json, which is not
        # inside a campaign save and not inside the checkout.
        assert "DCSRetribution" in str(path)

    def test_a_saved_key_round_trips(self, tmp_path: Path) -> None:
        store = SecretStore(tmp_path / SECRETS_FILENAME)
        assert store.load() is None
        assert not store.is_configured

        assert store.save(FAKE_KEY)
        assert store.load() == FAKE_KEY
        assert store.is_configured

    def test_surrounding_whitespace_from_a_paste_is_stripped(
        self, tmp_path: Path
    ) -> None:
        store = SecretStore(tmp_path / SECRETS_FILENAME)
        assert store.save(f"  {FAKE_KEY}\n")
        assert store.load() == FAKE_KEY

    def test_saving_an_empty_value_clears_the_key(self, tmp_path: Path) -> None:
        store = SecretStore(tmp_path / SECRETS_FILENAME)
        store.save(FAKE_KEY)
        assert store.save("   ")
        assert store.load() is None
        assert not store.path.exists()

    def test_clearing_a_key_that_was_never_stored_succeeds(
        self, tmp_path: Path
    ) -> None:
        store = SecretStore(tmp_path / SECRETS_FILENAME)
        assert store.clear()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits only")
    def test_the_file_is_owner_only(self, tmp_path: Path) -> None:
        store = SecretStore(tmp_path / SECRETS_FILENAME)
        store.save(FAKE_KEY)
        mode = stat.S_IMODE(store.path.stat().st_mode)
        assert mode == stat.S_IRUSR | stat.S_IWUSR

    def test_the_environment_wins_over_the_stored_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = SecretStore(tmp_path / SECRETS_FILENAME)
        store.save(FAKE_KEY)
        monkeypatch.setenv(ENV_VAR, "sk-or-v1-FROM-THE-ENVIRONMENT-INSTEAD")

        assert store.load() == "sk-or-v1-FROM-THE-ENVIRONMENT-INSTEAD"
        # ...but the stored key is still there, untouched.
        assert store.load_stored() == FAKE_KEY
        assert ENV_VAR in store.source

    def test_the_source_names_the_file_when_the_environment_is_unset(
        self, tmp_path: Path
    ) -> None:
        store = SecretStore(tmp_path / SECRETS_FILENAME)
        assert store.source == "not configured"
        store.save(FAKE_KEY)
        assert store.source == str(store.path)

    def test_a_corrupt_file_is_ignored_rather_than_raising(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / SECRETS_FILENAME
        path.write_text("this is not json", encoding="utf-8")
        assert SecretStore(path).load() is None

    def test_a_json_file_of_the_wrong_shape_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / SECRETS_FILENAME
        path.write_text('["a list, not an object"]', encoding="utf-8")
        assert SecretStore(path).load() is None

        path.write_text('{"openrouter_api_key": 12345}', encoding="utf-8")
        assert SecretStore(path).load() is None

    def test_repr_never_shows_the_key(self, tmp_path: Path) -> None:
        store = SecretStore(tmp_path / SECRETS_FILENAME)
        store.save(FAKE_KEY)
        assert FAKE_KEY not in repr(store)
        assert REDACTED in repr(store)

    def test_describe_never_shows_the_key(self, tmp_path: Path) -> None:
        store = SecretStore(tmp_path / SECRETS_FILENAME)
        store.save(FAKE_KEY)
        described = store.describe()
        assert FAKE_KEY not in described
        assert str(store.path) in described

    def test_unrelated_fields_in_the_file_survive_a_save(self, tmp_path: Path) -> None:
        path = tmp_path / SECRETS_FILENAME
        path.write_text('{"some_future_provider_key": "keep me"}', encoding="utf-8")
        store = SecretStore(path)
        assert store.save(FAKE_KEY)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["some_future_provider_key"] == "keep me"


class TestConfigRedaction:
    def test_the_serialisable_config_drops_the_key_entirely(self) -> None:
        config = make_config(api_key=FAKE_KEY)
        payload = config.to_dict()

        # Not masked in place -- absent, so no downstream consumer can print it.
        assert "api_key" not in payload
        assert payload["api_key_configured"] is True
        assert FAKE_KEY not in canonical_json(payload)

    def test_the_config_summary_is_safe_to_log(self) -> None:
        described = make_config(api_key=FAKE_KEY).describe()
        assert FAKE_KEY not in described
        assert "model=test/model" in described

    def test_settings_read_by_the_config_never_carry_the_key(
        self, tmp_path: Path
    ) -> None:
        # Settings is pickled into the save and dumped to JSON by the settings
        # window, so the key has to come from the store, not from Settings.
        settings = cast(
            "Any",
            type(
                "FakeSettings",
                (),
                {
                    "ai_commander_enabled": True,
                    "ai_commander_model": "test/model",
                    "ai_commander_intel_policy": IntelPolicy.REALISTIC.value,
                },
            )(),
        )
        store = SecretStore(tmp_path / SECRETS_FILENAME)
        store.save(FAKE_KEY)

        config = AiCommanderConfig.from_settings(settings, secret_store=store)

        assert config.api_key == FAKE_KEY
        assert not any(FAKE_KEY in str(value) for value in vars(settings).values())
        assert FAKE_KEY not in canonical_json(config.to_dict())

    def test_a_missing_key_is_a_recorded_problem_not_an_exception(self) -> None:
        settings = cast(
            "Any",
            type("FakeSettings", (), {"ai_commander_enabled": True})(),
        )
        config = AiCommanderConfig.from_settings(settings, secret_store=None)

        assert config.api_key is None
        assert not config.is_usable
        assert config.problems

    def test_the_client_repr_redacts_the_key(self) -> None:
        client = ChatCompletionClient(
            api_key=FAKE_KEY, model="test/model", base_url="https://example.invalid/v1"
        )
        assert FAKE_KEY not in repr(client)
        # "<redacted:set>" -- it says a key is present without showing it.
        assert REDACTED.strip("<>") in repr(client)
        assert FAKE_KEY not in client.describe()


class TestDecisionLogIsAuditable:
    def test_a_record_written_to_disk_reads_back_unchanged(
        self, tmp_path: Path
    ) -> None:
        log = AuditLog(tmp_path)
        record = AiDecisionRecord(campaign_id_hash="abc123", turn_id=4)
        record.attempts.append(
            LlmAttempt(
                attempt=1,
                kind="initial",
                prompt_tokens=1000,
                completion_tokens=200,
                actual_cost=0.0018,
            )
        )
        record.accepted = True

        path = log.write(record)

        assert path is not None
        assert path.name == "turn_0004_00.json"
        assert path.parent.name == "abc123"
        assert path.parent.parent.name == AUDIT_DIRECTORY_NAME

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["record_schema_version"] == RECORD_SCHEMA_VERSION
        assert payload["turn_id"] == 4
        assert payload["accepted"] is True
        assert payload["attempts"][0]["prompt_tokens"] == 1000

        summary = summary_from_payload(payload)
        assert summary is not None

    def test_repeat_records_for_one_turn_are_numbered_not_overwritten(
        self, tmp_path: Path
    ) -> None:
        log = AuditLog(tmp_path)
        first = log.write(AiDecisionRecord(campaign_id_hash="abc123", turn_id=2))
        second = log.write(AiDecisionRecord(campaign_id_hash="abc123", turn_id=2))

        assert first is not None and second is not None
        assert first != second
        # records_for_turn hands back parsed payloads, each tagged with its file.
        found = {Path(r["_path"]).name for r in log.records_for_turn("abc123", 2)}
        assert found == {"turn_0002_00.json", "turn_0002_01.json"}

    def test_spending_accumulates_across_records_for_the_same_turn(
        self, tmp_path: Path
    ) -> None:
        log = AuditLog(tmp_path)
        for cost in (0.01, 0.02):
            record = AiDecisionRecord(campaign_id_hash="abc123", turn_id=3)
            record.actual_cost = cost
            log.write(record)

        assert log.spent_this_turn("abc123", 3) == pytest.approx(0.03)
        # A different turn starts from zero again.
        assert log.spent_this_turn("abc123", 4) == pytest.approx(0.0)

    def test_prompt_logging_can_be_turned_off_without_losing_the_audit_trail(
        self, tmp_path: Path
    ) -> None:
        campaign, game = synthetic_game()
        config = make_config(log_prompts=False)
        log = AuditLog(tmp_path)
        client = ScriptedClient([""], catalog=CATALOG_PAYLOAD)

        turn = RedCommanderTurn(cast("Any", game), config, audit_log=log, client=client)
        brief = turn._project_brief()
        assert brief is not None
        client.script.append(example_decision_json(brief))
        result = turn.run()

        assert result.accepted
        assert result.log_path is not None
        payload = json.loads(result.log_path.read_text(encoding="utf-8"))

        assert payload["prompt_logging_enabled"] is False
        attempt = payload["attempts"][0]
        # The body is withheld, but the hash still proves which prompt was sent.
        assert attempt["prompt_messages"] is None
        assert attempt["prompt_hash"]
        assert attempt["prompt_tokens"] == 1000
        # And the accepted orders are always recorded, however prompts are set.
        assert payload["accepted_directive"]

    def test_the_same_prompt_always_hashes_the_same_way(self) -> None:
        messages = [
            {"role": "system", "content": "you are red"},
            {"role": "user", "content": "brief"},
        ]
        assert prompt_digest(messages) == prompt_digest(list(messages))
        assert prompt_digest(messages) != prompt_digest(messages[:1])

    def test_a_reviewer_can_reconstruct_the_reported_cost_from_the_file(
        self, tmp_path: Path
    ) -> None:
        campaign, game = synthetic_game()
        log = AuditLog(tmp_path)
        client = ScriptedClient([""], catalog=CATALOG_PAYLOAD)
        turn = RedCommanderTurn(
            cast("Any", game), make_config(), audit_log=log, client=client
        )
        brief = turn._project_brief()
        assert brief is not None
        client.script.append(example_decision_json(brief))
        result = turn.run()

        described = describe_turn_result(result)
        assert result.log_path is not None
        payload = json.loads(result.log_path.read_text(encoding="utf-8"))

        assert payload["actual_cost"] == pytest.approx(described["actual_cost"])
        assert payload["cost_cap_per_turn"] == pytest.approx(0.5)
        attempt = payload["attempts"][0]
        # 1000 prompt tokens at $1.00/M plus 200 completion at $4.00/M.
        assert attempt["actual_cost"] == pytest.approx(0.0018)
        assert attempt["prompt_tokens"] == 1000
        assert attempt["completion_tokens"] == 200
        # This provider returned token counts but no cost, so the money figure is
        # ours, computed from the catalogue price -- and the file says so.
        assert attempt["cost_is_estimated"] is True

    def test_a_provider_reported_cost_is_marked_as_authoritative(
        self, tmp_path: Path
    ) -> None:
        campaign, game = synthetic_game()
        log = AuditLog(tmp_path)
        client = ScriptedClient(
            [""],
            catalog=CATALOG_PAYLOAD,
            usage=TokenUsage(1000, 200, 1200, cost=0.0031),
        )
        turn = RedCommanderTurn(
            cast("Any", game), make_config(), audit_log=log, client=client
        )
        brief = turn._project_brief()
        assert brief is not None
        client.script.append(example_decision_json(brief))
        result = turn.run()

        assert result.log_path is not None
        attempt = json.loads(result.log_path.read_text(encoding="utf-8"))["attempts"][0]
        # The provider's own number wins over our catalogue arithmetic.
        assert attempt["actual_cost"] == pytest.approx(0.0031)
        assert attempt["cost_is_estimated"] is False

    def test_a_refused_turn_records_why_it_was_refused(self, tmp_path: Path) -> None:
        campaign, game = synthetic_game()
        log = AuditLog(tmp_path)
        client = ScriptedClient(
            ["not json at all", "still not json"], catalog=CATALOG_PAYLOAD
        )
        result = RedCommanderTurn(
            cast("Any", game), make_config(), audit_log=log, client=client
        ).run()

        assert not result.accepted
        assert result.fallback_reason == FallbackReason.MALFORMED_RESPONSE
        assert result.log_path is not None
        payload = json.loads(result.log_path.read_text(encoding="utf-8"))

        assert payload["accepted"] is False
        assert payload["fallback_reason"] == FallbackReason.MALFORMED_RESPONSE.value
        assert payload["fallback_policy"]
        assert payload["rejections"]
        # Every rejection says which element failed and why, in plain words.
        for rejection in payload["rejections"]:
            assert rejection["element"]
            assert rejection["reason"]

    def test_the_key_is_never_written_into_the_decision_log(
        self, tmp_path: Path
    ) -> None:
        campaign, game = synthetic_game()
        log = AuditLog(tmp_path)
        client = ScriptedClient([""], catalog=CATALOG_PAYLOAD)
        turn = RedCommanderTurn(
            cast("Any", game),
            make_config(api_key=FAKE_KEY, log_prompts=True),
            audit_log=log,
            client=client,
        )
        brief = turn._project_brief()
        assert brief is not None
        client.script.append(example_decision_json(brief))
        result = turn.run()

        assert result.log_path is not None
        body = result.log_path.read_text(encoding="utf-8")
        assert FAKE_KEY not in body
        assert "sk-or" not in body

    def test_an_unwritable_root_never_breaks_the_turn(self, tmp_path: Path) -> None:
        # A file where the audit directory should be: every write must fail.
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory", encoding="utf-8")
        log = AuditLog(blocked)

        assert log.write(AiDecisionRecord(campaign_id_hash="abc123", turn_id=1)) is None
        assert log.turns("abc123") == []
        assert log.spent_this_turn("abc123", 1) == pytest.approx(0.0)
