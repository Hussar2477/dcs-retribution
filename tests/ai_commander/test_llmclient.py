"""Tests for :mod:`game.ai_commander.llmclient` response handling.

These focus on the reasoning-model robustness of ``complete()``: recovering the
answer from the reasoning channel when ``content`` is empty, and the request-side
reasoning cap on structured (``response_format``) calls. They drive the real
``ChatCompletionClient`` through an injected fake HTTP opener so the request
payload and the normalised response can both be inspected.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from game.ai_commander.llmclient import ChatCompletionClient


class _FakeResponse:
    """Minimal stand-in for the object ``urllib`` yields as a context manager."""

    def __init__(self, body: Mapping[str, Any]) -> None:
        self.status = 200
        self._raw = json.dumps(body).encode("utf-8")
        self.headers: dict[str, str] = {}

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _FakeOpener:
    """Records the request payload and replays a scripted response body."""

    def __init__(self, body: Mapping[str, Any]) -> None:
        self._body = body
        self.sent_payloads: list[dict[str, Any]] = []

    def __call__(self, request: Any, timeout: Optional[float] = None) -> _FakeResponse:
        if request.data:
            self.sent_payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse(self._body)


def _client(opener: _FakeOpener) -> ChatCompletionClient:
    return ChatCompletionClient(api_key="unit-test-key-never-real", _opener=opener)


def _messages() -> Sequence[Mapping[str, str]]:
    return [{"role": "user", "content": "decide"}]


_JSON_ANSWER = '{"schema_version": "red-commander-decision/1", "ok": true}'


def _body_with_message(message: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": "gen-1",
        "model": "deepseek/deepseek-v4-flash-0731",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


def test_empty_content_recovers_reasoning_content_string() -> None:
    """content empty but reasoning_content holds the JSON -> returned as text."""

    opener = _FakeOpener(
        _body_with_message({"content": "", "reasoning_content": _JSON_ANSWER})
    )
    result = _client(opener).complete(_messages())
    assert result.text == _JSON_ANSWER


def test_empty_content_recovers_reasoning_list_of_parts() -> None:
    """Reasoning delivered as content parts is joined just like content parts."""

    opener = _FakeOpener(
        _body_with_message(
            {
                "content": None,
                "reasoning": [
                    {"text": '{"schema_version": '},
                    {"text": '"red-commander-decision/1"}'},
                ],
            }
        )
    )
    result = _client(opener).complete(_messages())
    assert result.text == '{"schema_version": "red-commander-decision/1"}'


def test_content_is_preferred_over_reasoning_when_both_present() -> None:
    """A normal reply wins; the reasoning channel is ignored when content exists."""

    opener = _FakeOpener(
        _body_with_message(
            {"content": _JSON_ANSWER, "reasoning_content": "irrelevant thinking"}
        )
    )
    result = _client(opener).complete(_messages())
    assert result.text == _JSON_ANSWER


def test_reasoning_content_channel_takes_precedence_over_reasoning() -> None:
    """reasoning_content is checked before reasoning."""

    opener = _FakeOpener(
        _body_with_message(
            {
                "content": "",
                "reasoning_content": _JSON_ANSWER,
                "reasoning": "should not be used",
            }
        )
    )
    result = _client(opener).complete(_messages())
    assert result.text == _JSON_ANSWER


def test_reasoning_hint_only_added_with_response_format() -> None:
    """The reasoning cap is sent only on structured calls, never otherwise."""

    plain = _FakeOpener(_body_with_message({"content": _JSON_ANSWER}))
    _client(plain).complete(_messages(), max_output_tokens=12000)
    assert "reasoning" not in plain.sent_payloads[0]

    structured = _FakeOpener(_body_with_message({"content": _JSON_ANSWER}))
    _client(structured).complete(
        _messages(),
        max_output_tokens=12000,
        response_format={"type": "json_object"},
    )
    payload = structured.sent_payloads[0]
    assert payload["reasoning"] == {"max_tokens": 2000}


def test_reasoning_hint_scales_with_small_budget() -> None:
    """The cap is a fraction of the budget so headroom always remains."""

    structured = _FakeOpener(_body_with_message({"content": _JSON_ANSWER}))
    _client(structured).complete(
        _messages(),
        max_output_tokens=1000,
        response_format={"type": "json_object"},
    )
    assert structured.sent_payloads[0]["reasoning"] == {"max_tokens": 500}
