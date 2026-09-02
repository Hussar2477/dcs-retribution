"""A minimal, dependency-free OpenAI-compatible chat-completions client.

Retribution's ``requirements.txt`` does not include ``requests``, and adding a
networking dependency to a mission generator is not worth it for one endpoint,
so this uses :mod:`urllib.request` from the standard library.

Provider-agnostic by construction: anything exposing ``POST {base_url}/chat/completions``
and ``GET {base_url}/models`` works. Defaults target OpenRouter; pointing
``base_url`` at ``http://localhost:11434/v1`` runs against a local Ollama server
with no API key.

The API key is only ever placed in an ``Authorization`` header. It is never
logged, never included in an exception message, and never written to the audit
log -- see :meth:`ChatCompletionClient.describe`, which is what the audit record
uses to identify the endpoint.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from game.ai_commander.serialization import jsonable

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_TIMEOUT_SECONDS = 90
#: The air-tasking stage is the most token-hungry, and reasoning models spend a
#: large slice of the budget on hidden chain-of-thought before the JSON answer.
#: A real Decision Log showed a reasoning model loop on the air-tasking stage and
#: burn the whole 12000-token budget in the reasoning channel, truncating the
#: answer (finish_reason="length") before any JSON was produced. This is raised
#: so reasoning + JSON both fit comfortably; the per-turn cost cap (enforced by
#: the ledger before every call) still bounds spend regardless of this value.
DEFAULT_MAX_OUTPUT_TOKENS = 18000

#: Hard ceiling for the one *enlarged* repair a stage may make after a truncated
#: response (see the controller's truncation-aware repair). Keeps the enlarged
#: budget bounded; the cost ledger is the actual guarantee against overspend.
MAX_OUTPUT_TOKENS_CEILING = 32000

#: Anti-repetition controls sent on every chat/completions request. Reasoning
#: models can fall into a loop that re-derives the same paragraph until the
#: budget is exhausted; a modest frequency/presence penalty discourages this.
#: Standard OpenAI-compatible parameters -- OpenRouter/DeepSeek honour them and
#: providers that do not simply ignore unknown fields.
DEFAULT_FREQUENCY_PENALTY = 0.4
DEFAULT_PRESENCE_PENALTY = 0.3

#: Ceiling on the reasoning-token budget for structured calls. The cap scales
#: with the output budget (``max_output_tokens // _REASONING_CAP_DIVISOR``) but
#: never exceeds this, so a big budget still leaves ample room for the JSON.
_REASONING_CAP_CEILING = 4000
_REASONING_CAP_DIVISOR = 2

#: Identifies the application to OpenRouter. Contains no user data.
APPLICATION_TITLE = "DCS Retribution LLM RED Commander"
APPLICATION_REFERER = "https://github.com/dcs-retribution/dcs-retribution"

#: Status codes that are worth retrying at transport level. 402 (insufficient
#: credit) and 401/403 (bad key) are not: retrying cannot help.
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
_MAX_TRANSPORT_ATTEMPTS = 3
_MAX_RETRY_AFTER_SECONDS = 10.0


class LlmError(Exception):
    """Base class for every failure this client reports."""


class LlmTimeout(LlmError):
    """The request did not complete inside the configured timeout."""


class LlmTransportError(LlmError):
    """DNS, TLS, connection or malformed-body failure."""


class LlmHttpError(LlmError):
    """The endpoint returned a non-2xx status."""

    def __init__(self, status: int, message: str, retry_after: Optional[float] = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.retry_after = retry_after

    @property
    def is_auth_failure(self) -> bool:
        return self.status in (401, 403)

    @property
    def is_payment_required(self) -> bool:
        return self.status == 402


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting as reported by the provider.

    ``cost`` is OpenRouter's own ``usage.cost`` when present; it is the
    authoritative charge and takes precedence over any local estimate.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_payload(cls, payload: Any) -> TokenUsage:
        if not isinstance(payload, Mapping):
            return cls()
        prompt_details = payload.get("prompt_tokens_details")
        cached = 0
        if isinstance(prompt_details, Mapping):
            cached = _as_int(prompt_details.get("cached_tokens"))
        completion_details = payload.get("completion_tokens_details")
        reasoning = 0
        if isinstance(completion_details, Mapping):
            reasoning = _as_int(completion_details.get("reasoning_tokens"))
        cost_raw = payload.get("cost")
        cost: Optional[float]
        try:
            cost = float(cost_raw) if cost_raw is not None else None
        except (TypeError, ValueError):
            cost = None
        return cls(
            input_tokens=_as_int(payload.get("prompt_tokens")),
            output_tokens=_as_int(payload.get("completion_tokens")),
            total_tokens=_as_int(payload.get("total_tokens")),
            cached_input_tokens=cached,
            reasoning_tokens=reasoning,
            cost=cost,
        )


@dataclass(frozen=True)
class LlmResponse:
    """A normalised chat-completion response."""

    text: str
    usage: TokenUsage
    model: str
    finish_reason: str
    request_id: Optional[str]
    latency_seconds: float
    attempts: int = 1
    had_tool_calls: bool = False

    @property
    def was_truncated(self) -> bool:
        """Whether the answer was cut off before completing.

        The unambiguous signal is ``finish_reason == "length"``: the provider
        hit the output budget. A reasoning model that loops can also exhaust the
        budget entirely inside its hidden reasoning channel, leaving no visible
        answer at all -- that shows up as an empty ``text`` together with a
        reasoning-token count that has consumed most of the budget.
        """

        return self.finish_reason == "length"

    def looks_truncated(self, max_output_tokens: int) -> bool:
        """Truncation, including the empty-answer-with-exhausted-reasoning case.

        ``max_output_tokens`` is the budget the request was made with, needed to
        judge whether the reasoning channel ate essentially all of it.
        """

        if self.was_truncated:
            return True
        if self.text.strip():
            return False
        budget = max(1, int(max_output_tokens))
        return self.usage.reasoning_tokens >= 0.8 * budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
            "latency_seconds": round(self.latency_seconds, 3),
            "attempts": self.attempts,
            "had_tool_calls": self.had_tool_calls,
            "usage": self.usage.to_dict(),
            "characters": len(self.text),
        }


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalise_finish_reason(raw: Any) -> str:
    """Reduce provider finish reasons to the five values we branch on."""

    known = {"stop", "length", "content_filter", "tool_calls", "error"}
    if isinstance(raw, str) and raw in known:
        return raw
    if isinstance(raw, str) and raw:
        return "stop" if raw in ("end_turn", "eos", "complete") else "error"
    return "error"


@dataclass
class ChatCompletionClient:
    """One configured endpoint/model pair.

    Instances are cheap; a new one is built for each turn so a settings change
    takes effect immediately.
    """

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    referer: str = APPLICATION_REFERER
    title: str = APPLICATION_TITLE
    session_id: Optional[str] = None
    #: Anti-repetition penalties applied to every request. Defaults discourage
    #: the reasoning-loop failure without any configuration.
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY
    _opener: Any = field(default=None, repr=False, compare=False)

    # -- identity ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Endpoint description for the audit log. Contains no secret."""

        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "api_key_present": bool(self.api_key),
            "session_id": self.session_id,
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"ChatCompletionClient(base_url={self.base_url!r}, model={self.model!r}, "
            f"api_key=<redacted:{'set' if self.api_key else 'unset'}>)"
        )

    @property
    def requires_api_key(self) -> bool:
        """Local providers (Ollama, LM Studio) do not need a key."""

        host = urllib.parse.urlparse(self.base_url).hostname or ""
        return host not in ("localhost", "127.0.0.1", "::1", "0.0.0.0")

    # -- requests ---------------------------------------------------------

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + "/" + path.lstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
            "X-OpenRouter-Title": self.title,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _open(self, request: urllib.request.Request) -> tuple[int, bytes, Any]:
        opener = self._opener or urllib.request.urlopen
        with opener(request, timeout=self.timeout_seconds) as response:
            return response.status, response.read(), response.headers

    def _send(self, path: str, payload: Optional[dict[str, Any]], method: str) -> Any:
        """Send one request, retrying only transport-level failures."""

        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        url = self._url(path)

        last_error: Optional[Exception] = None
        for attempt in range(1, _MAX_TRANSPORT_ATTEMPTS + 1):
            request = urllib.request.Request(
                url, data=body, headers=self._headers(), method=method
            )
            try:
                status, raw, headers = self._open(request)
            except urllib.error.HTTPError as err:  # noqa: PERF203 - branchy by nature
                detail = _read_error_body(err)
                retry_after = _retry_after_seconds(err.headers)
                if (
                    err.code in _RETRYABLE_STATUSES
                    and attempt < _MAX_TRANSPORT_ATTEMPTS
                ):
                    delay = retry_after if retry_after is not None else 2.0**attempt
                    logging.warning(
                        "LLM endpoint returned %s; retrying in %.1fs (attempt %s/%s)",
                        err.code,
                        delay,
                        attempt,
                        _MAX_TRANSPORT_ATTEMPTS,
                    )
                    time.sleep(min(delay, _MAX_RETRY_AFTER_SECONDS))
                    last_error = LlmHttpError(err.code, detail, retry_after)
                    continue
                raise LlmHttpError(err.code, detail, retry_after) from None
            except urllib.error.URLError as err:
                reason = getattr(err, "reason", err)
                if isinstance(reason, TimeoutError):
                    raise LlmTimeout(
                        f"request to {url} timed out after {self.timeout_seconds}s"
                    ) from None
                if attempt < _MAX_TRANSPORT_ATTEMPTS:
                    logging.warning(
                        "LLM endpoint unreachable (%s); retrying (attempt %s/%s)",
                        reason,
                        attempt,
                        _MAX_TRANSPORT_ATTEMPTS,
                    )
                    time.sleep(2.0**attempt)
                    last_error = LlmTransportError(str(reason))
                    continue
                raise LlmTransportError(f"could not reach {url}: {reason}") from None
            except TimeoutError:
                raise LlmTimeout(
                    f"request to {url} timed out after {self.timeout_seconds}s"
                ) from None

            del headers
            if status >= 400:
                raise LlmHttpError(status, raw.decode("utf-8", "replace")[:500])
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as err:
                raise LlmTransportError(f"endpoint returned a non-JSON body: {err}")

        raise last_error or LlmTransportError(f"could not reach {url}")

    # -- public API -------------------------------------------------------

    def fetch_model_catalog(self) -> Any:
        """Raw ``GET /models`` payload. Callers pass it to ``ModelCatalog``."""

        return self._send("models", None, "GET")

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.2,
        response_format: Optional[Mapping[str, Any]] = None,
    ) -> LlmResponse:
        """Request one completion and normalise the response.

        No tool definitions are ever sent, so a response containing tool calls is
        an anomaly. It is flagged on the result rather than acted on -- v1 never
        executes model-requested tools.
        """

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "max_tokens": int(max_output_tokens),
            "temperature": float(temperature),
            # Anti-repetition: discourages the reasoning-loop that re-derives the
            # same text until the output budget is exhausted. Standard
            # OpenAI-compatible parameters; ignored by providers that lack them.
            "frequency_penalty": float(self.frequency_penalty),
            "presence_penalty": float(self.presence_penalty),
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = dict(response_format)
            # Defense in depth for reasoning models: on structured decision
            # calls, cap how much of the budget the model may spend on hidden
            # reasoning so there is always headroom left for the JSON answer.
            # The cap scales with the (now larger) budget but is bounded, so the
            # thinking phase is not starved yet the JSON always has room. This is
            # an OpenRouter control; OpenRouter (and plain OpenAI / Ollama)
            # ignore unknown fields, so this is a safe no-op elsewhere.
            reasoning_cap = min(
                _REASONING_CAP_CEILING, int(max_output_tokens) // _REASONING_CAP_DIVISOR
            )
            if reasoning_cap > 0:
                payload["reasoning"] = {"max_tokens": reasoning_cap}
        if self.session_id:
            payload["user"] = self.session_id

        started = time.monotonic()
        body = self._send("chat/completions", payload, "POST")
        latency = time.monotonic() - started

        if not isinstance(body, Mapping):
            raise LlmTransportError("chat completion response was not a JSON object")
        error = body.get("error")
        if isinstance(error, Mapping):
            raise LlmHttpError(
                _as_int(error.get("code")) or 500,
                str(error.get("message", "provider reported an error"))[:500],
            )

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LlmTransportError("chat completion response contained no choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise LlmTransportError("chat completion choice was malformed")
        message = choice.get("message")
        text = ""
        had_tool_calls = False
        if isinstance(message, Mapping):
            text = _text_from_field(message.get("content"))
            if not text.strip():
                # Reasoning models emit their thinking into a separate channel
                # before the visible answer; when the budget is tight the JSON
                # answer can land there instead of in ``content``, leaving
                # ``content`` empty. Recover it from the reasoning channel only
                # when ``content`` yielded nothing, so normal replies are never
                # affected. OpenRouter uses ``reasoning``; some providers use
                # ``reasoning_content``.
                reasoning_text = _text_from_field(
                    message.get("reasoning_content")
                ) or _text_from_field(message.get("reasoning"))
                if reasoning_text.strip():
                    logging.info(
                        "LLM content was empty; recovered answer from the "
                        "reasoning channel (reasoning-model fallback)"
                    )
                    text = reasoning_text
            had_tool_calls = bool(message.get("tool_calls"))
        if had_tool_calls:
            logging.warning(
                "LLM returned tool calls although none were offered; ignoring them"
            )

        return LlmResponse(
            text=text,
            usage=TokenUsage.from_payload(body.get("usage")),
            model=str(body.get("model") or self.model),
            finish_reason=_normalise_finish_reason(choice.get("finish_reason")),
            request_id=(str(body["id"]) if isinstance(body.get("id"), str) else None),
            latency_seconds=latency,
            had_tool_calls=had_tool_calls,
        )


def _text_from_field(value: Any) -> str:
    """Extract text from a chat field that is either a string or content parts.

    Providers return message ``content`` (and reasoning channels) as either a
    plain string or a list of ``{"text": ...}`` parts; this normalises both.
    """

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(part.get("text", "")) for part in value if isinstance(part, Mapping)
        )
    return ""


def _read_error_body(err: urllib.error.HTTPError) -> str:
    try:
        raw = err.read()
    except Exception:  # pragma: no cover - defensive
        return err.reason if isinstance(err.reason, str) else "unknown error"
    text = raw.decode("utf-8", "replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text[:500]
    if isinstance(parsed, Mapping):
        error = parsed.get("error")
        if isinstance(error, Mapping):
            return str(error.get("message", text))[:500]
        if isinstance(error, str):
            return error[:500]
    return text[:500]


def _retry_after_seconds(headers: Any) -> Optional[float]:
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except Exception:  # pragma: no cover - defensive
        return None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None
