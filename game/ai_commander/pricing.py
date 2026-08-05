"""Model catalogue, cost estimation and the hard per-turn spend cap.

The cap is enforced by *reservation*, not by measurement after the fact: before
any request is sent, its worst-case uncached cost is reserved against the turn's
remaining budget, and the request is refused if

    spent_this_turn + reserved_in_flight + worst_case_next_call > cap

This is what makes a retry safe. Measuring cost after the response arrives would
allow a single expensive call, or a repair attempt, to blow straight through the
ceiling.

Prices come from the provider's live catalogue (``GET /models`` on an
OpenAI-compatible endpoint). When the catalogue is unavailable, a conservative
fallback price is used so an unknown model can never be treated as free.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from game.ai_commander.serialization import jsonable

#: Used when the catalogue cannot be reached or the model is missing from it.
#: Deliberately expensive (roughly the priciest preferred model in the design
#: inputs) so that an unknown model is treated as a worst case, never as free.
FALLBACK_INPUT_PRICE_PER_MILLION = 3.00
FALLBACK_OUTPUT_PRICE_PER_MILLION = 15.00

#: Rough characters-per-token used only for *pre-flight* estimates. Providers
#: bill with their own tokenizer, so actual usage always overrides this. Four
#: characters per token is the widely used English approximation; the 1.15
#: safety factor biases the estimate upwards so reservations do not under-shoot.
CHARACTERS_PER_TOKEN = 4.0
TOKEN_ESTIMATE_SAFETY_FACTOR = 1.15


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text``.

    Only used to reserve budget before a call. It is an upper-biased heuristic,
    not a tokenizer, and is replaced by the provider's reported usage as soon as
    a response arrives.
    """

    if not text:
        return 0
    return int(
        math.ceil(len(text) / CHARACTERS_PER_TOKEN * TOKEN_ESTIMATE_SAFETY_FACTOR)
    )


class CostCapExceeded(Exception):
    """Raised when a request would push the turn over its spend cap."""

    def __init__(self, message: str, would_be_total: float, cap: float) -> None:
        super().__init__(message)
        self.message = message
        self.would_be_total = would_be_total
        self.cap = cap


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token prices for one model."""

    model_id: str
    input_per_million: float
    output_per_million: float
    context_length: Optional[int] = None
    supported_parameters: tuple[str, ...] = ()
    is_fallback_estimate: bool = False

    @property
    def supports_json_schema(self) -> bool:
        return "structured_outputs" in self.supported_parameters

    @property
    def supports_response_format(self) -> bool:
        return "response_format" in self.supported_parameters

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million
            + output_tokens * self.output_per_million
        ) / 1_000_000.0

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def fallback_for(cls, model_id: str) -> ModelPrice:
        return cls(
            model_id=model_id,
            input_per_million=FALLBACK_INPUT_PRICE_PER_MILLION,
            output_per_million=FALLBACK_OUTPUT_PRICE_PER_MILLION,
            is_fallback_estimate=True,
        )


@dataclass
class ModelCatalog:
    """A snapshot of the provider's model list.

    ``retrieved_at`` is recorded so stale prices are visible in the audit log and
    are never silently treated as authoritative.
    """

    prices: dict[str, ModelPrice] = field(default_factory=dict)
    retrieved_at: Optional[float] = None
    source: str = "unavailable"
    error: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.prices)

    def price_for(self, model_id: str) -> ModelPrice:
        price = self.prices.get(model_id)
        if price is not None:
            return price
        logging.info(
            "Model %s not found in the provider catalogue; using the "
            "conservative fallback price for budgeting",
            model_id,
        )
        return ModelPrice.fallback_for(model_id)

    def contains(self, model_id: str) -> bool:
        return model_id in self.prices

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "retrieved_at": self.retrieved_at,
            "model_count": len(self.prices),
            "error": self.error,
        }

    @classmethod
    def from_payload(cls, payload: Any, source: str) -> ModelCatalog:
        """Parse an OpenAI-compatible ``/models`` payload.

        Tolerates both OpenRouter's ``pricing`` block (dollars per token, as
        strings) and endpoints that omit pricing entirely (Ollama), in which case
        the model is recorded at zero price because nothing is being billed.
        """

        prices: dict[str, ModelPrice] = {}
        entries: Iterable[Any]
        if isinstance(payload, Mapping):
            raw = payload.get("data", payload.get("models", []))
            entries = raw if isinstance(raw, list) else []
        elif isinstance(payload, list):
            entries = payload
        else:
            entries = []

        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            model_id = entry.get("id") or entry.get("name")
            if not isinstance(model_id, str) or not model_id:
                continue
            pricing = entry.get("pricing")
            input_price = 0.0
            output_price = 0.0
            if isinstance(pricing, Mapping):
                input_price = _per_million(pricing.get("prompt"))
                output_price = _per_million(pricing.get("completion"))
            context = entry.get("context_length") or entry.get("context_window")
            supported = entry.get("supported_parameters")
            prices[model_id] = ModelPrice(
                model_id=model_id,
                input_per_million=input_price,
                output_per_million=output_price,
                context_length=(
                    int(context) if isinstance(context, (int, float)) else None
                ),
                supported_parameters=(
                    tuple(str(p) for p in supported if isinstance(p, str))
                    if isinstance(supported, list)
                    else ()
                ),
            )
        return cls(prices=prices, retrieved_at=time.time(), source=source)

    @classmethod
    def unavailable(cls, reason: str) -> ModelCatalog:
        return cls(
            prices={}, retrieved_at=time.time(), source="unavailable", error=reason
        )


def _per_million(raw: Any) -> float:
    """Convert a per-token price (possibly a string) to dollars per million."""

    if raw is None:
        return 0.0
    try:
        return float(raw) * 1_000_000.0
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class CostEntry:
    """One settled request."""

    model_id: str
    input_tokens: int
    output_tokens: int
    cost: float
    reported_by_provider: bool
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class CostLedger:
    """Tracks and caps spend for a single campaign turn.

    ``already_spent`` lets the ledger be seeded from the audit log, so the cap
    survives the fact that Retribution may re-run turn initialisation several
    times (cheat capture, front-line cheats, buying a TGO) within one turn.
    """

    def __init__(self, cap: float, already_spent: float = 0.0) -> None:
        self.cap = max(0.0, float(cap))
        self._settled = max(0.0, float(already_spent))
        self._reserved = 0.0
        self.entries: list[CostEntry] = []

    # -- state ------------------------------------------------------------

    @property
    def settled(self) -> float:
        return self._settled

    @property
    def reserved(self) -> float:
        return self._reserved

    @property
    def committed(self) -> float:
        """Everything spent plus everything currently in flight."""

        return self._settled + self._reserved

    @property
    def remaining(self) -> float:
        return max(0.0, self.cap - self.committed)

    # -- cap enforcement --------------------------------------------------

    def worst_case_cost(
        self, price: ModelPrice, input_tokens: int, max_output_tokens: int
    ) -> float:
        """Uncached worst case for a request: full prompt plus a full output."""

        return price.cost_for(input_tokens, max_output_tokens)

    def can_afford(
        self, price: ModelPrice, input_tokens: int, max_output_tokens: int
    ) -> bool:
        projected = self.committed + self.worst_case_cost(
            price, input_tokens, max_output_tokens
        )
        return projected <= self.cap

    def reserve(
        self,
        price: ModelPrice,
        input_tokens: int,
        max_output_tokens: int,
        label: str = "",
    ) -> float:
        """Reserve a request's worst-case cost, or raise :class:`CostCapExceeded`."""

        worst_case = self.worst_case_cost(price, input_tokens, max_output_tokens)
        projected = self.committed + worst_case
        if projected > self.cap:
            raise CostCapExceeded(
                f"{label or 'request'} would bring this turn's spend to "
                f"${projected:.4f}, over the ${self.cap:.2f} cap "
                f"(already spent ${self._settled:.4f}, "
                f"in flight ${self._reserved:.4f}, "
                f"worst case for this call ${worst_case:.4f})",
                projected,
                self.cap,
            )
        self._reserved += worst_case
        return worst_case

    def release(self, reserved: float) -> None:
        """Release a reservation without settling (used when a call fails)."""

        self._reserved = max(0.0, self._reserved - max(0.0, reserved))

    def settle(
        self,
        reserved: float,
        price: ModelPrice,
        input_tokens: int,
        output_tokens: int,
        provider_cost: Optional[float] = None,
        model_id: Optional[str] = None,
        label: str = "",
    ) -> CostEntry:
        """Replace a reservation with the actual charge.

        ``provider_cost`` (OpenRouter's ``usage.cost``) is authoritative when
        present; otherwise the catalogue price is applied to the reported token
        counts. A provider-reported cost lower than the reservation is a saving
        and is recorded as such -- caching is never assumed in advance.
        """

        self.release(reserved)
        if provider_cost is not None and provider_cost >= 0:
            cost = float(provider_cost)
            reported = True
        else:
            cost = price.cost_for(input_tokens, output_tokens)
            reported = False
        self._settled += cost
        entry = CostEntry(
            model_id=model_id or price.model_id,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            cost=cost,
            reported_by_provider=reported,
            label=label,
        )
        self.entries.append(entry)
        return entry

    # -- reporting --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "cap": round(self.cap, 6),
            "spent": round(self._settled, 6),
            "reserved_in_flight": round(self._reserved, 6),
            "remaining": round(self.remaining, 6),
            "calls": [e.to_dict() for e in self.entries],
            "input_tokens": sum(e.input_tokens for e in self.entries),
            "output_tokens": sum(e.output_tokens for e in self.entries),
        }
