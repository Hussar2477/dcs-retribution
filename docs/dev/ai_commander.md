# LLM AI commander (OPFOR)

This document describes the optional language-model commander for the OPFOR
(RED) coalition: what it is allowed to do, how a turn flows through it, the
fairness guarantees it makes, how to audit a turn after the fact, how to
configure a provider, what a turn costs, and what is deliberately out of scope.

The feature is **off by default**. With `ai_commander_enabled` false, nothing in
`game/ai_commander/` runs, no file is written, no network request is made, and
turn processing is byte-for-byte the stock behaviour.

## 1. What it is, and what it is not

It **is** a turn-level strategic advisor for RED. Once per RED turn it is given
a sanitised briefing and returns a short structured document ranking priorities:

* an overall strategy posture,
* a ranked ordering of the active fronts,
* a per-front push posture (retreat / hold / probe / push / breakthrough),
* a ranked ordering of spending categories,
* a ranked ordering of *known* target sets with a mission purpose for each,
* a reserve policy, and
* a free-text `commander_intent` used only for the audit log and the UI.

It **is not** in the execution path. It never names a unit, a quantity, a price,
a coordinate, a waypoint, a loadout, a flight size, a time on target, a damage
result or a capture. Everything mechanical is still done by Retribution's own
deterministic code: the HTN planner, the procurement AI, the packages and flight
plans, the mission generator, the combat and capture resolution, and the save
file. The model's output only changes the *order* in which the existing planner
is offered its existing tasks, the stance set on a front, and the weighting the
existing procurement code uses.

Concretely, the accepted decision is turned into a `CommanderDirective`
(`game/ai_commander/directive.py`) which feeds three deterministic adapters in
`game/ai_commander/execution.py`:

| Adapter | Effect |
| --- | --- |
| `task_order_for(directive)` | Reorders the compound tasks offered to the existing HTN theater commander. |
| `apply_front_postures(...)` | Sets an existing `CombatStance` on a front, only if that stance is already legal there. |
| `DirectedProcurementAi` | Nudges the existing procurement split and reserve share. Prices, availability and capacity checks stay in stock code. |

If any of this is unavailable, refused, or fails, the built-in RED automation
takes the turn instead and the campaign continues normally.

## 2. Module map

All of it lives in `game/ai_commander/`, plus one call site and the UI.

| Module | Responsibility |
| --- | --- |
| `enums.py` | Every closed vocabulary: intel policy, strategy, posture, target-set category, procurement category, mission purpose, reserve policy, confidence, precision, strength/affordability bands, personality, fallback reason. |
| `serialization.py` | `jsonable()` and `stable_hash()` — deterministic, ordering-independent JSON and hashing used for every hash in the brief and the audit record. |
| `intel.py` | `IntelProjector` and `RedCommanderBrief`. Builds the fair briefing; **this is the fairness boundary**. |
| `postures.py` | Maps Retribution `CombatStance` values to/from the `FrontPosture` vocabulary, and computes which postures are legal on a front. |
| `decision.py` | The decision JSON schema, the example document, `extract_json_object`, and `validate_decision` — schema/ID/rank/enum validation. |
| `directive.py` | `CommanderDirective`, the validated, legality-checked intent object. |
| `legality.py` | `LegalityChecker` — rechecks an already schema-valid decision against *live* state (ownership, reachability, affordability, capacity, revision). |
| `pricing.py` | Token estimation, the `/models` catalogue, `ModelPrice`, and `CostLedger` (the per-turn cap). |
| `llmclient.py` | The only networking code. Stdlib `urllib` only. Timeouts, bounded retries, usage parsing, redacting `__repr__`. |
| `prompt.py` | System prompt, personality text, user prompt, repair prompt, and the `response_format` request. |
| `execution.py` | The deterministic adapters listed above. |
| `audit.py` | `AiDecisionRecord` and `AuditLog` — the on-disk decision log. |
| `secretstore.py` | Where the API key lives, and how it is masked. |
| `config.py` | `AiCommanderConfig`, built from `Settings` plus the secret store. |
| `controller.py` | `RedCommanderTurn` — orchestrates one turn. Never raises. |

Outside the package:

* `game/coalition.py` — the single call site, on the RED coalition's turn.
* `game/settings/settings.py` — ten `ai_commander_*` options on a new
  "AI Opponent" settings page (`AI_OPPONENT_PAGE`), split into an
  "LLM Commander" and a "Fairness and Audit" section.
* `game/settings/textoption.py` — a new free-text option kind, needed for the
  model identifier and base URL.
* `qt_ui/windows/settings/aicommanderkey.py` — the API key widget. Writes to the
  secret store, never to `Settings`, so the key can never reach a save file.
* `qt_ui/windows/aicommander/QAiCommanderLogWindow.py` — the read-only decision
  log viewer, opened from the "AI Log" toolbar action in the main window.

## 3. Per-turn control flow

`RedCommanderTurn.run()` is the whole turn. It is written so that **it never
raises**: every exit path either returns an accepted directive or a
`CommanderTurnResult` carrying a `FallbackReason`, and the caller in
`game/coalition.py` treats a fallback exactly like the stock code path.

1. **Disabled check.** If the feature is off, return immediately with
   `FallbackReason.DISABLED`. No brief is projected and *no log file is
   written*.
2. **Project the brief.** `IntelProjector(game, policy).project(...)` builds
   `RedCommanderBrief` from live state, with the fairness filter applied. This
   also produces `campaign_id_hash` (stable per campaign) and
   `campaign_revision` (a digest of turn number, RED budget, base ownership and
   RED deployable strength).
3. **Replay guard.** If the audit log already holds an accepted decision for
   *this* `campaign_revision`, that decision is replayed rather than re-bought.
   Re-entering a turn therefore cannot double-spend the budget or produce a
   different plan.
4. **Usability check.** A misconfigured setup (`is_usable` false — e.g. a remote
   base URL with no key) falls back with `FallbackReason.NOT_CONFIGURED`.
5. **Catalogue lookup.** One unbilled `GET /models` request records the live
   price, context length and notes for the configured model. If it fails, a
   deliberately pessimistic fallback price is used
   ($3.00/$15.00 per million in/out) and the record says the price is a
   fallback. Stale prices are never silently treated as authoritative.
6. **Seed the ledger.** `CostLedger(cap, already_spent=audit_log.spent_this_turn)`
   — the cap is enforced across the whole turn, not per call.
7. **Initial attempt.** Reserve the worst-case cost *before* sending. If the
   reservation would exceed the remaining cap, nothing is sent and the turn
   falls back with `FallbackReason.COST_CAP`.
8. **Validate.** `extract_json_object` then `validate_decision`. Fatal problems
   (non-object root, wrong `schema_version`, wrong `turn_id`, revision
   mismatch, unparseable enums) reject the whole document. Non-fatal problems
   (an unknown ID, a duplicate ID, a duplicate rank, ranks not starting at 1, an
   illegal posture, an over-long intent, unexpected keys) drop or truncate just
   the offending element and record a `Rejection` for it.
9. **At most one repair attempt.** If and only if validation failed *and* the
   remaining budget covers another worst-case call, one repair request is sent
   containing the validation errors and the same legal IDs — no extra hidden
   state. There is no retry loop beyond this.
10. **Legality check.** `LegalityChecker` re-tests the surviving decision
    against live state: is that front still ours, is that base still reachable
    and owned, can we actually afford the cheapest item in that spending
    category, does the base have capacity, is the revision still current. Each
    refusal carries a human-readable reason and is logged.
11. **Accept or fall back.** If anything legal survives, the directive is built
    and handed to the deterministic adapters. If nothing legal survives, the
    turn falls back with `FallbackReason.NO_LEGAL_CONTENT`.
12. **Write the record.** One JSON file per attempt-set per turn. `AuditLog.write()`
    never raises; a failure to write returns `None` and the turn still completes.

Any transport error, HTTP error, timeout, unexpected exception, or repeatedly
malformed output lands on the same deterministic fallback. The list of reasons
is the `FallbackReason` enum: `DISABLED`, `NOT_CONFIGURED`, `COST_CAP`,
`TRANSPORT_ERROR`, `HTTP_ERROR`, `TIMEOUT`, `MALFORMED_RESPONSE`,
`STALE_RESPONSE`, `NO_LEGAL_CONTENT`, `UNEXPECTED_ERROR`.

## 4. Fairness and anti-cheat guarantees

The design goal is that the commander is *interesting*, not *omniscient*. Four
separate mechanisms enforce that.

### 4.1 It only ever sees the projected brief

The model is given the `RedCommanderBrief` and nothing else. There is no code
path that hands it a `Game`, a `ControlPoint`, a save file, a file path, an
environment variable or a credential. The brief is an immutable dataclass tree
of primitives and closed enums, produced by `IntelProjector`, and it is the
*only* input to `prompt.build_messages`.

Everything RED can see about BLUE is deliberately coarse:

* Enemy strength on a front is a **band** (`_band_for_ratio`, thresholds at
  0.4 / 0.8 / 1.25 / 2.5 of the friendly-to-enemy deployable ratio), not a
  count.
* Target sets are **classes with a count** (`TS-1`, "enemy air defences", 3
  known), with a `confidence` of `PROBABLE` and a `location_precision` of
  `AREA` — never a coordinate, never a unit name, never a group ID.
* Affordability of a spending category is a **band** (`NONE` / `LIMITED` /
  `COMFORTABLE`), computed against the cheapest legal item, so the model is
  never handed a price list to arbitrage.
* Anything BLUE that RED could not plausibly observe is dropped entirely before
  the brief is built: `_is_observable()` keeps an object only if it sits inside
  RED's own threat zone or within 120 km (`_REALISTIC_OBSERVATION_RANGE_METERS`)
  of a RED-held control point.

### 4.2 It can only speak the closed decision vocabulary

The decision schema has no free-numeric and no free-identifier field. Every ID
the model returns must be one of the opaque, turn-scoped IDs that were in the
brief (`FRONT-n`, `TS-n`, `PROC-n`); every enum value must be a member of the
enum; ranks are integers that must be unique and start at 1; `commander_intent`
is truncated at `MAX_INTENT_CHARACTERS` (600) and lists at `MAX_LIST_ENTRIES`
(32). An invented ID is not a bug to be tolerated — it is a logged rejection.

Because the IDs are re-minted per turn from the projected brief, an ID cannot be
used to smuggle knowledge from one turn to the next, and it cannot be used to
address anything the brief did not already mention.

### 4.3 Legality is rechecked against live state, after validation

Schema validity is not authority. `LegalityChecker` runs against the live
`Game` immediately before application and refuses, with a reason:

* a posture that is not legal on that front given its force balance
  (the message names the legal values, e.g. *"legal values were retreat"*),
* a spending priority RED cannot pay for — the message quotes the real number,
  e.g. *"cheapest available airframe costs 22 but only 5 is available"*,
  *"cheapest ground unit costs 12 but only 5 is available"*,
  *"runway repair costs 100 but only 5 is available"*,
* a front, base or target set that is no longer ours / reachable / present,
* a `campaign_revision` that no longer matches:
  *"campaign state changed between briefing and application"*.

If nothing legal survives, the refusal message says so explicitly —
*"nothing in the decision was legal against live state, so the built-in RED
automation keeps control of this turn"* — and the built-in automation runs.

### 4.4 Model output is untrusted data

The response is parsed as JSON and matched against the schema. It is never
`eval`'d, never used to build a path, never written into a save, and never used
as a format string. Tool calls are refused in v1: if a response arrives with
`tool_calls`, that is recorded as a `<tool_calls>` rejection. The save file is
still written exclusively by Retribution's own persistence code from its own
objects.

## 5. What `REALISTIC` withholds versus `FULL_PARITY`

Two intel policies are selectable in the settings UI
(`ai_commander_intel_policy`, "Fairness and Audit" section).

> **This is a new restriction that Retribution itself does not have.**
> Stock Retribution has no campaign-layer fog of war. The human player's
> "Enemy Info" checkbox in the Intel window exposes the full enemy economy,
> aircraft inventory and ground forces, and the built-in OPFOR auto-planner
> reads live game state directly with no filtering at all. `REALISTIC` is
> therefore **not** a mirror of an existing Retribution feature and it does not
> re-implement one. It is a restriction invented for this feature, applied only
> to what the language model is shown, and it makes the LLM commander *less*
> informed than the built-in OPFOR auto-planner it replaces. The
> `opfor_autoplanner_aggressiveness` setting is a threat-radius ratio used by
> the stock planner; it is unrelated to this and is not a strategy dial.

The brief always carries a `withheld_fields` list naming what was removed, so
an auditor can see the policy that was in force without diffing two runs.

Under `REALISTIC` (recommended, the default) nine field groups are withheld:

| Withheld field | Meaning |
| --- | --- |
| `enemy_budget` | BLUE's cash on hand. |
| `enemy_income` | BLUE's income per turn. |
| `enemy_squadron_rosters` | BLUE squadron names, aircraft types and pilot counts. |
| `enemy_aircraft_inventory` | How many of what BLUE has, and where. |
| `enemy_exact_ground_unit_counts` | Replaced by a strength band per front. |
| `enemy_planned_flights` | BLUE's ATO / planned packages for the coming turn. |
| `enemy_unit_coordinates` | Exact positions; replaced by `AREA` precision. |
| `enemy_pending_purchases` | What BLUE has on order. |
| `campaign_save_data` | The save/`Game` object itself, in any form. |

Under `FULL_PARITY` only three remain withheld — `enemy_planned_flights`,
`enemy_unit_coordinates` and `campaign_save_data`. `FULL_PARITY` deliberately
matches what the human player can already read from the Intel window with
"Enemy Info" ticked: exact enemy counts, budget and inventory. It exists so
that a player who considers the Intel window part of normal play can give the
AI the same picture. Under `FULL_PARITY` the projector also stops filtering by
observability (`_is_observable` returns `True` unconditionally), upgrades
confidence to `CONFIRMED` and precision to `PRECISE`, and populates
`FrontView.enemy_unit_count`.

Even under `FULL_PARITY`, BLUE's planned flights, exact coordinates and the save
data are still never sent, and the authority boundary of section 4 is unchanged
— the model still cannot do anything except rank priorities.

The intel-leak tests in `tests/ai_commander/test_intel_leak.py` enforce this by
seeding recognisable sentinel values on the BLUE side of a synthetic campaign
(a canary budget, income, squadron name, aircraft name, pilot count, planned
package name, unit counts, capacity, an undetected TGO and a base placed far
outside RED's observation range) and then recursively scanning the fully
serialised brief for any of them. They also assert that RED's *own* equivalents
**are** present, so the filter cannot pass by being trivially empty.

## 6. Auditing a turn

Every turn that reaches the projector writes one JSON file. Nothing is silently
dropped: refusals, fallbacks and cost are all in the record.

**Where.** `<audit root>/AiDecisions/<campaign_id_hash>/turn_<NNNN>_<II>.json`,
where `<NNNN>` is the turn number and `<II>` distinguishes multiple records for
the same turn (capped at `MAX_RECORDS_PER_TURN` = 50). The audit root defaults
to the user data directory (`~/.local/share/DCSRetribution` on POSIX,
`%LOCALAPPDATA%\DCSRetribution` on Windows) and can be overridden.
`campaign_id_hash` is a hash of theater, faction names and base names — it
contains no path and no personal data.

**Reading it.** In the GUI, use the **AI Log** toolbar action in the main
window; it lists the records for the current campaign and shows one decision at
a time, read-only. On disk the file is plain JSON, so `jq` works fine.

To reconstruct a turn, walk the record in this order:

1. **Identity** — `record_schema_version`, `written_at`, `campaign_id_hash`,
   `turn_id`, `campaign_revision`, `personality`.
2. **What the commander knew** — `intel_policy`, `intel_hash`, `intel_brief`
   (the entire brief as sent, so fairness is auditable after the fact),
   `intel_rendered` (the compact text actually placed in the prompt), and
   `intel_brief.withheld_fields`. If you want to prove no BLUE-private value
   was ever sent, grep *this* object; it is the complete input.
3. **What it was asked** — `decision_schema_hash`, plus `attempts[*].prompt_messages`
   when raw prompt logging is enabled (`prompt_logging_enabled` records whether
   it was). When it is off, `attempts[*].prompt_hash` still lets you prove which
   prompt was sent without storing the campaign detail it contains.
4. **The provider round trips** — `base_url`, `configured_model`,
   `catalog_retrieved_at`, `catalog_input_price_per_million`,
   `catalog_output_price_per_million`, `catalog_context_length`,
   `catalog_notes`, then per attempt: `attempt`, `kind` (`initial` / `repair`),
   `started_at`, `latency_seconds`, `requested_model`, **`actual_model`**
   (so provider-side alias resolution is visible), `response_id`, `provider`,
   `finish_reason`, `retries`, `http_status`, `error`, the token counts
   (`prompt_tokens`, `completion_tokens`, `cached_tokens`, `reasoning_tokens`,
   `total_tokens`), `reserved_cost`, `actual_cost`, `cost_is_estimated`, and
   `response_text` / `response_hash`.
5. **What it proposed** — `parsed_decision` (schema-valid, before legality).
6. **What was refused** — `rejections`, a list of `{element, reason, value}`
   objects. This is the interesting part when the AI "did nothing": every
   dropped element is here with its reason.
7. **What was accepted** — `accepted_directive`.
8. **What deterministic code then did** — `planner_task_order` (the order the
   HTN planner was offered its tasks), `posture_applications` (one entry per
   requested posture, applied or not, with why), `procurement_notes`.
9. **Cost** — `cost_cap_per_turn`, `prior_cost_this_turn`, `estimated_cost`,
   `reserved_cost`, `actual_cost`, and the derived `total_prompt_tokens` /
   `total_completion_tokens`. `cost_is_estimated` false means the figure came
   from the provider's own `usage` block and is authoritative.
10. **Outcome** — `accepted`, `fallback_reason`, `fallback_policy`,
    `replayed_from_turn_record`, `notes`.

Useful checks:

```bash
# Which turns fell back, and why?
jq -r '[.turn_id, (.fallback_reason // "accepted")] | @tsv' \
  ~/.local/share/DCSRetribution/AiDecisions/*/turn_*.json

# Everything that was refused this campaign, with reasons.
jq -r '.rejections[] | [.element, .reason] | @tsv' \
  ~/.local/share/DCSRetribution/AiDecisions/*/turn_*.json

# Prove the brief carried no BLUE budget/inventory (REALISTIC).
jq '.intel_brief.withheld_fields' \
  ~/.local/share/DCSRetribution/AiDecisions/*/turn_0007_00.json

# What did a turn actually cost?
jq '{cap: .cost_cap_per_turn, actual: .actual_cost, estimated: .cost_is_estimated}' \
  ~/.local/share/DCSRetribution/AiDecisions/*/turn_0007_00.json
```

The API key never appears in any of this. It is not in the record, not in the
config's `to_dict()` (which emits only a boolean `api_key_configured`), and not
in any log line — `ChatCompletionClient.__repr__` prints
`api_key=<redacted:set>` and `AiCommanderConfig.describe()` masks the value.

## 7. Provider and model configuration

All of the following is on the **AI Opponent** settings page.

| Setting | Default | Notes |
| --- | --- | --- |
| `ai_commander_enabled` | `False` | Master switch. Off means the feature does not exist at runtime. |
| `ai_commander_model` | `deepseek/deepseek-v4-flash-0731` | The provider's exact model ID. Prefer a dated ID over a `latest` alias for reproducibility. |
| `ai_commander_base_url` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible chat-completions endpoint. |
| `ai_commander_personality` | `balanced` | `cautious` / `balanced` / `aggressive` / `attritional`. Changes only how the commander is briefed, never what it is allowed to do. |
| `ai_commander_intel_policy` | `realistic` | See section 5. |
| `ai_commander_cost_cap_per_turn` | `0.5` | US dollars, per RED turn, worst case, enforced *before* sending. |
| `ai_commander_timeout_seconds` | `90` | Exceeded means fall back. |
| `ai_commander_max_output_tokens` | `2000` | Caps the size, and therefore the cost, of each response. |
| `ai_commander_log_prompts` | `True` | Store raw prompts in the audit record. They contain campaign information (though no BLUE-private data), so this is user-configurable. |
| `ai_commander_fallback_to_builtin` | `True` | Keep this on. |

### Where the key lives

The API key is **not** a setting and is **not** part of the campaign. It is read,
in order:

1. the `OPENROUTER_API_KEY` environment variable, if set; otherwise
2. `ai_commander_secrets.json` in the user data directory
   (`~/.local/share/DCSRetribution/` on POSIX,
   `%LOCALAPPDATA%\DCSRetribution\` on Windows), under the key
   `openrouter_api_key`.

The file is created with mode `0600` and unrelated fields in it are preserved on
save. Because the key lives outside `Settings`, it can never be pickled into a
`.retribution` save, shared with a save file, or committed. It is also never
logged: `secretstore.mask()` renders a short key as all asterisks and a long one
as `abcdef...wxyz (N chars)`, and every `__repr__` on the path is redacting.

### OpenRouter

Point `ai_commander_base_url` at `https://openrouter.ai/api/v1` (the default),
set a model ID, and paste the key into the API key box on the AI Opponent
settings page — or export it:

```bash
export OPENROUTER_API_KEY=<PLACEHOLDER>
```

The client sends `Authorization: Bearer <key>`, requests an explicit `model`
(never the account default), asks for a JSON response format, caps output
tokens, and does not enable any provider plugins. HTTP 429 and 5xx get a small
bounded backoff; 401 and 402 do not retry — they fall back immediately, since a
bad key or an empty balance will not fix itself within a turn.

### Local models (Ollama and friends)

Any OpenAI-compatible server works. For Ollama:

```
ai_commander_base_url = http://localhost:11434/v1
ai_commander_model    = <whatever you have pulled>
```

`AiCommanderConfig.is_local` recognises loopback/local base URLs; a local
endpoint needs no API key (`requires_api_key` is false) and has no per-turn cost.
The `/models` catalogue lookup is still attempted, and if it is missing or
shaped differently the pessimistic fallback price is used for reservation
arithmetic only — with a local model the cap effectively never binds. Local
models are the right choice for anyone who does not want a paid dependency; the
only requirement is that the model can return a JSON object, and a malformed
reply just means the built-in automation takes the turn.

### The cost cap

`CostLedger` (in `pricing.py`) enforces

```
spent_this_turn + reserved_in_flight + worst_case_next_call <= cost_cap_per_turn
```

Worst case is computed *before* the request from the measured prompt length and
the configured output cap, using the live catalogue price (or the pessimistic
fallback). If the reservation does not fit, **the request is never sent** and the
turn falls back with `FallbackReason.COST_CAP`. When a response arrives, the
reservation is released and the actual charge settled — from the provider's own
`usage.cost` when present (recorded with `cost_is_estimated` false), otherwise
estimated from catalogue prices and returned token counts. Caching is treated as
a saving that may or may not materialise; it is never assumed when enforcing the
cap.

## 8. What a turn costs

There is no provider tokeniser in this repository and this feature adds **zero
runtime dependencies**, so token counts come from `pricing.estimate_tokens()` —
`ceil(len(text) / 4.0 * 1.15)`, the same estimator the controller uses to size
its reservation. The 1.15 factor deliberately over-counts so the reservation
errs high. The character counts below are exact measurements of the real
assembled prompt taken by `tools/ai_commander_dryrun.py`; the token counts are
that estimator applied to them. A provider that reports real usage overrides the
estimate for billing.

Measured prompt sizes on the harness's synthetic campaign:

| Text | Characters | Est. tokens |
| --- | --- | --- |
| initial prompt (cautious) | 7230 | 2079 |
| initial prompt (balanced) | 7169 | 2062 |
| initial prompt (aggressive) | 7203 | 2071 |
| initial prompt (attritional) | 7184 | 2066 |
| repair prompt (balanced) | 7449 | 2142 |
| representative response | 636 | 183 |

How the briefing grows with campaign size (measured by widening the synthetic
theater and reassembling the prompt each time):

| Fronts | Bases | Target sets | Characters | Est. tokens |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 6 | 7169 | 2062 |
| 2 | 6 | 6 | 7327 | 2107 |
| 4 | 10 | 6 | 7645 | 2198 |
| 8 | 18 | 6 | 8277 | 2380 |
| 16 | 34 | 6 | 9585 | 2756 |
| 32 | 66 | 6 | 12193 | 3506 |

The brief is a fixed-shape summary — one line per front, one per known target
class, one per procurement option — so it scales with the number of *fronts*,
not with the number of units, aircraft or objects in the campaign. A 32-front
theater is far larger than any stock campaign.

**Call-count assumption.** At most **two** billed chat completions per RED turn:
the initial request, plus exactly one repair request if and only if the first
response fails validation. There is no retry loop beyond that. One unbilled
`GET /models` lookup happens per turn. Provider-side transport retries (429/5xx)
re-send the same request and are billed only if a response is served. If a
future build adds a second decision point per turn, double the figures below.

* typical turn = 1 call: 2079 in + 183 out
* worst-case turn = 2 calls: 4221 in + 4000 out (both replies at the 2000-token cap)

| Model | $/M in | $/M out | Typical turn | Worst-case turn | Headroom vs $0.50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deepseek/deepseek-v4-flash-0731` (shipped default) | 0.09 | 0.18 | $0.00022 | $0.00110 | 455x |
| `qwen/qwen3.7-flash` (cheapest surveyed) | 0.03 | 0.13 | $0.00009 | $0.00065 | 773x |
| `openai/gpt-5.6-luna` | 0.10 | 0.60 | $0.00032 | $0.00282 | 177x |
| `z-ai/glm-5.2` | 0.76 | 2.42 | $0.00202 | $0.01289 | 39x |
| `moonshotai/kimi-k3` | 3.00 | 15.00 | $0.00898 | $0.07266 | 7x |
| catalogue unavailable (built-in pessimistic price) | 3.00 | 15.00 | $0.00898 | $0.07266 | 7x |

**Verdict: every surveyed model's worst-case turn stays under the $0.50 ceiling,
including the pessimistic fallback price used when the catalogue cannot be
read.** Even the most expensive model surveyed leaves roughly a 7x margin, so
the cap is a safety net against a pathological prompt or a price change, not a
routine constraint. A hundred-turn campaign on the shipped default costs on the
order of two US cents.

These are estimates from measured character counts and published prices, not
observed invoices. Prices were recorded on 2026-08-05 and change without notice;
the controller always prices from the live `/models` catalogue at runtime and
records the price it used in the audit record.

## 9. Tests and the dry-run harness

### Unit tests

`tests/ai_commander/` follows the existing pytest conventions (no `conftest.py`,
`__init__.py` per package, `MagicMock(spec=...)` / `cast("Game", ...)` stubs,
fully annotated). `tests/ai_commander/fakes.py` builds a synthetic RED-vs-BLUE
campaign entirely out of stubs — no DCS, no mission files, no network:

| File | Covers |
| --- | --- |
| `test_intel_leak.py` | The fairness boundary. BLUE sentinel values must not appear anywhere in the serialised brief under `REALISTIC`; RED's own equivalents must appear. |
| `test_decision_schema.py` | Malformed JSON, empty bodies, prose-wrapped JSON, missing/wrong `schema_version` and `turn_id`, unknown IDs, duplicate IDs, duplicate ranks, ranks not starting at 1, unknown enum values, unexpected keys, over-long intent, list-length caps. |
| `test_legality.py` | Overspending, fronts and bases RED does not own or cannot reach, buying aircraft with no squadron to fly them, bases with no supply source or unrepairable runway, illegal postures, and stale revisions — each rejected *with a logged reason* rather than applied. Also asserts the directive carries no amounts, so quantity and base-capacity limits stay with the engine. |
| `test_cost_cap.py` | Ledger arithmetic, refusal before sending when the worst-case reserve exceeds the cap, reserve/release/settle, and that the controller falls back instead of raising. |
| `test_fallback.py` | Transport error, HTTP error, timeout, repeatedly malformed output, unexpected exception — every one ends in the deterministic fallback so a turn never breaks. |
| `test_secrets_and_audit.py` | Key masking and redaction, that the key is absent from configs/records/reprs, audit record shape, per-turn cost accounting, and the replay guard. |

### CLI dry-run harness

`tools/ai_commander_dryrun.py` runs a complete offline RED turn against the
synthetic campaign and reports what happened. It works with **no API key** and
exits 0.

```bash
python tools/ai_commander_dryrun.py            # everything
python -m tools.ai_commander_dryrun --no-live  # skip the optional live call
python tools/ai_commander_dryrun.py --list
python tools/ai_commander_dryrun.py --scenario cheating-intel --verbose
python tools/ai_commander_dryrun.py --json /tmp/dryrun.json
```

It prints: the synthetic campaign layout; an intel-fairness sentinel scan; a
scenario table; a field-by-field dump of a real audit file; the token
measurements and the cost table in section 8; and a summary. The scenarios are

`valid`, `repair`, `malformed`, `truncated-json`, `cheating-intel`,
`cheating-overspend`, `illegal-posture`, `stale-state`, `tool-calls`,
`transport-failure`, `timeout`, `catalogue-unavailable`, `cost-cap`,
`not-configured`, `disabled`.

Each scenario asserts an expected outcome, so the harness doubles as an
integration smoke test. If `OPENROUTER_API_KEY` is set it will additionally make
**one** real request to confirm the provider path end to end; without a key it
says so and skips it. The key is never printed — only whether one was found.

## 10. Deferred scope

Deliberately **not** in this version:

* **Naval warfare.** Enemy shipping appears as a target *class* the commander can
  rank, but there is no naval-specific decision, no carrier group tasking and no
  sea-control reasoning.
* **Carrier operations.** No decisions about carrier station, recovery windows or
  air-wing composition.
* **Transports and cargo logistics.** Convoys and cargo ships are visible as
  target classes; RED's own transport, cargo and airlift planning is untouched
  and stays with the existing deterministic code.
* **BLUE as an AI commander.** The hook in `game/coalition.py` fires only for
  RED. Nothing in the package assumes RED, but running it for BLUE would need a
  fairness review of its own (the player's own information would become the
  model's), a second cost budget, and UI work.
* **A granular "active" role.** The commander decides once per turn at the
  strategic level. It does not shape individual packages, flight plans or
  target assignments.

### How the interfaces are shaped to accept the active role later

The v1 boundary is *authority*, not *granularity*, and the seams were left where
a later active role would need them:

* **The brief is the only input.** A finer role adds fields to
  `RedCommanderBrief` and entries to `withheld_fields`; it does not change what
  the model is allowed to touch. `intel_hash` already versions the brief so a
  richer brief is distinguishable in the audit log.
* **`IntelProjector` is the single filter.** Read-only planning *tools* for a
  future tool-calling role would be built on top of it, so they can only ever
  expose the already-sanitised RED view. `_is_observable()` is the one place
  observability is decided.
* **The decision schema is generated from the brief.** `decision_json_schema(brief)`
  and `decision_schema_hash(brief)` are derived, not hard-coded, so adding a
  decision block (for example package-level intent) does not require rewriting
  the validator, and existing audit records stay interpretable via their
  recorded schema hash.
* **`LegalityChecker` is the choke point.** Any new decision block gets a new
  legality rule and a new `Rejection` reason. The contract — *schema validity is
  not authority; live state decides* — does not change.
* **`CostLedger` is per turn, not per call.** It is seeded from
  `audit_log.spent_this_turn`, so a role that makes several calls (or per-tool
  authorisation with call-count limits and cumulative turn budgeting) plugs into
  the existing reserve/release/settle cycle without touching the cap semantics.
* **`LlmAttempt` records are a list.** `AiDecisionRecord.attempts` already holds
  many attempts with per-attempt cost and token accounting, so a multi-call
  role needs no record-format change.
* **The execution adapters are additive.** `execution.py` converts a directive
  into existing planner inputs. A finer role adds another adapter next to
  `task_order_for` / `apply_front_postures` / `DirectedProcurementAi`; it does
  not need a new execution path.

Real-time DCS unit control, mission scripting, arbitrary coordinates and any
direct mutation of persistence stay outside the LLM boundary until they are
separately designed and tested.

## 11. What has not been validated here

This feature was developed and tested on Linux. DCS World and the packaged
Retribution build are Windows-only, so the following is written and reviewed but
**not** exercised:

* a real campaign turn end to end inside DCS, including mission generation,
* the packaged (PyInstaller) build,
* Qt rendering of the new settings page, the key widget and the AI Log window,
* save/load round-tripping of a campaign created with the new settings present,
* real provider latency, real invoices and real `usage` blocks from OpenRouter.

Everything in sections 3 to 9 is covered by the headless tests and the dry-run
harness, which do not need DCS.
