# OpenRouter: small model, large model, and sequential routing

Measured on 2026-11-06 using a Mac mini M4 with 16 GB and local French Pocket TTS. Only synthetic text prompts were sent to OpenRouter; microphone audio and voice references remained local.

- Small model: `google/gemini-2.5-flash-lite` (observed provider: Google AI Studio).
- Large model: `anthropic/claude-sonnet-4.6` (observed provider: Claude Platform on AWS).
- Routed: the small model either streams its own answer or emits an unspoken delegation marker; only then is the large model called.

## Results

Text submission to first generated audio, including WAN and routing, excluding recognition, endpoint silence, and actual playback. Medians of **two requests per cell**: a smoke test, not a robust model ranking.

| Prompt | Small only | Large only | Small decides, then delegates if needed | Selected route |
| --- | ---: | ---: | ---: | --- |
| Capital-city fact | 491 ms | 928 ms | 526 ms | small |
| Single addition | 479 ms | 837 ms | 508 ms | small |
| Three modular constraints | 582 ms | 1044 ms | 1529 ms | large |

OpenRouter-reported total cost for **18 measured turns / 20 API calls**: **$0.0073784 USD**. No additional paid model warm-up requests were made. Costs come from response usage fields, not a token-price estimate.

## What this shows

The small model makes simple requests faster and cheaper than always calling the large model. Sequential delegation does **not** make the difficult path faster: the modular-arithmetic request took 1,529 ms to first audio when routed, compared with 1,044 ms directly, approximately 485 ms extra. The small model answered that same question correctly on its own in both trials. This corpus therefore does not establish a quality advantage for delegation.

Answers were checked against Paris, 42, and 23 respectively. The reasoning case is a compact arithmetic exercise, not a representative difficult-task evaluation. Routing policy explicitly delegates multi-step constraints; it is not a calibrated uncertainty model. The direct and routed simple answers also differ in length (for example a full sentence versus “Paris”), which affects TTS latency.

For a voice product, this supports a small-model fast path plus direct routing to a larger model when complexity is already known. A speculative parallel large-model request could avoid the sequential delay, but would increase cost; it was **not** implemented or benchmarked here. Never count a filler acknowledgement as the substantive response.

All modes use real streaming model responses, temperature 0, a 128-token cap, and real local speech synthesis. Mode order rotates. There is no explicit provider pin or latency-based provider sort. Network conditions, provider selection, response lengths, and small sample size limit generalization. The comparison does not claim these are the fastest available OpenRouter models.

## Run it

Store `OPENROUTER_API_KEY` in your local `.env` (ignored by Git), with owner-only permissions. The benchmark loads it without printing it. Requires an account with enough credit; runs make paid API requests.

```sh
uv sync --extra mac --group benchmark
uv run --extra mac --group benchmark python benchmarks/openrouter_cascade.py \
  --small google/gemini-2.5-flash-lite --large anthropic/claude-sonnet-4.6 \
  --runs 2 --output benchmarks/results/my-openrouter-cascade.json
```

For ordinary live conversations using the small model directly:

```sh
SPEED_LLM_URL=https://openrouter.ai/api/v1 \
SPEED_LLM_MODEL=google/gemini-2.5-flash-lite \
SPEED_LANGUAGE=fr SPEED_TTS_LANGUAGE=french SPEED_VOICE=estelle \
  uv run --env-file .env --extra mac speed-voice
```

The server maps `OPENROUTER_API_KEY` only when the configured endpoint exactly matches OpenRouter. Explicit `SPEED_LLM_KEY` still takes precedence. Sequential routing is an experimental benchmark adapter, not the default production conversation policy.

[Raw responses, routing decisions, providers, usage and timing](results/openrouter-cascade.json). The integration uses the documented [OpenRouter streaming Chat Completions API](https://openrouter.ai/docs/api-reference/overview).
