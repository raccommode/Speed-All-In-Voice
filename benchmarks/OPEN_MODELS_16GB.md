# Open models for a 16 GiB VRAM voice stack

Tested on 2026-11-06 through **OpenRouter**, with local French Pocket TTS on a Mac mini M4. These are cloud inference latencies, **not** measurements on a local 16 GiB GPU. All four selected model families have Apache 2.0 weights; no closed model was used in this batch.

## First audio and answer checks

Medians in milliseconds from committed text to first generated audio, including WAN, excluding speech recognition, endpoint silence, and physical playback. Two observations per cell, except the 3B column combines four observations per prompt from two separate pair experiments. Providers and precision were not pinned.

| Direct model | Capital-city fact | Single addition | Modular arithmetic | Modular arithmetic answer check |
| --- | ---: | ---: | ---: | --- |
| Ministral 3 3B | 502 | 611 | 530 | Wrong first answer in 4/4 |
| Ministral 3 8B | 814 | 576 | 472 | Wrong first answer in 2/2 |
| Ministral 3 14B | 554 | 739 | 620 | Correct answer in 2/2 |
| Qwen3.5 9B | 582 | 763 | 882 | Correct answer in 2/2 |

The correct answers are Paris, 42, and 23. In direct mode, Ministral 3B began the constraint answer with 14, then made inconsistent corrections; 8B began with 32 and also made incorrect corrections. These early spoken errors matter more than the low first-audio number. Ministral 14B and Qwen3.5 9B gave 23 and checked its remainders in both observations. This is a single reasoning exercise, not an accuracy benchmark.

Qwen3.5 was requested with `reasoning.enabled=false` for voice latency; this requests disabling thinking rather than merely hiding it. Observed providers included multiple backends, and its two reasoning latencies ranged from 425 to 1,339 ms despite identical answer text. Do not interpret the median as a reliable provider-independent speed ranking.

## Small-to-large routing result

Two complete 18-turn experiments compared 3B with 8B and 3B with 14B. The 3B router chose to answer itself in **all 12 routed turns**, including all four multi-step tasks that the system policy told it to delegate. Its routed arithmetic answers happened to be correct, unlike its direct answers. This demonstrates prompt sensitivity and routing-policy noncompliance, not a dependable classifier.

**No actual delegation to 8B or 14B occurred in these experiments.** The routed timings therefore do not measure the speed of a two-model handoff. Do not deploy this gate as a reliable complexity detector based on these results.

## Will they fit in 16 GiB?

The following are planning estimates for **one text-only session, about 4,096 context tokens, Q4_K_M weights, and room for voice inference**. They are not measured peak allocations. GPU runtime workspaces, KV cache, concurrency, context length, and optional vision projectors can materially change the footprint. A model file size is not total VRAM.

| Model | Published Q4_K_M file | Estimated single-model stack envelope |
| --- | ---: | ---: |
| Ministral 3 3B | 2.00 GiB | roughly 4–6 GiB |
| Ministral 3 8B | 4.84 GiB | roughly 7–10 GiB |
| Ministral 3 14B | 7.67 GiB | roughly 10–14 GiB |
| Qwen3.5 9B | 5.29 GiB | roughly 8–11 GiB |

These envelopes are engineering estimates derived from published checkpoint sizes plus allowances, not vendor guarantees. For Ministral, dense FP16 KV storage at 4K is approximately 0.406 / 0.531 / 0.625 GiB for 3B / 8B / 14B. Runtime and speech allowances account for the rest. The dense KV formula was not applied to Qwen3.5's hybrid architecture.

Loading **3B + 8B together** is a plausible 16 GiB configuration at short context: approximately 6.84 GiB of GGUF weights plus 0.94 GiB FP16 KV, leaving room for runtime and voice. **3B + 14B** starts at approximately 9.67 GiB weights plus 1.03 GiB KV before runtime/voice; it has much less margin and is not a guaranteed fit. Longer context or vision can exceed the budget. Prefer a single 9B or 14B model, or place the small router on CPU, until local measurements prove otherwise.

The available Proxmox test GPU is a shared 12 GiB RTX 3060, not an idle 16 GiB card. No new local NVIDIA fit or speed test was performed in this batch. OpenRouter may serve different precision and hardware; local Q4 quality and speed must be checked separately.

## Selection for the next local test

Qwen3.5 9B Q4_K_M offers more memory headroom than Ministral 14B and passed this small check. Ministral 14B is another candidate with less headroom. The faster 3B/8B results are not a reason to prefer them for answers requiring reliable reasoning. This is a shortlist for local validation, not a claim of general superiority.

## Reproduce

```sh
uv run --extra mac --group benchmark python benchmarks/openrouter_cascade.py \
  --small mistralai/ministral-3b-2512 --large mistralai/ministral-8b-2512 \
  --runs 2 --output benchmarks/results/my-open-3b-8b.json

uv run --extra mac --group benchmark python benchmarks/openrouter_cascade.py \
  --small mistralai/ministral-3b-2512 --large mistralai/ministral-14b-2512 \
  --runs 2 --output benchmarks/results/my-open-3b-14b.json

uv run --extra mac --group benchmark python benchmarks/openrouter_cascade.py \
  --large qwen/qwen3.5-9b --modes large --disable-reasoning \
  --runs 2 --output benchmarks/results/my-open-qwen9b.json
```

This batch made **42 measured turns / 42 API calls**. Total OpenRouter-reported cost: **$0.000608209 USD**. The benchmark now defaults to open models. The earlier closed-model report is retained only as historical evidence.

## Sources and raw evidence

- Apache 2.0 model cards: [Ministral 3B](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512), [8B](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512), [14B](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512), [Qwen3.5 9B](https://huggingface.co/Qwen/Qwen3.5-9B).
- Quantized checkpoints: official [Ministral 3B GGUF](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF), [8B GGUF](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF), [14B GGUF](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-GGUF); [Unsloth Qwen3.5 9B conversion](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF). These files are not included in this repository.
- [File sizes, revisions, and KV calculation](results/open-model-memory-metadata.json).
- Raw timings and answers: [3B/8B](results/openrouter-open-3b-8b.json), [3B/14B](results/openrouter-open-3b-14b.json), [Qwen3.5 9B](results/openrouter-open-qwen9b.json).
