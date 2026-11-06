# Ollama response latency and early speech

Measured on 2026-11-06 on a **Mac mini M4, 16 GB**, using Pocket TTS French/Estelle on two CPU threads, two sampler steps, and Ollama 0.17.0. Only one LLM was resident at a time; the raw report records residency and model digests. All three checkpoints use Q4_K_M quantization.

**The voice response starts while the LLM is still generating longer answers.** The important comparison is first generated audio versus the **last text token**, not merely when the HTTP stream closes. A short answer may already be complete before TTS can emit audio.

These measurements start with a committed **text** prompt. They exclude recognition, endpoint silence, network transport, and speaker playback. Do not compare them directly with the earlier 1.035 s paced-speech result. No prerecorded filler is counted.

## Two-sentence explanation

All times below are medians of three requests per mode, in milliseconds.

| Model | LLM-only first token | LLM-only complete text stream | Streaming first audio | Last text token with streaming | First audio when waiting for all text | Audio before last token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen2.5:1.5b` | 84 | 1348 | **240** | 1627 | 1405 | 3/3 |
| `qwen2.5:7b` | 124 | 4220 | **744** | 5882 | 4564 | 3/3 |
| `llama3.1:latest` | 161 | 5319 | **690** | 6064 | 5308 | 3/3 |

## Short factual response

All times below are medians of three requests per mode, in milliseconds.

| Model | LLM-only first token | LLM-only complete text stream | Streaming first audio | Last text token with streaming | First audio when waiting for all text | Audio before last token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen2.5:1.5b` | 85 | 213 | **225** | 196 | 264 | 0/3 |
| `qwen2.5:7b` | 117 | 576 | **563** | 525 | 650 | 0/3 |
| `llama3.1:latest` | 128 | 617 | **585** | 554 | 676 | 1/3 |

## What changed

- The first speech chunk now flushes at a clause boundary or approximately 32 characters at a word boundary, down from 60. Later chunks retain a 140-character cap. This trades some prosodic continuity for an earlier start; it is not a voice-quality improvement claim.
- The gateway measures first token, last observed text token, stream completion, and first audio separately. Streaming speech and LLM generation overlap through bounded queues.
- A benchmark-only buffered mode uses the same synthesis path but waits for all LLM text. The public conversation endpoint continues to use streaming.
- ONNX Runtime is now an explicit dependency so a fresh Apple install can load the VAD without accidentally relying on the NVIDIA extra.

## Method and limits

Two French prompts (a capital-city question and a sky-color explanation), three repetitions each, three modes, three LLMs: **54 measured requests** plus one initial LLM warm-up per model. Each mode makes a real independent request; mode order rotates. Temperature is 0.3, maximum output is 96 tokens, and the same short spoken-response system prompt is used. Different responses can have different lengths, so this is a practical comparison, not a paired identical-token replay experiment.

The standalone LLM mode does not synthesize audio. The streaming mode shares the machine with TTS; its generation duration can increase through contention and queue backpressure. The last-token timestamp is when the gateway observes that token, not an internal GPU timestamp. The first complete stream measurement includes the end marker. Raw first-call timings include model loading and are kept separate from warm results.

Three repetitions per prompt are a small sample; this is not a tail-latency or concurrency certification. Smaller models produced factual errors in the explanation, and some answers hit the output cap. Speed does not establish answer quality. Early word-boundary segmentation may alter pauses; acoustic listening and human speech tests remain necessary.

An initial exploratory run was discarded because multiple Ollama models remained resident and caused memory pressure. The published run requires an idle Ollama instance and unloads only each model it has just benchmarked using the [documented keep_alive control](https://docs.ollama.com/api/generate). The pre-existing speech model remains resident across the comparison.

## Reproduce

Run on an idle Ollama instance after pulling the three model names below. Do not run another model workload during the comparison.

```sh
uv sync --extra mac --group benchmark
uv run --extra mac --group benchmark python benchmarks/compare_llms.py \
  --models qwen2.5:1.5b qwen2.5:7b llama3.1:latest --runs 3 \
  --output benchmarks/results/my-ollama-comparison.json
```

[Raw results, generated responses, model digests, residency, versions, and source hashes](results/m4-ollama-comparison.json).
