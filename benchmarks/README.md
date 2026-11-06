# Benchmarks

Measured on **2026-11-06**. These are small synthetic-fixture measurements, not a general conversational quality study or a claim to be the fastest voice stack.

For an explicit comparison of LLM-only generation, early streaming speech, and waiting for the full answer across three Ollama models, see **[LLM response latency and early speech](LLM_COMPARISON.md)**.

The [latest latency tuning](LATENCY_TUNING.md) compares the prior scheduling policy with shorter speech chunks and recognition overlapped with endpoint silence. Historical numbers below remain unchanged.

The [OpenRouter experiment](OPENROUTER.md) compares Gemini Flash-Lite, Claude Sonnet, and sequential small-to-large delegation, including actual API costs.

The current [open-model shortlist for 16 GiB](OPEN_MODELS_16GB.md) reports Apache-2.0 models only, including observed errors and separate local-memory estimates.

## Mac mini M4 — 16 GB unified memory

10-core CPU (4 performance + 6 efficiency), MLX Whisper small, Pocket TTS 4.0.1 on two CPU threads, two sampler steps, local Ollama `qwen2.5:1.5b`. Ten warm runs per row. The LLM responds to a short capital-city question with a short sentence.

| Measurement | English p50 / p95 | French p50 / p95 |
| --- | --- | --- |
| Recognition of the complete fixture | 198 / 201 ms | 212 / 220 ms |
| TTS first chunk for fixed reply text | 42 / 45 ms | 45 / 48 ms |
| Committed turn → first generated audio, in process | 437 / 444 ms | 487 / 499 ms |
| TTS real-time factor (lower is better) | 0.136 / 0.139 | 0.141 / 0.145 |

**Closer to conversation latency:** a separate paced-PCM test through the actual WebSocket server measured **1,035 ms p50 / 1,131 ms p95** from estimated end of speech to first audio received, in French, over ten warm runs. This includes VAD/endpoint delay (320 ms configured), streaming input and loopback transport. It still excludes browser audio scheduling, sound hardware, and acoustic echo. A separate initial socket test measured 1,362 ms. Do not call any of these numbers acoustic mouth-to-ear latency.

The in-process benchmark uses already-committed audio without the endpoint detector. Its first French pipeline call measured 2,030 ms; the first English call measured 488 ms with Ollama already used by the French run. These are **not equivalent cold-start experiments**. Model load plus warm-up measured 2.78 s for the French run and 5.97 s for English (cache conditions differ). An earlier first setup including downloads took longer. Warm timings exclude loading and downloads.

A one-step TTS experiment gave 492 ms p50 / 506 ms p95 to first pipeline audio in French, versus 487 / 499 ms with two steps. Different generated fixtures and ordinary run variation prevent a controlled quality comparison. No useful conversation-latency gain was demonstrated, so the default remains two steps.

## NVIDIA RTX 3060 — 12 GB VRAM

Shared Proxmox LXC on Intel Core i7-6700. Approximately 9,531 MiB VRAM was already occupied by other services before these tests. Those services were not restarted. Faster-whisper small, int8_float16, beam size 1; French, ten warm runs. These are **component measurements only**, not an NVIDIA speech-to-LLM-to-speech benchmark.

| TTS placement | CUDA recognition p50 / p95 | TTS first chunk p50 / p95 | TTS RTF p50 / p95 |
| --- | --- | --- | --- |
| CPU, two threads | 106 / 119 ms | 95 / 135 ms | 0.374 / 0.413 |
| CUDA, same shared RTX 3060 | 97 / 116 ms | 50 / 69 ms | 0.214 / 0.248 |

The NVIDIA test used an existing isolated inference environment (Python 3.13.5, Torch 2.9.1+cu126), while the packaged project targets Python 3.12. Exact versions are recorded in the JSON. This validates these adapter operations on that hardware; it does not certify the complete locked install or standalone deployment there. GPU contention makes these unsuitable as dedicated-GPU capacity numbers.

## Reproduce

```sh
uv sync --extra mac --group benchmark
uv run --extra mac --group benchmark python benchmarks/run.py \
  --output benchmarks/results/my-machine-en.json --runs 10 --language en \
  --label 'My exact hardware'
```

Run with `--language fr` for French. The script synthesizes its own question fixture, records its hash and duration, measures full STT and streamed TTS, then runs the real Ollama pipeline. First pipeline invocation is separate from the ten measured runs. `--components-only` skips the LLM. `SPEED_BACKEND=cuda SPEED_TTS_DEVICE=cuda` selects NVIDIA.

For transport timing, start the server in the desired language profile, then use a known, authorized WAV recording:

```sh
uv run --group benchmark python benchmarks/live.py /path/to/question.wav \
  --output benchmarks/results/my-websocket.json
```

This sends PCM at real-time speed and appends silence. The last sample above absolute amplitude 0.02 estimates the speech end; it is not a human phonetic annotation. Run repeatedly for a distribution. Never publish private recordings.

## Interpretation and limitations

- Raw observations and configuration are in [results/](results/); local-only cache paths are replaced with public checkpoint IDs.
- p50 is the median; p95 is NumPy's interpolated percentile of ten samples. Ten runs do not characterize rare tail latency.
- RTF is synthesis wall time divided by generated audio duration. RTF below 1 means faster than playback, not an end-to-end response under one second.
- The fixture is generated from the same TTS family used for output, so recognition is easier than noisy human speech. French transcriptions include a minor grammatical variation. No WER, MOS, multilingual quality parity, or speaker-similarity score is claimed.
- Voice cloning from an arbitrary local recording was attempted but blocked because gated cloning weights were not authorized in this environment. Preset-conditioned synthesis is verified; custom-voice quality is not. `/healthz` reports whether cloning weights are actually loaded.
- Real browser microphone/playback QA was unavailable because local browser control was disabled. The server WebSocket path was exercised with real models and synthetic speech. AudioWorklet behavior still needs cross-browser/hardware validation.
- Initial-development results identify their revision as uncommitted. They are retained as an honest baseline rather than retroactively assigning a later Git commit to earlier measurements.
