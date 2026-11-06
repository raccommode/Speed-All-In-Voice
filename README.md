# Speed All in Voice

An independent, local voice runtime for **Ollama and AI agents**. Speak, get a streamed reply, and interrupt it. Written from scratch; not an Unmute fork and not a newly trained speech model.

**Experimental v0.1.** One active conversation per worker. The runtime combines existing speech models with a small orchestration layer. Performance claims are backed by [raw benchmark results](benchmarks/results) and [methodology](benchmarks/README.md).

Latest paired test on a **Mac mini M4, 16 GB**: **686 ms median** from estimated end of synthetic French speech to first audio received, versus 912 ms with the previous scheduling policy (five warm runs each; excludes speaker playback). See [latency tuning and its limitations](benchmarks/LATENCY_TUNING.md). Older measurements remain in the [benchmark archive](benchmarks/README.md).


## Quick start — Apple Silicon

Install [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/), then:

```sh
git clone https://github.com/raccommode/Speed-All-In-Voice.git
cd Speed-All-In-Voice
ollama pull qwen2.5:1.5b
uv sync --extra mac
uv run --extra mac speed-voice
```

Open **http://127.0.0.1:8765**, press **Start conversation**, and allow the microphone. Headphones reduce false interruptions caused by speaker echo. Speech model downloads and warm-up, plus loading the selected local Ollama model, happen before the server becomes ready. Hugging Face may require accepting Pocket TTS model terms and logging in (`hf auth login`) for cloning weights. No account credentials or reference recordings are included in this repository.

The default uses MLX Whisper on the Apple GPU and Pocket TTS on two CPU threads, leaving GPU capacity for Ollama. The TTS device choice is intentional: GPU offload is not automatically faster for a small streaming model.

## NVIDIA Linux

Use a CUDA-enabled PyTorch environment and the CUDA/cuDNN libraries required by [faster-whisper](https://github.com/SYSTRAN/faster-whisper#gpu). Then:

```sh
uv sync --extra cuda
SPEED_BACKEND=cuda SPEED_TTS_DEVICE=cuda uv run --extra cuda speed-voice
```

`SPEED_TTS_DEVICE=cpu` is also supported to leave GPU memory for your LLM. CUDA failures do not silently switch recognition to CPU. See [deployment](docs/deployment.md).

## Ollama configuration

```sh
SPEED_LLM_MODEL=qwen2.5:7b uv run --extra mac speed-voice
```

Ollama must already be running (`ollama serve` if needed). Any installed chat model can be selected. Prefer a small instruct model for latency. Reasoning-heavy models can delay the first spoken answer.

For another OpenAI-compatible server, set `SPEED_LLM_URL`, `SPEED_LLM_MODEL`, and optionally `SPEED_LLM_KEY`. Hermes/OpenClaw require an adapter that exposes streaming chat completions; this release does not claim a native agent integration. See [protocol](docs/protocol.md).

## OpenRouter and small/large model routing

OpenRouter works through the same streaming interface. The current [open-model comparison for 16 GiB VRAM](benchmarks/OPEN_MODELS_16GB.md) covers Ministral 3B/8B/14B and Qwen3.5 9B, answer errors, cloud timings, and local memory estimates. The [earlier closed-model comparison](benchmarks/OPENROUTER.md) is historical evidence only. The local `.env` is never committed. Sequential delegation is an experiment, not the default routing policy.

## Languages and voice cloning

Run one language profile per worker. Switching profiles requires restarting that worker; this release does not auto-switch TTS models mid-conversation.

```sh
SPEED_LANGUAGE=fr SPEED_TTS_LANGUAGE=french SPEED_VOICE=estelle \
  uv run --extra mac speed-voice
```

Pocket TTS 4.0.1 supports English, French, German, Portuguese, Italian, Spanish, and Dutch; availability and quality depend on the selected upstream checkpoint. Whisper supports broader recognition coverage. Set `SPEED_LANGUAGE` to its ISO code and `SPEED_TTS_LANGUAGE` to the corresponding Pocket configuration name. Supply an appropriate licensed reference voice for each language.

To clone a voice, use a clean local recording whose speaker has authorized this use:

```sh
SPEED_VOICE=/absolute/path/to/authorized-reference.wav uv run --extra mac speed-voice
```

Custom voice cloning requires the gated cloning weights; it has not been validated in the initial benchmark environment. `/healthz` exposes `voice_cloning_available`. Preset voices remain usable without those weights.

The reference is encoded once at startup. It is not uploaded to the LLM. Voice profiles are operator configuration, never arbitrary client-provided paths or URLs. See [licenses and model provenance](THIRD_PARTY.md).

## How it stays responsive

- Resident speech models and precomputed voice conditioning.
- 32 ms microphone frames; Silero VAD with pre-roll and configurable endpoint silence.
- Recognition starts after 128 ms of silence while the 320 ms endpoint is still pending. Resumed speech invalidates this attempt; no LLM or agent call happens before commitment.
- LLM token streaming overlaps a bounded phrase queue and TTS audio streaming.
- The first short clause, or roughly 16 characters ending at a word boundary, can be spoken before the complete answer exists. Subsequent chunks allow up to 64 characters for smoother phrasing.
- Metrics explicitly compare first audio with the end of the LLM stream; see the [Ollama comparison](benchmarks/LLM_COMPARISON.md).
- Barge-in cancels the LLM request, signals native synthesis, and clears client playback.
- A single active session avoids unbounded GPU queues. Deploy separate workers for more users.

Whisper recognition is **utterance-based**, not incremental streaming ASR. This is an interruptible streaming-output pipeline, not a full-duplex speech-to-speech neural model. A 320 ms silence endpoint is the default, not a guarantee of complete semantic turns. No speculative transcript is sent to an agent.

## Development

```sh
uv sync --extra mac --group dev
uv run pytest
uv run ruff check .
```

All repository prose, code comments, and UI are English. Contributions should include measured latency and quality effects, not just a faster microbenchmark. MIT for original code; upstream models, libraries, and voices retain their own licenses.
