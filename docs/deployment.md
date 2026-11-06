# Deployment and tuning

Python 3.12 is the supported packaged runtime. Use `uv sync --frozen --extra mac` on Apple Silicon or `uv sync --frozen --extra cuda` on Linux with NVIDIA. The CUDA extra selects faster-whisper; CUDA drivers and compatible cuBLAS/cuDNN are operator prerequisites. Follow faster-whisper's current GPU instructions. Run natively on macOS to access Metal; Docker Desktop does not expose Apple Metal.

The default bind is loopback. `--host 0.0.0.0` requires `SPEED_TOKEN`. A remote browser requires HTTPS for microphone access. Forward WebSocket upgrades through your reverse proxy, retain the original Host/Origin, and avoid response buffering. Do not expose the server directly to the public Internet. Authentication limits access but does not turn this experimental single-worker runtime into a multi-tenant service.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `SPEED_BACKEND` | `mlx` on macOS, `cuda` elsewhere | Recognition device (`cpu` also available with cuda extra) |
| `SPEED_STT_MODEL` | MLX whisper-small / faster-whisper small | Model ID or trusted local checkpoint |
| `SPEED_LANGUAGE` | `en` | Whisper language; empty enables detection |
| `SPEED_TTS_LANGUAGE` | `english` | Pocket TTS configuration |
| `SPEED_VOICE` | `alba` | Upstream voice name or authorized local recording |
| `SPEED_TTS_DEVICE` | `cpu` | `cpu` or `cuda` |
| `SPEED_THREADS` | `2` | Torch CPU / faster-whisper CPU threads |
| `SPEED_DECODE_STEPS` | `2` | TTS sampler steps; benchmark quality before reducing |
| `SPEED_FIRST_PHRASE_CHARS` | `16` | Initial speech chunk limit at a word boundary |
| `SPEED_PHRASE_CHARS` | `64` | Subsequent chunk limit |
| `SPEED_EARLY_STT_MS` | `128` | Start recognition during endpoint silence; `0` disables overlap |
| `SPEED_SILENCE_MS` | `320` | Endpoint silence (160–2000 ms), rounded to 32 ms frames |
| `SPEED_MAX_TURN_SECONDS` | `30` | Forced endpoint limit |
| `SPEED_MAX_REPLY_TOKENS` | `160` | LLM output cap |
| `SPEED_LLM_URL` | `http://127.0.0.1:11434/v1` | Trusted chat-completions server |
| `SPEED_LLM_MODEL` | `qwen2.5:1.5b` | Ollama model name |
| `SPEED_LLM_WARMUP` | `1` | Preload the default local Ollama endpoint at startup; `0` disables |
| `SPEED_LLM_KEY` | `ollama` | Server-side bearer credential |
| `SPEED_SYSTEM_PROMPT` | Short spoken replies | Conversation system prompt |
| `SPEED_TOKEN` | empty | Gateway access token |

Keep models warm, use small instruct LLMs, and avoid large context windows. Benchmark with realistic speech and simultaneous workloads before claiming production capacity. Do not run multiple TTS generations concurrently on the same Pocket instance. This runtime serializes native inference and uses bounded queues for backpressure.

Upstream model download configurations pin Pocket weight revisions. Whisper model IDs can change upstream; for strict reproducibility download a snapshot and set `SPEED_STT_MODEL` to its local path. Preserve cache metadata and license files. Once caches are complete, `HF_HUB_OFFLINE=1` prevents Hugging Face network downloads.

No automatic service installation, global GPU reconfiguration, or modification of another application is required.

Early recognition does not shorten the endpoint threshold. An interrupted hypothesis is discarded and at most one native recognition attempt is active. Set `SPEED_EARLY_STT_MS=0 SPEED_FIRST_PHRASE_CHARS=32 SPEED_PHRASE_CHARS=140` to reproduce the previous scheduling policy. Shorter speech chunks can affect prosody; tune with listening tests.

Local Ollama warm-up loads weights without generating a dummy response. It applies only to the default localhost/127.0.0.1 port 11434 endpoint, and uses five-minute residency. After a longer idle period, Ollama can unload the model and the next turn may be cold again. Other providers are not probed with Ollama-specific requests.
