# Earlier speech and overlapped recognition

Measured on 2026-11-06 on a **Mac mini M4, 16 GB**. These changes reduce waiting without changing speech or language models or shortening the endpoint threshold.

## Paced speech through the real server

The same synthetic French question WAV was sent at microphone speed through a real WebSocket server. Both configurations use MLX Whisper small, local Ollama Qwen2.5 1.5B, Pocket TTS French/Estelle on two CPU threads, and a **320 ms silence endpoint**. One initial request per configuration was discarded, followed by five measured requests.

| Configuration | End of speech → first audio received p50 | p95 |
| --- | ---: | ---: |
| Previous: 32 / 140 character chunks, recognition after commitment | 912 ms | 983 ms |
| Optimized: 16 / 64 character chunks, recognition after 128 ms silence | **686 ms** | **699 ms** |

Median reduction: **227 ms (25%)**. This is a new paired test, not a subtraction from the older 1.035 s baseline captured in a different run. The five optimized requests all used early recognition and returned the same transcript as the five reference requests. The input hash, transcripts, events, and measured timings are in [the raw report](results/m4-endpoint-tuning.json).

The timing includes endpointing and loopback transport. It excludes physical microphone capture and actual speaker playback. Speech end is estimated from the last input sample above amplitude 0.02, as in the original benchmark. This remains a small, clean synthetic-speech test, not a noisy human speech accuracy study.

## Controlled speech scheduling comparison

Three real warm Qwen2.5 7B token traces were captured locally and replayed at their recorded timings into real Pocket TTS. Each trace supplies **identical text and token deadlines** to both chunking policies. Test order alternates. This isolates scheduling from different LLM answers; it does not measure simultaneous live LLM/TTS resource contention.

| Speech chunk limits | First audio p50 | Simulated playback gaps |
| --- | ---: | ---: |
| Previous: 32 / 140 characters | 907 ms | 0 across 3 traces |
| Optimized: 16 / 64 characters | **549 ms** | 0 across 3 traces |

Median reduction: **358 ms (39%)**. Playback is simulated from emitted PCM durations with a 20 ms initial buffer. Zero simulated gaps is not proof of glitch-free browser or acoustic playback. TTS sampling and chunking change output duration and prosody; human listening quality has not been scored. [Raw token traces and synthesis measurements](results/m4-streaming-tuning.json).

## Implementation

Recognition now begins during endpoint silence, after 128 ms, while the endpoint continues waiting until 320 ms. It is a local speculative recognition attempt only: **the agent/LLM is never called before commitment**. If speech resumes, the hypothesis is invalidated. At most one native speculative recognition operation runs at a time. A failed early attempt falls back to recognition of the complete committed audio. Disconnect waits for the native attempt to finish before releasing the worker.

The first speech chunk uses a 16-character threshold at word boundaries; subsequent chunks use 64 characters so the second chunk does not wait for a long sentence. Punctuation can flush sooner. These limits are configurable. Smaller chunks may make speech less natural, even when there are no playback gaps.

The server also preloads the selected default local Ollama model before accepting conversations, without generating an acknowledgement. This moves model loading to startup; the five-minute residency can expire after idle time. This startup improvement is separate from the warm measurements above.

The default endpoint remains 320 ms: the measured gain does not rely on prematurely cutting off a person's pause. No prerecorded acknowledgement or filler is counted as the answer.

## Reproduce and revert

```sh
uv sync --extra mac --group benchmark
uv run --extra mac --group benchmark python benchmarks/tune_streaming.py
uv run --extra mac --group benchmark python benchmarks/compare_endpoint.py \
  /path/to/authorized-question.wav --runs 5 \
  --output benchmarks/results/my-endpoint-tuning.json
```

Use an idle Mac and Ollama instance, with the required models already downloaded. The endpoint script starts and stops only its own server on port 8766, which must be available. Both configurations reuse the same supplied WAV. The scheduling script captures new traces, so its numerical results can vary.

To restore the previous scheduling policy:

```sh
SPEED_EARLY_STT_MS=0 SPEED_FIRST_PHRASE_CHARS=32 SPEED_PHRASE_CHARS=140 \
  uv run --extra mac speed-voice
```

Remaining latency includes VAD's own speech-end delay, endpoint silence, any unfinished recognition, the LLM's time to useful words, and TTS startup. The observed gain is not a promise of sub-200 ms conversational latency.
