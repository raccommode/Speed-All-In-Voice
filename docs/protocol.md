# Conversation protocol v1

Connect a WebSocket to `/v1/conversation`. This is the project's own protocol, **not** an implementation of the OpenAI Realtime API.

Send `{"type":"hello","token":""}` within ten seconds. A configured `SPEED_TOKEN` must match. Credentials belong in this first message, not URL query strings. Browser Origin must match Host. Use TLS and an authenticated reverse proxy for remote deployment.

Wait for `{"type":"ready","input_rate":16000,"sample_rate":24000}`. Send binary little-endian PCM16 mono at 16 kHz. The recommended packet is 512 samples (32 ms). Packets up to 16000 samples are allowed. Stream at microphone speed; this is not a file upload endpoint.

Alternatively send `{"type":"text","text":"Hello"}`. This bypasses recognition. Send `{"type":"cancel"}` to interrupt. Closing the socket cancels the current turn. A worker accepts one conversation; a second connection receives an error and close code 1013.

Server JSON events:

| Type | Fields | Meaning |
| --- | --- | --- |
| `speech_started` | — | VAD detected sustained speech; clear playback immediately |
| `transcript` | `turn_id`, `text` | Final utterance transcript |
| `text` | `turn_id`, `text` | Assistant text delta |
| `audio` | `turn_id`, `sample_rate`, `pcm` | Base64 PCM16 mono audio chunk |
| `interrupted` | `turn_id` | Discard queued and future audio for this turn |
| `done` | `turn_id`, `metrics` | Server generation completed; playback may still be running |
| `error` | `message`, optional `turn_id` | Request failed |

`stt_ms`, `first_token_ms`, `first_audio_ms`, and `total_ms` use a monotonic clock from the **committed turn**. They exclude microphone capture, endpoint silence, transport, and device playback. `first_token_ms` includes STT. `first_audio_ms` includes STT, LLM time to a speakable phrase, queue wait, and TTS first chunk. For text turns `stt_ms` is bookkeeping only.

The gateway sends transcripts and the last six completed exchanges to the configured `/chat/completions` endpoint. It does not forward raw microphone audio or voice references to the LLM. Interrupted partial replies are not added to history. Tool calls are not implemented; an agent adapter must execute tools and return speakable streamed text. Configure credentials through environment variables, never public frontend code.

Use `examples/client.py` for a minimal text-to-voice client. The bundled browser demonstrates microphone input, playback, and cancellation. Echo cancellation is browser-dependent; this is not a validated acoustic echo canceller for speakerphone use.
