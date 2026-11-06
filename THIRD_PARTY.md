# Third-party software, weights, and voices

The MIT license in this repository covers **original application code only**. No Chell or Unmute application source, model weights, reference recordings, or proprietary system voices are vendored. Unmute is a product reference, not a dependency. Chell informed the requirements; the orchestration here was independently implemented.

Model files are fetched by upstream libraries at first startup. Installing this project does not relicense them. Keep upstream license and attribution files when redistributing models, dependency wheels, containers, or voice recordings.

| Component | Role | Upstream license / source |
| --- | --- | --- |
| Pocket TTS 4.0.1 | Streaming speech and voice conditioning | [MIT code](https://github.com/kyutai-labs/pocket-tts/blob/main/LICENSE) |
| Pocket TTS weights | Language-specific speech models | [CC BY 4.0, Kyutai model card](https://huggingface.co/kyutai/pocket-tts); access terms may apply |
| Pocket TTS without voice cloning | Upstream fallback checkpoints | [Model card and terms](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning) — not a substitute for cloning support |
| Alba MacKenna, casual | Default English voice | [CC BY 4.0, Alba MacKenna, Kyutai voice repository](https://huggingface.co/kyutai/tts-voices#alba-mackenna) |
| Estelle, `unmute-prod-website/developpeuse-3.wav` | French example voice | [CC0, Kyutai's own recording](https://huggingface.co/kyutai/tts-voices#unmute-prod-website) |
| Whisper | Speech recognition weights | [MIT, OpenAI](https://github.com/openai/whisper/blob/main/LICENSE) |
| MLX Whisper | Apple GPU inference | [MIT, MLX examples](https://github.com/ml-explore/mlx-examples/blob/main/LICENSE) |
| MLX Whisper small conversion | Default Apple recognition checkpoint | [Model card](https://huggingface.co/mlx-community/whisper-small-mlx) |
| faster-whisper / CTranslate2 | CUDA or CPU recognition | [MIT](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE), [MIT](https://github.com/OpenNMT/CTranslate2/blob/master/LICENSE) |
| Silero VAD | Voice activity detection | [MIT, Silero Team](https://github.com/snakers4/silero-vad/blob/master/LICENSE) |
| PyTorch | Tensor runtime | [BSD-style license and notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| FastAPI, Uvicorn, HTTPX | Transport | [MIT](https://github.com/fastapi/fastapi/blob/master/LICENSE), [BSD-3-Clause](https://github.com/encode/uvicorn/blob/main/LICENSE.md), [BSD-3-Clause](https://github.com/encode/httpx/blob/master/LICENSE.md) |
| NumPy, SoundFile | Audio arrays and decoding | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt), [BSD-3-Clause](https://github.com/bastibe/python-soundfile/blob/master/LICENSE) |
| Ollama | Separately installed LLM server | [MIT](https://github.com/ollama/ollama/blob/main/LICENSE) |
| Qwen2.5 1.5B Instruct | Suggested LLM, separately downloaded by Ollama | [Apache 2.0 model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |

The lockfile records transitive dependencies. Some binary packages include additional third-party libraries and notices (for example libsndfile, FFmpeg, and CUDA); preserve the notices distributed with those packages. CUDA components have NVIDIA terms, not this project's MIT license. The NVIDIA runtime is optional.

**Attribution for example audio:** generated using Pocket TTS by Kyutai, CC BY 4.0 weights; English voice conditioned on Alba MacKenna's casual recording, CC BY 4.0; French voice conditioned on Kyutai's Estelle recording, CC0. Generated benchmark audio is not committed. Output is synthetic and has not been endorsed by the speakers.

Other upstream voices can have noncommercial restrictions (notably Expresso and EARS). Do not assume that every voice listed by a TTS library is permissively licensed. Only the two examples above were selected for this release. For custom recordings, obtain permission for cloning and follow the recording's license. Use local files in trusted operator configuration; never publish private recordings or Hugging Face tokens.

License links were reviewed on 2026-11-06. A different model revision or voice requires its own review. This is an attribution inventory, not a claim that model weights are OSI-licensed software.

## Additional open-model benchmark candidates

The 16 GiB shortlist benchmarks [Ministral 3 3B](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512), [8B](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512), [14B](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512), and [Qwen3.5 9B](https://huggingface.co/Qwen/Qwen3.5-9B). Each linked model card identifies Apache 2.0. No weights or quantized conversions are redistributed. Preserve the original license and notices when downloading or redistributing a local checkpoint; quantization does not replace the original license. OpenRouter inference timings do not certify local GPU memory or performance.
