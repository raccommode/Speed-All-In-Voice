"""Replay identical, recorded real LLM token timings through real TTS."""

import asyncio
import base64
import json
import time
from dataclasses import replace
from pathlib import Path

from speed_voice.backends import SpeechModels
from speed_voice.config import Settings
from speed_voice.llm import LLM
from speed_voice.pipeline import respond


async def main():
    s = replace(
        Settings(),
        language="fr",
        tts_language="french",
        voice="estelle",
        llm_model="qwen2.5:7b",
        max_reply_tokens=96,
    )
    models = SpeechModels(s)
    await asyncio.to_thread(models.load)
    llm = LLM(s)
    prompt = "Explique en deux phrases pourquoi le ciel est bleu."
    messages = [{"role": "system", "content": s.system_prompt}, {"role": "user", "content": prompt}]
    # Keep loading and prompt prefill out of the three warm captures.
    async for _ in llm.tokens(messages):
        pass
    results = {
        "hardware": "Mac mini M4, 16 GB",
        "model": s.llm_model,
        "method": "Real warm Ollama token traces replayed at recorded timings into real CPU Pocket TTS. Identical text and token deadlines across configurations. Playback gaps are a server-side simulation with 20 ms initial buffer, not acoustic measurements.",
        "traces": [],
        "runs": [],
    }
    for trial in range(3):
        trace = []
        start = time.perf_counter()
        async for token in llm.tokens(messages):
            trace.append(((time.perf_counter() - start), token))
        results["traces"].append(trace)

        class Replay:
            async def tokens(self, messages, trace=trace):
                start = time.perf_counter()
                for deadline, token in trace:
                    await asyncio.sleep(max(0, deadline - (time.perf_counter() - start)))
                    yield token

        configs = [("previous", 32, 140), ("early", 16, 64)]
        if trial % 2:
            configs.reverse()
        for name, first, subsequent in configs:
            events = []
            audio = []
            start = time.perf_counter()

            async def emit(event, events=events, audio=audio, start=start):
                if event["type"] == "audio":
                    audio.append(
                        (
                            (time.perf_counter() - start),
                            len(base64.b64decode(event["pcm"])) / 2 / models.sample_rate,
                        )
                    )
                else:
                    events.append(event)

            await respond(
                models,
                Replay(),
                replace(s, first_phrase_chars=first, phrase_chars=subsequent),
                [],
                emit,
                1,
                text=prompt,
            )
            end = audio[0][0] + 0.02
            gaps = []
            for arrival, duration in audio:
                if arrival > end:
                    gaps.append((arrival - end) * 1000)
                end = max(end, arrival) + duration
            row = {
                "trial": trial,
                "configuration": name,
                "first_phrase_chars": first,
                "phrase_chars": subsequent,
                **events[-1]["metrics"],
                "simulated_gap_count": len(gaps),
                "simulated_gap_ms": sum(gaps),
                "output_audio_seconds": sum(d for _, d in audio),
            }
            results["runs"].append(row)
            print(json.dumps(row), flush=True)
            Path("benchmarks/results/m4-streaming-tuning.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
    await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
