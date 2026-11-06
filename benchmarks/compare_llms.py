"""Compare real Ollama LLM latency with streaming and buffered speech integration."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

from speed_voice.backends import SpeechModels
from speed_voice.config import Settings
from speed_voice.llm import LLM
from speed_voice.pipeline import respond

PROMPTS = {
    "short": "Quelle est la capitale de la France ? Réponds en une phrase.",
    "explanation": "Explique en deux phrases pourquoi le ciel est bleu et pourquoi il devient rouge au coucher du soleil.",
}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["qwen2.5:1.5b", "qwen2.5:7b", "llama3.1:latest"]
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("Use at least two repetitions per prompt")
    settings = replace(
        Settings.from_env(),
        language="fr",
        tts_language="french",
        voice="estelle",
        max_reply_tokens=96,
    )
    speech = SpeechModels(settings)
    await asyncio.to_thread(speech.load)
    async with httpx.AsyncClient() as client:
        if (await client.get("http://127.0.0.1:11434/api/ps")).json()["models"]:
            raise RuntimeError(
                "Use an idle Ollama instance with no resident models for this benchmark"
            )
        tags = (await client.get("http://127.0.0.1:11434/api/tags")).json()["models"]
        ollama_version = (await client.get("http://127.0.0.1:11434/api/version")).json()["version"]
    metadata = {x["name"]: x for x in tags if x["name"] in args.models}
    result = {
        "date_utc": datetime.now(UTC).isoformat(),
        "hardware": "Mac mini M4, 10-core CPU, 16 GB unified memory",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["torch", "pocket-tts", "mlx-whisper", "httpx"]
        },
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("src/speed_voice").glob("*.py"))
        },
        "model_metadata": metadata,
        "ollama_version": ollama_version,
        "prompts": PROMPTS,
        "settings": {
            "system_prompt": settings.system_prompt,
            "max_reply_tokens": 96,
            "temperature": 0.3,
            "tts_language": "french",
            "voice": "estelle",
            "tts_device": "cpu",
            "tts_threads": 2,
            "decode_steps": 2,
            "first_phrase_characters": settings.first_phrase_chars,
            "subsequent_phrase_characters": settings.phrase_chars,
        },
        "measurement": "Text committed to gateway -> first token / first generated audio / observed LLM stream end. Excludes recognition, endpoint, transport and playback. Real inference, not replay.",
        "method": "Three independent real requests per repetition: LLM only, streaming speech, buffered speech. Order rotates; outputs may differ. First LLM request per model is separate warm-up. Six measured requests per mode per model (two prompts x three repetitions).",
        "residency": {},
        "warmup": [],
        "runs": [],
        "summary": {},
    }

    def save():
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")

    for model in args.models:
        s = replace(settings, llm_model=model)
        llm = LLM(s)

        async def llm_only(prompt, llm=llm, s=s):
            start = time.perf_counter()
            first = None
            chunks = []
            async for token in llm.tokens(
                [
                    {"role": "system", "content": s.system_prompt},
                    {"role": "user", "content": prompt},
                ]
            ):
                if first is None:
                    first = (time.perf_counter() - start) * 1000
                chunks.append(token)
            return {
                "first_token_ms": first,
                "llm_stream_done_ms": (time.perf_counter() - start) * 1000,
                "reply": "".join(chunks),
            }

        try:
            result["warmup"].append({"model": model, **await llm_only(PROMPTS["short"])})
            async with httpx.AsyncClient() as client:
                result["residency"][model] = (
                    await client.get("http://127.0.0.1:11434/api/ps")
                ).json()
            save()
            for prompt_id, prompt in PROMPTS.items():
                for iteration in range(args.runs):
                    modes = ["llm_only", "streaming", "buffered"]
                    modes = modes[iteration % 3 :] + modes[: iteration % 3]
                    for mode in modes:
                        if mode == "llm_only":
                            row = await llm_only(prompt)
                        else:
                            events = []

                            async def emit(event, events=events):
                                if event["type"] != "audio":
                                    events.append(event)

                            await respond(
                                speech,
                                llm,
                                s,
                                [],
                                emit,
                                1,
                                text=prompt,
                                buffered_speech=mode == "buffered",
                            )
                            row = {
                                **events[-1]["metrics"],
                                "reply": "".join(e["text"] for e in events if e["type"] == "text"),
                            }
                        row.update(
                            model=model, prompt=prompt_id, repetition=iteration + 1, mode=mode
                        )
                        result["runs"].append(row)
                        print(json.dumps(row), flush=True)
                        save()
        finally:
            await llm.close()
            async with httpx.AsyncClient(timeout=30) as client:
                unloaded = await client.post(
                    "http://127.0.0.1:11434/api/generate", json={"model": model, "keep_alive": 0}
                )
                unloaded.raise_for_status()
                for _ in range(100):
                    if not (await client.get("http://127.0.0.1:11434/api/ps")).json()["models"]:
                        break
                    await asyncio.sleep(0.1)
                else:
                    raise RuntimeError("Model remained resident; refusing a confounded comparison")
    for model in args.models:
        result["summary"][model] = {}
        for mode in ["llm_only", "streaming", "buffered"]:
            rows = [r for r in result["runs"] if r["model"] == model and r["mode"] == mode]
            summary = {"n": len(rows)}
            for key in ["first_token_ms", "first_audio_ms", "llm_stream_done_ms", "audio_lead_ms"]:
                values = [r[key] for r in rows if key in r]
                if values:
                    summary[key] = {
                        "p50": float(np.median(values)),
                        "p95": float(np.percentile(values, 95)),
                    }
            if mode != "llm_only":
                summary["audio_before_llm_done_count"] = sum(
                    r["audio_before_llm_done"] for r in rows
                )
            result["summary"][model][mode] = summary
    save()


if __name__ == "__main__":
    asyncio.run(main())
