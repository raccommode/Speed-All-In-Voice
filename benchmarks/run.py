"""Real model benchmark; generated fixture, no human recordings are published."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from speed_voice.backends import SpeechModels
from speed_voice.config import Settings
from speed_voice.llm import LLM
from speed_voice.pipeline import respond


def percentile(values, q):
    return float(np.percentile(values, q))


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--language", choices=["en", "fr"], default="en")
    p.add_argument("--label", required=True)
    p.add_argument("--components-only", action="store_true")
    args = p.parse_args()
    if args.runs < 2:
        p.error("Use at least two measured runs")
    s = Settings.from_env()
    s = replace(
        s,
        language=args.language,
        tts_language="english" if args.language == "en" else "french",
        voice="alba" if args.language == "en" else "estelle",
    )
    models = SpeechModels(s)
    start = time.perf_counter()
    models.load()
    load_ms = (time.perf_counter() - start) * 1000
    question = (
        "What is the capital of France?"
        if args.language == "en"
        else "Quelle est la capitale de la France ?"
    )
    reply = (
        "The capital of France is Paris."
        if args.language == "en"
        else "La capitale de la France est Paris."
    )
    raw = b"".join(models.synthesize(question, threading.Event()))
    audio = resample_poly(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768, 2, 3).astype(
        np.float32
    )
    llm = LLM(s)

    async def turn():
        events = []

        async def emit(e):
            events.append(e)

        await respond(models, llm, s, [], emit, 1, audio=audio)
        done = next(e for e in events if e["type"] == "done")
        pcm = b"".join(
            __import__("base64").b64decode(e["pcm"]) for e in events if e["type"] == "audio"
        )
        return {
            **done["metrics"],
            "transcript": next(e["text"] for e in events if e["type"] == "transcript"),
            "reply": "".join(e["text"] for e in events if e["type"] == "text"),
            "output_audio_seconds": len(pcm) / 2 / models.sample_rate,
        }

    cold = None
    if not args.components_only:
        cold = await turn()  # First LLM call is reported separately, not mixed into warm runs.
    rows = []
    for index in range(args.runs):
        t = time.perf_counter()
        transcript = models.transcribe(audio)
        stt_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        first = None
        size = 0
        for chunk in models.synthesize(reply, threading.Event()):
            if first is None:
                first = (time.perf_counter() - t) * 1000
            size += len(chunk)
        total = (time.perf_counter() - t) * 1000
        row = {
            "stt_ms": stt_ms,
            "transcript": transcript,
            "tts_first_chunk_ms": first,
            "tts_total_ms": total,
            "tts_audio_seconds": size / 2 / models.sample_rate,
            "tts_real_time_factor": total / 1000 / (size / 2 / models.sample_rate),
        }
        if not args.components_only:
            row["pipeline"] = await turn()
        rows.append(row)
        print(json.dumps({"run": index + 1, **row}), flush=True)
    await llm.close()
    metrics = ["stt_ms", "tts_first_chunk_ms", "tts_real_time_factor"]
    summary = {
        k: {
            "p50": statistics.median([r[k] for r in rows]),
            "p95": percentile([r[k] for r in rows], 95),
        }
        for k in metrics
    }
    if not args.components_only:
        for k in ["first_token_ms", "first_audio_ms", "total_ms"]:
            summary["pipeline_" + k] = {
                "p50": statistics.median([r["pipeline"][k] for r in rows]),
                "p95": percentile([r["pipeline"][k] for r in rows], 95),
            }
    packages = {}
    for name in ["torch", "pocket-tts", "mlx-whisper", "faster-whisper", "numpy", "silero-vad"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    config = asdict(s)
    for key in ["llm_key", "token", "llm_url"]:
        config.pop(key)
    result = {
        "schema": 1,
        "date_utc": datetime.now(UTC).isoformat(),
        "hardware_label": args.label,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "config": config,
        "model_load_and_warmup_ms": load_ms,
        "first_pipeline_call": cold,
        "fixture": {
            "source": "Generated locally by the selected TTS voice; not a human accuracy corpus",
            "text": question,
            "pcm_sha256": hashlib.sha256(raw).hexdigest(),
            "seconds": len(audio) / 16000,
        },
        "measurement": "In-process server timing; excludes endpoint silence, network, microphone and playback buffering",
        "summary": summary,
        "runs": rows,
    }
    try:
        result["git_revision"] = subprocess.check_output(  # noqa: ASYNC221 — outside measured inference
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        result["git_revision"] = "uncommitted-initial-development"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
