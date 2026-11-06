"""Paired paced-audio benchmark using the same fixture and isolated server processes."""

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

import httpx
import numpy as np


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.runs < 2:
        p.error("Use at least two measured requests per configuration")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 8766))
    results = {
        "hardware": "Mac mini M4, 16 GB",
        "llm": "qwen2.5:1.5b",
        "fixture_sha256": hashlib.sha256(Path(args.wav).read_bytes()).hexdigest(),
        "method": "Same synthetic WAV paced through real WebSocket, French Whisper-small MLX + Pocket TTS CPU. 320 ms endpoint in both configurations. One discarded warm-up per configuration. No microphone or acoustic playback.",
        "runs": [],
        "summary": {},
    }
    with tempfile.TemporaryDirectory(prefix="speed-endpoint-") as tmp:
        for name, first, subsequent, early in [
            ("previous", 32, 140, 0),
            ("optimized", 16, 64, 128),
        ]:
            env = {
                **os.environ,
                "PYTHONPATH": str(Path("src").resolve()),
                "SPEED_LANGUAGE": "fr",
                "SPEED_TTS_LANGUAGE": "french",
                "SPEED_VOICE": "estelle",
                "SPEED_LLM_MODEL": "qwen2.5:1.5b",
                "SPEED_FIRST_PHRASE_CHARS": str(first),
                "SPEED_PHRASE_CHARS": str(subsequent),
                "SPEED_EARLY_STT_MS": str(early),
                "SPEED_SILENCE_MS": "320",
            }
            with open(Path(tmp) / f"{name}.log", "w") as log:
                server = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    "from speed_voice.cli import main; main()",
                    "--port",
                    "8766",
                    env=env,
                    stdout=log,
                    stderr=log,
                )
                try:
                    async with httpx.AsyncClient() as client:
                        for _ in range(300):
                            if server.returncode is not None:
                                raise RuntimeError(Path(log.name).read_text())
                            try:
                                ready = await client.get("http://127.0.0.1:8766/healthz")
                                if ready.status_code == 200:
                                    break
                            except httpx.ConnectError:
                                pass
                            await asyncio.sleep(0.2)
                        else:
                            raise RuntimeError("Voice server startup timed out")
                    for run in range(args.runs + 1):
                        output = Path(tmp) / "result.json"
                        process = await asyncio.create_subprocess_exec(
                            sys.executable,
                            "benchmarks/live.py",
                            str(Path(args.wav).resolve()),
                            "--url",
                            "ws://127.0.0.1:8766/v1/conversation",
                            "--output",
                            str(output),
                            env=env,
                            stdout=asyncio.subprocess.DEVNULL,
                        )
                        if await process.wait():
                            raise RuntimeError("Paced audio test failed")
                        if run:
                            result = json.loads(output.read_text())
                            result.update(configuration=name, run=run)
                            results["runs"].append(result)
                            print(
                                name,
                                run,
                                result["speech_end_to_first_received_audio_ms"],
                                flush=True,
                            )
                    rows = [
                        r["speech_end_to_first_received_audio_ms"]
                        for r in results["runs"]
                        if r["configuration"] == name
                    ]
                    results["summary"][name] = {
                        "p50_ms": float(np.median(rows)),
                        "p95_ms": float(np.percentile(rows, 95)),
                    }
                    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
                finally:
                    with contextlib.suppress(ProcessLookupError):
                        server.terminate()
                    await server.wait()


if __name__ == "__main__":
    asyncio.run(main())
