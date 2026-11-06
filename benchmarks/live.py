"""Measure a paced PCM utterance through the real WebSocket server."""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/v1/conversation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audio, rate = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    import math

    divisor = math.gcd(rate, 16000)
    audio = resample_poly(audio, 16000 // divisor, rate // divisor)
    # Last substantial sample is an estimate, not a phonetic speech-end annotation.
    end_index = int(np.flatnonzero(np.abs(audio) > 0.02)[-1]) + 1
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    pcm += b"\0\0" * 16000
    events = []
    first = None
    send_start = None
    async with websockets.connect(args.url) as ws:
        await ws.send(json.dumps({"type": "hello", "token": os.getenv("SPEED_TOKEN", "")}))
        assert json.loads(await ws.recv())["type"] == "ready"

        async def send():
            nonlocal send_start
            send_start = time.perf_counter()
            for offset in range(0, len(pcm), 1024):
                await ws.send(pcm[offset : offset + 1024])
                await asyncio.sleep(
                    max(0, send_start + (offset + 1024) / 32000 - time.perf_counter())
                )

        sender = asyncio.create_task(send())
        try:
            async with asyncio.timeout(60):
                async for message in ws:
                    event = json.loads(message)
                    if event["type"] == "audio":
                        if first is None:
                            first = time.perf_counter()
                        event = {"type": "audio", "bytes": len(base64.b64decode(event["pcm"]))}
                    events.append(event)
                    if event["type"] in {"done", "error"}:
                        break
        finally:
            sender.cancel()
            import contextlib

            with contextlib.suppress(asyncio.CancelledError):
                await sender
    if first is None:
        raise RuntimeError(events)
    result = {
        "measurement": "Paced synthetic PCM to first received WebSocket audio; no browser playback",
        "fixture_sha256": hashlib.sha256(Path(args.wav).read_bytes()).hexdigest(),
        "speech_end_estimator": "Last input sample above absolute amplitude 0.02",
        "speech_end_to_first_received_audio_ms": (first - send_start - end_index / 16000) * 1000,
        "events": events,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(result["speech_end_to_first_received_audio_ms"])


if __name__ == "__main__":
    asyncio.run(main())
