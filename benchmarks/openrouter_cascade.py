"""Small-vs-large-vs-routed voice benchmark. Only synthetic prompts leave the host."""

import argparse
import asyncio
import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

from speed_voice.backends import SpeechModels
from speed_voice.config import Settings
from speed_voice.pipeline import respond

CASES = {
    "fact": "Quelle est la capitale de la France ? Réponds brièvement.",
    "arithmetic": "Combien font dix-sept plus vingt-cinq ? Réponds brièvement.",
    "reasoning": "Quel est le plus petit entier positif qui donne un reste de deux quand on le divise par trois, un reste de trois quand on le divise par cinq et un reste de deux quand on le divise par sept ? Donne le résultat et une courte vérification.",
}
ROUTER = (
    "You are the fast front end of a voice assistant. For a simple factual question or "
    "a single arithmetic operation, respond with ANSWER on its own first line, then give "
    "the actual short answer in the user's language. For multi-step reasoning, complex "
    "constraints, uncertainty, or anything you cannot confidently answer, output only "
    "DELEGATE. Never provide filler, a preamble, or explain the routing. Do not answer "
    "the complex question yourself. Routing markers are not spoken."
)


class Cloud:
    def __init__(self, key, small, large, disable_reasoning=False):
        self.client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1/",
            headers={
                "Authorization": "Bearer " + key,
                "X-OpenRouter-Title": "Speed All in Voice benchmark",
            },
            timeout=httpx.Timeout(90, connect=10),
        )
        self.disable_reasoning = disable_reasoning
        self.small, self.large = small, large
        self.calls = []
        self.route = None
        self.route_ms = None
        self.mode = "large"

    async def request(self, model, messages):
        call = {"model": model, "usage": None, "generation_id": None, "provider": None}
        self.calls.append(call)
        start = time.perf_counter()
        async with self.client.stream(
            "POST",
            "chat/completions",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "max_tokens": 128,
                "temperature": 0,
                "stream_options": {"include_usage": True},
                **({"reasoning": {"enabled": False}} if self.disable_reasoning else {}),
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(
                        "OpenRouter generation failed: "
                        + str(event["error"].get("code", "unknown"))
                    )
                call["generation_id"] = event.get("id", call["generation_id"])
                call["provider"] = event.get("provider", call["provider"])
                if event.get("usage"):
                    call["usage"] = event["usage"]
                choices = event.get("choices", [])
                if choices:
                    token = choices[0].get("delta", {}).get("content")
                    if token:
                        call.setdefault("first_token_ms", (time.perf_counter() - start) * 1000)
                        yield token
        call["total_ms"] = (time.perf_counter() - start) * 1000

    async def tokens(self, messages):
        if self.mode != "routed":
            async for token in self.request(
                self.small if self.mode == "small" else self.large, messages
            ):
                yield token
            return
        start = time.perf_counter()
        prefix = ""
        decision = None
        routed = [{"role": "system", "content": ROUTER}] + messages[1:]
        async for token in self.request(self.small, routed):
            if decision == "ANSWER":
                yield token
            elif decision is None:
                prefix += token
                if "\n" in prefix:
                    marker, rest = prefix.split("\n", 1)
                    decision = marker.strip()
                    self.route_ms = (time.perf_counter() - start) * 1000
                    if decision == "ANSWER" and rest:
                        yield rest
                elif len(prefix) > 64:
                    decision = "DELEGATE"
        if decision is None:
            decision = prefix.strip()
            self.route_ms = (time.perf_counter() - start) * 1000
        if decision == "ANSWER":
            self.route = "small"
        else:
            self.route = "large"
            async for token in self.request(self.large, messages):
                yield token

    async def close(self):
        await self.client.aclose()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", default="mistralai/ministral-3b-2512")
    parser.add_argument("--large", default="mistralai/ministral-8b-2512")
    parser.add_argument("--disable-reasoning", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["small", "large", "routed"],
        default=["large", "small", "routed"],
    )
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", default="benchmarks/results/openrouter-open-3b-8b.json")
    args = parser.parse_args()
    if not 1 <= args.runs <= 3:
        parser.error("Use one to three repetitions for this small paid experiment")
    from dotenv import load_dotenv

    load_dotenv(override=False)
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        parser.error("Set OPENROUTER_API_KEY locally; never commit credentials")
    cloud = Cloud(key, args.small, args.large, args.disable_reasoning)
    settings = replace(
        Settings(), language="fr", tts_language="french", voice="estelle", max_reply_tokens=128
    )
    speech = SpeechModels(settings)
    await asyncio.to_thread(speech.load)
    result = {
        "date_utc": datetime.now(UTC).isoformat(),
        "hardware": "Mac mini M4, 16 GB; local French Pocket TTS CPU",
        "small": args.small,
        "large": args.large,
        "router_prompt": ROUTER,
        "system_prompt": settings.system_prompt,
        "prompts": CASES,
        "modes": args.modes,
        "max_tokens_per_call": 128,
        "temperature": 0,
        "reasoning_disabled_requested": args.disable_reasoning,
        "measurement": "Committed synthetic text to first generated speech. Includes WAN and routing; excludes STT, endpointing and acoustic playback. No prerecorded acknowledgement.",
        "runs": [],
        "summary": {},
    }
    try:
        for case, prompt in CASES.items():
            for repeat in range(args.runs):
                modes = args.modes
                offset = repeat % len(modes)
                modes = modes[offset:] + modes[:offset]
                for mode in modes:
                    cloud.mode = mode
                    cloud.calls = []
                    cloud.route = None
                    cloud.route_ms = None
                    events = []

                    async def emit(event, events=events):
                        if event["type"] != "audio":
                            events.append(event)

                    await respond(speech, cloud, settings, [], emit, 1, text=prompt)
                    row = {
                        "case": case,
                        "repeat": repeat + 1,
                        "mode": mode,
                        **events[-1]["metrics"],
                        "reply": "".join(e["text"] for e in events if e["type"] == "text"),
                        "route": cloud.route,
                        "route_ms": cloud.route_ms,
                        "calls": cloud.calls,
                    }
                    usage = [c["usage"] for c in cloud.calls]
                    row["cost_usd"] = (
                        sum(u["cost"] for u in usage)
                        if all(u and u.get("cost") is not None for u in usage)
                        else None
                    )
                    result["runs"].append(row)
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in ["case", "mode", "route", "first_audio_ms", "cost_usd"]
                            }
                        ),
                        flush=True,
                    )
                    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        for case in CASES:
            result["summary"][case] = {}
            for mode in args.modes:
                rows = [r for r in result["runs"] if r["case"] == case and r["mode"] == mode]
                result["summary"][case][mode] = {
                    "first_audio_p50_ms": float(np.median([r["first_audio_ms"] for r in rows])),
                    "routes": [r["route"] for r in rows],
                }
        costs = [r["cost_usd"] for r in result["runs"]]
        result["total_cost_usd"] = sum(costs) if all(v is not None for v in costs) else None
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    finally:
        await cloud.close()


if __name__ == "__main__":
    asyncio.run(main())
