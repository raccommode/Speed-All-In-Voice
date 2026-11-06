"""A minimal text-to-voice client. Writes synthetic audio to response.wav."""

import asyncio
import base64
import json
import os
import wave

import websockets


async def main():
    async with websockets.connect("ws://127.0.0.1:8765/v1/conversation") as ws:
        await ws.send(json.dumps({"type": "hello", "token": os.getenv("SPEED_TOKEN", "")}))
        ready = json.loads(await ws.recv())
        if ready["type"] != "ready":
            raise RuntimeError(ready)
        await ws.send(json.dumps({"type": "text", "text": "What is the capital of France?"}))
        with wave.open("response.wav", "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(ready["sample_rate"])
            async for message in ws:
                event = json.loads(message)
                if event["type"] == "audio":
                    wav.writeframes(base64.b64decode(event["pcm"]))
                elif event["type"] == "text":
                    print(event["text"], end="", flush=True)
                elif event["type"] == "error":
                    raise RuntimeError(event["message"])
                elif event["type"] == "done":
                    print("\n", event["metrics"])
                    break


if __name__ == "__main__":
    asyncio.run(main())
