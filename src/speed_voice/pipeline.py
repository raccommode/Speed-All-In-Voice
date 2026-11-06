"""Overlap token generation and synthesis with bounded queues and cancellation."""

import asyncio
import base64
import threading
import time

from .llm import PhraseBuffer


async def audio_stream(models, text, stop):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=4)

    def produce():
        def put(value):
            future = asyncio.run_coroutine_threadsafe(queue.put(value), loop)
            while not stop.is_set():
                try:
                    future.result(timeout=0.1)
                    return True
                except TimeoutError:
                    continue
            future.cancel()
            return False

        try:
            for chunk in models.synthesize(text, stop):
                if not put(chunk):
                    break
        except Exception as exc:  # noqa: BLE001 — forward native backend failures to the async consumer
            put(exc)
        finally:
            if not stop.is_set():
                put(None)

    worker = asyncio.create_task(asyncio.to_thread(produce))
    completed = False
    try:
        while True:
            item = await queue.get()
            if item is None:
                completed = True
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        if not completed:
            stop.set()
        # Do not cancel the worker: native inference must unwind before reuse.
        if stop.is_set():
            await asyncio.shield(worker)
        else:
            await worker


async def respond(models, llm, settings, history, emit, turn_id, audio=None, text=None):
    start = time.perf_counter()
    stop = threading.Event()
    metrics = {"turn_id": turn_id}
    phrases = asyncio.Queue(maxsize=3)
    transcript = text
    try:
        if transcript is None:
            transcript = await asyncio.to_thread(models.transcribe, audio)
        metrics["stt_ms"] = (time.perf_counter() - start) * 1000
        await emit({"type": "transcript", "turn_id": turn_id, "text": transcript})
        if not transcript:
            await emit({"type": "done", "turn_id": turn_id, "metrics": metrics})
            return
        messages = (
            [{"role": "system", "content": settings.system_prompt}]
            + history[-12:]
            + [{"role": "user", "content": transcript}]
        )
        answer = []

        async def generate():
            buffer = PhraseBuffer()
            async for token in llm.tokens(messages):
                metrics.setdefault("first_token_ms", (time.perf_counter() - start) * 1000)
                answer.append(token)
                await emit({"type": "text", "turn_id": turn_id, "text": token})
                for phrase in buffer.push(token):
                    await phrases.put(phrase)
            for phrase in buffer.push("", final=True):
                await phrases.put(phrase)
            await phrases.put(None)

        async def speak():
            while (phrase := await phrases.get()) is not None:
                async for chunk in audio_stream(models, phrase, stop):
                    metrics.setdefault("first_audio_ms", (time.perf_counter() - start) * 1000)
                    await emit(
                        {
                            "type": "audio",
                            "turn_id": turn_id,
                            "sample_rate": models.sample_rate,
                            "pcm": base64.b64encode(chunk).decode("ascii"),
                        }
                    )

        async with asyncio.TaskGroup() as group:
            group.create_task(generate())
            group.create_task(speak())
        history.extend(
            [
                {"role": "user", "content": transcript},
                {"role": "assistant", "content": "".join(answer)},
            ]
        )
        del history[:-12]
        metrics["total_ms"] = (time.perf_counter() - start) * 1000
        await emit({"type": "done", "turn_id": turn_id, "metrics": metrics})
    finally:
        stop.set()
