import asyncio
import base64
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from speed_voice.audio import Endpoint, decode_pcm
from speed_voice.config import Settings
from speed_voice.llm import LLM, PhraseBuffer
from speed_voice.pipeline import respond
from speed_voice.server import create_app


class Models:
    sample_rate = 24000

    def load(self):
        pass

    def reset_vad(self):
        pass

    def transcribe(self, audio):
        return "Hello"

    def probability(self, frame):
        return float(np.abs(frame).mean() > 0.1)

    def synthesize(self, text, stop):
        for _ in range(3):
            if stop.is_set():
                return
            yield np.ones(100, dtype="<i2").tobytes()


class Tokens:
    async def tokens(self, messages):
        for word in ["Hello, ", "how are ", "you?"]:
            yield word
            await asyncio.sleep(0.001)

    async def close(self):
        pass


def test_endpoint_preserves_preroll_and_requires_sustained_speech():
    ep = Endpoint(silence_ms=96)
    frame = np.zeros(512, np.float32)
    for _ in range(8):
        assert ep.feed(frame, 0) == (False, None)
    assert ep.feed(frame + 1, 1) == (False, None)
    assert ep.feed(frame + 1, 1)[0]
    for _ in range(3):
        ep.feed(frame + 1, 1)
    assert ep.feed(frame, 0)[1] is None
    assert ep.feed(frame, 0)[1] is None
    audio = ep.feed(frame, 0)[1]
    assert len(audio) == 14 * 512
    assert not ep.frames


def test_audio_bounds():
    for data in [b"", b"x", b"0" * 32002]:
        with pytest.raises(ValueError):
            decode_pcm(data)
    assert decode_pcm(b"\x00\x80")[0] == -1


def test_phrase_buffer_never_loses_text():
    text = "Hello, welcome to the conversation. This is a longer response without punctuation " * 4
    b = PhraseBuffer()
    out = []
    for c in text:
        out.extend(b.push(c))
    out.extend(b.push("", final=True))
    assert "".join("".join(out).split()) == "".join(text.split())
    assert out[0] == "Hello,"
    assert max(map(len, out)) <= 140


async def test_pipeline_streams_audio_before_llm_finishes():
    events = []
    history = []
    llm_done = False

    class SlowLLM:
        async def tokens(self, messages):
            nonlocal llm_done
            yield "Hello, "
            await asyncio.sleep(0.1)
            llm_done = True
            yield "welcome."

    async def emit(e):
        if e["type"] == "audio" and not any(x["type"] == "audio" for x in events):
            assert not llm_done
        events.append(e)

    await respond(Models(), SlowLLM(), Settings(), history, emit, 1, text="Hi")
    assert events[-1]["type"] == "done"
    assert events[-1]["metrics"]["first_audio_ms"] < events[-1]["metrics"]["total_ms"]
    assert history[-1]["content"] == "Hello, welcome."
    assert len(base64.b64decode(next(x["pcm"] for x in events if x["type"] == "audio"))) == 200


async def test_cancellation_unwinds_native_worker_and_preserves_history():
    started = threading.Event()
    stopped = threading.Event()
    events = []
    history = []

    class Blocking(Models):
        def synthesize(self, text, stop):
            started.set()
            try:
                while not stop.is_set():
                    yield b"\0\0" * 100
                    time.sleep(0.005)
            finally:
                stopped.set()

    async def emit(e):
        events.append(e)

    task = asyncio.create_task(
        respond(Blocking(), Tokens(), Settings(), history, emit, 1, text="Hi")
    )
    await asyncio.wait_for(asyncio.to_thread(started.wait), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert stopped.is_set()
    assert history == []
    assert not any(e["type"] == "done" for e in events)


async def test_llm_error_does_not_hang_speaker():
    class Broken:
        async def tokens(self, messages):
            yield "Hello, "
            raise RuntimeError("offline")

    async def emit(e):
        pass

    with pytest.raises(ExceptionGroup):
        await asyncio.wait_for(respond(Models(), Broken(), Settings(), [], emit, 1, text="Hi"), 1)


def test_websocket_auth_turn_and_busy():
    app = create_app(Settings(token="test"), Models(), Tokens())
    with TestClient(app) as client:
        assert client.get("/healthz").json()["status"] == "ready"
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({"token": "test"})
            assert ws.receive_json()["type"] == "ready"
            with client.websocket_connect("/v1/conversation") as second:
                second.send_json({"token": "test"})
                assert second.receive_json()["type"] == "error"
            ws.send_json({"type": "text", "text": "Hi"})
            types = []
            while "done" not in types:
                types.append(ws.receive_json()["type"])
            assert "audio" in types
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/v1/conversation") as bad,
        ):
            bad.send_json({"token": "wrong"})
            bad.receive_json()


async def test_sse_parser_handles_metadata_and_unicode():
    import httpx

    llm = LLM(Settings())
    await llm.client.aclose()
    llm.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                text=(
                    ': ping\n\ndata: {"choices":[]}\n\ndata: {"choices":[{"delta":{"content":"Bonjour"}}]}\n\ndata: [DONE]\n\n'
                ),
            )
        )
    )
    assert [x async for x in llm.tokens([])] == ["Bonjour"]
    await llm.close()


@pytest.mark.parametrize("buffered", [False, True])
async def test_first_audio_relative_to_llm_completion_without_punctuation(buffered):
    class Delayed:
        async def tokens(self, messages):
            yield "This explanation begins with several useful words "
            await asyncio.sleep(0.15)
            yield "and continues after the initial speech has started."

    events = []

    async def emit(event):
        events.append(event)

    await respond(
        Models(), Delayed(), Settings(), [], emit, 1, text="Explain", buffered_speech=buffered
    )
    metrics = events[-1]["metrics"]
    assert metrics["audio_before_llm_done"] is (not buffered)
    assert (metrics["audio_lead_ms"] > 0) is (not buffered)
    assert metrics["first_token_ms"] < metrics["llm_stream_done_ms"]


def test_early_phrase_does_not_split_an_unfinished_word():
    buffer = PhraseBuffer(32, 140)
    first = buffer.push("The capital of this beautiful coun")
    assert first == ["The capital of this beautiful"]
    rest = buffer.push("try is Paris.", final=True)
    assert rest == ["country is Paris."]


async def test_stream_end_is_not_mistaken_for_last_text_token():
    class SlowEndMarker:
        async def tokens(self, messages):
            yield "Here is the complete short response, "
            await asyncio.sleep(0.1)

    events = []

    async def emit(event):
        events.append(event)

    await respond(Models(), SlowEndMarker(), Settings(), [], emit, 1, text="Hi")
    metrics = events[-1]["metrics"]
    assert metrics["audio_before_llm_done"]
    assert metrics["first_audio_ms"] > metrics["last_token_ms"]


async def test_early_recognition_discards_resumed_speech_and_bounds_work():
    from speed_voice.recognition import EarlyRecognition

    started = threading.Event()
    finish = threading.Event()
    calls = []

    class Recognizer:
        def transcribe(self, audio):
            calls.append(len(audio))
            started.set()
            finish.wait(1)
            return "old transcript"

    early = EarlyRecognition(Recognizer(), 64)
    endpoint = Endpoint()
    endpoint.frames = [np.ones(512, np.float32)] * 6
    endpoint.voiced = 4
    endpoint.silent = 2
    early.observe(endpoint, False)
    await asyncio.to_thread(started.wait, 1)
    old = early.task
    early.observe(endpoint, True)
    early.observe(endpoint, False)
    assert early.task is old
    assert early.take() is None
    finish.set()
    await early.close()
    assert len(calls) == 1


async def test_valid_early_transcript_is_reused_only_at_committed_turn():
    from speed_voice.recognition import EarlyRecognition

    class Recognizer(Models):
        def transcribe(self, audio):
            return "Already recognized"

    model = Recognizer()
    early = EarlyRecognition(model, 64)
    endpoint = Endpoint()
    endpoint.frames = [np.zeros(512, np.float32)] * 6
    endpoint.voiced = 4
    endpoint.silent = 2
    early.observe(endpoint, False)
    await early.close()
    events = []

    async def emit(event):
        events.append(event)

    await respond(
        model, Tokens(), Settings(), [], emit, 1, audio=np.zeros(512), transcription=early.take()
    )
    assert events[0]["text"] == "Already recognized"
    assert events[-1]["metrics"]["early_stt_used"]
    assert early.take() is None


async def test_failed_early_recognition_falls_back_to_final_audio():
    async def fail():
        raise RuntimeError("speculative failure")

    events = []

    async def emit(event):
        events.append(event)

    await respond(
        Models(),
        Tokens(),
        Settings(),
        [],
        emit,
        1,
        audio=np.zeros(512),
        transcription=asyncio.create_task(fail()),
    )
    assert events[0]["text"] == "Hello"
    assert not events[-1]["metrics"].get("early_stt_used")


def test_resumed_speech_never_sends_early_transcript_to_llm():
    first_attempt = threading.Event()
    requests = []

    class Recognizer(Models):
        def transcribe(self, audio):
            first_attempt.set()
            return "complete thought" if np.max(audio) > 0.4 else "unfinished thought"

    class Agent(Tokens):
        async def tokens(self, messages):
            requests.append(messages[-1]["content"])
            async for token in super().tokens(messages):
                yield token

    def packet(voice, silent):
        samples = np.concatenate([np.full(8 * 512, voice), np.zeros(silent * 512)])
        return (samples * 32767).astype("<i2").tobytes()

    with TestClient(create_app(Settings(), Recognizer(), Agent())) as client:
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({"token": ""})
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(packet(0.2, 4))
            assert ws.receive_json()["type"] == "speech_started"
            assert first_attempt.wait(1)
            assert requests == []
            ws.send_bytes(packet(0.5, 10))
            events = []
            while not events or events[-1]["type"] != "done":
                events.append(ws.receive_json())
            assert (
                next(e["text"] for e in events if e["type"] == "transcript") == "complete thought"
            )
            assert requests == ["complete thought"]


async def test_ollama_warmup_loads_without_prompt_and_skips_other_providers():
    from dataclasses import replace

    import httpx

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"done": True})

    llm = LLM(Settings())
    await llm.client.aclose()
    llm.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await llm.warm()
    import json

    assert json.loads(requests[0].content) == {"model": "qwen2.5:1.5b", "keep_alive": "5m"}
    assert requests[0].url.path == "/api/generate"
    llm.settings = replace(Settings(), llm_url="https://example.com/v1")
    await llm.warm()
    assert len(requests) == 1
    llm.settings = replace(Settings(), llm_warmup=0)
    await llm.warm()
    assert len(requests) == 1
    await llm.close()
