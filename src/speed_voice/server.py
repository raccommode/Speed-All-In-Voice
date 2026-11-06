import asyncio
import contextlib
import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .audio import FRAME, Endpoint, decode_pcm
from .backends import SpeechModels
from .config import Settings
from .llm import LLM
from .pipeline import respond

logger = logging.getLogger(__name__)
WEB = Path(__file__).parent / "web"


def create_app(settings=None, models=None, llm=None):
    settings = settings or Settings.from_env()
    speech = models or SpeechModels(settings)
    language_model = llm or LLM(settings)
    occupied = False

    @asynccontextmanager
    async def lifespan(app):
        await asyncio.to_thread(speech.load)
        yield
        await language_model.close()

    app = FastAPI(title="Speed All in Voice", lifespan=lifespan)

    @app.get("/healthz")
    async def health():
        return {
            "status": "ready",
            "backend": settings.backend,
            "tts_device": settings.tts_device,
            "language": settings.language,
            "busy": occupied,
            "voice_cloning_available": getattr(speech, "can_clone", False),
        }

    @app.get("/")
    async def index():
        return FileResponse(WEB / "index.html")

    @app.get("/capture.js")
    async def capture():
        return FileResponse(WEB / "capture.js", media_type="text/javascript")

    @app.websocket("/v1/conversation")
    async def conversation(ws: WebSocket):
        nonlocal occupied
        origin = ws.headers.get("origin")
        # Block drive-by web pages even on loopback. Non-browser clients omit Origin.
        if origin and origin.split("://", 1)[-1] != ws.headers.get("host"):
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            hello = await asyncio.wait_for(ws.receive_json(), timeout=10)
            supplied = hello.get("token", "")
            if not isinstance(supplied, str) or not hmac.compare_digest(supplied, settings.token):
                await ws.close(code=1008)
                return
        except (TimeoutError, ValueError, AttributeError, WebSocketDisconnect):
            await ws.close(code=1008)
            return
        if occupied:
            await ws.send_json({"type": "error", "message": "Voice worker busy; retry later"})
            await ws.close(code=1013)
            return
        occupied = True
        task = None
        turn = 0
        history = []
        pending = np.empty(0, np.float32)
        endpoint = Endpoint(settings.silence_ms, settings.max_turn_seconds)
        speech.reset_vad()
        send_lock = asyncio.Lock()

        async def emit(event):
            async with send_lock:
                await ws.send_json(event)

        async def interrupt():
            nonlocal task
            if task and not task.done():
                task.cancel()
                # Tell the client immediately, before native inference has unwound.
                await emit({"type": "interrupted", "turn_id": turn})
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            task = None

        async def run_turn(audio=None, text=None):
            try:
                await respond(
                    speech, language_model, settings, history, emit, turn, audio=audio, text=text
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Voice turn failed")
                await emit(
                    {
                        "type": "error",
                        "turn_id": turn,
                        "message": "Turn failed; inspect the server log",
                    }
                )

        try:
            await emit({"type": "ready", "input_rate": 16000, "sample_rate": speech.sample_rate})
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    pending = np.concatenate([pending, decode_pcm(message["bytes"])])
                    while len(pending) >= FRAME:
                        frame, pending = pending[:FRAME], pending[FRAME:]
                        started, audio = endpoint.feed(frame, speech.probability(frame))
                        if started:
                            await interrupt()
                            await emit({"type": "speech_started"})
                        if audio is not None:
                            await interrupt()
                            turn += 1
                            task = asyncio.create_task(run_turn(audio=audio))
                elif message.get("text"):
                    import json

                    event = json.loads(message["text"])
                    if event.get("type") == "cancel":
                        await interrupt()
                        endpoint.reset()
                        pending = np.empty(0, np.float32)
                    elif event.get("type") == "text":
                        text = event.get("text")
                        if not isinstance(text, str) or not 1 <= len(text) <= 4000:
                            raise ValueError("Text must contain 1–4000 characters")
                        await interrupt()
                        turn += 1
                        task = asyncio.create_task(run_turn(text=text))
        except (WebSocketDisconnect, RuntimeError):
            pass
        except (ValueError, AttributeError):
            await ws.close(code=1008)
        finally:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            occupied = False

    return app
