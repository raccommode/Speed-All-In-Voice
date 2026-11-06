"""Hide recognition work inside endpoint silence without calling the agent early."""

import asyncio

import numpy as np


class EarlyRecognition:
    def __init__(self, models, silence_ms=128):
        self.models = models
        self.frames = int(np.ceil(silence_ms / 32)) if silence_ms else 0
        self.task = None
        self.valid = False
        self.attempted = False

    def invalidate(self):
        self.valid = False
        self.attempted = False

    def observe(self, endpoint, speaking):
        if speaking:
            self.invalidate()
        if (
            not self.frames
            or endpoint.silent < self.frames
            or endpoint.voiced < 4
            or self.attempted
            or (self.task is not None and not self.task.done())
        ):
            return
        self.attempted = True
        self.valid = True
        audio = np.concatenate(endpoint.frames).copy()
        self.task = asyncio.create_task(asyncio.to_thread(self.models.transcribe, audio))
        # A discarded speculation can fail without ever being awaited by a turn.
        self.task.add_done_callback(lambda task: None if task.cancelled() else task.exception())

    def take(self):
        candidate = self.task if self.valid else None
        self.invalidate()
        return candidate

    async def close(self):
        if self.task is not None:
            await asyncio.gather(asyncio.shield(self.task), return_exceptions=True)
