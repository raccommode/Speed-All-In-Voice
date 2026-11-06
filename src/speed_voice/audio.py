"""Audio framing and deterministic endpoint state (16 kHz mono PCM16)."""

from collections import deque

import numpy as np

RATE = 16000
FRAME = 512


def decode_pcm(raw: bytes) -> np.ndarray:
    if not raw or len(raw) % 2 or len(raw) > 32000:
        raise ValueError("Expected 1–16000 PCM16 samples")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


class Endpoint:
    def __init__(self, silence_ms=320, max_seconds=30):
        self.silence_frames = int(np.ceil(silence_ms / 32))
        self.max_frames = max_seconds * RATE // FRAME
        self.preroll = deque(maxlen=8)
        self.frames = []
        self.silent = 0
        self.voiced = 0
        self.candidate = 0

    def feed(self, frame, probability):
        """Return (speech_started, completed_audio). Two frames reject short clicks."""
        speaking = probability >= 0.5
        started = False
        if not self.frames:
            self.preroll.append(frame.copy())
            self.candidate = self.candidate + 1 if speaking else 0
            if self.candidate < 2:
                return False, None
            self.frames = list(self.preroll)
            self.preroll.clear()
            self.voiced = 2
            self.silent = 0
            started = True
        else:
            self.frames.append(frame.copy())
            self.voiced += int(speaking)
            self.silent = 0 if speaking else self.silent + 1
        if self.silent >= self.silence_frames or len(self.frames) >= self.max_frames:
            audio = np.concatenate(self.frames) if self.voiced >= 4 else None
            self.reset()
            return started, audio
        return started, None

    def reset(self):
        self.frames = []
        self.preroll.clear()
        self.candidate = self.silent = self.voiced = 0
