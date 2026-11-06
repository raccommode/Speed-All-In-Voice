import os
import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    backend: str = "mlx" if platform.system() == "Darwin" else "cuda"
    stt_model: str = ""
    language: str = "en"
    tts_language: str = "english"
    voice: str = "alba"
    tts_device: str = "cpu"
    decode_steps: int = 2
    threads: int = 2
    llm_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "qwen2.5:1.5b"
    llm_key: str = "ollama"
    token: str = ""
    silence_ms: int = 320
    max_turn_seconds: int = 30
    max_reply_tokens: int = 160
    system_prompt: str = (
        "You are a helpful voice assistant. Reply in the user's language. "
        "Use one or two short spoken sentences. No markdown, lists, or hidden reasoning."
    )

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for name in cls.__dataclass_fields__:
            raw = os.getenv("SPEED_" + name.upper())
            if raw is not None:
                values[name] = int(raw) if isinstance(getattr(defaults, name), int) else raw
        result = cls(**values)
        if result.backend not in {"mlx", "cuda", "cpu"}:
            raise ValueError("SPEED_BACKEND must be mlx, cuda, or cpu")
        if not 160 <= result.silence_ms <= 2000:
            raise ValueError("SPEED_SILENCE_MS must be between 160 and 2000")
        if result.tts_device not in {"cpu", "cuda"}:
            raise ValueError("SPEED_TTS_DEVICE must be cpu or cuda")
        return result

    @property
    def recognizer(self):
        return self.stt_model or (
            "mlx-community/whisper-small-mlx" if self.backend == "mlx" else "small"
        )
