"""Resident models. No application code is copied from Chell or Unmute."""

import threading

import numpy as np

from .audio import FRAME


class SpeechModels:
    def __init__(self, settings):
        self.settings = settings
        self.stt_lock = threading.Lock()
        self.tts_lock = threading.Lock()

    def load(self):
        import torch
        from pocket_tts import TTSModel
        from silero_vad import load_silero_vad

        s = self.settings
        torch.set_num_threads(s.threads)
        if s.tts_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        self.tts = TTSModel.load_model(
            language=s.tts_language, sampler_decode_steps=s.decode_steps
        ).to(s.tts_device)
        self.can_clone = self.tts.has_voice_cloning
        self.voice = self.tts.get_state_for_audio_prompt(s.voice)
        self.sample_rate = self.tts.sample_rate
        self.vad = load_silero_vad(onnx=True)
        if s.backend != "mlx":
            from faster_whisper import WhisperModel

            self.stt = WhisperModel(
                s.recognizer,
                device=s.backend,
                compute_type="int8_float16" if s.backend == "cuda" else "int8",
                cpu_threads=s.threads,
                num_workers=1,
            )
        self.transcribe(np.zeros(16000, dtype=np.float32))
        list(self.synthesize("Hello." if s.language == "en" else "Bonjour.", threading.Event()))
        self.reset_vad()

    def reset_vad(self):
        self.vad.reset_states()

    def probability(self, frame):
        import torch

        if len(frame) != FRAME:
            raise ValueError("VAD requires exactly 512 samples")
        return float(self.vad(torch.from_numpy(frame.copy()), 16000).item())

    def transcribe(self, audio):
        s = self.settings
        with self.stt_lock:
            if s.backend == "mlx":
                import mlx_whisper

                result = mlx_whisper.transcribe(
                    audio,
                    path_or_hf_repo=s.recognizer,
                    language=s.language or None,
                    condition_on_previous_text=False,
                    temperature=0,
                    verbose=False,
                )
                return result["text"].strip()
            segments, _ = self.stt.transcribe(
                audio,
                language=s.language or None,
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()

    def synthesize(self, text, stop):
        # Keep the lock until the generator unwinds, including cancellation.
        with self.tts_lock:
            if stop.is_set():
                return
            for chunk in self.tts.generate_audio_stream(self.voice, text, stop=stop):
                if stop.is_set():
                    return
                audio = chunk.detach().cpu().numpy().reshape(-1)
                yield (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
