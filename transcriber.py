"""Whisper transcription pipeline.

Accepts raw PCM audio (16-bit signed LE, 48kHz stereo — Discord's native
format) and returns transcribed text via the openai-whisper package.
"""

from __future__ import annotations

import io
import logging
import struct
import tempfile
import time
import wave
from pathlib import Path
from threading import Lock

import numpy as np

log = logging.getLogger("transcriber")

# Lazy-loaded whisper model (first call downloads weights)
_model = None
_model_lock = Lock()
_model_name: str = "small"


def set_model(name: str) -> None:
    global _model_name
    _model_name = name


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import whisper

                log.info("Loading whisper model '%s' (first run downloads weights)...", _model_name)
                _model = whisper.load_model(_model_name)
                log.info("Whisper model loaded.")
    return _model


def pcm_to_wav_bytes(pcm: bytes, *, sample_rate: int = 48000, channels: int = 2, sample_width: int = 2) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def pcm_to_float32(pcm: bytes, *, sample_rate: int = 48000, channels: int = 2) -> np.ndarray:
    """Convert raw PCM (s16le, 48kHz) to float32 mono at 16kHz for Whisper."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    # Stereo to mono
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    # Resample 48kHz -> 16kHz (simple decimation by 3)
    samples = samples[::3]
    return samples


def transcribe_pcm(pcm: bytes, *, sample_rate: int = 48000, channels: int = 2) -> dict:
    """Transcribe raw PCM audio.

    Returns:
        {"text": str, "language": str, "duration_ms": int}
        text is empty string if nothing detected.
    """
    if len(pcm) < 1600:
        return {"text": "", "language": "unknown", "duration_ms": 0}

    model = _get_model()
    audio = pcm_to_float32(pcm, sample_rate=sample_rate, channels=channels)
    duration_ms = int(len(audio) / 16000 * 1000)

    if duration_ms < 200:
        return {"text": "", "language": "unknown", "duration_ms": duration_ms}

    t0 = time.monotonic()
    result = model.transcribe(
        audio,
        language=None,  # auto-detect
        fp16=False,  # CPU-safe
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        condition_on_previous_text=False,
    )
    elapsed = time.monotonic() - t0
    text = result.get("text", "").strip()
    lang = result.get("language", "unknown")

    log.debug(
        "Transcribed %dms audio in %.1fs: lang=%s text=%r",
        duration_ms, elapsed, lang, text[:80],
    )
    return {"text": text, "language": lang, "duration_ms": duration_ms}
