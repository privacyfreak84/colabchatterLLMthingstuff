"""Pitch-preserving speech-rate adjustment for Chatterbox-generated narration.

This runs FFmpeg's atempo/WSOLA filter on each completed speech chunk. It does
not alter the sample rate or add reverb/delay, and is intentionally applied
before the notebook inserts its explicit inter-chunk/paragraph pauses.
"""
from __future__ import annotations

import math
import shutil
import subprocess

import numpy as np

MIN_SPEECH_RATE = 0.75
MAX_SPEECH_RATE = 1.15
DEFAULT_SPEECH_RATE = 0.85
PROCESSOR_ID = "ffmpeg-atempo-wsola-v1"


def adjust_speech_rate(
    waveform,
    *,
    sample_rate: int,
    rate: float = DEFAULT_SPEECH_RATE,
    ffmpeg_binary: str | None = None,
    timeout_seconds: int = 300,
) -> np.ndarray:
    """Adjust mono float audio tempo while keeping pitch and sample rate unchanged.

    rate=0.85 makes speech run at 85% of its original tempo (about 17.6%
    longer). rate=1.0 returns a copy without invoking FFmpeg.
    """
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Speech-rate processing requires a non-empty mono waveform.")
    if not np.isfinite(audio).all():
        raise ValueError("Speech-rate input contains non-finite samples.")
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Sample rate must be a positive integer.") from exc
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")

    try:
        rate = float(rate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Speech rate must be a finite number.") from exc
    if not math.isfinite(rate) or not MIN_SPEECH_RATE <= rate <= MAX_SPEECH_RATE:
        raise ValueError(
            f"Speech rate must be between {MIN_SPEECH_RATE:.2f}x and {MAX_SPEECH_RATE:.2f}x."
        )

    if math.isclose(rate, 1.0, rel_tol=0.0, abs_tol=1e-7):
        return audio.copy()

    ffmpeg = ffmpeg_binary or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg is required for pitch-preserving speech-rate control but was not found. "
            "Restart the notebook runtime or install FFmpeg, then rerun."
        )

    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel", "error",
        "-f", "f32le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-i", "pipe:0",
        "-af", f"atempo={rate:.6f}",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            input=audio.astype("<f4", copy=False).tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"FFmpeg speech-rate processing exceeded {timeout_seconds} seconds."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Could not start FFmpeg for speech-rate processing: {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "FFmpeg pitch-preserving speech-rate processing failed: "
            + (detail[-1200:] or f"exit code {result.returncode}")
        )
    if not result.stdout or len(result.stdout) % 4:
        raise RuntimeError("FFmpeg returned an empty or malformed speech-rate waveform.")

    stretched = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)
    if stretched.size == 0 or not np.isfinite(stretched).all():
        raise RuntimeError("FFmpeg returned an invalid speech-rate waveform.")

    expected_samples = audio.size / rate
    duration_ratio = stretched.size / expected_samples
    if not 0.90 <= duration_ratio <= 1.10:
        raise RuntimeError(
            "FFmpeg returned an unexpected duration after speech-rate processing "
            f"(expected about {expected_samples / sample_rate:.2f}s, "
            f"received {stretched.size / sample_rate:.2f}s)."
        )
    return stretched
