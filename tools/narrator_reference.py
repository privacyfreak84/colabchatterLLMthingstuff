"""CPU-only checks for a candidate audiobook narrator reference recording.

Chatterbox conditions on a short reference clip: the T3 text-to-token model
uses roughly the first 6 seconds and S3Gen uses roughly the first 10 seconds.
This module inspects a PCM WAV with the standard library only, so it runs in
CI and anywhere Python runs. It reports measurable problems; it cannot judge
whether the voice itself is the right one for the book.
"""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Any

# Chatterbox reads only the start of a reference clip.
USED_WINDOW_SECONDS = 10.0
MIN_SECONDS_BLOCKING = 6.0
MIN_SECONDS_RECOMMENDED = 8.0
FRAME_SECONDS = 0.02
SILENCE_DBFS = -50.0


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(value) if value > 1e-9 else -180.0


def _read_mono_samples(path: Path, max_seconds: float | None) -> tuple[list[float], dict[str, Any]]:
    """Return mono float samples in [-1, 1] plus basic file facts (PCM WAV only)."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        total_frames = handle.getnframes()
        wanted = total_frames if max_seconds is None else min(total_frames, int(max_seconds * rate))
        raw = handle.readframes(wanted)
    if width == 1:
        values = [(b - 128) / 128.0 for b in raw]
    elif width == 2:
        values = [v / 32768.0 for v in struct.unpack(f"<{len(raw) // 2}h", raw)]
    elif width == 3:
        values = []
        for i in range(0, len(raw) - 2, 3):
            n = int.from_bytes(raw[i:i + 3], "little", signed=True)
            values.append(n / 8388608.0)
    elif width == 4:
        values = [v / 2147483648.0 for v in struct.unpack(f"<{len(raw) // 4}i", raw)]
    else:
        raise ValueError(f"Unsupported PCM sample width: {width} bytes.")
    if channels > 1:
        usable = len(values) - len(values) % channels
        values = [
            sum(values[i:i + channels]) / channels for i in range(0, usable, channels)
        ]
    facts = {
        "channels": channels,
        "sample_rate": rate,
        "sample_width_bytes": width,
        "duration_seconds": round(total_frames / rate, 3) if rate else 0.0,
    }
    return values, facts


def analyze_reference(path: str | Path) -> dict[str, Any]:
    """Measure the part of the recording Chatterbox will actually use."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Narrator reference not found: {path}")
    try:
        samples, facts = _read_mono_samples(path, USED_WINDOW_SECONDS)
    except wave.Error as exc:
        raise ValueError(
            f"{path.name} is not a readable PCM WAV ({exc}). "
            "Convert it to 16- or 24-bit PCM WAV (for example with ffmpeg) and try again."
        ) from exc
    if not samples:
        raise ValueError(f"{path.name} contains no audio samples.")

    rate = facts["sample_rate"]
    peak = max(abs(s) for s in samples)
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    clipped = sum(1 for s in samples if abs(s) >= 0.999) / len(samples)

    frame = max(1, int(FRAME_SECONDS * rate))
    frame_db = []
    for i in range(0, len(samples) - frame + 1, frame):
        chunk = samples[i:i + frame]
        frame_db.append(_dbfs(math.sqrt(sum(s * s for s in chunk) / frame)))
    if not frame_db:
        raise ValueError(f"{path.name} is too short to analyse.")

    leading = 0
    while leading < len(frame_db) and frame_db[leading] < SILENCE_DBFS:
        leading += 1
    trailing = 0
    while trailing < len(frame_db) - leading and frame_db[-1 - trailing] < SILENCE_DBFS:
        trailing += 1

    ordered = sorted(frame_db)
    noise_floor = ordered[int(0.10 * (len(ordered) - 1))]
    speech_level = ordered[int(0.90 * (len(ordered) - 1))]
    voiced = sum(1 for d in frame_db if d > SILENCE_DBFS + 15) / len(frame_db)

    return {
        **facts,
        "analysed_seconds": round(len(samples) / rate, 3),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "rms_dbfs": round(_dbfs(rms), 1),
        "clipped_fraction": round(clipped, 8),
        "leading_silence_seconds": round(leading * FRAME_SECONDS, 2),
        "trailing_silence_seconds": round(trailing * FRAME_SECONDS, 2),
        "noise_floor_dbfs": round(noise_floor, 1),
        "speech_level_dbfs": round(speech_level, 1),
        "speech_to_noise_db": round(speech_level - noise_floor, 1),
        "voiced_fraction": round(voiced, 3),
    }


def validate_narrator_reference(path: str | Path) -> dict[str, Any]:
    """Return metrics plus blocking `problems` and advisory `warnings`.

    `ok` means no measurable blocker was found. It is not a quality approval:
    the voice, accent, and delivery still need a human decision.
    """
    metrics = analyze_reference(path)
    problems: list[str] = []
    warnings: list[str] = []

    seconds = metrics["duration_seconds"]
    if seconds < MIN_SECONDS_BLOCKING:
        problems.append(
            f"Recording is {seconds:.1f} s; at least {MIN_SECONDS_BLOCKING:.0f} s is needed "
            f"(about {MIN_SECONDS_RECOMMENDED:.0f}-{USED_WINDOW_SECONDS:.0f} s is recommended)."
        )
    elif seconds < MIN_SECONDS_RECOMMENDED:
        warnings.append(f"Recording is {seconds:.1f} s; 8-10 s of speech is recommended.")
    if seconds > USED_WINDOW_SECONDS:
        warnings.append(
            f"Only the first {USED_WINDOW_SECONDS:.0f} s is used; make sure that opening is representative."
        )

    if metrics["sample_rate"] < 16000:
        problems.append(f"Sample rate {metrics['sample_rate']} Hz is below 16 kHz.")
    elif metrics["sample_rate"] < 24000:
        warnings.append(f"Sample rate {metrics['sample_rate']} Hz is below the 24 kHz model output rate.")
    if metrics["channels"] > 1:
        warnings.append("Recording has more than one channel; it will be mixed to mono.")

    if metrics["clipped_fraction"] > 0.001:
        problems.append(f"Clipping: {metrics['clipped_fraction'] * 100:.2f}% of samples are at full scale.")
    elif metrics["clipped_fraction"] > 0:
        warnings.append("A few samples reach full scale; re-record with lower input gain if possible.")
    if metrics["rms_dbfs"] < -35:
        warnings.append(f"Level is very low ({metrics['rms_dbfs']} dBFS RMS); consider a louder recording.")

    if metrics["leading_silence_seconds"] > 1.0:
        warnings.append(f"{metrics['leading_silence_seconds']} s of leading silence wastes the used window.")
    if metrics["trailing_silence_seconds"] > 1.0 and seconds <= USED_WINDOW_SECONDS:
        warnings.append(f"{metrics['trailing_silence_seconds']} s of trailing silence wastes the used window.")

    if metrics["speech_to_noise_db"] < 15:
        problems.append(
            f"Speech is only {metrics['speech_to_noise_db']} dB above the noise floor; the recording is too noisy."
        )
    elif metrics["speech_to_noise_db"] < 25:
        warnings.append(
            f"Speech is {metrics['speech_to_noise_db']} dB above the noise floor; a quieter room would be better."
        )
    if metrics["voiced_fraction"] < 0.5:
        warnings.append("Less than half the analysed window contains speech.")

    return {"ok": not problems, "problems": problems, "warnings": warnings, "metrics": metrics}
