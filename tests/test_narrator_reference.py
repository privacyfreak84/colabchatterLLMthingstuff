import math
import struct
import wave

import pytest

from tools.narrator_reference import analyze_reference, validate_narrator_reference

RATE = 24000


def _write_wav(path, samples, rate=RATE, channels=1, width=2):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        if width == 2:
            frames = b"".join(struct.pack("<h", max(-32768, min(32767, int(s * 32767)))) for s in samples)
        else:
            frames = b"".join(bytes([max(0, min(255, int(128 + s * 127)))]) for s in samples)
        handle.writeframes(frames)
    return path


def _speechlike(seconds, amplitude=0.4, noise=0.0005, rate=RATE):
    """Tone bursts with pauses: a stand-in for speech with a quiet noise floor."""
    out = []
    for i in range(int(seconds * rate)):
        t = i / rate
        burst = 1.0 if (t % 0.5) < 0.35 else 0.0
        tone = math.sin(2 * math.pi * 180 * t) * amplitude * burst
        hiss = noise * math.sin(2 * math.pi * 3137 * t)
        out.append(tone + hiss)
    return out


def test_good_recording_passes_without_problems(tmp_path):
    path = _write_wav(tmp_path / "ok.wav", _speechlike(9.0))
    result = validate_narrator_reference(path)
    assert result["ok"], result["problems"]
    assert result["problems"] == []
    assert result["metrics"]["duration_seconds"] == pytest.approx(9.0, abs=0.01)
    assert result["metrics"]["speech_to_noise_db"] >= 25


def test_too_short_recording_is_blocked(tmp_path):
    path = _write_wav(tmp_path / "short.wav", _speechlike(3.0))
    result = validate_narrator_reference(path)
    assert not result["ok"]
    assert any("at least 6 s" in p for p in result["problems"])


def test_between_six_and_eight_seconds_only_warns(tmp_path):
    path = _write_wav(tmp_path / "mid.wav", _speechlike(7.0))
    result = validate_narrator_reference(path)
    assert result["ok"]
    assert any("8-10 s" in w for w in result["warnings"])


def test_clipping_is_blocked(tmp_path):
    loud = [max(-1.0, min(1.0, s * 4)) for s in _speechlike(9.0)]
    path = _write_wav(tmp_path / "clip.wav", loud)
    result = validate_narrator_reference(path)
    assert not result["ok"]
    assert any("Clipping" in p for p in result["problems"])


def test_noisy_recording_is_blocked(tmp_path):
    path = _write_wav(tmp_path / "noisy.wav", _speechlike(9.0, amplitude=0.05, noise=0.04))
    result = validate_narrator_reference(path)
    assert not result["ok"]
    assert any("noise" in p for p in result["problems"])


def test_low_sample_rate_is_blocked_and_24k_warning_is_separate(tmp_path):
    low = _write_wav(tmp_path / "low.wav", _speechlike(9.0, rate=8000), rate=8000)
    assert any("16 kHz" in p for p in validate_narrator_reference(low)["problems"])
    mid = _write_wav(tmp_path / "mid.wav", _speechlike(9.0, rate=16000), rate=16000)
    result = validate_narrator_reference(mid)
    assert result["ok"]
    assert any("24 kHz" in w for w in result["warnings"])


def test_only_the_first_ten_seconds_are_analysed_and_a_warning_is_given(tmp_path):
    path = _write_wav(tmp_path / "long.wav", _speechlike(14.0))
    result = validate_narrator_reference(path)
    assert result["metrics"]["analysed_seconds"] == pytest.approx(10.0, abs=0.01)
    assert result["metrics"]["duration_seconds"] == pytest.approx(14.0, abs=0.01)
    assert any("first 10 s" in w for w in result["warnings"])


def test_long_leading_silence_warns(tmp_path):
    samples = [0.0] * (2 * RATE) + _speechlike(8.0)
    path = _write_wav(tmp_path / "lead.wav", samples)
    result = validate_narrator_reference(path)
    assert result["metrics"]["leading_silence_seconds"] >= 1.9
    assert any("leading silence" in w for w in result["warnings"])


def test_stereo_is_mixed_down_with_a_warning(tmp_path):
    mono = _speechlike(9.0)
    stereo = [s for sample in mono for s in (sample, sample)]
    path = _write_wav(tmp_path / "st.wav", stereo, channels=2)
    result = validate_narrator_reference(path)
    assert result["metrics"]["channels"] == 2
    assert any("mixed to mono" in w for w in result["warnings"])


def test_missing_and_non_wav_files_give_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_reference(tmp_path / "nope.wav")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav file at all")
    with pytest.raises(ValueError, match="PCM WAV"):
        analyze_reference(bad)
