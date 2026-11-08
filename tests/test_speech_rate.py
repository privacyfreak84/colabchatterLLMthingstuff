import numpy as np
import pytest

from tools import speech_rate


def test_rate_one_returns_identical_copy_without_calling_ffmpeg(monkeypatch):
    source = np.linspace(-0.5, 0.5, 100, dtype=np.float32)

    def unexpected(*args, **kwargs):
        raise AssertionError("FFmpeg must not run at the original rate")

    monkeypatch.setattr(speech_rate.subprocess, "run", unexpected)
    output = speech_rate.adjust_speech_rate(source, sample_rate=24000, rate=1.0)

    np.testing.assert_array_equal(output, source)
    assert output.dtype == np.float32
    assert output is not source


def test_slow_rate_uses_ffmpeg_atempo_without_changing_sample_rate(monkeypatch):
    source = np.linspace(-0.25, 0.25, 1000, dtype=np.float32)
    captured = {}
    expected_output = np.linspace(-0.25, 0.25, round(source.size / 0.8), dtype="<f4")

    class FakeResult:
        returncode = 0
        stdout = expected_output.tobytes()
        stderr = b""

    monkeypatch.setattr(speech_rate.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeResult()

    monkeypatch.setattr(speech_rate.subprocess, "run", fake_run)
    output = speech_rate.adjust_speech_rate(source, sample_rate=24000, rate=0.8)

    assert "-af" in captured["command"]
    assert "atempo=0.800000" in captured["command"]
    assert captured["command"][captured["command"].index("-ar") + 1] == "24000"
    assert captured["command"][-1] == "pipe:1"
    assert captured["kwargs"]["input"] == source.astype("<f4", copy=False).tobytes()
    assert captured["kwargs"]["timeout"] == 300
    assert output.dtype == np.float32
    assert output.size == expected_output.size


@pytest.mark.parametrize("rate", [0.0, 0.5, 0.749, 1.151, 2.0, float("nan"), float("inf")])
def test_invalid_speech_rates_are_rejected(rate):
    source = np.ones(32, dtype=np.float32)
    with pytest.raises(ValueError, match="Speech rate must"):
        speech_rate.adjust_speech_rate(source, sample_rate=24000, rate=rate)


def test_empty_multichannel_and_nonfinite_waveforms_are_rejected():
    with pytest.raises(ValueError, match="non-empty mono"):
        speech_rate.adjust_speech_rate(np.array([], dtype=np.float32), sample_rate=24000, rate=0.9)
    with pytest.raises(ValueError, match="non-empty mono"):
        speech_rate.adjust_speech_rate(np.ones((2, 8), dtype=np.float32), sample_rate=24000, rate=0.9)
    with pytest.raises(ValueError, match="non-finite"):
        speech_rate.adjust_speech_rate(np.array([0.0, np.nan], dtype=np.float32), sample_rate=24000, rate=0.9)


def test_missing_ffmpeg_has_actionable_error(monkeypatch):
    monkeypatch.setattr(speech_rate.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="FFmpeg is required"):
        speech_rate.adjust_speech_rate(np.ones(100, dtype=np.float32), sample_rate=24000, rate=0.9)


def test_ffmpeg_errors_are_reported(monkeypatch):
    class FailedResult:
        returncode = 1
        stdout = b""
        stderr = b"atempo filter failed"

    monkeypatch.setattr(speech_rate.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(speech_rate.subprocess, "run", lambda *args, **kwargs: FailedResult())
    with pytest.raises(RuntimeError, match="atempo filter failed"):
        speech_rate.adjust_speech_rate(np.ones(100, dtype=np.float32), sample_rate=24000, rate=0.9)


def test_duration_mismatch_is_not_silently_accepted(monkeypatch):
    class FakeResult:
        returncode = 0
        stdout = np.ones(100, dtype="<f4").tobytes()
        stderr = b""

    monkeypatch.setattr(speech_rate.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(speech_rate.subprocess, "run", lambda *args, **kwargs: FakeResult())
    with pytest.raises(RuntimeError, match="unexpected duration"):
        speech_rate.adjust_speech_rate(np.ones(1000, dtype=np.float32), sample_rate=24000, rate=0.8)
