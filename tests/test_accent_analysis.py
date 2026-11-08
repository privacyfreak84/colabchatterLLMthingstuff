import csv
import io
import json
from pathlib import Path
import subprocess
import types

import numpy as np
import pytest

from tools import accent_analysis as accent


def make_audio_file(tmp_path, name="book.wav", payload=b"test audio"):
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def test_audio_validation_accepts_wav_and_mp3_and_rejects_other_types(tmp_path):
    assert accent.validate_audio_path(make_audio_file(tmp_path, "book.wav")).suffix == ".wav"
    assert accent.validate_audio_path(make_audio_file(tmp_path, "book.mp3")).suffix == ".mp3"
    with pytest.raises(ValueError, match="WAV and MP3"):
        accent.validate_audio_path(make_audio_file(tmp_path, "book.flac"))
    with pytest.raises(ValueError, match="empty"):
        accent.validate_audio_path(make_audio_file(tmp_path, "empty.mp3", b""))


def test_audio_validation_rejects_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        accent.validate_audio_path(tmp_path / "missing.wav")


def test_timestamp_format_handles_hours_and_fractional_seconds():
    assert accent.format_timestamp(0) == "00:00:00"
    assert accent.format_timestamp(65.9) == "00:01:05"
    assert accent.format_timestamp(3661.2) == "01:01:01"


def test_streaming_decoder_yields_overlapping_windows_and_final_tail(tmp_path, monkeypatch):
    path = make_audio_file(tmp_path)
    sample_rate = 16_000
    seconds = 41
    audio = np.full(seconds * sample_rate, 0.05, dtype="<f4")
    raw = audio.tobytes()

    class FakeProcess:
        def __init__(self):
            self.stdout = io.BytesIO(raw)
            self.stderr = io.BytesIO(b"")
            self._returncode = None
            self.terminated = False

        def wait(self):
            self._returncode = 0
            return 0

        def poll(self):
            return self._returncode

        def terminate(self):
            self.terminated = True
            self._returncode = -15

    process = FakeProcess()
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(accent.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(accent.subprocess, "Popen", fake_popen)

    windows = list(accent.iter_audio_windows(path))
    assert [row["start_seconds"] for row in windows] == [0.0, 10.0, 20.0, 30.0]
    assert [row["end_seconds"] for row in windows] == [15.0, 25.0, 35.0, 41.0]
    assert all(row["audio"].dtype == np.float32 for row in windows)
    assert all(len(row["audio"]) <= 15 * sample_rate for row in windows)
    assert captured["command"][captured["command"].index("-ar") + 1] == "16000"
    assert str(path) in captured["command"]


def test_streaming_decoder_includes_short_file_when_longer_than_minimum(tmp_path, monkeypatch):
    path = make_audio_file(tmp_path, "short.mp3")
    raw = np.full(8 * 16_000, 0.04, dtype="<f4").tobytes()

    class FakeProcess:
        def __init__(self):
            self.stdout = io.BytesIO(raw)
            self.stderr = io.BytesIO(b"")
            self.code = None

        def wait(self):
            self.code = 0
            return 0

        def poll(self):
            return self.code

        def terminate(self):
            self.code = -15

    monkeypatch.setattr(accent.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(accent.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    windows = list(accent.iter_audio_windows(path))
    assert len(windows) == 1
    assert windows[0]["start_seconds"] == 0
    assert windows[0]["end_seconds"] == 8


def test_streaming_decoder_requires_ffmpeg(tmp_path, monkeypatch):
    path = make_audio_file(tmp_path)
    monkeypatch.setattr(accent.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="FFmpeg"):
        list(accent.iter_audio_windows(path))


def test_report_groups_adjacent_predictions_and_flags_only_sustained_changes(tmp_path):
    path = make_audio_file(tmp_path)
    rows = [
        {"start_seconds": 0, "end_seconds": 15, "accent": "north_american", "score": 0.82,
         "scores": {"north_american": 0.82, "british_isles": 0.18}},
        {"start_seconds": 10, "end_seconds": 25, "accent": "north_american", "score": 0.78,
         "scores": {"north_american": 0.78, "british_isles": 0.22}},
        {"start_seconds": 20, "end_seconds": 35, "accent": "british_isles", "score": 0.71,
         "scores": {"north_american": 0.29, "british_isles": 0.71}},
        {"start_seconds": 30, "end_seconds": 45, "accent": "british_isles", "score": 0.76,
         "scores": {"north_american": 0.24, "british_isles": 0.76}},
    ]
    report = accent.build_accent_report(
        rows,
        audio_path=path,
        audio_duration_seconds=45,
    )
    assert report["dominant_accent"] == "north_american"
    assert report["dominant_top_label_share"] == 0.5
    assert len(report["regions"]) == 2
    assert report["regions"][0]["accent_display"] == "North American English"
    assert len(report["possible_sustained_changes"]) == 1
    assert report["possible_sustained_changes"][0]["from"] == "North American English"
    assert report["transcription_performed"] is False


def test_report_does_not_claim_single_window_change_is_sustained(tmp_path):
    path = make_audio_file(tmp_path)
    rows = [
        {"start_seconds": 0, "end_seconds": 15, "accent": "north_american", "score": 0.8,
         "scores": {"north_american": 0.8, "british_isles": 0.2}},
        {"start_seconds": 10, "end_seconds": 25, "accent": "british_isles", "score": 0.6,
         "scores": {"north_american": 0.4, "british_isles": 0.6}},
        {"start_seconds": 20, "end_seconds": 35, "accent": "north_american", "score": 0.7,
         "scores": {"north_american": 0.7, "british_isles": 0.3}},
    ]
    report = accent.build_accent_report(rows, audio_path=path, audio_duration_seconds=35)
    assert report["possible_sustained_changes"] == []


def test_report_rejects_no_usable_segments(tmp_path):
    path = make_audio_file(tmp_path)
    with pytest.raises(ValueError, match="No usable speech windows"):
        accent.build_accent_report([], audio_path=path, audio_duration_seconds=0)


def test_report_writes_json_and_csv_without_audio_copy(tmp_path):
    audio = make_audio_file(tmp_path)
    report = accent.build_accent_report(
        [{
            "start_seconds": 0, "end_seconds": 15, "start": "00:00:00", "end": "00:00:15",
            "accent": "north_american", "accent_display": "North American English",
            "score": 0.8, "runner_up": "british_isles",
            "runner_up_display": "British Isles English", "runner_up_score": 0.2,
            "margin": 0.6,
            "scores": {"north_american": 0.8, "british_isles": 0.2},
        }],
        audio_path=audio,
        audio_duration_seconds=15,
    )
    json_path, csv_path = accent.write_accent_report(report, tmp_path / "reports")
    assert Path(json_path).is_file()
    assert Path(csv_path).is_file()
    saved = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert saved["source_audio_sha256"] == accent.sha256_file(audio)
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["accent_display"] == "North American English"
    assert not (tmp_path / "reports" / audio.name).exists()


def test_markdown_report_is_timestamped_and_has_no_transcript_or_source_text(tmp_path):
    audio = make_audio_file(tmp_path)
    report = accent.build_accent_report(
        [{
            "start_seconds": 0, "end_seconds": 15, "accent": "north_american",
            "score": 0.8, "scores": {"north_american": 0.8, "british_isles": 0.2},
        }],
        audio_path=audio,
        audio_duration_seconds=15,
    )
    markdown = accent.format_report_markdown(report)
    assert "00:00:00 to 00:00:15" in markdown
    assert "North American English" in markdown
    assert "No transcription is performed" in markdown
    assert "The Buddha speaks" not in markdown


def test_accent_model_loader_pins_source_and_checkpoint_artifacts():
    import inspect
    source = inspect.getsource(accent.load_accentcl_model)
    assert len(accent.ACCENTCL_COMMIT) == 40
    assert len(accent.ACCENTCL_CHECKPOINT_REVISION) == 40
    assert len(accent.ACCENTCL_CHECKPOINT_SHA256) == 64
    assert accent.ACCENTCL_CHECKPOINT_SIZE_BYTES == 2_563_884_656
    assert "revision=ACCENTCL_CHECKPOINT_REVISION" in source
    assert "load_verified_accent_checkpoint(" in source
    assert "snapshot_download" in source
    assert "openai/whisper-large-v3" in source
    assert "model_name=str(whisper_path)" in source


def test_legacy_accent_checkpoint_is_deserialized_only_after_hash_verification(tmp_path, monkeypatch):
    import hashlib

    payload = b"official AccentCL checkpoint fixture"
    checkpoint_path = tmp_path / "accentcl.pt"
    checkpoint_path.write_bytes(payload)
    monkeypatch.setattr(
        accent, "ACCENTCL_CHECKPOINT_SIZE_BYTES", len(payload)
    )
    monkeypatch.setattr(
        accent, "ACCENTCL_CHECKPOINT_SHA256", hashlib.sha256(payload).hexdigest()
    )

    calls = {}

    class FakeTorch:
        @staticmethod
        def load(path, **kwargs):
            calls["path"] = path
            calls["kwargs"] = kwargs
            return {"label2id": {"north_american": 0}, "model": {"weight": "tensor"}}

    checkpoint = accent.load_verified_accent_checkpoint(
        checkpoint_path, torch_module=FakeTorch
    )
    assert checkpoint["label2id"] == {"north_american": 0}
    assert calls["kwargs"] == {
        "map_location": "cpu",
        "weights_only": False,
    }


def test_unverified_accent_checkpoint_is_never_unpickled(tmp_path, monkeypatch):
    payload = b"tampered checkpoint"
    checkpoint_path = tmp_path / "accentcl.pt"
    checkpoint_path.write_bytes(payload)
    monkeypatch.setattr(accent, "ACCENTCL_CHECKPOINT_SIZE_BYTES", len(payload))
    monkeypatch.setattr(accent, "ACCENTCL_CHECKPOINT_SHA256", "0" * 64)

    class FakeTorch:
        called = False

        @classmethod
        def load(cls, *args, **kwargs):
            cls.called = True
            raise AssertionError("torch.load must not run for an unverified checkpoint")

    with pytest.raises(RuntimeError, match="SHA-256 verification failed"):
        accent.load_verified_accent_checkpoint(checkpoint_path, torch_module=FakeTorch)
    assert FakeTorch.called is False
