"""Exercise run_narrator_retest end to end with a stand-in model (no GPU, no weights).

This checks control flow only: which voice conditions which task, resuming,
partial runs, and the guards. It says nothing about how real audio sounds.
"""
import json
import math
import struct
import sys
import types
import wave
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from tools.narrator_retest import (  # noqa: E402
    CPU_SMOKE_TERMS,
    make_aligned_reference,
    make_transfer_bundle,
    run_narrator_retest,
    select_terms,
)

PARAMS = {"exaggeration": 0.5, "cfg_weight": 0.5}


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class Conds:
    t3 = None
    gen = None


class FakeModel:
    sr = 24000
    device = "cpu"
    s3gen = object()

    def __init__(self, log):
        self.log = log
        self.conds = Conds()

    def prepare_conditionals(self, path, exaggeration=0.5):
        self.conds = Conds()
        self.conds.t3 = f"t3:{Path(path).name}"
        self.conds.gen = f"gen:{Path(path).name}"

    def generate(self, text, language_id, **params):
        self.log.append(("gen", language_id, self.conds.t3, self.conds.gen, text))
        n = int(0.05 * 24000 * (len(text) % 9 + 4))
        tone = 0.3 * np.sin(2 * np.pi * 200 * np.arange(n) / 24000)
        pad = np.zeros(3000)
        return FakeTensor(np.concatenate([pad, tone, pad]))


@pytest.fixture
def stubs(monkeypatch):
    log = []
    torch = types.ModuleType("torch")
    torch.__version__ = "2.6.0+cpu"
    torch.manual_seed = lambda seed: None
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, manual_seed=lambda s: None, manual_seed_all=lambda s: None
    )
    chatterbox = types.ModuleType("chatterbox")
    vc = types.ModuleType("chatterbox.vc")

    class FakeVC:
        def __init__(self, s3gen, device, ref_dict=None):
            self.ref = ref_dict

        def generate(self, audio):
            data, _ = sf.read(audio, dtype="float32")
            log.append(("vc", Path(audio).name, self.ref))
            return FakeTensor(data * 0.9)

    vc.ChatterboxVC = FakeVC
    chatterbox.vc = vc
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "chatterbox", chatterbox)
    monkeypatch.setitem(sys.modules, "chatterbox.vc", vc)
    return log


def _wav(path, seconds=9.0):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * 180 * i / 24000)
                                  * (1 if (i / 24000) % 0.5 < 0.35 else 0)))
            for i in range(int(seconds * 24000))
        )
        handle.writeframes(frames)


@pytest.fixture
def kwargs(tmp_path, stubs):
    _wav(tmp_path / "narrator.wav")
    _wav(tmp_path / "hindi.wav")
    return dict(
        narrator_reference_audio=tmp_path / "narrator.wav",
        hindi_reference_audio=tmp_path / "hindi.wav",
        output_dir=tmp_path / "out",
        model_revision="rev",
        code_revision="commit",
        base_generation_parameters=PARAMS,
    )


def test_cpu_partial_run_uses_the_right_voice_for_each_task(kwargs, stubs):
    result = run_narrator_retest(model=FakeModel(stubs), terms=select_terms(CPU_SMOKE_TERMS), **kwargs)
    assert (result["task_count"], result["generated_count"], result["failed_count"]) == (14, 14, 0)
    assert result["runtime"]["device"] == "cpu"
    gens = [c for c in stubs if c[0] == "gen"]
    assert all((c[2], c[3]) == ("t3:narrator.wav", "gen:narrator.wav") for c in gens if c[1] == "en")
    assert {(c[2], c[3]) for c in gens if c[1] == "hi"} == {
        ("t3:hindi.wav", "gen:hindi.wav"),  # Hindi source clip
        ("t3:hindi.wav", "gen:narrator.wav"),  # HYBRID: Hindi T3, narrator S3Gen
    }
    assert {c[2] for c in stubs if c[0] == "vc"} == {"gen:narrator.wav"}
    manifest = json.loads((kwargs["output_dir"] / "narrator_retest_manifest.json").read_text())
    assert manifest["full_book_synthesis_authorized"] is False
    assert all("generation_seconds" in t for t in manifest["tasks"])
    assert not (kwargs["output_dir"] / "N001_EN_PREFIX.wav").exists()  # terms not requested


def test_resume_reuses_everything_and_extending_keeps_earlier_clips(kwargs, stubs):
    run_narrator_retest(model=FakeModel(stubs), terms=select_terms(CPU_SMOKE_TERMS), **kwargs)
    out = kwargs["output_dir"]
    before = {p.name: p.stat().st_mtime_ns for p in out.glob("N*.wav")}

    stubs.clear()
    run_narrator_retest(model=FakeModel(stubs), terms=select_terms(CPU_SMOKE_TERMS), **kwargs)
    assert stubs == []  # nothing regenerated

    stubs.clear()
    full = run_narrator_retest(model=FakeModel(stubs), **kwargs)
    assert full["generated_count"] == 28
    assert len([c for c in stubs if c[0] == "gen"]) == 8  # only the two new terms
    assert all((out / name).stat().st_mtime_ns == stamp for name, stamp in before.items())

    stubs.clear()
    subset = run_narrator_retest(model=FakeModel(stubs), terms=select_terms(["Maitreya"]), **kwargs)
    assert subset["task_count"] == 7 and stubs == []
    manifest = json.loads((out / "narrator_retest_manifest.json").read_text())
    assert len(manifest["tasks"]) == 28  # earlier terms stay recorded


def test_deleted_clip_is_regenerated_alone(kwargs, stubs):
    run_narrator_retest(model=FakeModel(stubs), terms=select_terms(["Maitreya"]), **kwargs)
    (kwargs["output_dir"] / "N002_EN_PREFIX.wav").unlink()
    stubs.clear()
    result = run_narrator_retest(model=FakeModel(stubs), terms=select_terms(["Maitreya"]), **kwargs)
    assert len([c for c in stubs if c[0] == "gen"]) == 1
    assert result["generated_count"] == 7


def test_guards_refuse_conflicting_or_unsafe_runs(kwargs, stubs, tmp_path):
    run_narrator_retest(model=FakeModel(stubs), terms=select_terms(["Maitreya"]), **kwargs)
    changed = {"Maitreya": {"devanagari": "मैत्रेय", "sentence": "Then Maitreya said something else entirely."}}
    with pytest.raises(RuntimeError, match="differs from the one already"):
        run_narrator_retest(model=FakeModel(stubs), terms=changed, **kwargs)
    with pytest.raises(RuntimeError, match="different narrator/model/settings"):
        run_narrator_retest(
            model=FakeModel(stubs), terms=select_terms(["Maitreya"]),
            **{**kwargs, "base_generation_parameters": {"exaggeration": 0.9}},
        )
    _wav(tmp_path / "short.wav", seconds=2.0)
    with pytest.raises(ValueError, match="blocking problems"):
        run_narrator_retest(
            model=FakeModel(stubs), terms=select_terms(["Maitreya"]),
            **{**kwargs, "narrator_reference_audio": tmp_path / "short.wav", "output_dir": tmp_path / "o2"},
        )


def test_bundle_contains_manifest_and_every_clip(kwargs, stubs, tmp_path):
    import base64
    import io
    import zipfile

    run_narrator_retest(model=FakeModel(stubs), terms=select_terms(["Maitreya"]), **kwargs)
    info = make_transfer_bundle(kwargs["output_dir"], tmp_path / "b.zip", tmp_path / "b.txt")
    assert info["wav_count"] == 7
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(Path(info["text"]).read_bytes())))
    assert "narrator_retest_manifest.json" in archive.namelist()
    assert len([n for n in archive.namelist() if n.endswith(".wav")]) == 7


def test_aligned_reference_is_first_ten_seconds_padded_to_40ms(tmp_path):
    _wav(tmp_path / "long.wav", seconds=14.0)
    seconds = make_aligned_reference(tmp_path / "long.wav", tmp_path / "aligned.wav")
    data, rate = sf.read(tmp_path / "aligned.wav")
    assert rate == 24000 and len(data) % 960 == 0
    assert seconds == pytest.approx(10.0, abs=0.001)
    _wav(tmp_path / "odd.wav", seconds=7.013)
    make_aligned_reference(tmp_path / "odd.wav", tmp_path / "odd_aligned.wav")
    assert len(sf.read(tmp_path / "odd_aligned.wav")[0]) % 960 == 0
