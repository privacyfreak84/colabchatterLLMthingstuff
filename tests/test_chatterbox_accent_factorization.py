"""CPU-only tests for the Chatterbox speaker/accent conditioning overlay."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = REPO_ROOT / "tools" / "chatterbox_accent_factorization.py"


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to(self, *args, **kwargs):
        return self

    def mean(self, axis=0, keepdim=False):
        result = self.value.mean(axis=axis, keepdims=keepdim)
        return FakeTensor(result)

    def __rmul__(self, other):
        return FakeTensor(other * self.value)

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.value, axis=axis))


class FakeT3Cond:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to(self, **kwargs):
        return self


class FakeConditionals:
    def __init__(self, t3, gen):
        self.t3 = t3
        self.gen = gen


class FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")

    @staticmethod
    def from_numpy(value):
        return FakeTensor(value)

    @staticmethod
    def atleast_2d(value):
        if isinstance(value, FakeTensor):
            value = value.value
        return FakeTensor(np.atleast_2d(value))

    @staticmethod
    def ones(*shape):
        return FakeTensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def as_tensor(value, **kwargs):
        return FakeTensor(value)


def install_fake_chatterbox_modules(monkeypatch):
    loads = {}

    fake_librosa = types.ModuleType("librosa")

    def load(path, sr, mono=True):
        path = str(path)
        if "identity" in path:
            audio = np.full(sr * 25, 0.2, dtype=np.float32)
        else:
            # A time ramp makes later prompt offsets observable in the tests.
            audio = np.linspace(0.0, 1.0, sr * 25, dtype=np.float32)
        loads[path] = audio.copy()
        return audio, sr

    def resample(audio, orig_sr, target_sr):
        audio = np.asarray(audio, dtype=np.float32)
        target_length = round(len(audio) * target_sr / orig_sr)
        return np.interp(
            np.linspace(0, len(audio) - 1, target_length),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    fake_librosa.load = load
    fake_librosa.resample = resample

    chatterbox = types.ModuleType("chatterbox")
    chatterbox.__path__ = []
    models = types.ModuleType("chatterbox.models")
    models.__path__ = []
    s3tokenizer = types.ModuleType("chatterbox.models.s3tokenizer")
    s3tokenizer.S3_SR = 16000
    s3gen_module = types.ModuleType("chatterbox.models.s3gen")
    s3gen_module.S3GEN_SR = 24000
    t3 = types.ModuleType("chatterbox.models.t3")
    t3.__path__ = []
    t3_modules = types.ModuleType("chatterbox.models.t3.modules")
    t3_modules.__path__ = []
    cond_enc = types.ModuleType("chatterbox.models.t3.modules.cond_enc")
    cond_enc.T3Cond = FakeT3Cond
    mtl = types.ModuleType("chatterbox.mtl_tts")
    mtl.Conditionals = FakeConditionals

    for name, module in {
        "librosa": fake_librosa,
        "torch": FakeTorch(),
        "chatterbox": chatterbox,
        "chatterbox.models": models,
        "chatterbox.models.s3tokenizer": s3tokenizer,
        "chatterbox.models.s3gen": s3gen_module,
        "chatterbox.models.t3": t3,
        "chatterbox.models.t3.modules": t3_modules,
        "chatterbox.models.t3.modules.cond_enc": cond_enc,
        "chatterbox.mtl_tts": mtl,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return loads


def load_patch_module():
    spec = importlib.util.spec_from_file_location("chatterbox_accent_factorization_test", PATCH_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyTTS:
    ENC_COND_LEN = 6 * 16000
    DEC_COND_LEN = 10 * 24000
    device = "cpu"

    def __init__(self):
        self.t3 = types.SimpleNamespace(
            hp=types.SimpleNamespace(speech_cond_prompt_len=150)
        )
        self.calls = {}

        class VoiceEncoder:
            def embeds_from_wavs(inner_self, wavs, sample_rate):
                self.calls["voice_identity"] = np.asarray(wavs[0]).copy()
                return np.full((1, 256), float(np.mean(wavs[0])), dtype=np.float32)

        class S3SpeakerEncoder:
            def inference(inner_self, wav):
                self.calls["s3_identity"] = wav.value.copy()
                return FakeTensor([[float(np.mean(wav.value))]])

        class S3Tokenizer:
            def forward(inner_self, wavs, max_len):
                self.calls["t3_style"] = np.asarray(wavs[0]).copy()
                return FakeTensor([[17]]), None

        class S3Gen:
            dtype = "float32"

            def __init__(inner_self):
                inner_self.tokenizer = S3Tokenizer()
                inner_self.speaker_encoder = S3SpeakerEncoder()

            def embed_ref(inner_self, wav, sr, device):
                self.calls["s3_style"] = np.asarray(wav).copy()
                return {
                    "prompt_token": FakeTensor([[1]]),
                    "prompt_token_len": FakeTensor([1]),
                    "prompt_feat": FakeTensor([[[1.0]]]),
                    "prompt_feat_len": None,
                    "embedding": FakeTensor([[999.0]]),
                }

        self.ve = VoiceEncoder()
        self.s3gen = S3Gen()
        self.conds = None
        self.generate_call = None

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
    ):
        self.generate_call = {
            "text": text,
            "language_id": language_id,
            "audio_prompt_path": audio_prompt_path,
        }
        return "generated"


def test_source_patch_separates_voice_identity_from_accent_prompt(monkeypatch, tmp_path):
    loads = install_fake_chatterbox_modules(monkeypatch)
    patch = load_patch_module()
    installed = patch.install_chatterbox_accent_factorization(DummyTTS)
    assert installed["installed"] is True
    assert installed["already_installed"] is False

    identity_path = tmp_path / "identity.wav"
    accent_path = tmp_path / "accent.wav"
    identity_path.write_bytes(b"identity")
    accent_path.write_bytes(b"accent")

    model = DummyTTS()
    model.prepare_conditionals(identity_path, accent_prompt_path=accent_path)

    assert np.isclose(np.mean(model.calls["voice_identity"]), 0.2)
    assert np.isclose(np.mean(model.calls["s3_identity"]), 0.2)
    assert np.isclose(np.mean(model.calls["s3_style"]), 0.5)
    assert np.isclose(np.mean(model.calls["t3_style"]), 0.5)
    assert np.isclose(
        float(model.conds.t3.speaker_emb.value.mean()), 0.2
    )
    assert np.isclose(
        float(model.conds.gen["embedding"].value.mean()), 0.2
    )

    # The style reference must not overwrite the narrator's speaker identity.
    assert not np.isclose(
        float(model.conds.gen["embedding"].value.mean()),
        float(model.calls["s3_style"].mean()),
    )
    # Installer is idempotent in notebook restarts/re-execution.
    second = patch.install_chatterbox_accent_factorization(DummyTTS)
    assert second["already_installed"] is True


def test_prompt_offset_uses_later_same_speaker_segment_for_style_only(monkeypatch, tmp_path):
    install_fake_chatterbox_modules(monkeypatch)
    patch = load_patch_module()
    patch.install_chatterbox_accent_factorization(DummyTTS)

    identity_path = tmp_path / "identity.wav"
    accent_path = tmp_path / "accent.wav"
    identity_path.write_bytes(b"identity")
    accent_path.write_bytes(b"accent")

    model = DummyTTS()
    model.prepare_conditionals(
        identity_path,
        accent_prompt_path=accent_path,
        accent_prompt_start_seconds=6.0,
    )

    # Six seconds into a 25-second ascending guide is roughly 0.24.
    assert np.isclose(float(model.calls["t3_style"][0]), 0.24, atol=0.002)
    assert np.isclose(float(model.calls["s3_style"][0]), 0.24, atol=0.002)
    assert len(model.calls["t3_style"]) == 6 * 16000
    assert len(model.calls["s3_style"]) == 10 * 24000

    # Both T3 and S3Gen speaker identity vectors still come from the narrator.
    assert np.isclose(float(model.calls["voice_identity"].mean()), 0.2)
    assert np.isclose(float(model.calls["s3_identity"].mean()), 0.2)


def test_prompt_offset_rejects_negative_and_incomplete_windows(monkeypatch, tmp_path):
    install_fake_chatterbox_modules(monkeypatch)
    patch = load_patch_module()
    patch.install_chatterbox_accent_factorization(DummyTTS)

    identity_path = tmp_path / "identity.wav"
    accent_path = tmp_path / "accent.wav"
    identity_path.write_bytes(b"identity")
    accent_path.write_bytes(b"accent")
    model = DummyTTS()

    try:
        model.prepare_conditionals(
            identity_path, accent_prompt_path=accent_path, accent_prompt_start_seconds=-1
        )
    except ValueError as exc:
        assert "cannot be negative" in str(exc)
    else:
        raise AssertionError("negative prompt offsets must be rejected")

    try:
        model.prepare_conditionals(
            identity_path, accent_prompt_path=accent_path, accent_prompt_start_seconds=20
        )
    except ValueError as exc:
        assert "full configured" in str(exc)
    else:
        raise AssertionError("a prompt offset without a full prompt window must be rejected")


def test_patched_generate_accepts_separate_accent_reference(monkeypatch, tmp_path):
    install_fake_chatterbox_modules(monkeypatch)
    patch = load_patch_module()
    patch.install_chatterbox_accent_factorization(DummyTTS)

    identity_path = tmp_path / "identity.wav"
    accent_path = tmp_path / "accent.wav"
    identity_path.write_bytes(b"identity")
    accent_path.write_bytes(b"accent")

    model = DummyTTS()
    result = model.generate(
        "A sample English sentence.",
        language_id="en",
        audio_prompt_path=str(identity_path),
        accent_prompt_path=str(accent_path),
    )

    assert result == "generated"
    assert model.conds is not None
    assert model.generate_call["audio_prompt_path"] is None
    assert model.generate_call["text"] == "A sample English sentence."
    assert model.generate_call["language_id"] == "en"
