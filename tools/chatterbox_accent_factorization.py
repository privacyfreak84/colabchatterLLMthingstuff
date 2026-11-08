"""Accent/identity factorization patch for the pinned Chatterbox Multilingual source.

Chatterbox's stock zero-shot path derives both speaker-identity embeddings and the
speech/mel prompt from one recording. This overlay adds an optional second recording:
identity_reference_path supplies the target speaker embeddings; accent_prompt_path
supplies T3 speech-prompt tokens and S3Gen prompt tokens/mels.

No model weights are changed or retrained. This is a source-level conditioning
factorization, not a guarantee that the learned model will obey a dialect perfectly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


PATCH_VERSION = "1.1.0"
PATCH_FEATURE = "separate-speaker-identity-and-accent-style-conditioning"


def _load_mono_24k(librosa: Any, path: str | Path, *, label: str):
    audio, _ = librosa.load(str(path), sr=24000, mono=True)
    if audio is None or getattr(audio, "ndim", 0) != 1 or len(audio) < 12_000:
        raise ValueError(f"{label} must contain at least 0.5 seconds of mono speech.")
    try:
        import numpy as np
        if not np.isfinite(audio).all():
            raise ValueError(f"{label} contains non-finite audio samples.")
        if float(np.max(np.abs(audio))) < 1e-5:
            raise ValueError(f"{label} is silent or effectively silent.")
    except ImportError:
        pass
    return audio


def install_chatterbox_accent_factorization(model_class=None) -> dict:
    """Patch ChatterboxMultilingualTTS in-place; safe to call more than once."""
    if model_class is None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as model_class

    if getattr(model_class, "_accent_factorization_patch_version", None) == PATCH_VERSION:
        return {
            "installed": True,
            "version": PATCH_VERSION,
            "feature": PATCH_FEATURE,
            "already_installed": True,
        }

    import librosa
    import torch
    from chatterbox.mtl_tts import Conditionals
    from chatterbox.models.s3tokenizer import S3_SR
    from chatterbox.models.s3gen import S3GEN_SR
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    original_generate = model_class.generate

    def prepare_conditionals(
        self,
        wav_fpath,
        exaggeration=0.5,
        accent_prompt_path=None,
        accent_prompt_start_seconds: float = 0.0,
    ):
        """Prepare identity from wav_fpath and acoustic style from accent_prompt_path.

        When accent_prompt_path is omitted, it falls back to the narrator file and
        reproduces the upstream single-reference conditioning layout.
        """
        identity_path = Path(wav_fpath)
        style_path = Path(accent_prompt_path) if accent_prompt_path else identity_path
        prompt_offset_seconds = float(accent_prompt_start_seconds)
        if prompt_offset_seconds < 0:
            raise ValueError("Accent/style prompt start time cannot be negative.")
        if prompt_offset_seconds and accent_prompt_path is None:
            raise ValueError(
                "A non-zero accent/style prompt offset requires a separate accent/style guide file. "
                "For the later part of the narrator recording, upload that same original recording as the guide."
            )
        if not identity_path.is_file():
            raise ValueError(f"Narrator identity reference does not exist: {identity_path}")
        if not style_path.is_file():
            raise ValueError(f"Accent/style reference does not exist: {style_path}")

        identity_24k = _load_mono_24k(
            librosa, identity_path, label="Narrator identity reference"
        )
        style_24k = (
            identity_24k
            if style_path.resolve() == identity_path.resolve()
            else _load_mono_24k(librosa, style_path, label="Accent/style reference")
        )

        identity_16k = librosa.resample(
            identity_24k, orig_sr=S3GEN_SR, target_sr=S3_SR
        )
        style_16k = (
            identity_16k
            if style_path.resolve() == identity_path.resolve()
            else librosa.resample(style_24k, orig_sr=S3GEN_SR, target_sr=S3_SR)
        )
        style_start_24k = int(round(prompt_offset_seconds * S3GEN_SR))
        style_start_16k = int(round(prompt_offset_seconds * S3_SR))
        style_prompt_24k = style_24k[
            style_start_24k : style_start_24k + int(self.DEC_COND_LEN)
        ]
        style_prompt_16k = style_16k[
            style_start_16k : style_start_16k + int(self.ENC_COND_LEN)
        ]
        if len(style_prompt_24k) < 12_000 or len(style_prompt_16k) < 8_000:
            raise ValueError(
                "Accent/style guide is too short after the selected start time. "
                "Choose an earlier offset or provide a longer recording."
            )

        # T3 speaker vector is derived only from the narrator identity recording.
        ve_embed = torch.from_numpy(
            self.ve.embeds_from_wavs([identity_16k], sample_rate=S3_SR)
        )
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        # T3's acoustic speech prompt comes from the accent/style guide.
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [style_prompt_16k],
                max_len=plen,
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=float(exaggeration) * torch.ones(1, 1, 1),
        ).to(device=self.device)

        # S3Gen's acoustic prompt tokens/features come from the style guide.
        s3gen_ref_dict = self.s3gen.embed_ref(
            style_prompt_24k,
            S3GEN_SR,
            device=self.device,
        )

        # But S3Gen's speaker x-vector comes only from the narrator identity clip.
        # This prevents the accent guide's speaker identity from replacing the narrator.
        identity_s3_samples = int(self.DEC_COND_LEN * (S3_SR / S3GEN_SR))
        identity_for_s3 = torch.as_tensor(
            identity_16k[:identity_s3_samples],
            dtype=self.s3gen.dtype,
            device=self.device,
        ).unsqueeze(0)
        s3gen_ref_dict["embedding"] = self.s3gen.speaker_encoder.inference(identity_for_s3)

        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

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
        accent_prompt_path=None,
        accent_prompt_start_seconds: float = 0.0,
    ):
        """Upstream generate API plus optional separate accent/style reference."""
        if accent_prompt_path and not audio_prompt_path:
            raise ValueError(
                "accent_prompt_path requires audio_prompt_path: the latter supplies "
                "the target speaker's identity."
            )
        if audio_prompt_path:
            self.prepare_conditionals(
                audio_prompt_path,
                exaggeration=exaggeration,
                accent_prompt_path=accent_prompt_path,
                accent_prompt_start_seconds=accent_prompt_start_seconds,
            )
            # Conditionals are prepared above; let the original generator use them
            # without preparing again from a single reference.
            audio_prompt_path = None

        return original_generate(
            self,
            text,
            language_id,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )

    model_class._accent_factorization_original_prepare_conditionals = (
        getattr(model_class, "prepare_conditionals", None)
    )
    model_class._accent_factorization_original_generate = original_generate
    model_class.prepare_conditionals = prepare_conditionals
    model_class.generate = generate
    model_class._accent_factorization_patch_version = PATCH_VERSION

    return {
        "installed": True,
        "version": PATCH_VERSION,
        "feature": PATCH_FEATURE,
        "already_installed": False,
    }
