"""Controlled Hindi-to-English bridge auditions for Chatterbox Multilingual V3.

This module tests two routes without editing the manuscript:
1. Hindi T3 + English S3Gen reference conditioning.
2. Hindi TTS source audio converted with ChatterboxVC to an English target voice.

All model-heavy imports are lazy so the pure task builder can be tested in CPU-only CI.
"""
from __future__ import annotations

import hashlib
import json
import random
import traceback
from pathlib import Path
from typing import Any, Mapping


DEFAULT_PACE_SETTINGS = {
    "exaggeration": 0.7,
    "cfg_weight": 0.3,
}


def build_bridge_task_specs(
    devanagari_forms: Mapping[str, str],
    base_generation_parameters: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Build a deterministic, four-term, 24-task comparison matrix."""
    if len(devanagari_forms) != 4:
        raise ValueError("The bridge audition requires exactly four representative terms.")
    if not base_generation_parameters:
        raise ValueError("Base generation parameters must not be empty.")

    base = dict(base_generation_parameters)
    pace = {**base, **DEFAULT_PACE_SETTINGS}
    specs: list[dict[str, Any]] = []

    for index, (term, devanagari) in enumerate(devanagari_forms.items(), start=1):
        term = str(term).strip()
        devanagari = str(devanagari).strip()
        if not term or not devanagari:
            raise ValueError("Every bridge term needs a non-empty source term and Devanagari form.")

        prefix = f"B{index:03d}"
        prompt = f"इस शब्द का उच्चारण {devanagari} है।"
        for mode, suffix, variant, params in (
            ("hindi_source", "HI_SOURCE_DEFAULT", "hindi_source_default", base),
            ("hindi_source", "HI_SOURCE_PACE", "hindi_source_pace", pace),
            ("hybrid", "HYBRID_DEFAULT", "hybrid_hindi_t3_english_s3gen_default", base),
            ("hybrid", "HYBRID_PACE", "hybrid_hindi_t3_english_s3gen_pace", pace),
        ):
            specs.append({
                "task_id": f"{prefix}_{suffix}",
                "term": term,
                "kind": "tts",
                "mode": mode,
                "variant": variant,
                "language_id": "hi",
                "text": prompt,
                "generation_parameters": dict(params),
                "source_task_id": "",
                "seed": 20261101 + index * 100 + len(specs) % 100,
            })

        specs.extend([
            {
                "task_id": f"{prefix}_VC_DEFAULT",
                "term": term,
                "kind": "voice_conversion",
                "mode": "voice_conversion",
                "variant": "voice_converted_default",
                "language_id": "",
                "text": "",
                "generation_parameters": {},
                "source_task_id": f"{prefix}_HI_SOURCE_DEFAULT",
                "seed": 20261101 + index * 100 + 5,
            },
            {
                "task_id": f"{prefix}_VC_PACE",
                "term": term,
                "kind": "voice_conversion",
                "mode": "voice_conversion",
                "variant": "voice_converted_pace",
                "language_id": "",
                "text": "",
                "generation_parameters": {},
                "source_task_id": f"{prefix}_HI_SOURCE_PACE",
                "seed": 20261101 + index * 100 + 6,
            },
        ])

    ids = [spec["task_id"] for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Bridge task IDs are not unique.")
    return specs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _audio_metrics(waveform: Any) -> dict[str, float]:
    import numpy as np

    audio = np.asarray(waveform, dtype=np.float32).squeeze()
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"Expected a non-empty mono waveform, received shape {audio.shape}.")
    if not np.isfinite(audio).all():
        raise ValueError("Generated audio contains NaN or infinite samples.")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    return {
        "duration_seconds": round(audio.size / 24000, 3),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "clipped_fraction": round(clipped, 8),
    }


def run_hindi_english_bridge_auditions(
    *,
    model: Any,
    english_reference_audio: str | Path,
    hindi_reference_audio: str | Path,
    output_dir: str | Path,
    devanagari_forms: Mapping[str, str],
    model_revision: str,
    code_revision: str,
    base_generation_parameters: Mapping[str, float],
) -> dict[str, Any]:
    """Run/resume all 24 bridge tasks and return a compact result summary.

    The English reference is used only by S3Gen/ChatterboxVC. Hindi T3
    conditionals are always prepared from the Hindi reference. The original
    model conditionals are restored before returning.
    """
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from chatterbox.vc import ChatterboxVC

    en_ref = Path(english_reference_audio)
    hi_ref = Path(hindi_reference_audio)
    out = Path(output_dir)
    if not en_ref.is_file() or not hi_ref.is_file():
        raise FileNotFoundError("Both English and Hindi reference WAVs must exist.")
    # The model carries its own device (CUDA or CPU); CPU runs are valid but slow.
    if int(model.sr) != 24000:
        raise RuntimeError(f"Expected the Chatterbox V3 24 kHz output rate, received {model.sr}.")

    out.mkdir(parents=True, exist_ok=True)
    specs = build_bridge_task_specs(devanagari_forms, base_generation_parameters)
    manifest_path = out / "hindi_english_bridge_manifest.json"
    identity = {
        "schema_version": "1.0.0",
        "model": "ResembleAI/chatterbox, Multilingual V3",
        "model_revision": str(model_revision),
        "chatterbox_source_commit": str(code_revision),
        "english_reference": {"file": en_ref.name, "sha256": _sha256(en_ref)},
        "hindi_reference": {"file": hi_ref.name, "sha256": _sha256(hi_ref)},
        "devanagari_forms": dict(devanagari_forms),
        "base_generation_parameters": dict(base_generation_parameters),
        "pace_generation_parameters": {
            **dict(base_generation_parameters),
            **DEFAULT_PACE_SETTINGS,
        },
        "full_book_synthesis_authorized": False,
    }

    old_by_id: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise RuntimeError(
                "This bridge output folder belongs to different model/reference/settings inputs. "
                "Preserve it and choose a fresh output folder."
            )
        old_by_id = {
            item["task_id"]: item
            for item in previous.get("tasks", [])
            if item.get("task_id")
        }

    manifest: dict[str, Any] = {
        "identity": identity,
        "tasks": [],
        "full_book_synthesis_authorized": False,
    }
    tasks_by_id: dict[str, dict[str, Any]] = {}
    outputs_by_id: dict[str, Path] = {}

    def save_manifest() -> None:
        manifest["tasks"] = list(tasks_by_id.values())
        _atomic_json(manifest_path, manifest)

    # Prepare Hindi T3 and Hindi S3Gen conditionals together, as in ordinary
    # same-language Hindi TTS. Preserve the T3 object, then independently
    # derive the English S3Gen reference dictionary.
    model.prepare_conditionals(str(hi_ref), exaggeration=0.5)
    hindi_t3_conditionals = model.conds.t3
    hindi_s3gen_reference = model.conds.gen
    english_wave, _ = librosa.load(str(en_ref), sr=int(model.sr), mono=True)
    english_wave = english_wave[: 10 * int(model.sr)]
    if english_wave.size < int(model.sr):
        raise ValueError("English target reference must contain at least one second of audio.")
    english_s3gen_reference = model.s3gen.embed_ref(
        english_wave,
        int(model.sr),
        device=model.device,
    )
    voice_converter = ChatterboxVC(
        model.s3gen,
        model.device,
        ref_dict=english_s3gen_reference,
    )

    def run_one(spec: dict[str, Any]) -> None:
        task_id = spec["task_id"]
        prior = old_by_id.get(task_id, {})
        filename = f"{task_id}.wav"
        wav_path = out / filename
        if (
            prior.get("status") == "generated"
            and prior.get("output_file") == filename
            and wav_path.is_file()
            and prior.get("sha256")
            and _sha256(wav_path) == prior["sha256"]
        ):
            record = {**spec, **prior}
            tasks_by_id[task_id] = record
            outputs_by_id[task_id] = wav_path
            print(f"Reusing verified bridge clip: {task_id}")
            return

        record = {
            **spec,
            "status": "running",
            "output_file": "",
            "sha256": "",
            "error": "",
            "traceback": "",
        }
        tasks_by_id[task_id] = record
        save_manifest()
        _set_seed(int(spec["seed"]))

        try:
            if spec["kind"] == "tts":
                model.conds.t3 = hindi_t3_conditionals
                model.conds.gen = (
                    hindi_s3gen_reference
                    if spec["mode"] == "hindi_source"
                    else english_s3gen_reference
                )
                generated = model.generate(
                    spec["text"],
                    language_id=spec["language_id"],
                    **spec["generation_parameters"],
                )
            elif spec["kind"] == "voice_conversion":
                source_id = spec["source_task_id"]
                source_path = outputs_by_id.get(source_id)
                if source_path is None or not source_path.is_file():
                    raise RuntimeError(
                        f"Voice-conversion source {source_id} is missing; "
                        "the corresponding Hindi source task must succeed first."
                    )
                generated = voice_converter.generate(audio=str(source_path))
            else:
                raise ValueError(f"Unknown bridge task kind: {spec['kind']}")

            waveform = generated.detach().float().cpu().numpy().squeeze()
            metrics = _audio_metrics(waveform)
            temp_path = out / f"{task_id}.tmp.wav"
            sample_rate = int(model.sr)
            sf.write(str(temp_path), waveform, sample_rate, format="WAV", subtype="PCM_16")
            temp_path.replace(wav_path)

            record.update(metrics)
            record["sample_rate"] = sample_rate
            record["output_file"] = filename
            record["sha256"] = _sha256(wav_path)
            record["status"] = "generated"
            outputs_by_id[task_id] = wav_path
            save_manifest()
            print(f"{task_id}: {metrics['duration_seconds']} s")
        except BaseException as exc:
            record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "generation_failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
            save_manifest()
            raise

    try:
        for spec in specs:
            run_one(spec)
    finally:
        # Leave the in-memory model in its original same-language Hindi state.
        model.conds.t3 = hindi_t3_conditionals
        model.conds.gen = hindi_s3gen_reference

    generated_count = sum(1 for item in tasks_by_id.values() if item.get("status") == "generated")
    failed_count = sum(1 for item in tasks_by_id.values() if item.get("status") == "generation_failed")
    save_manifest()
    return {
        "manifest_path": str(manifest_path),
        "output_dir": str(out),
        "task_count": len(specs),
        "generated_count": generated_count,
        "failed_count": failed_count,
        "tasks": list(tasks_by_id.values()),
        "full_book_synthesis_authorized": False,
    }
