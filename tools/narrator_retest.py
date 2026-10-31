"""Narrator-voice retest: splice bridged Sanskrit terms into real book sentences.

The first bridge audition used public demo voices and isolated Hindi carrier
sentences. This module asks the question the audiobook actually depends on:
with the real narrator recording as the target voice, does a bridged term
sound natural when it sits inside an English sentence from the book?

For each term the sentence is split at the term. The English prefix and suffix
are rendered in the narrator voice, the term is rendered through both bridge
routes (HYBRID and VC), and each route is spliced between the English parts.

Everything that needs a model is imported lazily so the task builder, sentence
splitter, and audio assembly can be tested in CI. The GPU runner has not been
run yet. Nothing here edits the manuscript or authorizes full-book synthesis.
"""
from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any, Mapping

from tools.chatterbox_voice_bridge import _atomic_json, _audio_metrics, _set_seed, _sha256

SCHEMA_VERSION = "1.0.0"
SAMPLE_RATE = 24000

# Real sentences from the approved manuscript (taken from the source-English
# pronunciation audition tasks). Devanagari forms match the bridge audition.
DEFAULT_TERMS: dict[str, dict[str, str]] = {
    "Avalokiteshvara": {
        "devanagari": "अवलोकितेश्वर",
        "sentence": (
            "Then the enlightening being Avalokiteshvara said to the Buddha, "
            "\"The Buddha speaks of ten stages of enlightening beings: the stage of intense joy, "
            "the stage of freedom from defilement, the stage of radiance, the stage of blazing intellect, "
            "the stage most difficult to conquer, the stage of presence, the stage of far going, "
            "the stage of immovability, the stage of good intellect, and the stage of cloud of teaching."
        ),
    },
    "Maitreya": {
        "devanagari": "मैत्रेय",
        "sentence": (
            "Then the enlightening being Maitreya said to the Buddha, "
            "\"Based on what, abiding in what, do enlightening beings practice tranquillity "
            "and observation in the great vehicle?"
        ),
    },
    "Manjushri": {
        "devanagari": "मञ्जुश्री",
        "sentence": (
            "Then the great enlightening beings Manjushri, the Glorious One, said to the Buddha, "
            "\"You speak of the reality body of the enlightened, who have arrived at thusness."
        ),
    },
    "Vijnanavada": {
        "devanagari": "विज्ञानवाद",
        "sentence": (
            "This is one of the main texts of that stream of Buddhist tradition known as Vijnanavada, "
            "\"The Doctrine of Consciousness,\" or Yogachara, \"The Practice of Yoga.\""
        ),
    },
}

# Splice defaults. Both are recorded in the manifest so a listening decision
# can be tied to the exact join settings.
DEFAULT_ASSEMBLY = {
    "gap_ms": 60,
    "comma_gap_ms": 180,
    "edge_fade_ms": 10,
    "trim_threshold_db": -45.0,
    "trim_keep_ms": 30,
}


def split_context_sentence(sentence: str, term: str) -> tuple[str, str, str]:
    """Split a sentence around exactly one whole-word occurrence of `term`.

    Returns (prefix, suffix, suffix_lead). `suffix_lead` is any leading
    separator punctuation removed from the suffix (for example a comma); it is
    kept so the join can use a longer pause. Wording is never altered.
    """
    pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
    matches = list(pattern.finditer(sentence))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one whole-word occurrence of {term!r}, found {len(matches)}."
        )
    match = matches[0]
    prefix = sentence[: match.start()].strip()
    remainder = sentence[match.end():]
    suffix = remainder.lstrip(" ,;:")
    suffix_lead = remainder[: len(remainder) - len(suffix)].strip()
    if not prefix or not suffix:
        raise ValueError(f"{term!r} must have English text on both sides in its sentence.")
    return prefix, suffix, suffix_lead


def build_narrator_task_specs(
    terms: Mapping[str, Mapping[str, str]],
    base_generation_parameters: Mapping[str, float],
    *,
    term_text_template: str = "{devanagari}",
    assembly: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the deterministic retest matrix: seven tasks per term (five clips, two splices)."""
    if not terms:
        raise ValueError("At least one term is required.")
    if not base_generation_parameters:
        raise ValueError("Base generation parameters must not be empty.")
    settings = {**DEFAULT_ASSEMBLY, **dict(assembly or {})}
    params = dict(base_generation_parameters)
    specs: list[dict[str, Any]] = []

    for index, (term, info) in enumerate(terms.items(), start=1):
        term = str(term).strip()
        devanagari = str(info.get("devanagari", "")).strip()
        sentence = str(info.get("sentence", "")).strip()
        if not term or not devanagari or not sentence:
            raise ValueError("Every term needs a name, a Devanagari form, and a context sentence.")
        prefix, suffix, lead = split_context_sentence(sentence, term)
        gap_ms = settings["comma_gap_ms"] if "," in lead else settings["gap_ms"]
        p = f"N{index:03d}"
        seed = 20261201 + index * 100

        def tts(suffix_id: str, mode: str, text: str, language: str, offset: int) -> dict[str, Any]:
            return {
                "task_id": f"{p}_{suffix_id}",
                "term": term,
                "kind": "tts",
                "mode": mode,
                "language_id": language,
                "text": text,
                "generation_parameters": dict(params),
                "source_task_id": "",
                "parts": [],
                "seed": seed + offset,
            }

        specs.append(tts("EN_PREFIX", "english_narrator", prefix, "en", 1))
        specs.append(tts("EN_SUFFIX", "english_narrator", suffix, "en", 2))
        specs.append(tts("HI_SOURCE", "hindi_source", term_text_template.format(devanagari=devanagari), "hi", 3))
        specs.append(tts("HYBRID", "hybrid", term_text_template.format(devanagari=devanagari), "hi", 4))
        specs.append({
            "task_id": f"{p}_VC",
            "term": term,
            "kind": "voice_conversion",
            "mode": "voice_conversion",
            "language_id": "",
            "text": "",
            "generation_parameters": {},
            "source_task_id": f"{p}_HI_SOURCE",
            "parts": [],
            "seed": seed + 5,
        })
        for route in ("HYBRID", "VC"):
            specs.append({
                "task_id": f"{p}_SPLICE_{route}",
                "term": term,
                "kind": "splice",
                "mode": f"splice_{route.lower()}",
                "language_id": "",
                "text": f"{prefix} [{term}] {suffix}",
                "generation_parameters": {},
                "source_task_id": "",
                "parts": [f"{p}_EN_PREFIX", f"{p}_{route}", f"{p}_EN_SUFFIX"],
                "join_gap_ms": gap_ms,
                "seed": seed + 6,
            })

    ids = [s["task_id"] for s in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Narrator retest task IDs are not unique.")
    return specs


def trim_silence(audio: Any, sample_rate: int, threshold_db: float, keep_ms: int) -> Any:
    """Trim leading/trailing samples below `threshold_db`, keeping `keep_ms` of margin."""
    import numpy as np

    data = np.asarray(audio, dtype=np.float32)
    if data.size == 0:
        raise ValueError("Cannot trim empty audio.")
    floor = 10 ** (threshold_db / 20.0)
    loud = np.flatnonzero(np.abs(data) >= floor)
    if loud.size == 0:
        raise ValueError("Clip is entirely below the silence threshold.")
    keep = int(sample_rate * keep_ms / 1000)
    start = max(0, int(loud[0]) - keep)
    stop = min(data.size, int(loud[-1]) + 1 + keep)
    return data[start:stop]


def assemble_splice(
    parts: list[Any],
    sample_rate: int,
    *,
    gap_ms: int,
    edge_fade_ms: int,
    trim_threshold_db: float,
    trim_keep_ms: int,
) -> Any:
    """Trim each part's edges, fade the joins, and concatenate with a short pause."""
    import numpy as np

    if len(parts) != 3:
        raise ValueError("A splice needs exactly three parts: prefix, term, suffix.")
    fade = max(1, int(sample_rate * edge_fade_ms / 1000))
    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
    pieces = []
    for number, part in enumerate(parts):
        clip = trim_silence(part, sample_rate, trim_threshold_db, trim_keep_ms).copy()
        n = min(fade, clip.size // 2)
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        if number > 0:
            clip[:n] *= ramp
        if number < len(parts) - 1:
            clip[clip.size - n:] *= ramp[::-1]
        pieces.append(clip)
        if number < len(parts) - 1:
            pieces.append(gap)
    return np.concatenate(pieces)


def run_narrator_retest(
    *,
    model: Any,
    narrator_reference_audio: str | Path,
    hindi_reference_audio: str | Path,
    output_dir: str | Path,
    model_revision: str,
    code_revision: str,
    base_generation_parameters: Mapping[str, float],
    terms: Mapping[str, Mapping[str, str]] | None = None,
    term_text_template: str = "{devanagari}",
    assembly: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run/resume the narrator retest and return a compact result summary.

    English parts use the narrator for both T3 and S3Gen. Hindi source clips use
    the Hindi reference for both. HYBRID uses Hindi T3 with narrator S3Gen. VC
    converts the Hindi source clip to the narrator voice. The model's original
    conditionals are restored before returning. Not yet run on a GPU.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from chatterbox.vc import ChatterboxVC

    from tools.narrator_reference import validate_narrator_reference

    narrator_ref = Path(narrator_reference_audio)
    hindi_ref = Path(hindi_reference_audio)
    out = Path(output_dir)
    if not narrator_ref.is_file() or not hindi_ref.is_file():
        raise FileNotFoundError("Both the narrator and Hindi reference recordings must exist.")
    check = validate_narrator_reference(narrator_ref)
    if not check["ok"]:
        raise ValueError("Narrator reference has blocking problems: " + "; ".join(check["problems"]))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the narrator retest.")
    if int(model.sr) != SAMPLE_RATE:
        raise RuntimeError(f"Expected the Chatterbox V3 24 kHz output rate, received {model.sr}.")

    terms = dict(terms or DEFAULT_TERMS)
    settings = {**DEFAULT_ASSEMBLY, **dict(assembly or {})}
    out.mkdir(parents=True, exist_ok=True)
    specs = build_narrator_task_specs(
        terms, base_generation_parameters, term_text_template=term_text_template, assembly=settings
    )
    manifest_path = out / "narrator_retest_manifest.json"
    identity = {
        "schema_version": SCHEMA_VERSION,
        "model": "ResembleAI/chatterbox, Multilingual V3",
        "model_revision": str(model_revision),
        "chatterbox_source_commit": str(code_revision),
        "narrator_reference": {"file": narrator_ref.name, "sha256": _sha256(narrator_ref)},
        "hindi_reference": {"file": hindi_ref.name, "sha256": _sha256(hindi_ref)},
        "terms": {name: dict(info) for name, info in terms.items()},
        "term_text_template": term_text_template,
        "base_generation_parameters": dict(base_generation_parameters),
        "assembly": settings,
        "full_book_synthesis_authorized": False,
    }

    old_by_id: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise RuntimeError(
                "This output folder belongs to a different narrator/model/terms/settings combination. "
                "Preserve it and choose a fresh output folder."
            )
        old_by_id = {t["task_id"]: t for t in previous.get("tasks", []) if t.get("task_id")}

    manifest: dict[str, Any] = {
        "identity": identity,
        "narrator_reference_check": check,
        "tasks": [],
        "full_book_synthesis_authorized": False,
    }
    tasks_by_id: dict[str, dict[str, Any]] = {}
    paths_by_id: dict[str, Path] = {}

    def save_manifest() -> None:
        manifest["tasks"] = list(tasks_by_id.values())
        _atomic_json(manifest_path, manifest)

    # Prepare both voices once and keep the pieces. Original state is restored at the end.
    model.prepare_conditionals(str(narrator_ref), exaggeration=0.5)
    narrator_t3, narrator_gen = model.conds.t3, model.conds.gen
    model.prepare_conditionals(str(hindi_ref), exaggeration=0.5)
    hindi_t3, hindi_gen = model.conds.t3, model.conds.gen
    voice_converter = ChatterboxVC(model.s3gen, model.device, ref_dict=narrator_gen)

    def reusable(spec: dict[str, Any], filename: str, part_hashes: list[str]) -> dict[str, Any] | None:
        prior = old_by_id.get(spec["task_id"], {})
        wav_path = out / filename
        if (
            prior.get("status") == "generated"
            and prior.get("output_file") == filename
            and wav_path.is_file()
            and prior.get("sha256")
            and _sha256(wav_path) == prior["sha256"]
            and prior.get("part_sha256", []) == part_hashes
        ):
            return prior
        return None

    def run_one(spec: dict[str, Any]) -> None:
        task_id = spec["task_id"]
        filename = f"{task_id}.wav"
        wav_path = out / filename
        part_hashes: list[str] = []
        if spec["kind"] == "splice":
            for part_id in spec["parts"]:
                part_path = paths_by_id.get(part_id)
                if part_path is None:
                    raise RuntimeError(f"Splice {task_id} needs {part_id}, which has not been generated.")
                part_hashes.append(_sha256(part_path))
        prior = reusable(spec, filename, part_hashes)
        if prior is not None:
            tasks_by_id[task_id] = {**spec, **prior}
            paths_by_id[task_id] = wav_path
            print(f"Reusing verified clip: {task_id}")
            return

        record = {**spec, "status": "running", "output_file": "", "sha256": "", "error": "", "traceback": ""}
        tasks_by_id[task_id] = record
        save_manifest()
        _set_seed(int(spec["seed"]))
        try:
            if spec["kind"] == "tts":
                if spec["mode"] == "english_narrator":
                    model.conds.t3, model.conds.gen = narrator_t3, narrator_gen
                elif spec["mode"] == "hindi_source":
                    model.conds.t3, model.conds.gen = hindi_t3, hindi_gen
                elif spec["mode"] == "hybrid":
                    model.conds.t3, model.conds.gen = hindi_t3, narrator_gen
                else:
                    raise ValueError(f"Unknown TTS mode: {spec['mode']}")
                generated = model.generate(
                    spec["text"], language_id=spec["language_id"], **spec["generation_parameters"]
                )
                waveform = generated.detach().float().cpu().numpy().squeeze()
            elif spec["kind"] == "voice_conversion":
                source_path = paths_by_id.get(spec["source_task_id"])
                if source_path is None:
                    raise RuntimeError(f"Voice-conversion source {spec['source_task_id']} is missing.")
                generated = voice_converter.generate(audio=str(source_path))
                waveform = generated.detach().float().cpu().numpy().squeeze()
            elif spec["kind"] == "splice":
                clips = []
                for part_id in spec["parts"]:
                    data, rate = sf.read(str(paths_by_id[part_id]), dtype="float32")
                    if rate != SAMPLE_RATE:
                        raise RuntimeError(f"{part_id} has sample rate {rate}, expected {SAMPLE_RATE}.")
                    clips.append(np.asarray(data, dtype=np.float32).squeeze())
                waveform = assemble_splice(
                    clips,
                    SAMPLE_RATE,
                    gap_ms=int(spec["join_gap_ms"]),
                    edge_fade_ms=int(settings["edge_fade_ms"]),
                    trim_threshold_db=float(settings["trim_threshold_db"]),
                    trim_keep_ms=int(settings["trim_keep_ms"]),
                )
                record["part_sha256"] = part_hashes
            else:
                raise ValueError(f"Unknown task kind: {spec['kind']}")

            metrics = _audio_metrics(waveform)
            temp_path = out / f"{task_id}.tmp.wav"
            sf.write(str(temp_path), waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            temp_path.replace(wav_path)
            record.update(metrics)
            record["sample_rate"] = SAMPLE_RATE
            record["output_file"] = filename
            record["sha256"] = _sha256(wav_path)
            record["status"] = "generated"
            paths_by_id[task_id] = wav_path
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
        model.conds.t3, model.conds.gen = hindi_t3, hindi_gen

    save_manifest()
    items = list(tasks_by_id.values())
    return {
        "manifest_path": str(manifest_path),
        "output_dir": str(out),
        "task_count": len(specs),
        "generated_count": sum(1 for t in items if t.get("status") == "generated"),
        "failed_count": sum(1 for t in items if t.get("status") == "generation_failed"),
        "tasks": items,
        "narrator_reference_check": check,
        "full_book_synthesis_authorized": False,
    }
