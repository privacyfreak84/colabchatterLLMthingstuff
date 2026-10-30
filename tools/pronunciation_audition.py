#!/usr/bin/env python3
"""Checkpointed pronunciation auditions for the pinned Chatterbox V3 runtime.

No monkey-patching, implicit retries, manuscript edits, or pronunciation
approvals occur here. One failed sample is recorded with a traceback and stops
the run; successful samples are checkpointed and reused on resume.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import traceback
from pathlib import Path
from typing import Any, Iterable
import inspect

IAST_FORMS = {
    "Avalokiteshvara": "Avalokiteśvara",
    "Avatamsaka": "Avataṃsaka",
    "Buddha": "Buddha",
    "Gautama": "Gautama",
    "Maitreya": "Maitreya",
    "Manjushri": "Mañjuśrī",
    "nirvana": "nirvāṇa",
    "samsara": "saṃsāra",
    "Sandhinirmochana-sutra": "Saṃdhinirmocana-sūtra",
    "Subhuti": "Subhūti",
    "sutra": "sūtra",
    "Vijnanavada": "Vijñānavāda",
    "Yogachara": "Yogācāra",
}

# Provisional Devanagari spellings for Hindi-language auditions only.
# They must be judged by listening; no spelling is approved automatically.
DEVANAGARI_FORMS = {
    "Avalokiteshvara": "अवलोकितेश्वर",
    "Avatamsaka": "अवतंसक",
    "Buddha": "बुद्ध",
    "Gautama": "गौतम",
    "Maitreya": "मैत्रेय",
    "Manjushri": "मञ्जुश्री",
    "nirvana": "निर्वाण",
    "samsara": "संसार",
    "Sandhinirmochana-sutra": "संधिनिर्मोचन-सूत्र",
    "Subhuti": "सुभूति",
    "sutra": "सूत्र",
    "Vijnanavada": "विज्ञानवाद",
    "Yogachara": "योगाचार",
}

DEFAULT_GENERATION_PARAMETERS = {
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "min_p": 0.05,
    "top_p": 0.95,
}
GENERATION_PARAMETER_NAMES = set(DEFAULT_GENERATION_PARAMETERS)

DECISION_FIELDS = [
    "audition_id", "source_term", "occurrences", "pages",
    "source_english_audio", "iast_english_audio", "devanagari_hindi_audio",
    "chosen_variant", "approved_spoken_form", "review_status", "review_notes",
]
TASK_FIELDS = [
    "task_id", "audition_id", "source_term", "occurrences", "pages",
    "variant", "language_id", "candidate_form", "text", "reference_language",
    "reference_audio", "status", "unknown_token_count", "output_file",
    "duration_seconds", "seed", "sha256", "error", "traceback_file",
]


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


def _replace_once(text: str, term: str, replacement: str) -> str:
    updated, count = _term_pattern(term).subn(lambda _: replacement, text, count=1)
    if count != 1:
        raise ValueError(
            f"Expected exactly one occurrence of {term!r} in its audition context; found {count}."
        )
    return updated


def build_audition_tasks(sheet_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create original-English, IAST-English, and Devanagari-Hindi tasks."""
    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in sheet_rows:
        audition_id = str(row.get("audition_id", "")).strip()
        term = str(row.get("source_term", "")).strip()
        context = str(row.get("context_prompt", "")).strip()
        if not audition_id or not term or not context:
            raise ValueError("Each audition row needs audition_id, source_term, and context_prompt.")
        if audition_id in seen_ids:
            raise ValueError(f"Duplicate audition_id: {audition_id}")
        seen_ids.add(audition_id)
        if term not in IAST_FORMS or term not in DEVANAGARI_FORMS:
            raise ValueError(f"No IAST/Devanagari audition forms configured for {term!r}.")
        if not _term_pattern(term).search(context):
            raise ValueError(f"The source term {term!r} is absent from its supplied context.")

        common = {
            "audition_id": audition_id,
            "source_term": term,
            "occurrences": str(row.get("occurrences", "")),
            "pages": str(row.get("pages", "")),
        }
        tasks.append({
            **common,
            "task_id": f"{audition_id}_EN_SOURCE",
            "variant": "source_english",
            "language_id": "en",
            "candidate_form": term,
            "text": context,
            "reference_language": "en",
            "reference_audio": "official_demo_en_f1_padded.wav",
            "status": "pending_preflight",
        })

        iast = IAST_FORMS[term]
        if iast.casefold() != term.casefold():
            tasks.append({
                **common,
                "task_id": f"{audition_id}_EN_IAST",
                "variant": "iast_english",
                "language_id": "en",
                "candidate_form": iast,
                "text": _replace_once(context, term, iast),
                "reference_language": "en",
                "reference_audio": "official_demo_en_f1.wav",
                "status": "pending_preflight",
            })

        devanagari = DEVANAGARI_FORMS[term]
        tasks.append({
            **common,
            "task_id": f"{audition_id}_HI_DEVANAGARI",
            "variant": "devanagari_hindi",
            "language_id": "hi",
            "candidate_form": devanagari,
            "text": f"इस शब्द का उच्चारण {devanagari} है।",
            "reference_language": "hi",
            "reference_audio": "official_demo_hi_f1_padded.wav",
            "status": "pending_preflight",
        })
    return tasks


def preflight_tasks(model: Any, tasks: list[dict[str, Any]], punc_norm: Any) -> list[dict[str, Any]]:
    """Reject tokenizer inputs containing [UNK] before they reach TTS."""
    vocab = model.tokenizer.tokenizer.get_vocab()
    unk_id = vocab.get("[UNK]")
    if unk_id is None:
        raise RuntimeError("The loaded tokenizer has no [UNK] token; cannot audit coverage safely.")

    checked: list[dict[str, Any]] = []
    for original in tasks:
        task = dict(original)
        encoded = model.tokenizer.text_to_tokens(
            punc_norm(task["text"]), language_id=task["language_id"]
        )
        if hasattr(encoded, "eq"):
            unknowns = int(encoded.eq(unk_id).sum().item())
        else:
            flat = encoded.tolist() if hasattr(encoded, "tolist") else encoded
            while isinstance(flat, list) and flat and isinstance(flat[0], list):
                flat = flat[0]
            unknowns = sum(1 for item in flat if int(item) == int(unk_id))
        task["unknown_token_count"] = unknowns
        task["status"] = "ready" if unknowns == 0 else "skipped_unknown_tokens"
        task["error"] = "" if unknowns == 0 else f"Tokenizer produced {unknowns} [UNK] token(s)."
        task.setdefault("traceback_file", "")
        checked.append(task)
    return checked


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _decision_rows(sheet_rows: list[dict[str, str]], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sheet_rows:
        tid = row["audition_id"]
        task_paths = {
            task["variant"]: task.get("output_file", "")
            for task in tasks
            if task["audition_id"] == tid and task.get("status") == "generated"
        }
        rows.append({
            "audition_id": tid,
            "source_term": row["source_term"],
            "occurrences": row.get("occurrences", ""),
            "pages": row.get("pages", ""),
            "source_english_audio": task_paths.get("source_english", ""),
            "iast_english_audio": task_paths.get("iast_english", ""),
            "devanagari_hindi_audio": task_paths.get("devanagari_hindi", ""),
            "chosen_variant": "",
            "approved_spoken_form": "",
            "review_status": "awaiting_listening",
            "review_notes": "Listen first. Record a spoken form only after review; source spelling is unchanged.",
        })
    return rows


def generate_audition_set(
    *,
    model: Any,
    audition_sheet: str | Path,
    output_dir: str | Path,
    english_reference_audio: str | Path,
    hindi_reference_audio: str | Path,
    punc_norm: Any,
    model_revision: str,
    code_revision: str,
    seed_base: int = 20261030,
    generation_parameters: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Generate/resume a checkpointed audition set; stop on the first real error."""
    import numpy as np
    import soundfile as sf

    sheet_path, out = Path(audition_sheet), Path(output_dir)
    en_ref, hi_ref = Path(english_reference_audio), Path(hindi_reference_audio)
    if not sheet_path.is_file():
        raise FileNotFoundError(f"Missing pronunciation audition sheet: {sheet_path}")
    if not en_ref.is_file() or not hi_ref.is_file():
        raise FileNotFoundError("Both prepared English and Hindi reference WAV files are required.")
    out.mkdir(parents=True, exist_ok=True)

    parameters = dict(DEFAULT_GENERATION_PARAMETERS)
    if generation_parameters:
        unknown = set(generation_parameters) - GENERATION_PARAMETER_NAMES
        if unknown:
            raise ValueError(f"Unsupported Chatterbox generation parameters: {sorted(unknown)}")
        parameters.update(generation_parameters)

    if "generate" in getattr(model, "__dict__", {}):
        raise RuntimeError(
            "model.generate has been overridden on this model instance. "
            "Restart the Colab runtime and load a fresh model; do not stack monkey-patches."
        )
    signature = inspect.signature(model.generate)
    unsupported = set(parameters) - set(signature.parameters)
    if unsupported:
        raise RuntimeError(
            "The installed Chatterbox generate() API does not accept "
            f"{sorted(unsupported)}. Check that the pinned upstream source was installed."
        )

    with sheet_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sheet_rows = list(csv.DictReader(handle))
    if len(sheet_rows) != 13:
        raise ValueError(f"Expected the approved 13-term pronunciation sheet; found {len(sheet_rows)} rows.")

    sheet_hash = _sha256(sheet_path)
    tasks = preflight_tasks(model, build_audition_tasks(sheet_rows), punc_norm)
    manifest_path = out / "pronunciation_audition_manifest.json"
    task_csv = out / "pronunciation_audition_tasks.csv"
    summary_path = out / "pronunciation_audition_summary.json"
    decisions_path = out / "pronunciation_audition_decisions.csv"

    # Reuse only successful, hash-verified samples from this exact pronunciation sheet.
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("pronunciation_sheet_sha256") != sheet_hash:
            raise ValueError(
                "This audition directory belongs to a different pronunciation sheet. "
                "Choose a fresh output directory rather than overwriting old audition evidence."
            )
        if (
            previous.get("schema_version") != "2.0.0"
            or previous.get("model_revision") != model_revision
            or previous.get("chatterbox_source_commit") != code_revision
            or previous.get("generation_parameters") != parameters
        ):
            raise ValueError(
                "This audition directory was created with a different Chatterbox/model/settings version. "
                "Choose a fresh output directory; old audio will not be reused across model revisions."
            )
        prior_by_id = {item.get("task_id"): item for item in previous.get("tasks", [])}
        for task in tasks:
            prior = prior_by_id.get(task["task_id"], {})
            filename = prior.get("output_file", "")
            audio_path = out / filename if filename else None
            if (
                prior.get("status") == "generated"
                and audio_path is not None
                and audio_path.is_file()
                and prior.get("sha256")
                and _sha256(audio_path) == prior["sha256"]
            ):
                task.update(prior)
            elif task["status"] == "ready":
                task["status"] = "ready"
                task["output_file"] = ""
                task["duration_seconds"] = ""
                task["seed"] = ""
                task["sha256"] = ""
                task["error"] = ""
                task["traceback_file"] = ""

    state = {"status": "running"}

    def checkpoint() -> None:
        manifest = {
            "schema_version": "2.0.0",
            "model": "ResembleAI/chatterbox, Multilingual V3",
            "model_revision": model_revision,
            "chatterbox_source_commit": code_revision,
            "pronunciation_sheet_sha256": sheet_hash,
            "voice_references": {
                "en": {"file": en_ref.name, "sha256": _sha256(en_ref), "source": "Resemble AI public demo sample"},
                "hi": {"file": hi_ref.name, "sha256": _sha256(hi_ref), "source": "Resemble AI public demo sample"},
            },
            "generation_parameters": parameters,
            "seed_base": seed_base,
            "status": state["status"],
            "tasks": tasks,
            "full_book_synthesis_authorized": False,
        }
        _atomic_json(manifest_path, manifest)
        _atomic_csv(task_csv, TASK_FIELDS, tasks)
        _atomic_csv(decisions_path, DECISION_FIELDS, _decision_rows(sheet_rows, tasks))
        summary = {
            "status": state["status"],
            "task_count": len(tasks),
            "generated_count": sum(x["status"] == "generated" for x in tasks),
            "skipped_unknown_tokens": sum(x["status"] == "skipped_unknown_tokens" for x in tasks),
            "generation_failures": sum(x["status"] in {"generation_failed", "interrupted"} for x in tasks),
            "pending_count": sum(x["status"] in {"ready", "pending_preflight", "running"} for x in tasks),
            "decision_rows": len(sheet_rows),
            "output_dir": str(out),
            "full_book_synthesis_authorized": False,
        }
        _atomic_json(summary_path, summary)

    checkpoint()
    references = {"en": en_ref, "hi": hi_ref}
    for language_id in ("en", "hi"):
        group = [
            task for task in tasks
            if task["language_id"] == language_id
            and task["status"] == "ready"
        ]
        if not group:
            continue
        try:
            model.prepare_conditionals(str(references[language_id]), exaggeration=parameters["exaggeration"])
        except BaseException as exc:
            state["status"] = "failed"
            error_file = out / f"reference_{language_id}_error.txt"
            error_file.write_text(traceback.format_exc(), encoding="utf-8")
            for task in group:
                task["status"] = "generation_failed"
                task["error"] = f"Reference conditioning failed: {type(exc).__name__}: {exc}"
                task["traceback_file"] = error_file.name
            checkpoint()
            raise RuntimeError(
                f"Failed to prepare the {language_id!r} reference audio. "
                f"Full traceback: {error_file}"
            ) from exc

        for task in group:
            language_offset = 50 if language_id == "hi" else 0
            seed = seed_base + int(task["audition_id"][1:]) * 100 + language_offset
            task["seed"] = seed
            task["status"] = "running"
            task["error"] = ""
            checkpoint()
            try:
                _set_seed(seed)
                wav = model.generate(
                    task["text"],
                    language_id=language_id,
                    **parameters,
                )
                waveform = wav.detach().float().cpu().numpy().squeeze()
                if waveform.ndim != 1 or waveform.size == 0:
                    raise ValueError(f"Unexpected audio shape: {tuple(wav.shape)}")
                if not np.isfinite(waveform).all():
                    raise ValueError("Generated waveform contains NaN or infinite samples.")

                filename = f"{task['task_id']}.wav"
                audio_path = out / filename
                temporary_audio = out / f"{task['task_id']}.tmp.wav"
                sf.write(
                    str(temporary_audio),
                    waveform,
                    int(model.sr),
                    format="WAV",
                    subtype="PCM_16",
                )
                temporary_audio.replace(audio_path)
                task["output_file"] = filename
                task["duration_seconds"] = round(float(waveform.size / int(model.sr)), 3)
                task["sha256"] = _sha256(audio_path)
                task["status"] = "generated"
                task["error"] = ""
                task["traceback_file"] = ""
                checkpoint()
            except BaseException as exc:
                task["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "generation_failed"
                task["error"] = f"{type(exc).__name__}: {exc}"
                traceback_path = out / f"{task['task_id']}_traceback.txt"
                traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
                task["traceback_file"] = traceback_path.name
                state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                checkpoint()
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise RuntimeError(
                    f"Audition {task['task_id']} failed. Earlier successful WAVs are checkpointed. "
                    f"Inspect {task['traceback_file']} before retrying."
                ) from exc

    state["status"] = "complete"
    checkpoint()
    return json.loads(summary_path.read_text(encoding="utf-8"))
