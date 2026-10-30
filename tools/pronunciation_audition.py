#!/usr/bin/env python3
"""Prepare and generate reviewable Chatterbox Multilingual V3 pronunciation auditions.

This module never edits the narration manuscripts or the approved pronunciation
sheet. It creates temporary listening variants and an explicit decision template.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import random
from pathlib import Path
from typing import Any, Iterable

# Exact Sanskrit transliterations (IAST) used only as audition candidates.
# They are not approval decisions and are never written back into the book.
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

# Standard Devanagari forms for Hindi-language auditions.
# These are candidate spellings for listening tests, not forced pronunciations.
DEVANAGARI_FORMS = {
    "Avalokiteshvara": "अवलोकितेश्वर",
    "Avatamsaka": "अवतंसक",
    "Buddha": "बुद्ध",
    "Gautama": "गौतम",
    "Maitreya": "मैत्रेय",
    "Manjushri": "मञ्जुश्री",
    "nirvana": "निर्वाण",
    "samsara": "संसार",
    "Sandhinirmochana-sutra": "सन्धिनिर्मोचन-सूत्र",
    "Subhuti": "सुभूति",
    "sutra": "सूत्र",
    "Vijnanavada": "विज्ञानवाद",
    "Yogachara": "योगाचार",
}

DECISION_FIELDS = [
    "audition_id", "source_term", "occurrences", "pages",
    "source_english_audio", "iast_english_audio", "devanagari_hindi_audio",
    "chosen_variant", "approved_spoken_form", "review_status", "review_notes",
]
TASK_FIELDS = [
    "task_id", "audition_id", "source_term", "occurrences", "pages",
    "variant", "language_id", "candidate_form", "text", "reference_language",
    "reference_audio", "status", "unknown_token_count", "output_file",
    "duration_seconds", "seed", "sha256", "error",
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
    """Create English-source, English-IAST, and Hindi-Devanagari test tasks."""
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

        iast = IAST_FORMS[term]
        devanagari = DEVANAGARI_FORMS[term]
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
            "reference_audio": "official_demo_en_f1.flac",
            "status": "pending_preflight",
        })
        if iast.casefold() != term.casefold():
            tasks.append({
                **common,
                "task_id": f"{audition_id}_EN_IAST",
                "variant": "iast_english",
                "language_id": "en",
                "candidate_form": iast,
                "text": _replace_once(context, term, iast),
                "reference_language": "en",
                "reference_audio": "official_demo_en_f1.flac",
                "status": "pending_preflight",
            })
        tasks.append({
            **common,
            "task_id": f"{audition_id}_HI_DEVANAGARI",
            "variant": "devanagari_hindi",
            "language_id": "hi",
            "candidate_form": devanagari,
            "text": f"संस्कृत में इस शब्द का उच्चारण है {devanagari}।",
            "reference_language": "hi",
            "reference_audio": "official_demo_hi_f1.flac",
            "status": "pending_preflight",
        })
    return tasks


def preflight_tasks(model: Any, tasks: list[dict[str, Any]], punc_norm: Any) -> list[dict[str, Any]]:
    """Count tokenizer UNK tokens before spending time on generation."""
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
            flat = encoded
            if hasattr(flat, "tolist"):
                flat = flat.tolist()
            while isinstance(flat, list) and flat and isinstance(flat[0], list):
                flat = flat[0]
            unknowns = sum(1 for item in flat if int(item) == int(unk_id))
        task["unknown_token_count"] = unknowns
        task["status"] = "ready" if unknowns == 0 else "skipped_unknown_tokens"
        task["error"] = "" if unknowns == 0 else f"Tokenizer produced {unknowns} [UNK] token(s)."
        checked.append(task)
    return checked


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_audition_set(
    *,
    model: Any,
    audition_sheet: str | Path,
    output_dir: str | Path,
    english_reference_audio: str | Path,
    hindi_reference_audio: str | Path,
    punc_norm: Any,
    seed_base: int = 20261030,
) -> dict[str, Any]:
    """Generate a small, reproducible audition set with explicit language prompts."""
    import soundfile as sf

    sheet_path, out = Path(audition_sheet), Path(output_dir)
    en_ref, hi_ref = Path(english_reference_audio), Path(hindi_reference_audio)
    if not sheet_path.is_file():
        raise FileNotFoundError(f"Missing pronunciation audition sheet: {sheet_path}")
    if not en_ref.is_file() or not hi_ref.is_file():
        raise FileNotFoundError("Both official English and Hindi reference clips are required.")
    out.mkdir(parents=True, exist_ok=True)

    with sheet_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sheet_rows = list(csv.DictReader(handle))
    tasks = build_audition_tasks(sheet_rows)
    tasks = preflight_tasks(model, tasks, punc_norm)

    parameters = {
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "temperature": 0.8,
        "repetition_penalty": 1.2,
        "min_p": 0.05,
        "top_p": 1.0,
    }
    for language_id, reference in (("en", en_ref), ("hi", hi_ref)):
        group = [t for t in tasks if t["language_id"] == language_id and t["status"] == "ready"]
        if not group:
            continue
        model.prepare_conditionals(str(reference), exaggeration=parameters["exaggeration"])
        for task in group:
            # Paired English variants for one term reuse the same seed.
            language_offset = 50 if language_id == "hi" else 0
            seed = seed_base + (int(task["audition_id"][1:]) * 100) + language_offset
            task["seed"] = seed
            try:
                _set_seed(seed)
                wav = model.generate(task["text"], language_id=language_id, **parameters)
                waveform = wav.detach().float().cpu().numpy().squeeze()
                if waveform.ndim != 1 or waveform.size == 0:
                    raise ValueError(f"Unexpected audio shape: {tuple(wav.shape)}")
                filename = f"{task['task_id']}.wav"
                audio_path = out / filename
                sf.write(str(audio_path), waveform, int(model.sr), subtype="PCM_16")
                task["output_file"] = filename
                task["duration_seconds"] = round(float(waveform.size / int(model.sr)), 3)
                task["sha256"] = _sha256(audio_path)
                task["status"] = "generated"
                task["error"] = ""
            except Exception as exc:  # Preserve successful auditions if one task fails.
                task["status"] = "generation_failed"
                task["error"] = f"{type(exc).__name__}: {exc}"

    task_manifest = {
        "schema_version": "1.0.0",
        "model": "ResembleAI/chatterbox, Chatterbox Multilingual V3 (t3_model='v3')",
        "pronunciation_sheet_sha256": _sha256(sheet_path),
        "voice_references": {
            "en": {"file": en_ref.name, "sha256": _sha256(en_ref), "source": "Resemble AI public demo sample"},
            "hi": {"file": hi_ref.name, "sha256": _sha256(hi_ref), "source": "Resemble AI public demo sample"},
        },
        "parameters": parameters,
        "seed_base": seed_base,
        "tasks": tasks,
        "full_book_synthesis_authorized": False,
    }
    manifest_path = out / "pronunciation_audition_manifest.json"
    manifest_path.write_text(json.dumps(task_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out / "pronunciation_audition_tasks.csv", TASK_FIELDS, tasks)

    decision_rows = []
    for row in sheet_rows:
        tid = row["audition_id"]
        task_paths = {
            t["variant"]: t.get("output_file", "")
            for t in tasks if t["audition_id"] == tid and t.get("status") == "generated"
        }
        decision_rows.append({
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
            "review_notes": "Do not change the source spelling; record a chosen form only after listening.",
        })
    _write_csv(out / "pronunciation_audition_decisions.csv", DECISION_FIELDS, decision_rows)

    summary = {
        "task_count": len(tasks),
        "generated_count": sum(t["status"] == "generated" for t in tasks),
        "skipped_unknown_tokens": sum(t["status"] == "skipped_unknown_tokens" for t in tasks),
        "generation_failures": sum(t["status"] == "generation_failed" for t in tasks),
        "decision_rows": len(decision_rows),
        "output_dir": str(out),
        "full_book_synthesis_authorized": False,
        "files": [
            "pronunciation_audition_manifest.json",
            "pronunciation_audition_tasks.csv",
            "pronunciation_audition_decisions.csv",
        ],
    }
    (out / "pronunciation_audition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
