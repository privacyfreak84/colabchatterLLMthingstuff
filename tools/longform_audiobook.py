"""Resumable, provenance-aware long-form audiobook job helpers.

This module deliberately does not import or invoke Chatterbox. It handles planning,
checkpointing, logging, integrity checks, streaming WAV assembly, and job packaging.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf


MANIFEST_SCHEMA_VERSION = 1
MAX_SOURCE_CHARACTERS = 4_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_job_id(value: str | None = None) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return datetime.now(timezone.utc).strftime("book_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    if len(candidate) > 80 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", candidate):
        raise ValueError("Job ID must start with a letter or number and contain only letters, numbers, dots, underscores, or hyphens (max 80 characters).")
    return candidate


def slugify(value: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (slug[:max_length].strip("-") or "section")


def _heading_title(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if re.match(r"^#{1,6}\s+", stripped):
        return re.sub(r"^#{1,6}\s+", "", stripped).strip() or None
    if re.match(r"^(chapter|part|book)\s+(?:[0-9ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b", stripped, re.I):
        return stripped
    if re.match(
        r"^(preface|foreword|introduction|prologue|epilogue|afterword|conclusion|notes|glossary|bibliography|appendix(?:\s+[a-z0-9]+)?)\b",
        stripped,
        re.I,
    ):
        return stripped
    return None


def _split_paragraph(text: str, max_chars: int) -> list[tuple[str, bool]]:
    """Return (chunk text, split_inside_sentence) pairs without losing source words."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    result: list[tuple[str, bool]] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            result.append((current, False))
            current = ""

    for sentence in sentences:
        if len(sentence) <= max_chars:
            candidate = sentence if not current else current + " " + sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                flush()
                current = sentence
            continue

        flush()
        # Prefer clause boundaries. Fall back to word boundaries only when a clause
        # itself exceeds the configured limit. The original full source remains saved.
        clauses = [part.strip() for part in re.split(r"(?<=[,;:])\s+", sentence) if part.strip()]
        pieces: list[str] = []
        piece = ""
        for clause in clauses:
            if len(clause) <= max_chars:
                candidate = clause if not piece else piece + " " + clause
                if len(candidate) <= max_chars:
                    piece = candidate
                else:
                    if piece:
                        pieces.append(piece)
                    piece = clause
                continue
            if piece:
                pieces.append(piece)
                piece = ""
            words = clause.split()
            word_piece = ""
            for word in words:
                if len(word) > max_chars:
                    raise ValueError(f"A single token is longer than the {max_chars}-character chunk limit.")
                candidate = word if not word_piece else word_piece + " " + word
                if len(candidate) <= max_chars:
                    word_piece = candidate
                else:
                    if word_piece:
                        pieces.append(word_piece)
                    word_piece = word
            if word_piece:
                pieces.append(word_piece)
        if piece:
            pieces.append(piece)
        result.extend((part, True) for part in pieces)

    flush()
    return result


def plan_audiobook_chunks(
    source_text: str,
    max_chars: int = 320,
    *,
    speak_headings: bool = True,
    chunk_pause_seconds: float = 0.18,
    paragraph_pause_seconds: float = 0.45,
    section_pause_seconds: float = 1.0,
) -> dict:
    """Plan paragraph-aware chunks and section boundaries for one approved text file."""
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("The manuscript is empty.")
    if len(source_text) > MAX_SOURCE_CHARACTERS:
        raise ValueError(f"Manuscript exceeds the {MAX_SOURCE_CHARACTERS:,}-character safety limit.")
    if not 120 <= int(max_chars) <= 600:
        raise ValueError("Chunk size must be between 120 and 600 characters.")
    pauses = (chunk_pause_seconds, paragraph_pause_seconds, section_pause_seconds)
    if any(not 0 <= float(value) <= 5 for value in pauses):
        raise ValueError("Every pause must be between 0 and 5 seconds.")

    normalized = source_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    raw_sections: list[dict] = []
    current_title = "Full text"
    current_paragraph_lines: list[tuple[str, int]] = []
    current_paragraphs: list[dict] = []
    saw_heading = False

    def flush_paragraph() -> None:
        nonlocal current_paragraph_lines
        if current_paragraph_lines:
            joined = " ".join(part.strip() for part, _ in current_paragraph_lines if part.strip())
            if joined:
                current_paragraphs.append({
                    "text": joined,
                    "line_start": current_paragraph_lines[0][1],
                    "line_end": current_paragraph_lines[-1][1],
                })
        current_paragraph_lines = []

    def flush_section() -> None:
        nonlocal current_paragraphs
        flush_paragraph()
        if current_paragraphs:
            raw_sections.append({"title": current_title, "paragraphs": current_paragraphs})
        current_paragraphs = []

    for line_number, line in enumerate(lines, start=1):
        heading = _heading_title(line)
        if heading:
            flush_section()
            if not saw_heading and raw_sections and raw_sections[0]["title"] == "Full text":
                raw_sections[0]["title"] = "Opening section"
            current_title = heading
            saw_heading = True
            if speak_headings:
                current_paragraphs.append({
                    "text": heading,
                    "line_start": line_number,
                    "line_end": line_number,
                    "is_heading": True,
                })
        elif not line.strip():
            flush_paragraph()
        else:
            current_paragraph_lines.append((line, line_number))
    flush_section()

    if not raw_sections:
        raise ValueError("No readable paragraphs or sections were found.")

    sections = []
    chunks = []
    for section_index, section in enumerate(raw_sections, start=1):
        section_id = f"S{section_index:03d}"
        section_record = {
            "section_id": section_id,
            "title": section["title"],
            "chunk_ids": [],
        }
        for paragraph_index, paragraph in enumerate(section["paragraphs"], start=1):
            paragraph_chunks = _split_paragraph(paragraph["text"], int(max_chars))
            for part_index, (chunk_text, split_inside_sentence) in enumerate(paragraph_chunks, start=1):
                chunk_index = len(chunks) + 1
                chunk_id = f"C{chunk_index:05d}"
                is_last_in_paragraph = part_index == len(paragraph_chunks)
                is_last_in_section = (
                    is_last_in_paragraph
                    and paragraph_index == len(section["paragraphs"])
                )
                if is_last_in_section:
                    boundary_after = "section"
                    pause_after = float(section_pause_seconds)
                elif is_last_in_paragraph:
                    boundary_after = "paragraph"
                    pause_after = float(paragraph_pause_seconds)
                else:
                    boundary_after = "chunk"
                    pause_after = float(chunk_pause_seconds)

                chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "section_id": section_id,
                    "section_title": section["title"],
                    "paragraph_index": paragraph_index,
                    "line_start": paragraph["line_start"],
                    "line_end": paragraph["line_end"],
                    "text": chunk_text,
                    "text_sha256": sha256_text(chunk_text),
                    "split_inside_sentence": bool(split_inside_sentence),
                    "boundary_after": boundary_after,
                    "pause_after_seconds": pause_after,
                })
                section_record["chunk_ids"].append(chunk_id)
        sections.append(section_record)

    if chunks:
        chunks[-1]["boundary_after"] = "end"
        chunks[-1]["pause_after_seconds"] = 0.0

    return {
        "sections": sections,
        "chunks": chunks,
        "source_sha256": sha256_text(normalized),
        "source_characters": len(normalized),
        "source_lines": len(lines),
        "chunk_count": len(chunks),
        "section_count": len(sections),
        "max_chunk_characters": int(max_chars),
        "speak_headings": bool(speak_headings),
        "pause_settings": {
            "chunk": float(chunk_pause_seconds),
            "paragraph": float(paragraph_pause_seconds),
            "section": float(section_pause_seconds),
        },
    }


def atomic_write_json(path: str | Path, data: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def read_manifest(job_dir: str | Path) -> dict:
    path = Path(job_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No manifest exists for this job: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(job_dir: str | Path, manifest: dict) -> None:
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(Path(job_dir) / "manifest.json", manifest)


def append_job_event(
    job_dir: str | Path,
    event: str,
    message: str,
    *,
    level: str = "INFO",
    details: dict | None = None,
) -> None:
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp_utc": utc_now(),
        "level": level.upper(),
        "event": event,
        "message": message,
        "details": details or {},
    }
    with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    human = f"{entry['timestamp_utc']} [{entry['level']}] {event}: {message}\n"
    with (root / "job.log").open("a", encoding="utf-8") as handle:
        handle.write(human)
        handle.flush()


def read_log_tail(path: str | Path, max_lines: int = 150) -> str:
    target = Path(path)
    if not target.is_file():
        return "No log file exists yet."
    with target.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return "".join(lines[-max(1, int(max_lines)):]).strip() or "Log file is empty."


def _safe_relative_path(job_dir: Path, relative_path: str) -> Path:
    candidate = (job_dir / relative_path).resolve()
    try:
        candidate.relative_to(job_dir.resolve())
    except ValueError as exc:
        raise ValueError("Manifest contains a path outside its job directory.") from exc
    return candidate


def verify_chunk_wav(job_dir: str | Path, chunk: dict, sample_rate: int = 24000) -> bool:
    root = Path(job_dir)
    relative_path = chunk.get("wav_relpath")
    expected_hash = chunk.get("wav_sha256")
    if not relative_path or not expected_hash:
        return False
    try:
        path = _safe_relative_path(root, relative_path)
        if not path.is_file() or sha256_file(path) != expected_hash:
            return False
        info = sf.info(str(path))
        return info.samplerate == sample_rate and info.channels == 1 and info.frames > 0
    except (OSError, RuntimeError, ValueError):
        return False


def create_or_resume_job(
    jobs_root: str | Path,
    job_id: str | None,
    *,
    source_text: str,
    source_name: str,
    plan: dict,
    generation_config: dict,
    base_seed: int,
) -> tuple[Path, dict, bool]:
    """Create a job or safely reopen one only when source, plan, and settings match."""
    resolved_id = safe_job_id(job_id)
    root = Path(jobs_root) / resolved_id
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "source.txt"
    manifest_path = root / "manifest.json"
    source_hash = sha256_text(source_text.replace("\r\n", "\n").replace("\r", "\n"))
    config_json = json.dumps(generation_config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    config_hash = sha256_text(config_json)
    plan_identity = [
        {
            "chunk_id": item["chunk_id"],
            "text_sha256": item["text_sha256"],
            "section_id": item["section_id"],
            "paragraph_index": item["paragraph_index"],
        }
        for item in plan["chunks"]
    ]
    plan_hash = sha256_text(json.dumps(plan_identity, sort_keys=True, ensure_ascii=False))

    if manifest_path.exists():
        manifest = read_manifest(root)
        if manifest.get("source_sha256") != source_hash:
            raise ValueError("This job ID already belongs to different manuscript text. Use a new job ID.")
        if manifest.get("generation_config_sha256") != config_hash:
            raise ValueError("This job ID has different voice/settings/reference configuration. Use the original settings or a new job ID.")
        if manifest.get("plan_sha256") != plan_hash:
            raise ValueError("The chunk plan differs from the saved job. Use the original chunk settings or a new job ID.")
        if len(manifest.get("chunks", [])) != len(plan["chunks"]):
            raise ValueError("Saved job chunk count does not match the current plan.")
        for saved, planned in zip(manifest["chunks"], plan["chunks"]):
            if saved.get("chunk_id") != planned["chunk_id"] or saved.get("text_sha256") != planned["text_sha256"]:
                raise ValueError("Saved chunk identity differs from the current manuscript.")
        manifest["status"] = "resuming"
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        for chunk in manifest["chunks"]:
            if chunk.get("status") == "complete" and not verify_chunk_wav(root, chunk):
                chunk["status"] = "pending"
                chunk.pop("wav_sha256", None)
                chunk.pop("duration_seconds", None)
                chunk["error"] = "Saved audio missing or failed integrity verification; scheduled for regeneration."
        if not source_path.exists():
            source_path.write_text(source_text, encoding="utf-8")
        write_manifest(root, manifest)
        append_job_event(root, "job_resumed", "Reopened saved job after verifying source, settings, plan, and completed chunk files.", details={"resume_count": manifest["resume_count"]})
        return root, manifest, True

    source_path.write_text(source_text, encoding="utf-8")
    (root / "chunks").mkdir(exist_ok=True)
    (root / "sections").mkdir(exist_ok=True)
    chunks = []
    for planned in plan["chunks"]:
        item = dict(planned)
        item["status"] = "pending"
        item["wav_relpath"] = f"chunks/{item['chunk_id'].lower()}.wav"
        item["seed"] = (
            int(base_seed) + int(item["chunk_index"]) - 1
            if int(base_seed) >= 0
            else secrets.randbelow(2**31 - 1)
        )
        chunks.append(item)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "job_id": resolved_id,
        "status": "planned",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "resume_count": 0,
        "source_name": Path(source_name).name,
        "source_sha256": source_hash,
        "source_characters": len(source_text),
        "generation_config": generation_config,
        "generation_config_sha256": config_hash,
        "plan_sha256": plan_hash,
        "plan": {key: value for key, value in plan.items() if key != "chunks"},
        "sections": plan["sections"],
        "chunks": chunks,
        "artifacts": {},
        "errors": [],
    }
    write_manifest(root, manifest)
    append_job_event(
        root,
        "job_created",
        f"Created audiobook job with {len(chunks)} chunks across {len(plan['sections'])} section(s).",
        details={"source_sha256": source_hash, "plan_sha256": plan_hash},
    )
    return root, manifest, False


def assemble_wav(
    input_paths: list[str | Path],
    pauses_after: list[float],
    output_path: str | Path,
    *,
    sample_rate: int = 24000,
    block_frames: int = 131072,
) -> dict:
    """Stream mono WAV inputs into one PCM16 WAV without loading the whole book into RAM."""
    import numpy as np

    if len(input_paths) != len(pauses_after) or not input_paths:
        raise ValueError("Audio input paths and pause lengths must be non-empty and have equal length.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial" + output.suffix)
    total_frames = 0
    peak = 0.0
    if temporary.exists():
        temporary.unlink()
    try:
        with sf.SoundFile(str(temporary), mode="w", samplerate=sample_rate, channels=1, subtype="PCM_16", format="WAV") as writer:
            for index, (input_path, pause_seconds) in enumerate(zip(input_paths, pauses_after)):
                with sf.SoundFile(str(input_path), mode="r") as reader:
                    if reader.samplerate != sample_rate or reader.channels != 1:
                        raise ValueError(f"Audio chunk has unexpected format: {input_path}")
                    while True:
                        block = reader.read(block_frames, dtype="float32", always_2d=True)
                        if block.size == 0:
                            break
                        peak = max(peak, float(np.max(np.abs(block))))
                        writer.write(block)
                        total_frames += block.shape[0]
                pause_frames = int(round(max(0.0, float(pause_seconds)) * sample_rate))
                remaining = pause_frames
                silence = np.zeros((min(block_frames, max(1, pause_frames)), 1), dtype=np.float32)
                while remaining:
                    count = min(remaining, len(silence))
                    writer.write(silence[:count])
                    remaining -= count
                    total_frames += count
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(output),
        "sample_rate": sample_rate,
        "channels": 1,
        "frames": total_frames,
        "duration_seconds": round(total_frames / sample_rate, 3),
        "peak": round(peak, 6),
        "sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
    }


def assemble_job_masters(job_dir: str | Path, manifest: dict, sample_rate: int = 24000) -> dict:
    root = Path(job_dir)
    incomplete = [item["chunk_id"] for item in manifest["chunks"] if item.get("status") != "complete" or not verify_chunk_wav(root, item, sample_rate)]
    if incomplete:
        raise ValueError("Cannot assemble masters; incomplete or invalid chunks: " + ", ".join(incomplete[:12]))

    sections_out = []
    full_paths: list[Path] = []
    full_pauses: list[float] = []
    for section in manifest["sections"]:
        section_chunks = [
            item for item in manifest["chunks"]
            if item["section_id"] == section["section_id"]
        ]
        paths = [_safe_relative_path(root, item["wav_relpath"]) for item in section_chunks]
        pauses = [
            float(item["pause_after_seconds"]) if i < len(section_chunks) - 1 else 0.0
            for i, item in enumerate(section_chunks)
        ]
        filename = f"{section['section_id'].lower()}_{slugify(section['title'])}.wav"
        section_path = root / "sections" / filename
        info = assemble_wav(paths, pauses, section_path, sample_rate=sample_rate)
        sections_out.append({
            "section_id": section["section_id"],
            "title": section["title"],
            "path": str(section_path),
            "relpath": str(section_path.relative_to(root)),
            "audio": info,
        })
        for index, (path, item) in enumerate(zip(paths, section_chunks)):
            full_paths.append(path)
            full_pauses.append(float(item["pause_after_seconds"]))
    master_path = root / "whole_book_master.wav"
    master_info = assemble_wav(full_paths, full_pauses, master_path, sample_rate=sample_rate)
    artifacts = {
        "whole_book_master": {
            "relpath": str(master_path.relative_to(root)),
            "audio": master_info,
        },
        "section_masters": sections_out,
    }
    manifest["artifacts"] = artifacts
    manifest["status"] = "complete"
    write_manifest(root, manifest)
    append_job_event(root, "assembly_complete", "Created full-book and section master WAVs.", details={"duration_seconds": master_info["duration_seconds"], "section_count": len(sections_out)})
    return artifacts


def create_job_bundle(job_dir: str | Path) -> Path:
    root = Path(job_dir)
    manifest = read_manifest(root)
    if manifest.get("status") != "complete":
        raise ValueError("The audiobook job is not complete; finish all chunks before packaging.")
    bundle = root / "audiobook_delivery_bundle.zip"
    temporary = bundle.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for relative in ["source.txt", "manifest.json", "events.jsonl", "job.log", "whole_book_master.wav"]:
            path = root / relative
            if path.is_file():
                archive.write(path, arcname=relative)
        for folder_name in ("chunks", "sections"):
            folder = root / folder_name
            if folder.exists():
                for path in sorted(folder.rglob("*")):
                    if path.is_file() and path.suffix.lower() == ".wav":
                        archive.write(path, arcname=str(path.relative_to(root)))
    os.replace(temporary, bundle)
    return bundle
