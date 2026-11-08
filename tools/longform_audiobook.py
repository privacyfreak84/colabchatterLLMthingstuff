"""Resumable, provenance-aware long-form audiobook job helpers.

This module deliberately does not import or invoke Chatterbox. It handles planning,
checkpointing, logging, integrity checks, streaming WAV assembly, and job packaging.
"""
from __future__ import annotations

import hashlib
import json
import os
from io import BytesIO
import re
import secrets
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf


MANIFEST_SCHEMA_VERSION = 1
MAX_SOURCE_CHARACTERS = 4_000_000
MAX_PDF_BYTES = 100 * 1024 * 1024
MAX_PDF_PAGES = 2500


def normalize_manuscript_text(text: str) -> str:
    """Reflow physical line wraps into continuous paragraphs without changing punctuation.

    Blank lines remain paragraph boundaries. Recognized standalone headings remain
    separate. Soft hyphens and layout-only whitespace are removed; visible hyphens
    are retained and joined directly to the next wrapped line.
    """
    if not isinstance(text, str):
        raise TypeError("Manuscript text must be a string.")
    text = text.replace("\ufeff", "").replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    text = text.replace("\u00a0", " ")
    paragraphs: list[str] = []
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        if current_lines:
            joined = current_lines[0]
            for line in current_lines[1:]:
                if joined.endswith("-"):
                    joined += line
                else:
                    joined += " " + line
            joined = re.sub(r"[ \t]+", " ", joined).strip()
            if joined:
                paragraphs.append(joined)
        current_lines = []

    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            flush()
            continue
        if _heading_title(line):
            flush()
            paragraphs.append(line)
            continue
        current_lines.append(line)
    flush()

    normalized = "\n\n".join(paragraphs).strip()
    if len(normalized) > MAX_SOURCE_CHARACTERS:
        raise ValueError(
            f"Manuscript exceeds the {MAX_SOURCE_CHARACTERS:,}-character safety limit."
        )
    return normalized


def extract_pdf_manuscript(
    pdf_source: str | Path | bytes,
    *,
    page_start: int = 1,
    page_end: int | None = None,
) -> dict:
    """Extract selectable PDF text and preserve one-based page provenance; this is not OCR."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires pypdf. Rerun the notebook setup cell.") from exc

    source_name = "uploaded.pdf"
    if isinstance(pdf_source, bytes):
        pdf_bytes = pdf_source
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise ValueError(f"PDF exceeds the {MAX_PDF_BYTES // (1024 * 1024)} MiB safety limit.")
        stream = BytesIO(pdf_bytes)
    else:
        path = Path(pdf_source)
        source_name = path.name
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise ValueError("Select an existing PDF file.")
        if path.stat().st_size > MAX_PDF_BYTES:
            raise ValueError(f"PDF exceeds the {MAX_PDF_BYTES // (1024 * 1024)} MiB safety limit.")
        pdf_bytes = path.read_bytes()
        stream = BytesIO(pdf_bytes)

    try:
        reader = PdfReader(stream, strict=False)
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
            except Exception as exc:
                raise ValueError("This PDF is encrypted and cannot be opened with an empty password.") from exc
            if not decrypted:
                raise ValueError("This PDF is password-protected. Remove the password before uploading.")
        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ValueError("The PDF contains no pages.")
        if total_pages > MAX_PDF_PAGES:
            raise ValueError(f"PDF has {total_pages:,} pages; the safety limit is {MAX_PDF_PAGES:,}.")

        start_page = int(page_start)
        end_page = total_pages if page_end in (None, 0) else int(page_end)
        if start_page < 1 or end_page < start_page or end_page > total_pages:
            raise ValueError(
                f"Invalid PDF page range. Choose pages 1 through {total_pages}; "
                "end page 0 means the last page."
            )

        page_texts: list[str] = []
        page_records: list[dict] = []
        warnings: list[str] = []
        for page_number in range(start_page, end_page + 1):
            page = reader.pages[page_number - 1]
            extraction_mode = "layout"
            try:
                raw_page_text = page.extract_text(extraction_mode="layout") or ""
            except TypeError:
                extraction_mode = "default_fallback"
                raw_page_text = page.extract_text() or ""
            except Exception as exc:
                extraction_mode = "default_fallback"
                warnings.append(
                    f"Page {page_number}: layout extraction failed ({type(exc).__name__}); tried default extraction."
                )
                try:
                    raw_page_text = page.extract_text() or ""
                except Exception as fallback_exc:
                    warnings.append(
                        f"Page {page_number}: text extraction failed ({type(fallback_exc).__name__})."
                    )
                    raw_page_text = ""

            cleaned_page = normalize_manuscript_text(raw_page_text.replace("\x00", ""))
            page_characters = len(re.sub(r"\s+", "", cleaned_page))
            page_texts.append(cleaned_page)
            page_records.append({
                "page": page_number,
                "characters_without_whitespace": page_characters,
                "extraction_mode": extraction_mode,
                "likely_image_only_or_empty": page_characters < 20,
            })
            if page_characters == 0:
                warnings.append(f"Page {page_number}: no selectable text found.")
            elif page_characters < 80:
                warnings.append(
                    f"Page {page_number}: only {page_characters} non-whitespace characters extracted; inspect for scanned content or extraction gaps."
                )

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {type(exc).__name__}: {exc}") from exc

    nonempty_pages = [
        (record, text)
        for record, text in zip(page_records, page_texts)
        if text.strip()
    ]
    # A single newline between pages is deliberate: it is treated as a visual wrap,
    # not a paragraph break, so a sentence continuing across a page stays continuous.
    combined_page_text = "\n".join(text for _, text in nonempty_pages)
    normalized_text = normalize_manuscript_text(combined_page_text)

    total_characters = sum(item["characters_without_whitespace"] for item in page_records)
    if not normalized_text or (
        total_characters < 100
        and max((item["characters_without_whitespace"] for item in page_records), default=0) < 40
    ):
        raise ValueError(
            "Almost no selectable text was extracted. This PDF is probably scanned/image-only. "
            "The notebook extracts embedded text but does not perform OCR. Run OCR first, then upload the searchable PDF."
        )

    page_spans = []
    cursor = 0
    for index, (record, page_text) in enumerate(nonempty_pages):
        char_start = cursor
        char_end = char_start + len(page_text)
        page_spans.append({
            "page_number": record["page"],
            "char_start": char_start,
            "char_end": char_end,
        })
        joins_to_next_without_separator = (
            index < len(nonempty_pages) - 1
            and page_text.rstrip().endswith("-")
        )
        cursor = char_end + (
            0 if joins_to_next_without_separator else
            1 if index < len(nonempty_pages) - 1 else 0
        )

    empty_pages = [
        item["page"] for item in page_records
        if item["likely_image_only_or_empty"]
    ]
    low_text_pages = [
        item["page"] for item in page_records
        if item["characters_without_whitespace"] < 80
    ]
    metadata = getattr(reader, "metadata", None)
    pdf_metadata = {}
    if metadata is not None:
        for key, value in (
            ("title", getattr(metadata, "title", None)),
            ("author", getattr(metadata, "author", None)),
            ("subject", getattr(metadata, "subject", None)),
            ("creator", getattr(metadata, "creator", None)),
        ):
            if value:
                pdf_metadata[key] = str(value)[:500]

    return {
        "text": normalized_text,
        "source_name": source_name,
        "pdf_sha256": sha256_bytes(pdf_bytes),
        "pdf_size_bytes": len(pdf_bytes),
        "total_pdf_pages": total_pages,
        "selected_page_start": start_page,
        "selected_page_end": end_page,
        "selected_page_count": end_page - start_page + 1,
        "extracted_characters": len(normalized_text),
        "page_records": page_records,
        "page_spans": page_spans,
        "empty_or_image_only_pages": empty_pages,
        "low_text_pages": low_text_pages,
        "warnings": warnings,
        "pdf_metadata": pdf_metadata,
        "ocr_attempted": False,
        "extraction_method": "pypdf layout text extraction with default-mode fallback",
        "line_wraps_normalized": True,
    }


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
    source_pages: list[dict] | None = None,
    chunk_pause_seconds: float = 0.18,
    paragraph_pause_seconds: float = 0.45,
    section_pause_seconds: float = 1.0,
) -> dict:
    """Plan paragraph-aware chunks and section boundaries for one approved text file."""
    source_text = normalize_manuscript_text(source_text)
    if not source_text:
        raise ValueError("The manuscript is empty after line-wrap cleanup.")
    if not 120 <= int(max_chars) <= 600:
        raise ValueError("Chunk size must be between 120 and 600 characters.")
    pauses = (chunk_pause_seconds, paragraph_pause_seconds, section_pause_seconds)
    if any(not 0 <= float(value) <= 5 for value in pauses):
        raise ValueError("Every pause must be between 0 and 5 seconds.")

    normalized = normalize_manuscript_text(source_text)
    lines = normalized.split("\n")
    source_pages = source_pages or []
    raw_sections: list[dict] = []
    current_title = "Full text"
    current_paragraph_lines: list[tuple[str, int, int, int]] = []
    current_paragraphs: list[dict] = []
    saw_heading = False

    def flush_paragraph() -> None:
        nonlocal current_paragraph_lines
        if current_paragraph_lines:
            joined = " ".join(part.strip() for part, _, _, _ in current_paragraph_lines if part.strip())
            if joined:
                current_paragraphs.append({
                    "text": joined,
                    "line_start": current_paragraph_lines[0][1],
                    "line_end": current_paragraph_lines[-1][1],
                    "char_start": current_paragraph_lines[0][2],
                    "char_end": current_paragraph_lines[-1][3],
                })
        current_paragraph_lines = []

    def flush_section() -> None:
        nonlocal current_paragraphs
        flush_paragraph()
        if current_paragraphs:
            raw_sections.append({"title": current_title, "paragraphs": current_paragraphs})
        current_paragraphs = []

    line_offset = 0
    for line_number, line in enumerate(lines, start=1):
        char_start = line_offset
        char_end = char_start + len(line)
        line_offset = char_end + 1
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
                    "char_start": char_start,
                    "char_end": char_end,
                    "is_heading": True,
                })
        elif not line.strip():
            flush_paragraph()
        else:
            current_paragraph_lines.append((line, line_number, char_start, char_end))
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
            paragraph_search_cursor = 0
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

                local_char_start = paragraph["text"].find(chunk_text, paragraph_search_cursor)
                if local_char_start < 0:
                    local_char_start = paragraph_search_cursor
                local_char_end = local_char_start + len(chunk_text)
                paragraph_search_cursor = local_char_end
                chunk_char_start = paragraph["char_start"] + local_char_start
                chunk_char_end = paragraph["char_start"] + local_char_end
                chunk_source_pages = [
                    int(page["page_number"])
                    for page in source_pages
                    if int(page.get("char_start", 0)) < chunk_char_end
                    and int(page.get("char_end", 0)) > chunk_char_start
                ]

                tts_text = chunk_text
                if split_inside_sentence and tts_text and tts_text[-1] not in ".!?,":
                    # Chatterbox punc_norm otherwise appends a full stop to this
                    # artificial chunk boundary. Keep source text unchanged and
                    # record the continuation comma as a TTS-only aid.
                    tts_text += ","

                chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "section_id": section_id,
                    "section_title": section["title"],
                    "paragraph_index": paragraph_index,
                    "line_start": paragraph["line_start"],
                    "line_end": paragraph["line_end"],
                    "source_char_start": chunk_char_start,
                    "source_char_end": chunk_char_end,
                    "source_pages": chunk_source_pages,
                    "text": chunk_text,
                    "tts_text": tts_text,
                    "tts_text_sha256": sha256_text(tts_text),
                    "tts_boundary_comma_added": tts_text != chunk_text,
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
        "source_pages": source_pages,
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
    source_pdf_bytes: bytes | None = None,
    pdf_extraction_report: dict | None = None,
) -> tuple[Path, dict, bool]:
    """Create a job or safely reopen one only when source, plan, and settings match."""
    resolved_id = safe_job_id(job_id)
    root = Path(jobs_root) / resolved_id
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "source.txt"
    manifest_path = root / "manifest.json"
    source_text = normalize_manuscript_text(source_text)
    source_hash = sha256_text(source_text)
    source_pdf_hash = sha256_bytes(source_pdf_bytes) if source_pdf_bytes is not None else None
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
        if manifest.get("source_pdf_sha256") != source_pdf_hash:
            raise ValueError("This job ID was created from a different source PDF. Use the original PDF or a new job ID.")
        if manifest.get("generation_config_sha256") != config_hash:
            raise ValueError("This job ID has different voice/settings/reference configuration. Use the original settings or a new job ID.")
        if manifest.get("plan_sha256") != plan_hash:
            raise ValueError("The chunk plan differs from the saved job. Use the original chunk settings or a new job ID.")
        if manifest.get("pdf_extraction_report") != (pdf_extraction_report or None):
            raise ValueError("The PDF extraction report or selected page range differs from the saved job. Restore it or use a new Job ID.")
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
        if source_pdf_bytes is not None:
            original_pdf = root / "source_original.pdf"
            if original_pdf.exists() and sha256_file(original_pdf) != source_pdf_hash:
                raise ValueError("The archived original PDF is corrupt or does not match the uploaded PDF.")
            if not original_pdf.exists():
                original_pdf.write_bytes(source_pdf_bytes)
        write_manifest(root, manifest)
        append_job_event(root, "job_resumed", "Reopened saved job after verifying source, settings, plan, and completed chunk files.", details={"resume_count": manifest["resume_count"]})
        return root, manifest, True

    source_path.write_text(source_text, encoding="utf-8")
    if source_pdf_bytes is not None:
        (root / "source_original.pdf").write_bytes(source_pdf_bytes)
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
        "source_pdf_sha256": source_pdf_hash,
        "source_pdf_filename": Path(source_name).name if source_pdf_bytes is not None else None,
        "pdf_extraction_report": pdf_extraction_report or None,
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


def force_rerender_chunks(
    job_dir: str | Path,
    manifest: dict,
    chunk_ids: list[str],
    *,
    new_random_seed: bool = True,
) -> list[str]:
    """Invalidate selected completed chunks while preserving their prior WAV revisions."""
    requested = list(dict.fromkeys(str(value).strip().upper() for value in chunk_ids if str(value).strip()))
    if not requested:
        return []
    known = {item["chunk_id"].upper(): item for item in manifest.get("chunks", [])}
    unknown = [chunk_id for chunk_id in requested if chunk_id not in known]
    if unknown:
        raise ValueError("Unknown chunk ID(s): " + ", ".join(unknown))

    root = Path(job_dir)
    revisions_dir = root / "revisions"
    rerendered = []
    for chunk_id in requested:
        chunk = known[chunk_id]
        old_seed = int(chunk["seed"])
        old_path = _safe_relative_path(root, chunk["wav_relpath"])
        previous_renders = chunk.setdefault("previous_renders", [])

        if old_path.is_file():
            old_hash = sha256_file(old_path)
            revision_number = len(previous_renders) + 1
            revision_name = f"{chunk_id.lower()}_rev{revision_number:02d}_seed{old_seed}.wav"
            revision_path = revisions_dir / revision_name
            revisions_dir.mkdir(parents=True, exist_ok=True)
            os.replace(old_path, revision_path)
            previous_renders.append({
                "revision": revision_number,
                "seed": old_seed,
                "wav_sha256": old_hash,
                "duration_seconds": chunk.get("duration_seconds"),
                "relpath": str(revision_path.relative_to(root)),
                "preserved_at_utc": utc_now(),
                "was_verified_complete": bool(
                    chunk.get("status") == "complete"
                    and chunk.get("wav_sha256") == old_hash
                ),
            })

        if new_random_seed:
            chunk.setdefault("seed_history", []).append(old_seed)
            chunk["seed"] = secrets.randbelow(2**31 - 1)
        chunk["status"] = "pending"
        for key in (
            "wav_sha256",
            "duration_seconds",
            "elapsed_seconds",
            "peak_before_limiter",
            "limiter_gain",
            "completed_at_utc",
            "device",
            "error",
        ):
            chunk.pop(key, None)
        rerendered.append(chunk_id)

    manifest["status"] = "ready_for_rerender"
    manifest.setdefault("rerender_history", []).append({
        "timestamp_utc": utc_now(),
        "chunk_ids": rerendered,
        "new_random_seed": bool(new_random_seed),
    })
    write_manifest(root, manifest)
    append_job_event(
        root,
        "chunks_queued_for_rerender",
        "Queued selected chunks for a new render; any prior WAVs were preserved under revisions/.",
        details={"chunk_ids": rerendered, "new_random_seed": bool(new_random_seed)},
    )
    return rerendered


def build_audio_qa_report(job_dir: str | Path, manifest: dict, sample_rate: int = 24000) -> dict:
    """Run technical checks only; listening remains authoritative for speech quality."""
    root = Path(job_dir)
    items = []
    errors = []
    seen_hashes: dict[str, str] = {}
    total_duration = 0.0

    for chunk in manifest.get("chunks", []):
        flags = []
        valid = verify_chunk_wav(root, chunk, sample_rate)
        if not valid:
            flags.append("missing_or_invalid_wav")
            errors.append(chunk["chunk_id"])
            duration = chunk.get("duration_seconds")
            sample_peak = None
        else:
            wav_path = _safe_relative_path(root, chunk["wav_relpath"])
            info = sf.info(str(wav_path))
            duration = round(info.frames / info.samplerate, 3)
            if duration < 0.5:
                flags.append("very_short_chunk")
            if duration > 42.0:
                flags.append("unusually_long_chunk")
            words = len(str(chunk.get("text", "")).split())
            rate = (words / duration * 60.0) if duration > 0 else None
            if rate is not None and (rate < 65 or rate > 260):
                flags.append("unusual_estimated_speaking_rate")
            if chunk.get("split_inside_sentence"):
                flags.append("split_inside_sentence")
            unknown_count = int(chunk.get("unknown_text_token_count", 0) or 0)
            if unknown_count:
                flags.append("unknown_text_tokens")
            if float(chunk.get("limiter_gain", 1.0)) < 0.98:
                flags.append("peak_limiter_engaged")
            digest = chunk.get("wav_sha256")
            if digest:
                if digest in seen_hashes:
                    flags.append("duplicate_audio_hash")
                    flags.append(f"same_audio_as_{seen_hashes[digest]}")
                else:
                    seen_hashes[digest] = chunk["chunk_id"]
            sample_peak = chunk.get("peak_before_limiter")
            total_duration += duration

        items.append({
            "chunk_id": chunk["chunk_id"],
            "section_id": chunk.get("section_id"),
            "paragraph_index": chunk.get("paragraph_index"),
            "status": chunk.get("status"),
            "duration_seconds": duration,
            "word_count": len(str(chunk.get("text", "")).split()),
            "unknown_text_token_count": int(chunk.get("unknown_text_token_count", 0) or 0),
            "estimated_words_per_minute": (
                round(len(str(chunk.get("text", "")).split()) / duration * 60.0, 1)
                if duration and duration > 0 else None
            ),
            "peak_before_limiter": sample_peak,
            "flags": flags,
        })

    flagged = [item for item in items if item["flags"]]
    report = {
        "schema_version": 1,
        "job_id": manifest.get("job_id"),
        "created_at_utc": utc_now(),
        "sample_rate": sample_rate,
        "technical_integrity_pass": not errors,
        "chunk_count": len(items),
        "invalid_chunk_ids": errors,
        "flagged_chunk_count": len(flagged),
        "flagged_chunks": flagged,
        "total_chunk_audio_seconds": round(total_duration, 3),
        "human_review_required": True,
        "limitations": [
            "These checks cannot detect accent drift, mispronunciation, unnatural prosody, repetition, or semantic omissions.",
            "Duration and speaking-rate flags are heuristic review prompts, not automatic rejection criteria.",
            "Human listening against the approved manuscript remains mandatory.",
        ],
        "chunks": items,
    }
    atomic_write_json(root / "qa_report.json", report)
    manifest["qa_report"] = {
        "relpath": "qa_report.json",
        "technical_integrity_pass": report["technical_integrity_pass"],
        "flagged_chunk_count": report["flagged_chunk_count"],
        "human_review_required": True,
    }
    write_manifest(root, manifest)
    append_job_event(
        root,
        "audio_qa_complete",
        f"Technical QA checked {len(items)} chunks; {len(flagged)} chunk(s) were flagged for listening review.",
        details={
            "technical_integrity_pass": report["technical_integrity_pass"],
            "flagged_chunk_count": report["flagged_chunk_count"],
        },
    )
    return report


def create_job_bundle(job_dir: str | Path) -> Path:
    root = Path(job_dir)
    manifest = read_manifest(root)
    if manifest.get("status") != "complete":
        raise ValueError("The audiobook job is not complete; finish all chunks before packaging.")
    bundle = root / "audiobook_delivery_bundle.zip"
    temporary = bundle.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for relative in ["source.txt", "source_original.pdf", "manifest.json", "events.jsonl", "job.log", "whole_book_master.wav", "qa_report.json"]:
            path = root / relative
            if path.is_file():
                archive.write(path, arcname=relative)
        for folder_name in ("chunks", "sections", "revisions"):
            folder = root / folder_name
            if folder.exists():
                for path in sorted(folder.rglob("*")):
                    if path.is_file() and path.suffix.lower() == ".wav":
                        archive.write(path, arcname=str(path.relative_to(root)))
    os.replace(temporary, bundle)
    return bundle
