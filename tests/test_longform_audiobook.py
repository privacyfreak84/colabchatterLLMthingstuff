import json
import zipfile
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.longform_audiobook import (
    append_job_event,
    assemble_job_masters,
    assemble_selected_chunk_wavs,
    assemble_wav,
    build_audio_qa_report,
    create_job_bundle,
    create_or_resume_job,
    force_rerender_chunks,
    extract_pdf_manuscript,
    normalize_manuscript_text,
    plan_audiobook_chunks,
    read_log_tail,
    read_manifest,
    safe_job_id,
    safe_output_name,
    verify_chunk_wav,
    write_manifest,
)


def test_safe_job_id_rejects_path_traversal_and_generates_id():
    assert safe_job_id("chapter-01_take.2") == "chapter-01_take.2"
    generated = safe_job_id("")
    assert generated.startswith("book_")
    for invalid in ("../escape", "/tmp/file", "has spaces", ".hidden"):
        try:
            safe_job_id(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe job id accepted: {invalid}")


def test_plan_detects_sections_and_preserves_all_words():
    source = (
        "Preface\n\n"
        "A first paragraph has two sentences. The second sentence follows.\n\n"
        "Chapter One: The Path\n\n"
        "A longer paragraph discusses Avalokiteshvara, Maitreya, and Manjushri."
    )
    plan = plan_audiobook_chunks(source, max_chars=120)
    assert plan["section_count"] == 2
    assert plan["sections"][0]["title"] == "Preface"
    assert plan["sections"][1]["title"] == "Chapter One: The Path"
    assert plan["chunk_count"] >= 3
    combined = " ".join(item["text"] for item in plan["chunks"])
    for word in ("Avalokiteshvara", "Maitreya", "Manjushri", "follows."):
        assert word in combined
    assert plan["chunks"][-1]["pause_after_seconds"] == 0.0
    assert plan["chunks"][-1]["boundary_after"] == "end"


def test_plan_can_exclude_heading_audio_without_losing_heading_metadata():
    plan = plan_audiobook_chunks(
        "# Chapter 1: Introduction\n\nThe actual paragraph.",
        speak_headings=False,
    )
    assert plan["sections"][0]["title"] == "Chapter 1: Introduction"
    assert [item["text"] for item in plan["chunks"]] == ["The actual paragraph."]


def test_plan_splits_long_sentences_and_marks_internal_splits():
    text = " ".join(f"word{i}" for i in range(90)) + "."
    plan = plan_audiobook_chunks(text, max_chars=120)
    assert plan["chunk_count"] > 1
    assert all(len(item["text"]) <= 120 for item in plan["chunks"])
    assert any(item["split_inside_sentence"] for item in plan["chunks"])
    assert any(item["tts_boundary_comma_added"] for item in plan["chunks"])
    assert any(item["tts_text"].endswith(",") for item in plan["chunks"] if item["tts_boundary_comma_added"])
    assert " ".join(item["text"] for item in plan["chunks"]).replace(" ", "") == text.replace(" ", "")


def test_create_job_persists_random_seeds_and_resume_checks_source_and_config(tmp_path):
    source = "Preface\n\nA first paragraph."
    plan = plan_audiobook_chunks(source)
    config = {"language": "en", "temperature": 0.55, "reference_sha256": "abc"}
    root, manifest, resumed = create_or_resume_job(
        tmp_path,
        "my-book",
        source_text=source,
        source_name="approved.txt",
        plan=plan,
        generation_config=config,
        base_seed=-1,
    )
    assert not resumed
    assert (root / "source.txt").read_text(encoding="utf-8") == source
    seeds = [item["seed"] for item in manifest["chunks"]]
    assert len(seeds) == len(set(seeds))
    root2, manifest2, resumed2 = create_or_resume_job(
        tmp_path,
        "my-book",
        source_text=source,
        source_name="approved.txt",
        plan=plan,
        generation_config=config,
        base_seed=-1,
    )
    assert resumed2 and root2 == root
    assert [item["seed"] for item in manifest2["chunks"]] == seeds

    try:
        create_or_resume_job(
            tmp_path,
            "my-book",
            source_text=source + " changed",
            source_name="approved.txt",
            plan=plan_audiobook_chunks(source + " changed"),
            generation_config=config,
            base_seed=-1,
        )
    except ValueError as exc:
        assert "different manuscript" in str(exc)
    else:
        raise AssertionError("resuming a job with different source should fail")


def test_resume_invalidates_missing_or_modified_chunk_audio(tmp_path):
    source = "A small book."
    plan = plan_audiobook_chunks(source)
    root, manifest, _ = create_or_resume_job(
        tmp_path, "integrity", source_text=source, source_name="book.txt",
        plan=plan, generation_config={"voice": "narrator"}, base_seed=7,
    )
    chunk = manifest["chunks"][0]
    wav_path = root / chunk["wav_relpath"]
    sf.write(wav_path, np.zeros(2400, dtype=np.float32), 24000, subtype="PCM_16")
    from tools.longform_audiobook import sha256_file
    chunk["status"] = "complete"
    chunk["wav_sha256"] = sha256_file(wav_path)
    write_manifest(root, manifest)
    assert verify_chunk_wav(root, chunk)

    wav_path.write_bytes(wav_path.read_bytes() + b"corruption")
    assert not verify_chunk_wav(root, chunk)
    _, resumed, was_resumed = create_or_resume_job(
        tmp_path, "integrity", source_text=source, source_name="book.txt",
        plan=plan, generation_config={"voice": "narrator"}, base_seed=7,
    )
    assert was_resumed
    assert resumed["chunks"][0]["status"] == "pending"


def test_logs_are_append_only_and_readable(tmp_path):
    append_job_event(tmp_path, "job_started", "Starting the book.")
    append_job_event(tmp_path, "chunk_complete", "Chunk 1 saved.", details={"chunk": 1})
    tail = read_log_tail(tmp_path / "job.log", max_lines=1)
    assert "chunk_complete" in tail
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["job_started", "chunk_complete"]


def test_user_selected_wav_assembly_and_bundle_include_only_on_demand_master(tmp_path):
    source = "Preface\n\nFirst sentence. Second sentence.\n\nChapter One\n\nFinal paragraph."
    plan = plan_audiobook_chunks(source, max_chars=120)
    root, manifest, _ = create_or_resume_job(
        tmp_path, "assembly", source_text=source, source_name="book.txt",
        plan=plan, generation_config={"voice": "test"}, base_seed=12,
    )

    for index, chunk in enumerate(manifest["chunks"]):
        wav_path = root / chunk["wav_relpath"]
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            wav_path,
            np.full(2400, 0.1 * (index + 1), dtype=np.float32),
            24000,
            subtype="PCM_16",
        )
        from tools.longform_audiobook import sha256_file
        chunk["status"] = "complete"
        chunk["wav_sha256"] = sha256_file(wav_path)
        chunk["duration_seconds"] = 0.1
    manifest["status"] = "complete"
    write_manifest(root, manifest)

    selected_ids = [manifest["chunks"][-1]["chunk_id"], manifest["chunks"][0]["chunk_id"]]
    selected = assemble_selected_chunk_wavs(
        root,
        manifest,
        selected_ids,
        output_name="chosen",
    )
    assert selected["selected_chunk_ids"] == [
        manifest["chunks"][0]["chunk_id"],
        manifest["chunks"][-1]["chunk_id"],
    ]
    selected_path = root / selected["relpath"]
    assert selected_path.is_file()
    with sf.SoundFile(selected_path) as audio:
        # Two 0.1-second clips, plus the first selected chunk's saved join pause.
        assert audio.frames >= 4800

    all_audio = assemble_selected_chunk_wavs(
        root, manifest, include_all_completed=True
    )
    assert all_audio["include_all_completed"] is True
    assert all_audio["chunk_count"] == len(manifest["chunks"])
    assert len(manifest["assemblies"]) == 2

    report = build_audio_qa_report(root, manifest)
    assert report["technical_integrity_pass"] is True
    assert report["human_review_required"] is True
    assert (root / "qa_report.json").is_file()

    bundle = create_job_bundle(root)
    assert bundle.is_file()
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert "source.txt" in names
    assert "manifest.json" in names
    assert "qa_report.json" in names
    assert any(name.startswith("chunks/") for name in names)
    assert any(name.startswith("assemblies/") for name in names)
    assert "whole_book_master.wav" not in names


def test_selected_wav_assembly_rejects_empty_unknown_and_invalid_selections(tmp_path):
    source = "One short sentence."
    plan = plan_audiobook_chunks(source)
    root, manifest, _ = create_or_resume_job(
        tmp_path, "assembly-validation", source_text=source, source_name="book.txt",
        plan=plan, generation_config={"voice": "test"}, base_seed=12,
    )
    chunk = manifest["chunks"][0]
    wav_path = root / chunk["wav_relpath"]
    sf.write(wav_path, np.zeros(2400, dtype=np.float32), 24000, subtype="PCM_16")
    from tools.longform_audiobook import sha256_file
    chunk.update({
        "status": "complete",
        "wav_sha256": sha256_file(wav_path),
        "duration_seconds": 0.1,
    })
    write_manifest(root, manifest)

    import pytest
    with pytest.raises(ValueError, match="Select one or more"):
        assemble_selected_chunk_wavs(root, manifest)
    with pytest.raises(ValueError, match="Unknown chunk ID"):
        assemble_selected_chunk_wavs(root, manifest, ["C99999"])
    with pytest.raises(ValueError, match="all completed chunks"):
        assemble_selected_chunk_wavs(root, manifest, include_all_completed=True)

    # A completed entry with a modified WAV must not be included in an assembly.
    wav_path.write_bytes(wav_path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="incomplete or invalid"):
        assemble_selected_chunk_wavs(root, manifest, ["C00001"])


def test_assemble_wav_rejects_mismatched_input_format(tmp_path):
    stereo = tmp_path / "stereo.wav"
    sf.write(stereo, np.zeros((2400, 2), dtype=np.float32), 24000, subtype="PCM_16")
    try:
        assemble_wav([stereo], [0.0], tmp_path / "out.wav")
    except ValueError as exc:
        assert "unexpected format" in str(exc)
    else:
        raise AssertionError("stereo chunk should be rejected")



def test_force_rerender_keeps_old_active_wav_until_atomic_replacement(tmp_path):
    source = "A sentence to rerender."
    plan = plan_audiobook_chunks(source)
    root, manifest, _ = create_or_resume_job(
        tmp_path, "rerender", source_text=source, source_name="book.txt",
        plan=plan, generation_config={"voice": "test"}, base_seed=22,
    )
    chunk = manifest["chunks"][0]
    wav_path = root / chunk["wav_relpath"]
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, np.full(2400, 0.1, dtype=np.float32), 24000, subtype="PCM_16")
    from tools.longform_audiobook import sha256_file
    original_hash = sha256_file(wav_path)
    original_seed = chunk["seed"]
    chunk.update({
        "status": "complete",
        "wav_sha256": original_hash,
        "duration_seconds": 0.1,
    })
    write_manifest(root, manifest)

    rerendered = force_rerender_chunks(
        root, manifest, ["c00001"], new_random_seed=True
    )
    assert rerendered == ["C00001"]
    assert chunk["status"] == "pending"
    assert chunk["seed"] != original_seed
    # The old active WAV remains in place until the new .partial.wav is atomically
    # installed by the notebook's generator.
    assert wav_path.is_file()
    assert sha256_file(wav_path) == original_hash
    assert chunk["pending_replacement"]["wav_sha256"] == original_hash
    assert chunk.get("previous_renders", []) == []

    # Simulate the generator installing the replacement.
    replacement_audio = np.full(2400, 0.2, dtype=np.float32)
    temporary = wav_path.with_name(wav_path.stem + ".partial.wav")
    sf.write(temporary, replacement_audio, 24000, subtype="PCM_16")
    import os
    os.replace(temporary, wav_path)
    assert sha256_file(wav_path) != original_hash

    try:
        force_rerender_chunks(root, manifest, ["C99999"])
    except ValueError as exc:
        assert "Unknown chunk ID" in str(exc)
    else:
        raise AssertionError("unknown rerender chunk ID should fail")


def test_force_rerender_backup_is_opt_in_and_copies_instead_of_moving(tmp_path):
    source = "A sentence to rerender."
    plan = plan_audiobook_chunks(source)
    root, manifest, _ = create_or_resume_job(
        tmp_path, "rerender-backup", source_text=source, source_name="book.txt",
        plan=plan, generation_config={"voice": "test"}, base_seed=22,
    )
    chunk = manifest["chunks"][0]
    wav_path = root / chunk["wav_relpath"]
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, np.full(2400, 0.1, dtype=np.float32), 24000, subtype="PCM_16")
    from tools.longform_audiobook import sha256_file
    original_hash = sha256_file(wav_path)
    chunk.update({
        "status": "complete",
        "wav_sha256": original_hash,
        "duration_seconds": 0.1,
    })
    write_manifest(root, manifest)

    force_rerender_chunks(
        root, manifest, ["C00001"],
        new_random_seed=False,
        preserve_previous=True,
    )
    previous = chunk["previous_renders"][0]
    previous_path = root / previous["relpath"]
    assert previous_path.is_file()
    assert sha256_file(previous_path) == original_hash
    assert wav_path.is_file()
    assert sha256_file(wav_path) == original_hash


def test_filter_plan_sections_selects_multiple_sections_and_renumbers_chunks():
    from tools.longform_audiobook import filter_plan_sections

    source = (
        "Chapter 1\n\nFirst chapter sentence. More first chapter text.\n\n"
        "Chapter 2\n\nSecond chapter sentence. More second chapter text.\n\n"
        "Chapter 3\n\nThird chapter sentence."
    )
    plan = plan_audiobook_chunks(source, max_chars=120)
    assert [item["section_id"] for item in plan["sections"]] == ["S001", "S002", "S003"]

    selected = filter_plan_sections(plan, ["s003", "s001"])
    assert [item["section_id"] for item in selected["sections"]] == ["S001", "S003"]
    assert selected["all_section_count"] == 3
    assert selected["section_count"] == 2
    assert selected["selection_mode"] == "selected_sections"
    assert selected["selected_section_ids"] == ["S001", "S003"]
    assert [item["chunk_id"] for item in selected["chunks"]] == [
        f"C{index:05d}" for index in range(1, len(selected["chunks"]) + 1)
    ]
    combined_text = " ".join(item["text"] for item in selected["chunks"])
    assert "First chapter sentence" in combined_text
    assert "Third chapter sentence" in combined_text
    assert "Second chapter sentence" not in combined_text
    assert selected["chunks"][-1]["boundary_after"] == "end"


def test_filter_plan_sections_empty_means_all_and_unknown_ids_fail():
    from tools.longform_audiobook import filter_plan_sections

    plan = plan_audiobook_chunks(
        "Chapter 1\n\nFirst sentence.\n\nChapter 2\n\nSecond sentence."
    )
    all_sections = filter_plan_sections(plan, [])
    assert all_sections["section_count"] == 2
    assert all_sections["selection_mode"] == "all"
    try:
        filter_plan_sections(plan, ["S999"])
    except ValueError as exc:
        assert "Unknown section ID" in str(exc)
    else:
        raise AssertionError("unknown section ID should fail")

def test_audio_qa_flags_listening_review_without_claiming_quality(tmp_path):
    source = "A longish paragraph with enough words to form one chunk."
    plan = plan_audiobook_chunks(source)
    root, manifest, _ = create_or_resume_job(
        tmp_path, "qa", source_text=source, source_name="book.txt",
        plan=plan, generation_config={"voice": "test"}, base_seed=3,
    )
    chunk = manifest["chunks"][0]
    wav_path = root / chunk["wav_relpath"]
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, np.full(2400, 0.1, dtype=np.float32), 24000, subtype="PCM_16")
    from tools.longform_audiobook import sha256_file
    chunk.update({
        "status": "complete",
        "wav_sha256": sha256_file(wav_path),
        "duration_seconds": 0.1,
        "limiter_gain": 0.95,
        "unknown_text_token_count": 2,
    })
    report = build_audio_qa_report(root, manifest)
    assert report["technical_integrity_pass"] is True
    assert report["human_review_required"] is True
    assert report["flagged_chunk_count"] >= 1
    assert "very_short_chunk" in report["flagged_chunks"][0]["flags"]
    assert "unknown_text_tokens" in report["flagged_chunks"][0]["flags"]
    assert report["flagged_chunks"][0]["unknown_text_token_count"] == 2



def test_line_wrapped_prose_reflows_but_blank_lines_and_punctuation_survive():
    source = (
        'Then the enlightening being Avalokiteshvara said to the Buddha, "The Buddha '
        'speaks of ten stages of enlightening beings: the stage of intense joy, the stage '
        'of freedom from defilement, the stage of radiance, the stage of blazing intellect, '
        'the stage most difficult to conquer, the stage of presence, the stage of far going, '
        'the stage of immovability, the stage of good intellect, and the stage of cloud of '
        'teaching. You also speak of Buddhahood, making eleven stages. How many kinds of '
        'purity, in how many portions, are these stages contained in?"'
    )
    wrapped = (
        "Then the enlightening being Avalokiteshvara said to the Buddha, \n"
        '"The Buddha speaks of ten stages of enlightening beings: the stage of intense joy,\n'
        "the stage of freedom from defilement, the stage of radiance, the stage of blazing\n"
        "intellect, the stage most difficult to conquer, the stage of presence, the stage of\n"
        "far going, the stage of immovability, the stage of good intellect, and the stage\n"
        "of cloud of teaching. You also speak of Buddhahood, making eleven stages. How many\n"
        'kinds of purity, in how many portions, are these stages contained in?"'
    )
    normalized = normalize_manuscript_text(wrapped)
    expected = normalize_manuscript_text(source)
    assert normalized == expected
    assert "beings: the stage" in normalized
    assert "teaching. You also" in normalized
    assert normalized.endswith('contained in?"')

    paragraphs = normalize_manuscript_text(
        "First line, with punctuation\ncontinued on the next line.\n\n"
        "Second paragraph: still intact!"
    )
    assert paragraphs == (
        "First line, with punctuation continued on the next line.\n\n"
        "Second paragraph: still intact!"
    )


def test_line_wrap_cleanup_preserves_visible_hyphen_and_removes_soft_hyphen():
    normalized = normalize_manuscript_text("A well-\nknown term and an inter\u00adnational word.")
    assert normalized == "A well-known term and an international word."


def test_pdf_extraction_reports_page_text_and_normalizes_line_breaks(tmp_path):
    from reportlab.pdfgen import canvas
    from pypdf import PdfReader

    pdf_path = tmp_path / "line_wrapped_book.pdf"
    canvas_obj = canvas.Canvas(str(pdf_path), pagesize=(612, 792))
    canvas_obj.drawString(72, 720, "Then the enlightening being Avalokiteshvara said")
    canvas_obj.drawString(72, 700, "to the Buddha, who listened carefully.")
    canvas_obj.showPage()
    canvas_obj.drawString(72, 720, "Chapter One: The Path")
    canvas_obj.drawString(72, 700, "The second page contains selectable text.")
    canvas_obj.save()

    extracted = extract_pdf_manuscript(pdf_path)
    assert extracted["total_pdf_pages"] == 2
    assert extracted["selected_page_count"] == 2
    assert extracted["pdf_sha256"]
    assert extracted["extraction_method"].startswith("pypdf")
    assert "Avalokiteshvara said to the Buddha" in extracted["text"]
    assert "Chapter One: The Path" in extracted["text"]
    assert extracted["source_name"] == pdf_path.name
    assert extracted["ocr_attempted"] is False
    assert extracted["empty_or_image_only_pages"] == []


def test_pdf_job_persists_original_pdf_and_refuses_different_pdf_on_resume(tmp_path):
    source = "A PDF-derived sentence."
    pdf_bytes = b"%PDF-test-content"
    plan = plan_audiobook_chunks(source)
    config = {"language": "en", "source_kind": "pdf"}
    root, manifest, resumed = create_or_resume_job(
        tmp_path,
        "pdf-job",
        source_text=source,
        source_name="source.pdf",
        plan=plan,
        generation_config=config,
        base_seed=-1,
        source_pdf_bytes=pdf_bytes,
        pdf_extraction_report={"page_count": 1, "warnings": []},
    )
    assert not resumed
    assert (root / "source_original.pdf").read_bytes() == pdf_bytes
    assert manifest["source_pdf_sha256"]
    assert manifest["pdf_extraction_report"]["page_count"] == 1

    _, _, resumed = create_or_resume_job(
        tmp_path,
        "pdf-job",
        source_text=source,
        source_name="source.pdf",
        plan=plan,
        generation_config=config,
        base_seed=-1,
        source_pdf_bytes=pdf_bytes,
        pdf_extraction_report={"page_count": 1, "warnings": []},
    )
    assert resumed
    try:
        create_or_resume_job(
            tmp_path,
            "pdf-job",
            source_text=source,
            source_name="source.pdf",
            plan=plan,
            generation_config=config,
            base_seed=-1,
            source_pdf_bytes=b"%PDF-a-different-source",
        )
    except ValueError as exc:
        assert "different source PDF" in str(exc)
    else:
        raise AssertionError("resuming with a different PDF should fail")


def test_pdf_extraction_rejects_non_pdf_bytes():
    try:
        extract_pdf_manuscript(b"not a PDF")
    except ValueError as exc:
        assert "Could not read PDF" in str(exc)
    else:
        raise AssertionError("invalid PDF bytes should fail")



def test_visual_line_wraps_reflow_to_one_paragraph_without_changing_punctuation():
    raw = (
        'Then the enlightening being Avalokiteshvara said to the Buddha, "The Buddha speaks of ten stages of enlightening beings:\n'
        'the stage of intense joy, the stage of freedom from defilement, the stage of radiance,\n'
        'the stage of blazing intellect, the stage most difficult to conquer, the stage of presence,\n'
        'the stage of far going, the stage of immovability, the stage of good intellect, and the stage of cloud of teaching.\n'
        'You also speak of Buddhahood, making eleven stages. How many kinds of purity, in how many portions,\n'
        'are these stages contained in?"'
    )
    expected = (
        'Then the enlightening being Avalokiteshvara said to the Buddha, "The Buddha speaks of ten stages of enlightening beings: '
        'the stage of intense joy, the stage of freedom from defilement, the stage of radiance, '
        'the stage of blazing intellect, the stage most difficult to conquer, the stage of presence, '
        'the stage of far going, the stage of immovability, the stage of good intellect, and the stage of cloud of teaching. '
        'You also speak of Buddhahood, making eleven stages. How many kinds of purity, in how many portions, '
        'are these stages contained in?"'
    )
    assert normalize_manuscript_text(raw) == expected
    for punctuation in (':', ',', '.', '"', '?'):
        assert normalize_manuscript_text(raw).count(punctuation) == expected.count(punctuation)


def test_line_reflow_preserves_blank_paragraphs_and_visible_hyphens():
    raw = "A first line wraps-\ninto the next line.\n\nA second paragraph stays separate."
    assert normalize_manuscript_text(raw) == (
        "A first line wraps-into the next line.\n\nA second paragraph stays separate."
    )


def test_pdf_extraction_supports_page_ranges_and_records_page_provenance(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self, extraction_mode=None):
            return self.text

    class FakeReader:
        def __init__(self, stream, strict=False):
            self.pages = [
                FakePage("Ignored first page has enough text but is outside the range."),
                FakePage("A paragraph begins on the selected page and continues"),
                FakePage("on the next page with its punctuation preserved."),
            ]
            self.is_encrypted = False
            self.metadata = types.SimpleNamespace(title="Test source", author="Test author")

    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        types.SimpleNamespace(PdfReader=FakeReader),
    )
    result = extract_pdf_manuscript(
        b"%PDF-test-bytes",
        page_start=2,
        page_end=3,
    )
    assert result["text"] == (
        "A paragraph begins on the selected page and continues on the next page with its punctuation preserved."
    )
    assert result["selected_page_start"] == 2
    assert result["selected_page_end"] == 3
    assert result["selected_page_count"] == 2
    assert [item["page_number"] for item in result["page_spans"]] == [2, 3]
    assert all(item["char_end"] > item["char_start"] for item in result["page_spans"])
    assert result["pdf_metadata"]["title"] == "Test source"
    assert result["line_wraps_normalized"] is True


def test_scanned_pdf_with_no_selectable_text_is_rejected_with_ocr_guidance(monkeypatch):
    class FakePage:
        def extract_text(self, extraction_mode=None):
            return ""

    class FakeReader:
        def __init__(self, stream, strict=False):
            self.pages = [FakePage(), FakePage(), FakePage()]
            self.is_encrypted = False
            self.metadata = None

    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        types.SimpleNamespace(PdfReader=FakeReader),
    )
    try:
        extract_pdf_manuscript(b"%PDF-scanned")
    except ValueError as exc:
        assert "scanned/image-only" in str(exc)
        assert "OCR" in str(exc)
    else:
        raise AssertionError("image-only PDF should not be accepted as a complete text source")


def test_pdf_source_pages_are_attached_to_planned_chunks():
    source = "A first sentence is on page two. A second sentence continues on page three."
    split = source.index("A second")
    plan = plan_audiobook_chunks(
        source,
        source_pages=[
            {"page_number": 2, "char_start": 0, "char_end": split},
            {"page_number": 3, "char_start": split, "char_end": len(source)},
        ],
    )
    assert plan["source_pages"][0]["page_number"] == 2
    # This one chunk crosses the PDF page boundary, so both source pages are recorded.
    assert plan["chunks"][0]["source_pages"] == [2, 3]


def test_pdf_original_is_preserved_in_job_folder_and_delivery_bundle(tmp_path):
    import hashlib

    source = "Chapter 1\n\nThe cleaned PDF manuscript text."
    pdf_bytes = b"%PDF-original-source-for-test"
    pdf_report = {
        "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "selected_page_start": 1,
        "selected_page_end": 1,
    }
    plan = plan_audiobook_chunks(source)
    root, manifest, resumed = create_or_resume_job(
        tmp_path,
        "pdf-job",
        source_text=source,
        source_name="original.pdf",
        plan=plan,
        generation_config={"voice": "test"},
        base_seed=2,
        source_pdf_bytes=pdf_bytes,
        pdf_extraction_report=pdf_report,
    )
    assert not resumed
    assert (root / "source_original.pdf").read_bytes() == pdf_bytes
    assert manifest["source_pdf_sha256"] == pdf_report["pdf_sha256"]
    assert manifest["pdf_extraction_report"] == pdf_report

    for chunk in manifest["chunks"]:
        wav_path = root / chunk["wav_relpath"]
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, np.full(2400, 0.05, dtype=np.float32), 24000, subtype="PCM_16")
        from tools.longform_audiobook import sha256_file
        chunk.update({
            "status": "complete",
            "wav_sha256": sha256_file(wav_path),
            "duration_seconds": 0.1,
        })
    write_manifest(root, manifest)
    assemble_job_masters(root, manifest)
    build_audio_qa_report(root, manifest)
    bundle = create_job_bundle(root)
    with zipfile.ZipFile(bundle) as archive:
        assert "source_original.pdf" in archive.namelist()


def test_existing_job_remains_resumable_across_helper_commit_updates(tmp_path):
    source = "A sentence that remains the same."
    plan = plan_audiobook_chunks(source)
    root, manifest, resumed = create_or_resume_job(
        tmp_path, "helper-upgrade",
        source_text=source,
        source_name="book.txt",
        plan=plan,
        generation_config={"voice": "test", "helper_commit": "old-helper-sha"},
        base_seed=7,
    )
    assert not resumed
    _, updated_manifest, resumed = create_or_resume_job(
        tmp_path, "helper-upgrade",
        source_text=source,
        source_name="book.txt",
        plan=plan,
        generation_config={"voice": "test", "helper_commit": "new-helper-sha"},
        base_seed=7,
    )
    assert resumed
    assert updated_manifest["job_id"] == root.name


def test_legacy_full_source_job_resumes_with_new_default_selection_metadata(tmp_path):
    source = (
        "Chapter 1\n\nFirst chapter sentence.\n\n"
        "Chapter 2\n\nSecond chapter sentence."
    )
    plan = plan_audiobook_chunks(source)
    root, _, resumed = create_or_resume_job(
        tmp_path, "legacy-default-selection",
        source_text=source,
        source_name="book.txt",
        plan=plan,
        generation_config={"voice": "test"},
        base_seed=7,
    )
    assert not resumed

    _, manifest, resumed = create_or_resume_job(
        tmp_path, "legacy-default-selection",
        source_text=source,
        source_name="book.txt",
        plan=plan,
        generation_config={
            "voice": "test",
            "helper_commit": "new-helper",
            "selection_mode": "all",
            "selected_section_ids": ["S001", "S002"],
            "text_paragraph_start": 1,
            "text_paragraph_end": 0,
        },
        base_seed=7,
    )
    assert resumed
    assert manifest["job_id"] == root.name


def test_partial_section_or_paragraph_selection_remains_a_distinct_job_config(tmp_path):
    source = (
        "Chapter 1\n\nFirst chapter sentence.\n\n"
        "Chapter 2\n\nSecond chapter sentence."
    )
    plan = plan_audiobook_chunks(source)
    create_or_resume_job(
        tmp_path, "partial-selection-identity",
        source_text=source,
        source_name="book.txt",
        plan=plan,
        generation_config={"voice": "test"},
        base_seed=7,
    )
    try:
        create_or_resume_job(
            tmp_path, "partial-selection-identity",
            source_text=source,
            source_name="book.txt",
            plan=plan,
            generation_config={
                "voice": "test",
                "selection_mode": "selected_sections",
                "selected_section_ids": ["S002"],
            },
            base_seed=7,
        )
    except ValueError as exc:
        assert "different voice/settings/reference configuration" in str(exc)
    else:
        raise AssertionError("a partial selection must not resume a full-source job")


def test_selected_sections_assemble_into_a_named_combined_master_and_bundle(tmp_path):
    from tools.longform_audiobook import filter_plan_sections, sha256_file

    source = (
        "Chapter 1\n\nThis first chapter is intentionally excluded.\n\n"
        "Chapter 2\n\nThis second chapter should be spoken."
    )
    full_plan = plan_audiobook_chunks(source)
    selected_plan = filter_plan_sections(full_plan, ["S002"])
    root, manifest, resumed = create_or_resume_job(
        tmp_path,
        "selected-sections",
        source_text=source,
        source_name="book.txt",
        plan=selected_plan,
        generation_config={"voice": "test"},
        base_seed=4,
    )
    assert not resumed
    for chunk in manifest["chunks"]:
        wav_path = root / chunk["wav_relpath"]
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, np.full(2400, 0.05, dtype=np.float32), 24000, subtype="PCM_16")
        chunk.update({
            "status": "complete",
            "wav_sha256": sha256_file(wav_path),
            "duration_seconds": 0.1,
        })
    write_manifest(root, manifest)

    artifacts = assemble_job_masters(root, manifest)
    master_path = root / artifacts["whole_book_master"]["relpath"]
    assert master_path.name == "selected_sections_master.wav"
    assert artifacts["whole_book_master"]["is_full_source"] is False
    assert manifest["plan"]["selected_section_ids"] == ["S002"]

    bundle = create_job_bundle(root)
    with zipfile.ZipFile(bundle) as archive:
        assert "selected_sections_master.wav" in archive.namelist()
        assert "whole_book_master.wav" not in archive.namelist()


def test_safe_output_name_rejects_paths_and_reserved_filenames():
    assert safe_output_name("Buddhist Yoga") == "Buddhist Yoga"
    assert safe_output_name("  ") is None
    for invalid in ("../escape", "folder/name", r"folder\\name", "bad:name", "CON", "LPT1.wav", "name."):
        try:
            safe_output_name(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe output name accepted: {invalid}")


def test_custom_output_name_prefixes_chunks_sections_master_and_bundle(tmp_path):
    from tools.longform_audiobook import sha256_file

    source = (
        "Chapter One\\n\\nFirst chapter sentence.\\n\\n"
        "Chapter Two\\n\\nSecond chapter sentence."
    )
    plan = plan_audiobook_chunks(source)
    root, manifest, resumed = create_or_resume_job(
        tmp_path,
        "custom-output",
        source_text=source,
        source_name="book.txt",
        plan=plan,
        generation_config={"voice": "test"},
        base_seed=19,
        output_name="Buddhist Yoga",
    )
    assert not resumed
    assert manifest["output_name"] == "Buddhist Yoga"
    for chunk in manifest["chunks"]:
        assert Path(chunk["wav_relpath"]).name.startswith("Buddhist Yoga_c")
        wav_path = root / chunk["wav_relpath"]
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, np.full(2400, 0.05, dtype=np.float32), 24000, subtype="PCM_16")
        chunk.update({
            "status": "complete",
            "wav_sha256": sha256_file(wav_path),
            "duration_seconds": 0.1,
        })
    write_manifest(root, manifest)

    artifacts = assemble_job_masters(root, manifest)
    assert all(
        Path(item["relpath"]).name.startswith("Buddhist Yoga_s")
        for item in artifacts["section_masters"]
    )
    master_path = root / artifacts["whole_book_master"]["relpath"]
    assert master_path.name == "Buddhist Yoga_whole_book_master.wav"

    bundle = create_job_bundle(root)
    assert bundle.name == "Buddhist Yoga_audiobook_delivery_bundle.zip"
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert master_path.name in names
    assert any(name.startswith("chunks/Buddhist Yoga_c") for name in names)
    assert any(name.startswith("sections/Buddhist Yoga_s") for name in names)

    _, resumed_manifest, resumed = create_or_resume_job(
        tmp_path,
        "custom-output",
        source_text=source,
        source_name="book.txt",
        plan=plan,
        generation_config={"voice": "test"},
        base_seed=19,
        output_name=None,
    )
    assert resumed
    assert resumed_manifest["output_name"] == "Buddhist Yoga"

    try:
        create_or_resume_job(
            tmp_path,
            "custom-output",
            source_text=source,
            source_name="book.txt",
            plan=plan,
            generation_config={"voice": "test"},
            base_seed=19,
            output_name="Different Name",
        )
    except ValueError as exc:
        assert "different custom output name" in str(exc)
    else:
        raise AssertionError("a resumed job must not silently change its output filenames")


def test_text_paragraph_range_selects_one_based_inclusive_paragraphs_after_reflow():
    from tools.longform_audiobook import select_text_paragraph_range

    raw = (
        "First paragraph wraps over\nmultiple visual lines.\n\n"
        "Second paragraph is separate.\n\n"
        "Third paragraph is also separate."
    )
    selected, report = select_text_paragraph_range(raw, 2, 2)
    assert selected == "Second paragraph is separate."
    assert report == {
        "total_paragraphs": 3,
        "selected_paragraph_start": 2,
        "selected_paragraph_end": 2,
        "selected_paragraph_count": 1,
    }

    selected_all, report_all = select_text_paragraph_range(raw, 1, 0)
    assert selected_all.startswith("First paragraph wraps over multiple visual lines.")
    assert report_all["selected_paragraph_end"] == 3
    assert report_all["selected_paragraph_count"] == 3


def test_text_paragraph_range_rejects_invalid_or_empty_ranges():
    from tools.longform_audiobook import select_text_paragraph_range

    for start, end in ((0, 1), (2, 1), (1, 99)):
        try:
            select_text_paragraph_range("One.\n\nTwo.", start, end)
        except ValueError as exc:
            assert "Invalid text paragraph range" in str(exc)
        else:
            raise AssertionError("invalid paragraph range should fail")
    try:
        select_text_paragraph_range("   ")
    except ValueError as exc:
        assert "no paragraphs" in str(exc)
    else:
        raise AssertionError("empty text should fail")
