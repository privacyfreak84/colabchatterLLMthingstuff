import json
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.longform_audiobook import (
    append_job_event,
    assemble_job_masters,
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


def test_streaming_assembly_and_full_job_bundle(tmp_path):
    source = "Preface\n\nFirst sentence. Second sentence.\n\nChapter One\n\nFinal paragraph."
    plan = plan_audiobook_chunks(source, max_chars=120)
    root, manifest, _ = create_or_resume_job(
        tmp_path, "assembly", source_text=source, source_name="book.txt",
        plan=plan, generation_config={"voice": "test"}, base_seed=12,
    )

    for chunk in manifest["chunks"]:
        wav_path = root / chunk["wav_relpath"]
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, np.full(2400, 0.1, dtype=np.float32), 24000, subtype="PCM_16")
        from tools.longform_audiobook import sha256_file
        chunk["status"] = "complete"
        chunk["wav_sha256"] = sha256_file(wav_path)
        chunk["duration_seconds"] = 0.1
    write_manifest(root, manifest)

    artifacts = assemble_job_masters(root, manifest)
    assert manifest["status"] == "complete"
    assert Path(root / artifacts["whole_book_master"]["relpath"]).is_file()
    assert len(artifacts["section_masters"]) == 2
    for section in artifacts["section_masters"]:
        assert Path(root / section["relpath"]).is_file()

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
    assert "whole_book_master.wav" in names
    assert "qa_report.json" in names
    assert any(name.startswith("chunks/") for name in names)


def test_assemble_wav_rejects_mismatched_input_format(tmp_path):
    stereo = tmp_path / "stereo.wav"
    sf.write(stereo, np.zeros((2400, 2), dtype=np.float32), 24000, subtype="PCM_16")
    try:
        assemble_wav([stereo], [0.0], tmp_path / "out.wav")
    except ValueError as exc:
        assert "unexpected format" in str(exc)
    else:
        raise AssertionError("stereo chunk should be rejected")



def test_force_rerender_preserves_old_audio_and_assigns_new_seed(tmp_path):
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
    assert not wav_path.exists()
    previous = chunk["previous_renders"][0]
    previous_path = root / previous["relpath"]
    assert previous_path.is_file()
    assert previous["wav_sha256"] == original_hash
    assert sha256_file(previous_path) == original_hash
    assert previous["was_verified_complete"] is True

    try:
        force_rerender_chunks(root, manifest, ["C99999"])
    except ValueError as exc:
        assert "Unknown chunk ID" in str(exc)
    else:
        raise AssertionError("unknown rerender chunk ID should fail")


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
    assert extracted["page_count"] == 2
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
