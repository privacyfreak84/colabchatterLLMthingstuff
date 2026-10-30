#!/usr/bin/env python3
"""Record approved cross-page joins and prepare pronunciation audition prompts.

This stage copies the already-reviewed proposed tracks into explicitly approved
tracks. It never edits the extracted source manuscripts or the proposal files.
Same-page candidates remain excluded from automatic repair.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

APPROVAL_SCHEMA_VERSION = "1.0.0"
AUDITION_FIELDS = [
    "audition_id",
    "source_term",
    "category",
    "occurrences",
    "pages",
    "isolated_prompt",
    "carrier_prompt",
    "context_prompt",
    "chosen_spoken_form",
    "pronunciation_strategy",
    "review_status",
    "review_notes",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDITION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sentence_containing(text: str, term: str) -> str:
    """Return the complete punctuation-bounded sentence containing the term."""
    match = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    starts = list(re.finditer(r"[.!?][\"'”’)]*\s+", text[:match.start()]))
    start = starts[-1].end() if starts else 0
    end_match = re.search(r"[.!?][\"'”’)]*(?:\s+|$)", text[match.end():])
    end = match.end() + end_match.end() if end_match else len(text)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def _first_context(blocks: list[dict[str, Any]], term: str) -> str:
    for block in blocks:
        source = str(block.get("text", "")).strip()
        sentence = _sentence_containing(source, term)
        if sentence:
            return sentence
    return ""


def approve_repair_bundle(
    *,
    repair_summary: str | Path,
    repair_review: str | Path,
    repair_review_csv: str | Path,
    main_proposed: str | Path,
    notes_proposed: str | Path,
    pronunciation_draft: str | Path,
    source_blocks: str | Path,
    output_dir: str | Path,
    expected_cross_page_joins: int = 48,
    expected_manual_review_items: int = 1,
) -> dict[str, Any]:
    """Validate the approved repair set, copy final tracks, and build audition prompts.

    Approval is narrowly scoped to every successfully applied cross-page
    candidate. Within-page candidates are never included in the approval set.
    """
    summary_path = Path(repair_summary)
    review_path = Path(repair_review)
    review_csv_path = Path(repair_review_csv)
    main_source = Path(main_proposed)
    notes_source = Path(notes_proposed)
    pron_path = Path(pronunciation_draft)
    blocks_path = Path(source_blocks)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    candidates = review.get("join_candidates", [])
    csv_rows = _read_csv(review_csv_path)
    if not summary.get("source_files_preserved") or not summary.get("approval_required_before_synthesis"):
        raise ValueError("Repair summary does not confirm source preservation and approval gating.")
    if review.get("source_manifest_sha256") != summary.get("source_manifest_sha256"):
        raise ValueError("Repair summary and detailed review have different source hashes.")
    if review.get("candidate_count") != summary.get("candidate_count"):
        raise ValueError("Repair summary and detailed review candidate counts differ.")
    if len(candidates) != summary.get("candidate_count"):
        raise ValueError("Detailed review candidate list is incomplete.")
    if len(csv_rows) != len(candidates):
        raise ValueError("CSV review and JSON review candidate counts differ.")

    cross_page = [c for c in candidates if c.get("scope") == "cross_page"]
    within_page = [c for c in candidates if c.get("scope") == "within_page"]
    if len(cross_page) != expected_cross_page_joins:
        raise ValueError(
            f"Expected {expected_cross_page_joins} cross-page joins; found {len(cross_page)}."
        )
    if len(within_page) != expected_manual_review_items:
        raise ValueError(
            f"Expected {expected_manual_review_items} same-page review items; found {len(within_page)}."
        )

    candidate_ids = [c.get("candidate_id") for c in candidates]
    if len(candidate_ids) != len(set(candidate_ids)) or any(not x for x in candidate_ids):
        raise ValueError("Candidate IDs are missing or duplicated.")

    for candidate in cross_page:
        if candidate.get("application") != "applied_to_proposal":
            raise ValueError(
                f"{candidate.get('candidate_id')} was not uniquely applied to its proposal copy."
            )
        if candidate.get("status") != "proposed_pending_approval":
            raise ValueError(f"{candidate.get('candidate_id')} has an unexpected review status.")
    for candidate in within_page:
        if candidate.get("application") not in (None, ""):
            raise ValueError(
                f"{candidate.get('candidate_id')} must remain unapplied because it is within-page."
            )
        if candidate.get("status") != "manual_review_required":
            raise ValueError(f"{candidate.get('candidate_id')} is not marked for manual review.")

    # Cross-check the human-readable CSV against the machine-readable review.
    json_by_id = {c["candidate_id"]: c for c in candidates}
    csv_by_id = {r.get("candidate_id"): r for r in csv_rows}
    if set(csv_by_id) != set(json_by_id):
        raise ValueError("CSV and JSON candidate IDs do not match.")
    for candidate_id, item in json_by_id.items():
        row = csv_by_id[candidate_id]
        for field in ("scope", "application", "status", "previous_block_id", "next_block_id"):
            if str(item.get(field, "")) != str(row.get(field, "")):
                raise ValueError(f"CSV/JSON mismatch for {candidate_id}, field {field}.")

    main_text = main_source.read_text(encoding="utf-8")
    notes_text = notes_source.read_text(encoding="utf-8")
    main_approved = out / "main_manuscript_approved.txt"
    notes_approved = out / "notes_manuscript_approved.txt"
    main_approved.write_text(main_text, encoding="utf-8")
    notes_approved.write_text(notes_text, encoding="utf-8")

    # The 48 accepted joins are exactly the cross-page candidates. The single
    # same-page glossary boundary is explicitly excluded rather than modified.
    approved_ids = [c["candidate_id"] for c in cross_page]
    excluded = [
        {
            "candidate_id": c["candidate_id"],
            "reason": "Same-page glossary boundary; retain as separate entries.",
            "from_page": c.get("from_page"),
            "to_page": c.get("to_page"),
            "previous_block_id": c.get("previous_block_id"),
            "next_block_id": c.get("next_block_id"),
        }
        for c in within_page
    ]

    blocks = [
        json.loads(line)
        for line in blocks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pronunciation_rows = _read_csv(pron_path)
    if not pronunciation_rows:
        raise ValueError("Pronunciation draft is empty.")
    audition_rows: list[dict[str, Any]] = []
    for index, item in enumerate(pronunciation_rows, start=1):
        term = item.get("source_term", "").strip()
        if not term:
            raise ValueError("Pronunciation draft contains an empty source term.")
        context = _first_context(blocks, term)
        if not context:
            raise ValueError(f"No source context found for pronunciation term {term!r}.")
        audition_rows.append({
            "audition_id": f"P{index:03d}",
            "source_term": term,
            "category": item.get("category", ""),
            "occurrences": item.get("occurrences", ""),
            "pages": item.get("pages", ""),
            "isolated_prompt": f"{term}.",
            "carrier_prompt": f"Please pronounce the name or term {term} clearly.",
            "context_prompt": context,
            "chosen_spoken_form": "",
            "pronunciation_strategy": "",
            "review_status": "awaiting_audition",
            "review_notes": "Baseline uses the exact source spelling. No phonetic respelling has been assumed.",
        })

    audition_csv = out / "pronunciation_audition_sheet.csv"
    _write_csv(audition_csv, audition_rows)
    audition_text = [
        "# Buddhist Yoga: pronunciation audition prompts",
        "",
        "These prompts are for short, separate listening tests, not manuscript text.",
        "Each item has an isolated form, a neutral carrier phrase, and one exact source-context sentence.",
        "The source spelling is preserved in all three. Record a chosen spoken form only after listening.",
        "",
    ]
    for item in audition_rows:
        audition_text.extend([
            f"## {item['audition_id']}: {item['source_term']}",
            f"Category: {item['category']}",
            f"Occurrences: {item['occurrences']} | PDF pages: {item['pages']}",
            f"Isolated: {item['isolated_prompt']}",
            f"Carrier: {item['carrier_prompt']}",
            f"Context: {item['context_prompt']}",
            "Decision: ______________________________",
            "",
        ])
    audition_txt = out / "pronunciation_audition_prompts.txt"
    audition_txt.write_text("\n".join(audition_text).rstrip() + "\n", encoding="utf-8")

    approval_record = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "approval_status": "approved_for_manuscript_stage",
        "approval_scope": "all uniquely applied cross-page continuation proposals",
        "source_pdf_sha256": summary.get("source_manifest_sha256", ""),
        "source_block_count": summary.get("source_block_count"),
        "accepted_cross_page_join_count": len(approved_ids),
        "accepted_candidate_ids": approved_ids,
        "excluded_manual_review_candidates": excluded,
        "source_manuscripts_overwritten": False,
        "proposed_manuscripts_overwritten": False,
        "approved_tracks_are_exact_copies_of_proposed_tracks": True,
        "synthesis_authorized": False,
        "files": {
            "main_approved": main_approved.name,
            "notes_approved": notes_approved.name,
            "pronunciation_audition_csv": audition_csv.name,
            "pronunciation_audition_prompts": audition_txt.name,
        },
        "sha256": {
            "main_approved": _sha256(main_approved),
            "notes_approved": _sha256(notes_approved),
            "pronunciation_audition_csv": _sha256(audition_csv),
            "pronunciation_audition_prompts": _sha256(audition_txt),
        },
    }
    record_path = out / "manuscript_approval_record.json"
    record_path.write_text(
        json.dumps(approval_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "approval_status": approval_record["approval_status"],
        "accepted_cross_page_joins": len(approved_ids),
        "excluded_manual_review_items": len(excluded),
        "excluded_manual_review_candidate_ids": [item["candidate_id"] for item in excluded],
        "accepted_candidate_ids": approved_ids,
        "pronunciation_audition_items": len(audition_rows),
        "source_manuscripts_overwritten": False,
        "synthesis_authorized": False,
        "files": approval_record["files"],
    }
