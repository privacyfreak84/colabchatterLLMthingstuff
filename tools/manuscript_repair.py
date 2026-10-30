#!/usr/bin/env python3
"""Build reversible, provenance-aware manuscript repair proposals.

The source manuscript and existing reconnaissance files are never modified.
Cross-page joins are proposed in separate files. Same-page candidates are
reported for human review and are never applied automatically.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

ALGORITHM_VERSION = "0.1.0"
_TERMINAL = re.compile(r"[.!?;:][\"'”’)]*\s*$")
_LOWER_OR_QUOTE = re.compile(r"^[a-z'\"“‘(\[]")
_TRAILING_HYPHEN = re.compile(r"[-‐‑]\s*$")

# Additional terms missed by the capitalization heuristic. These are inventory
# entries only. No respellings or language routing are inferred.
EXTRA_TERMS = {
    "Buddha": "Buddhist name/title",
    "nirvana": "Buddhist/Sanskrit term",
    "samsara": "Buddhist/Sanskrit term",
    "sutra": "Buddhist text term",
    "vijnana": "Sanskrit transliteration",
    "Subhuti": "Buddhist name",
    "Avatamsaka": "Sanskrit scripture name",
}

CSV_FIELDS = [
    "candidate_id", "scope", "section_id", "section_title", "from_page",
    "to_page", "previous_block_id", "next_block_id", "join_action",
    "confidence", "status", "reason", "previous_text", "next_text",
    "proposed_join", "application", "application_note",
]
PRON_FIELDS = [
    "source_term", "category", "occurrences", "pages", "example_context",
    "spoken_form", "pronunciation_strategy", "target_language_or_script",
    "confidence", "notes", "approval_status",
]


def _normal_text(value: Any) -> str:
    return str(value or "").strip()


def _is_heading(block: dict[str, Any]) -> bool:
    kind = str(block.get("block_kind", "")).strip().lower()
    return kind in {"heading", "title", "page_number", "footer", "header"}


def _make_candidate(
    previous: dict[str, Any],
    following: dict[str, Any],
    section: dict[str, Any],
) -> dict[str, Any] | None:
    left = _normal_text(previous.get("text"))
    right = _normal_text(following.get("text"))
    if not left or not right or _is_heading(previous) or _is_heading(following):
        return None
    if not _LOWER_OR_QUOTE.match(right) or _TERMINAL.search(left):
        return None

    p1, p2 = int(previous.get("page", 0)), int(following.get("page", 0))
    if p2 == p1:
        scope, status, confidence = "within_page", "manual_review_required", "review_only"
    elif p2 == p1 + 1:
        scope, status, confidence = "cross_page", "proposed_pending_approval", "high_heuristic"
    else:
        return None

    dehyphenate = bool(_TRAILING_HYPHEN.search(left))
    action = "dehyphenate_and_join" if dehyphenate else "join_with_space"
    joined = (
        re.sub(r"[-‐‑]\s*$", "", left) + right
        if dehyphenate else left + " " + right
    )
    if scope == "within_page":
        reason = "Same-page blocks appear to continue a sentence; reported only because block boundaries may be intentional."
    elif dehyphenate:
        reason = "Previous block ends in a line-break hyphen and the next begins lowercase."
    else:
        reason = "Previous block lacks terminal punctuation and the next begins lowercase."

    return {
        "scope": scope,
        "section_id": section["section_id"],
        "section_title": section.get("title", ""),
        "from_page": p1,
        "to_page": p2,
        "previous_block_id": previous.get("block_id", ""),
        "next_block_id": following.get("block_id", ""),
        "join_action": action,
        "confidence": confidence,
        "status": status,
        "reason": reason,
        "previous_text": left,
        "next_text": right,
        "proposed_join": joined,
    }


def _apply_cross_page_proposals(
    source_text: str, candidates: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Apply only unique exact block-pair matches to a separate proposal copy."""
    revised, applied = source_text, []
    for candidate in candidates:
        if candidate["scope"] != "cross_page":
            continue
        left, right = candidate["previous_text"], candidate["next_text"]
        patterns = [left + separator + right for separator in ("\n\n", "\n")]
        counts = [(pattern, revised.count(pattern)) for pattern in patterns]
        matches = [pattern for pattern, count in counts if count == 1]
        total = sum(count for _, count in counts)
        if len(matches) == 1 and total == 1:
            revised = revised.replace(matches[0], candidate["proposed_join"], 1)
            candidate["application"] = "applied_to_proposal"
            candidate["application_note"] = "Unique exact adjacent-block match found in the source track."
            applied.append(candidate)
        else:
            candidate["application"] = "not_applied"
            candidate["application_note"] = (
                f"Expected one exact adjacent-block match across supported separators; found {total}. "
                "Retained for manual inspection."
            )
    return revised, applied


def _csv_write(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pronunciation_draft(
    blocks: list[dict[str, Any]], candidates_path: Path
) -> list[dict[str, Any]]:
    candidate_terms: dict[str, str] = {}
    if candidates_path.is_file():
        data = json.loads(candidates_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("candidates", [])
        for item in data if isinstance(data, list) else []:
            term = _normal_text(item.get("term"))
            language = _normal_text(item.get("language_guess"))
            if term and "sanskrit" in language.lower():
                candidate_terms[term] = language

    terms = dict(candidate_terms)
    terms.update(EXTRA_TERMS)
    rows = []
    for term, category in sorted(terms.items(), key=lambda item: item[0].casefold()):
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        matches = [pattern.search(_normal_text(block.get("text"))) for block in blocks]
        occurrences = sum(len(pattern.findall(_normal_text(block.get("text")))) for block in blocks)
        if not occurrences:
            continue
        pages = sorted({
            int(block.get("page", 0))
            for block, match in zip(blocks, matches)
            if match
        })
        context = ""
        for block, match in zip(blocks, matches):
            if match:
                text = _normal_text(block.get("text"))
                lo, hi = max(0, match.start() - 90), min(len(text), match.end() + 130)
                context = text[lo:hi].replace("\n", " ")
                break
        rows.append({
            "source_term": term,
            "category": category,
            "occurrences": occurrences,
            "pages": "; ".join(map(str, pages)),
            "example_context": context,
            "spoken_form": "",
            "pronunciation_strategy": "",
            "target_language_or_script": "",
            "confidence": "awaiting_audition",
            "notes": "Inventory only. Do not alter the manuscript before an auditioned form is approved.",
            "approval_status": "unreviewed",
        })
    return rows


def build_review_bundle(
    *,
    blocks_jsonl: str | Path,
    section_manifest: str | Path,
    main_manuscript: str | Path,
    notes_manuscript: str | Path,
    pronunciation_candidates: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write proposed narration tracks and review files without overwriting inputs."""
    blocks_path, manifest_path = Path(blocks_jsonl), Path(section_manifest)
    main_path, notes_path = Path(main_manuscript), Path(notes_manuscript)
    candidates_path, out = Path(pronunciation_candidates), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    blocks = [
        json.loads(line)
        for line in blocks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {block["block_id"]: block for block in blocks}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sections = manifest.get("sections", [])
    all_candidates: list[dict[str, Any]] = []

    for section in sections:
        ordered, missing = [], []
        for block_id in section.get("source_block_ids", []):
            if block_id not in by_id:
                missing.append(block_id)
            else:
                ordered.append(by_id[block_id])
        if missing:
            raise ValueError(f"Section {section.get('section_id')} has missing blocks: {missing[:5]}")
        for previous, following in zip(ordered, ordered[1:]):
            candidate = _make_candidate(previous, following, section)
            if candidate:
                all_candidates.append(candidate)

    for index, candidate in enumerate(all_candidates, start=1):
        candidate["candidate_id"] = f"J{index:03d}"

    main_source = main_path.read_text(encoding="utf-8")
    notes_source = notes_path.read_text(encoding="utf-8")
    main_candidates = [c for c in all_candidates if c["section_id"] != "back-01"]
    notes_candidates = [c for c in all_candidates if c["section_id"] == "back-01"]
    main_revised, main_applied = _apply_cross_page_proposals(main_source, main_candidates)
    notes_revised, notes_applied = _apply_cross_page_proposals(notes_source, notes_candidates)

    main_out = out / "main_manuscript_repaired_proposed.txt"
    notes_out = out / "notes_manuscript_repaired_proposed.txt"
    main_out.write_text(main_revised, encoding="utf-8")
    notes_out.write_text(notes_revised, encoding="utf-8")

    review_csv = out / "manuscript_repair_review.csv"
    _csv_write(review_csv, CSV_FIELDS, all_candidates)
    title_blocks = sorted(
        [b for b in blocks if int(b.get("page", 0)) == 1],
        key=lambda b: int(b.get("order", 0)),
    )
    title_text = "\n\n".join(
        _normal_text(b.get("text")) for b in title_blocks if _normal_text(b.get("text"))
    )
    title_path = out / "title_page_opening_proposal.txt"
    title_path.write_text(title_text + ("\n" if title_text else ""), encoding="utf-8")

    pron_rows = _pronunciation_draft(blocks, candidates_path)
    pron_path = out / "pronunciation_dictionary_draft.csv"
    _csv_write(pron_path, PRON_FIELDS, pron_rows)

    summary = {
        "algorithm_version": ALGORITHM_VERSION,
        "source_manifest_sha256": manifest.get("source_sha256", ""),
        "source_block_count": len(blocks),
        "section_count": len(sections),
        "candidate_count": len(all_candidates),
        "cross_page_candidates": sum(c["scope"] == "cross_page" for c in all_candidates),
        "within_page_review_candidates": sum(c["scope"] == "within_page" for c in all_candidates),
        "main_track_proposals_applied": len(main_applied),
        "notes_track_proposals_applied": len(notes_applied),
        "proposals_not_applied": sum(c.get("application") == "not_applied" for c in all_candidates),
        "pronunciation_dictionary_entries": len(pron_rows),
        "title_page_blocks": len(title_blocks),
        "source_files_preserved": True,
        "approval_required_before_synthesis": True,
        "files": {
            "main_proposed": main_out.name,
            "notes_proposed": notes_out.name,
            "join_review_csv": review_csv.name,
            "pronunciation_dictionary_draft": pron_path.name,
            "title_page_opening_proposal": title_path.name,
        },
    }
    (out / "manuscript_repair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    detailed = dict(summary)
    detailed["join_candidates"] = all_candidates
    (out / "manuscript_repair_review.json").write_text(
        json.dumps(detailed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
