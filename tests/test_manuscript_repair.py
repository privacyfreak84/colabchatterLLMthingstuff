import json
from pathlib import Path

from tools.manuscript_repair import _apply_cross_page_proposals, _make_candidate, build_review_bundle


def block(block_id, page, order, text, kind="text"):
    return {"block_id": block_id, "page": page, "order": order, "text": text, "block_kind": kind}


def section(block_ids, section_id="chapter-01"):
    return {"section_id": section_id, "title": "Chapter 1. Introduction", "source_block_ids": block_ids}


def test_cross_page_lowercase_continuation_is_proposed():
    candidate = _make_candidate(block("p010-b007", 10, 7, "beyond"),
                                block("p011-b001", 11, 1, "words, they temporarily set up names"),
                                section([]))
    assert candidate["scope"] == "cross_page"
    assert candidate["status"] == "proposed_pending_approval"
    assert candidate["proposed_join"] == "beyond words, they temporarily set up names"


def test_line_break_hyphen_is_removed_in_proposal():
    candidate = _make_candidate(block("p001-b001", 1, 1, "inter-"),
                                block("p002-b001", 2, 1, "dependent origination"), section([]))
    assert candidate["join_action"] == "dehyphenate_and_join"
    assert candidate["proposed_join"] == "interdependent origination"


def test_terminal_punctuation_prevents_automatic_join():
    assert _make_candidate(block("p001-b001", 1, 1, "This sentence ends."),
                           block("p002-b001", 2, 1, "another paragraph begins"), section([])) is None


def test_heading_blocks_are_not_joined():
    assert _make_candidate(block("p001-b001", 1, 1, "Chapter heading", "heading"),
                           block("p001-b002", 1, 2, "the body begins"), section([])) is None


def test_same_page_candidate_is_review_only():
    candidate = _make_candidate(block("p001-b001", 1, 1, "a thought which continues"),
                                block("p001-b002", 1, 2, "into this block"), section([]))
    assert candidate["scope"] == "within_page"
    assert candidate["status"] == "manual_review_required"


def test_exact_unique_pair_is_applied_only_to_proposal():
    candidate = {"scope": "cross_page", "previous_text": "before", "next_text": "after",
                 "proposed_join": "before after"}
    revised, applied = _apply_cross_page_proposals("Header\n\nbefore\n\nafter", [candidate])
    assert revised == "Header\n\nbefore after"
    assert len(applied) == 1


def test_ambiguous_pair_is_not_applied():
    candidate = {"scope": "cross_page", "previous_text": "before", "next_text": "after",
                 "proposed_join": "before after"}
    source = "before\n\nafter\n\nbefore\n\nafter"
    revised, applied = _apply_cross_page_proposals(source, [candidate])
    assert revised == source
    assert not applied
    assert candidate["application"] == "not_applied"


def test_bundle_preserves_sources_and_writes_review_outputs(tmp_path: Path):
    blocks = [
        block("p001-b001", 1, 1, "A Book", "heading"),
        block("p001-b002", 1, 2, "Translated by Someone", "text"),
        block("p002-b001", 2, 1, "Preface begins", "heading"),
        block("p002-b002", 2, 2, "The sentence continues"),
        block("p003-b001", 3, 1, "on the next page."),
        block("p116-b001", 116, 1, "Notes", "heading"),
        block("p116-b002", 116, 2, "A note ends."),
    ]
    block_path = tmp_path / "blocks.jsonl"
    block_path.write_text("\n".join(json.dumps(x) for x in blocks), encoding="utf-8")
    manifest_path = tmp_path / "sections.json"
    manifest_path.write_text(json.dumps({"source_sha256": "abc", "sections": [
        section(["p002-b001", "p002-b002", "p003-b001"]),
        {"section_id": "back-01", "title": "Notes",
         "source_block_ids": ["p116-b001", "p116-b002"]},
    ]}), encoding="utf-8")
    main_path = tmp_path / "main.txt"
    main_original = "Translator's Preface\n\nPreface begins\n\nThe sentence continues\n\non the next page."
    main_path.write_text(main_original, encoding="utf-8")
    notes_path = tmp_path / "notes.txt"
    notes_original = "Notes\n\nA note ends."
    notes_path.write_text(notes_original, encoding="utf-8")
    pron_path = tmp_path / "pron.json"
    pron_path.write_text("[]", encoding="utf-8")
    out = tmp_path / "out"

    summary = build_review_bundle(blocks_jsonl=block_path, section_manifest=manifest_path,
                                  main_manuscript=main_path, notes_manuscript=notes_path,
                                  pronunciation_candidates=pron_path, output_dir=out)

    assert summary["cross_page_candidates"] == 1
    assert (out / "main_manuscript_repaired_proposed.txt").read_text(encoding="utf-8") == (
        "Translator's Preface\n\nPreface begins\n\nThe sentence continues on the next page."
    )
    assert main_path.read_text(encoding="utf-8") == main_original
    assert notes_path.read_text(encoding="utf-8") == notes_original
    assert (out / "title_page_opening_proposal.txt").read_text(encoding="utf-8").startswith("A Book")
    assert (out / "pronunciation_dictionary_draft.csv").is_file()
    assert (out / "manuscript_repair_review.csv").is_file()
 

def test_quoted_lowercase_continuation_is_proposed():
    candidate = _make_candidate(
        block("p002-b006", 2, 6, 'neither "created" nor'),
        block("p003-b001", 3, 1, '"uncreated," neither mundane nor supernal.'),
        section([]),
    )
    assert candidate is not None
    assert candidate["proposed_join"] == 'neither "created" nor "uncreated," neither mundane nor supernal.'


def test_quoted_uppercase_new_sentence_is_not_auto_joined():
    candidate = _make_candidate(
        block("p002-b006", 2, 6, "The previous clause continues"),
        block("p003-b001", 3, 1, '"Another paragraph begins here.'),
        section([]),
    )
    assert candidate is None
