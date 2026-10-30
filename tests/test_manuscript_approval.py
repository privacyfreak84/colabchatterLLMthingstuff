import csv
import json
from pathlib import Path

import pytest

from tools.manuscript_approval import approve_repair_bundle


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_bundle(root: Path, bad_application: bool = False) -> dict[str, Path]:
    cross = {
        "candidate_id": "J001",
        "scope": "cross_page",
        "status": "proposed_pending_approval",
        "application": "not_applied" if bad_application else "applied_to_proposal",
        "previous_block_id": "p002-b001",
        "next_block_id": "p003-b001",
        "from_page": 2,
        "to_page": 3,
    }
    same = {
        "candidate_id": "J002",
        "scope": "within_page",
        "status": "manual_review_required",
        "application": "",
        "previous_block_id": "p119-b006",
        "next_block_id": "p119-b007",
        "from_page": 119,
        "to_page": 119,
    }
    summary = {
        "source_files_preserved": True,
        "approval_required_before_synthesis": True,
        "source_manifest_sha256": "abc123",
        "candidate_count": 2,
        "source_block_count": 3,
    }
    review = {**summary, "join_candidates": [cross, same]}
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "review.json").write_text(json.dumps(review), encoding="utf-8")
    fields = [
        "candidate_id", "scope", "status", "application",
        "previous_block_id", "next_block_id",
    ]
    rows = [{field: str(item.get(field, "")) for field in fields} for item in (cross, same)]
    write_csv(root / "review.csv", fields, rows)

    main = root / "main_proposed.txt"
    notes = root / "notes_proposed.txt"
    main.write_text("Preface\n\nA sentence continues across pages.", encoding="utf-8")
    notes.write_text("Notes\n\nGlossary entries remain separate.", encoding="utf-8")
    (root / "pron.csv").write_text(
        "\ufeffsource_term,category,occurrences,pages,spoken_form,approval_status\n"
        "Maitreya,Sanskrit name,2,3;4,,unreviewed\n",
        encoding="utf-8",
    )
    (root / "blocks.jsonl").write_text(
        "\n".join([
            json.dumps({"block_id": "a", "page": 3,
                        "text": "Then Maitreya said to the Buddha, "
                                "\"Please explain the practice.\""}),
            json.dumps({"block_id": "b", "page": 4,
                        "text": "The Buddha replied to Maitreya."}),
        ]),
        encoding="utf-8",
    )
    return {
        "summary": root / "summary.json",
        "review": root / "review.json",
        "csv": root / "review.csv",
        "main": main,
        "notes": notes,
        "pron": root / "pron.csv",
        "blocks": root / "blocks.jsonl",
    }


def run_approval(paths: dict[str, Path], out: Path) -> dict:
    return approve_repair_bundle(
        repair_summary=paths["summary"],
        repair_review=paths["review"],
        repair_review_csv=paths["csv"],
        main_proposed=paths["main"],
        notes_proposed=paths["notes"],
        pronunciation_draft=paths["pron"],
        source_blocks=paths["blocks"],
        output_dir=out,
        expected_cross_page_joins=1,
        expected_manual_review_items=1,
    )


def test_approval_copies_proposals_and_excludes_same_page_item(tmp_path: Path):
    paths = make_bundle(tmp_path)
    main_before = paths["main"].read_text(encoding="utf-8")
    notes_before = paths["notes"].read_text(encoding="utf-8")
    result = run_approval(paths, tmp_path / "out")

    assert result["accepted_cross_page_joins"] == 1
    assert result["excluded_manual_review_items"] == 1
    assert result["excluded_manual_review_candidate_ids"] == ["J002"]
    assert result["pronunciation_audition_items"] == 1
    assert result["synthesis_authorized"] is False
    assert (tmp_path / "out/main_manuscript_approved.txt").read_text(encoding="utf-8") == main_before
    assert (tmp_path / "out/notes_manuscript_approved.txt").read_text(encoding="utf-8") == notes_before
    assert paths["main"].read_text(encoding="utf-8") == main_before
    assert paths["notes"].read_text(encoding="utf-8") == notes_before

    record = json.loads((tmp_path / "out/manuscript_approval_record.json").read_text())
    assert record["accepted_candidate_ids"] == ["J001"]
    assert record["excluded_manual_review_candidates"][0]["candidate_id"] == "J002"
    assert record["synthesis_authorized"] is False


def test_non_applied_cross_page_candidate_blocks_approval(tmp_path: Path):
    paths = make_bundle(tmp_path, bad_application=True)
    with pytest.raises(ValueError, match="not uniquely applied"):
        run_approval(paths, tmp_path / "out")


def test_pronunciation_sheet_uses_source_spelling_and_real_context(tmp_path: Path):
    paths = make_bundle(tmp_path)
    run_approval(paths, tmp_path / "out")
    with (tmp_path / "out/pronunciation_audition_sheet.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))

    assert row["source_term"] == "Maitreya"
    assert row["isolated_prompt"] == "Maitreya."
    assert "Maitreya said to the Buddha" in row["context_prompt"]
    assert row["chosen_spoken_form"] == ""
    assert row["review_status"] == "awaiting_audition"


def test_missing_source_context_blocks_audition_sheet(tmp_path: Path):
    paths = make_bundle(tmp_path)
    paths["blocks"].write_text(
        json.dumps({"block_id": "x", "page": 1, "text": "No relevant term here."}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No source context"):
        run_approval(paths, tmp_path / "out")


def test_expected_approval_count_is_enforced(tmp_path: Path):
    paths = make_bundle(tmp_path)
    with pytest.raises(ValueError, match="Expected 48 cross-page joins"):
        approve_repair_bundle(
            repair_summary=paths["summary"],
            repair_review=paths["review"],
            repair_review_csv=paths["csv"],
            main_proposed=paths["main"],
            notes_proposed=paths["notes"],
            pronunciation_draft=paths["pron"],
            source_blocks=paths["blocks"],
            output_dir=tmp_path / "out",
        )
