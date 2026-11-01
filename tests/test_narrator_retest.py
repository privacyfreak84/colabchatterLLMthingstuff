import pytest

np = pytest.importorskip("numpy")

from tools.narrator_retest import (
    DEFAULT_ASSEMBLY,
    DEFAULT_TERMS,
    assemble_splice,
    build_narrator_task_specs,
    split_context_sentence,
    trim_silence,
)

BASE = {
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "min_p": 0.05,
    "top_p": 0.95,
}
SR = 24000


def test_splitter_returns_prefix_suffix_and_leading_punctuation():
    prefix, suffix, lead = split_context_sentence(
        "Then the great enlightening beings Manjushri, the Glorious One, said to the Buddha.",
        "Manjushri",
    )
    assert prefix == "Then the great enlightening beings"
    assert suffix == "the Glorious One, said to the Buddha."
    assert lead == ","


def test_splitter_without_punctuation_has_empty_lead():
    prefix, suffix, lead = split_context_sentence("Then Maitreya said to the Buddha.", "Maitreya")
    assert (prefix, suffix, lead) == ("Then", "said to the Buddha.", "")


@pytest.mark.parametrize(
    "sentence, message",
    [
        ("Nothing relevant here.", "found 0"),
        ("Maitreya met Maitreya.", "found 2"),
        ("Maitreyas are not the term.", "found 0"),
        ("Maitreya opens the sentence.", "both sides"),
    ],
)
def test_splitter_rejects_missing_repeated_partial_or_edge_terms(sentence, message):
    with pytest.raises(ValueError, match=message):
        split_context_sentence(sentence, "Maitreya")


def test_default_terms_all_split_cleanly_on_real_sentences():
    for term, info in DEFAULT_TERMS.items():
        prefix, suffix, _ = split_context_sentence(info["sentence"], term)
        assert prefix and suffix
        assert term not in prefix and term not in suffix


def test_builder_creates_seven_deterministic_tasks_per_term():
    specs = build_narrator_task_specs(DEFAULT_TERMS, BASE)
    assert len(specs) == 28
    assert len({s["task_id"] for s in specs}) == 28
    first = [s["task_id"] for s in specs[:7]]
    assert first == [
        "N001_EN_PREFIX", "N001_EN_SUFFIX", "N001_HI_SOURCE", "N001_HYBRID",
        "N001_VC", "N001_SPLICE_HYBRID", "N001_SPLICE_VC",
    ]
    assert build_narrator_task_specs(DEFAULT_TERMS, BASE) == specs


def test_task_modes_languages_and_dependencies():
    by_id = {s["task_id"]: s for s in build_narrator_task_specs(DEFAULT_TERMS, BASE)}
    assert by_id["N002_EN_PREFIX"]["mode"] == "english_narrator"
    assert by_id["N002_EN_PREFIX"]["language_id"] == "en"
    assert by_id["N002_HI_SOURCE"]["language_id"] == "hi"
    assert by_id["N002_HYBRID"]["mode"] == "hybrid"
    assert by_id["N002_HYBRID"]["text"] == by_id["N002_HI_SOURCE"]["text"] == "मैत्रेय"
    assert by_id["N002_VC"]["kind"] == "voice_conversion"
    assert by_id["N002_VC"]["source_task_id"] == "N002_HI_SOURCE"
    assert by_id["N002_SPLICE_VC"]["parts"] == ["N002_EN_PREFIX", "N002_VC", "N002_EN_SUFFIX"]
    assert by_id["N002_SPLICE_HYBRID"]["parts"] == ["N002_EN_PREFIX", "N002_HYBRID", "N002_EN_SUFFIX"]


def test_every_dependency_exists_and_comes_earlier():
    specs = build_narrator_task_specs(DEFAULT_TERMS, BASE)
    seen = set()
    for spec in specs:
        for needed in [*spec["parts"], spec["source_task_id"]]:
            if needed:
                assert needed in seen, f"{spec['task_id']} needs {needed} first"
        seen.add(spec["task_id"])


def test_comma_after_term_uses_the_longer_pause():
    by_id = {s["task_id"]: s for s in build_narrator_task_specs(DEFAULT_TERMS, BASE)}
    assert by_id["N003_SPLICE_HYBRID"]["join_gap_ms"] == DEFAULT_ASSEMBLY["comma_gap_ms"]  # Manjushri,
    assert by_id["N002_SPLICE_HYBRID"]["join_gap_ms"] == DEFAULT_ASSEMBLY["gap_ms"]  # Maitreya said


def test_english_parts_keep_the_manuscript_wording():
    by_id = {s["task_id"]: s for s in build_narrator_task_specs(DEFAULT_TERMS, BASE)}
    assert by_id["N002_EN_PREFIX"]["text"] == "Then the enlightening being"
    assert by_id["N002_EN_SUFFIX"]["text"].startswith("said to the Buddha")


def test_builder_does_not_mutate_inputs_and_validates_them():
    params = dict(BASE)
    build_narrator_task_specs(DEFAULT_TERMS, params)
    assert params == BASE
    with pytest.raises(ValueError):
        build_narrator_task_specs({}, BASE)
    with pytest.raises(ValueError):
        build_narrator_task_specs(DEFAULT_TERMS, {})
    with pytest.raises(ValueError, match="needs a name"):
        build_narrator_task_specs({"X": {"devanagari": "", "sentence": "A X b."}}, BASE)


def _tone(seconds, amplitude=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_trim_silence_removes_padding_but_keeps_a_margin():
    pad = np.zeros(SR, dtype=np.float32)
    clip = np.concatenate([pad, _tone(0.5), pad])
    trimmed = trim_silence(clip, SR, -45.0, 30)
    assert trimmed.size == pytest.approx(int(0.5 * SR) + 2 * int(0.03 * SR), abs=40)


def test_trim_silence_rejects_silent_or_empty_clips():
    with pytest.raises(ValueError):
        trim_silence(np.zeros(SR, dtype=np.float32), SR, -45.0, 30)
    with pytest.raises(ValueError):
        trim_silence(np.zeros(0, dtype=np.float32), SR, -45.0, 30)


def test_assemble_splice_length_gap_and_fades():
    parts = [_tone(0.4), _tone(0.3), _tone(0.5)]
    settings = dict(gap_ms=100, edge_fade_ms=10, trim_threshold_db=-45.0, trim_keep_ms=0)
    result = assemble_splice(parts, SR, **settings)
    expected = int(0.4 * SR) + int(0.3 * SR) + int(0.5 * SR) + 2 * int(0.1 * SR)
    assert result.size == pytest.approx(expected, abs=SR // 100)
    assert float(np.max(np.abs(result))) <= 0.5 + 1e-6
    assert abs(float(result[0])) < 0.05  # first part has no fade-in, but starts near zero crossing
    # The join is silent for the configured gap.
    first_end = int(0.4 * SR)
    window = result[first_end + 50: first_end + int(0.1 * SR) - 50]
    assert float(np.max(np.abs(window))) < 1e-4


def test_assemble_splice_needs_exactly_three_parts():
    with pytest.raises(ValueError, match="three parts"):
        assemble_splice(
            [_tone(0.2)] * 2, SR, gap_ms=60, edge_fade_ms=10, trim_threshold_db=-45.0, trim_keep_ms=30
        )


# ---- partial runs, term selection ----

from tools.narrator_retest import CPU_SMOKE_TERMS, select_terms


def test_select_terms_defaults_subsets_and_unknown_names():
    assert list(select_terms(None)) == list(DEFAULT_TERMS)
    assert list(select_terms(CPU_SMOKE_TERMS)) == ["Maitreya", "Vijnanavada"]
    with pytest.raises(ValueError, match="Unknown term"):
        select_terms(["Maitreya", "Nope"])


def test_select_terms_returns_copies():
    chosen = select_terms(["Maitreya"])
    chosen["Maitreya"]["sentence"] = "changed"
    assert DEFAULT_TERMS["Maitreya"]["sentence"] != "changed"


def test_task_ids_follow_the_term_not_its_position_in_a_subset():
    subset = build_narrator_task_specs(select_terms(CPU_SMOKE_TERMS), BASE)
    ids = [s["task_id"] for s in subset]
    assert len(ids) == 14
    assert {i[:4] for i in ids} == {"N002", "N004"}
    full = {s["task_id"]: s for s in build_narrator_task_specs(DEFAULT_TERMS, BASE)}
    for spec in subset:  # identical specs (text, seed, parts) whichever subset they come from
        assert full[spec["task_id"]] == spec


def test_custom_terms_are_numbered_after_the_defaults():
    custom = {"Zzterm": {"devanagari": "ज़", "sentence": "Then Zzterm spoke clearly today."}}
    ids = [s["task_id"] for s in build_narrator_task_specs(custom, BASE)]
    assert ids[0] == "N014_EN_PREFIX"


# ---- remaining nine terms, HYBRID-only mode, English control ----

from tools.narrator_retest import ALL_TERMS, REMAINING_TERM_NAMES, REMAINING_TERMS


def test_all_terms_keep_the_original_four_first_and_cover_all_thirteen():
    assert list(ALL_TERMS)[:4] == list(DEFAULT_TERMS)
    assert len(ALL_TERMS) == 13 and len(REMAINING_TERMS) == 9
    assert set(ALL_TERMS) == set(DEFAULT_TERMS) | set(REMAINING_TERMS)
    assert select_terms(None).keys() == DEFAULT_TERMS.keys()  # default stays the original four


def test_every_remaining_sentence_splits_around_its_term():
    for name, info in REMAINING_TERMS.items():
        prefix, suffix, _ = split_context_sentence(info["sentence"], info.get("surface", name))
        assert prefix and suffix, name
        assert info["devanagari"], name


def test_remaining_terms_get_stable_ids_after_the_original_four():
    specs = build_narrator_task_specs(select_terms(list(REMAINING_TERM_NAMES)), BASE, routes=("HYBRID",))
    assert sorted({s["task_id"][:4] for s in specs}) == [f"N{n:03d}" for n in range(5, 14)]
    assert [s["task_id"] for s in specs[:4]] == [
        "N005_EN_PREFIX", "N005_EN_SUFFIX", "N005_HYBRID", "N005_SPLICE_HYBRID",
    ]


def test_hybrid_only_has_no_hindi_source_voice_conversion_or_vc_splice():
    specs = build_narrator_task_specs(select_terms(["Maitreya"]), BASE, routes=("HYBRID",))
    assert [s["task_id"] for s in specs] == [
        "N002_EN_PREFIX", "N002_EN_SUFFIX", "N002_HYBRID", "N002_SPLICE_HYBRID",
    ]
    assert specs[-1]["parts"] == ["N002_EN_PREFIX", "N002_HYBRID", "N002_EN_SUFFIX"]


def test_vc_only_keeps_its_hindi_source():
    ids = [s["task_id"] for s in build_narrator_task_specs(select_terms(["Maitreya"]), BASE, routes=("VC",))]
    assert "N002_HI_SOURCE" in ids and "N002_VC" in ids and "N002_HYBRID" not in ids


def test_routes_are_validated():
    for bad in ((), ("HYBRID", "NOPE"), ("hybrid-ish",)):
        with pytest.raises(ValueError, match="routes"):
            build_narrator_task_specs(select_terms(["Maitreya"]), BASE, routes=bad)


def test_hybrid_only_tasks_are_identical_to_the_same_tasks_in_a_both_routes_run():
    both = {s["task_id"]: s for s in build_narrator_task_specs(select_terms(["Maitreya"]), BASE)}
    for spec in build_narrator_task_specs(select_terms(["Maitreya"]), BASE, routes=("HYBRID",)):
        assert both[spec["task_id"]] == spec  # so finished clips are reusable across modes


def test_english_control_reads_the_whole_sentence_in_english():
    specs = build_narrator_task_specs(select_terms(["nirvana"]), BASE, routes=("HYBRID",), include_english_control=True)
    control = specs[-1]
    assert control["task_id"] == "N011_EN_FULL"
    assert control["mode"] == "english_narrator" and control["language_id"] == "en"
    assert control["text"] == REMAINING_TERMS["nirvana"]["sentence"]
    without = build_narrator_task_specs(select_terms(["nirvana"]), BASE, routes=("HYBRID",))
    assert all(s["task_id"] != "N011_EN_FULL" for s in without)


def test_surface_spelling_is_used_for_the_split_but_the_inventory_name_for_ids():
    specs = build_narrator_task_specs(select_terms(["samsara"]), BASE, routes=("HYBRID",))
    by_id = {s["task_id"]: s for s in specs}
    assert by_id["N012_EN_PREFIX"]["text"] == "repetitious cycles:"
    assert by_id["N012_EN_SUFFIX"]["text"].startswith("the working out of routine habits")
    assert by_id["N012_SPLICE_HYBRID"]["join_gap_ms"] == DEFAULT_ASSEMBLY["comma_gap_ms"]
