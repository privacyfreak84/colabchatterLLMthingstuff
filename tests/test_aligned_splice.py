import json

import pytest

np = pytest.importorskip("numpy")

from tools.aligned_splice import (  # noqa: E402
    ALIGNMENT_PRONUNCIATIONS,
    ALIGNMENT_TEXT_OVERRIDES,
    AlignmentError,
    _snap_to_quiet,
    locate_term,
    normalize_words,
    splice_replacing_word,
    term_token_range,
)
from tools.narrator_retest import ALL_TERMS, REMAINING_TERMS, STAND_INS, standin_sentence  # noqa: E402

SR = 24000


def test_normalize_words_lowercases_splits_hyphens_and_drops_punctuation():
    assert normalize_words('The "Sandhinirmochana-sutra," a sourcebook.') == [
        "the", "sandhinirmochana", "sutra", "a", "sourcebook",
    ]
    assert normalize_words("don't stop") == ["don't", "stop"]


def test_normalize_words_rejects_digits():
    with pytest.raises(AlignmentError, match="Spell out numbers"):
        normalize_words("book 26")


def test_term_token_range_counts_words_before_and_inside_the_term():
    sentence = 'Then the enlightening being Maitreya said to the Buddha, "Based on what?'
    assert term_token_range(sentence, "Maitreya") == (4, 4)
    assert term_token_range("see the Sandhinirmochana-sutra, today", "Sandhinirmochana-sutra") == (2, 3)
    with pytest.raises(AlignmentError, match="exactly once"):
        term_token_range("Maitreya met Maitreya", "Maitreya")


@pytest.mark.parametrize("name", list(REMAINING_TERMS))
def test_locate_term_full_base_finds_the_term_words(name):
    info = ALL_TERMS[name]
    words, first, last = locate_term(name, info, "full")
    surface = normalize_words(str(info.get("surface") or name))
    assert words[first:last + 1] == surface


@pytest.mark.parametrize("name", list(STAND_INS))
def test_locate_term_standin_base_swaps_in_the_standin_words(name):
    words, first, last = locate_term(name, ALL_TERMS[name], "standin")
    assert words[first:last + 1] == normalize_words(STAND_INS[name])
    full_words, _, _ = locate_term(name, ALL_TERMS[name], "full")
    assert words[:first] == full_words[:first]  # nothing before the slot changed


def test_buddha_has_no_standin_because_it_stays_english():
    assert "Buddha" not in STAND_INS
    with pytest.raises(AlignmentError, match="no stand-in"):
        locate_term("Buddha", ALL_TERMS["Buddha"], "standin")
    with pytest.raises(ValueError, match="No stand-in"):
        standin_sentence("Buddha", ALL_TERMS["Buddha"])


def test_standin_sentence_replaces_only_the_term():
    text = standin_sentence("nirvana", ALL_TERMS["nirvana"])
    assert "banana" in text and "nirvana" not in text
    assert text.startswith("This is why it is said")


def test_overridden_sentence_spells_out_the_numbers():
    assert "Avatamsaka" in ALIGNMENT_TEXT_OVERRIDES
    words, first, last = locate_term("Avatamsaka", ALL_TERMS["Avatamsaka"], "full")
    assert words[first] == "avatamsaka" and first == last
    assert not any(ch.isdigit() for w in words for ch in w)


def test_every_word_of_every_sentence_can_be_aligned():
    pocketsphinx = pytest.importorskip("pocketsphinx")
    decoder = pocketsphinx.Decoder(samprate=16000, loglevel="FATAL")
    for word, phones in ALIGNMENT_PRONUNCIATIONS.items():
        decoder.add_word(word, phones, False)
    for name in ALL_TERMS:
        for base in ("full", "standin"):
            if base == "standin" and name not in STAND_INS:
                continue
            words, _, _ = locate_term(name, ALL_TERMS[name], base)
            missing = [w for w in dict.fromkeys(words) if decoder.lookup_word(w) is None]
            assert not missing, f"{name}/{base}: no pronunciation for {missing}"


def test_snap_moves_a_cut_into_the_nearest_quiet_gap():
    x = np.ones(SR // 2, dtype=np.float32) * 0.3
    x[int(SR * 0.25):int(SR * 0.27)] = 0.0  # a 20 ms gap
    cut = _snap_to_quiet(x, SR, int(SR * 0.255) + 200, 30)
    assert int(SR * 0.25) <= cut <= int(SR * 0.27)


# ---- audio manipulation (needs Praat) ----

parselmouth = pytest.importorskip("parselmouth")

from tools.aligned_splice import change_prosody, median_f0  # noqa: E402


def _voiced(hz, seconds):
    """A buzzy harmonic signal Praat treats as voiced speech."""
    t = np.arange(int(seconds * SR)) / SR
    wave = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 8))
    return (0.2 * wave / np.max(np.abs(wave))).astype(np.float32)


def test_change_prosody_moves_pitch_and_duration_in_the_requested_direction():
    base = _voiced(180, 1.0)
    f0 = median_f0(base, SR)
    assert f0 == pytest.approx(180, rel=0.03)
    shifted = change_prosody(base, SR, pitch_factor=1.2, duration_factor=0.8)
    assert median_f0(shifted, SR) == pytest.approx(f0 * 1.2, rel=0.05)
    assert len(shifted) / SR == pytest.approx(0.8, rel=0.08)


def test_change_prosody_with_unit_factors_returns_the_audio_unchanged():
    base = _voiced(180, 0.5)
    assert np.allclose(change_prosody(base, SR, 1.0, 1.0), base, atol=1e-6)


def _sentence_and_spans():
    """Three 'words' of tone with silence between: the middle one is the word to replace."""
    parts, spans, t = [], [], 0.0
    for name, hz in (("then", 200), ("maitreya", 230), ("said", 200)):
        parts.append(_voiced(hz, 0.5))
        spans.append((name, t, t + 0.5))
        t += 0.5
        gap = np.zeros(int(0.05 * SR), dtype=np.float32)
        parts.append(gap)
        t += 0.05
    return np.concatenate(parts), spans


def test_splice_replaces_only_the_word_and_matches_its_pitch():
    full, spans = _sentence_and_spans()
    term = _voiced(165, 0.9)  # lower and longer than the English word it replaces
    out, report = splice_replacing_word(full, SR, spans, 1, 1, term)
    assert report["replaced_words"] == ["maitreya"]
    assert 0.5 <= report["term_seconds_after"] <= 0.75  # pulled toward the 0.5 s English word
    assert report["pitch_factor"] > 1.2  # 230/165, within the shift limit
    start = int(report["cut_start_seconds"] * SR)
    assert median_f0(out[:start - 200], SR) == pytest.approx(200, rel=0.05)  # untouched English before
    assert median_f0(out[start + SR // 10: start + int(0.3 * SR)], SR) == pytest.approx(230, rel=0.1)
    assert float(np.max(np.abs(out))) <= 0.98 + 1e-6
    assert abs(len(out) - len(full)) < SR * 0.4


def test_splice_rejects_bad_ranges_and_tiny_spans():
    full, spans = _sentence_and_spans()
    with pytest.raises(AlignmentError, match="outside"):
        splice_replacing_word(full, SR, spans, 1, 5, _voiced(200, 0.5))
    tiny = [("a", 0.5, 0.52)] * 3
    with pytest.raises(AlignmentError, match="short"):
        splice_replacing_word(full, SR, tiny, 1, 1, _voiced(200, 0.5))


sf = pytest.importorskip("soundfile")

from tools.aligned_splice import build_aligned_splices  # noqa: E402


def test_build_reports_a_missing_clip_instead_of_crashing_and_keeps_a_report(tmp_path):
    reports = build_aligned_splices(tmp_path, ["nirvana", "sutra"], base="full")
    assert [r["status"] for r in reports] == ["failed", "failed"]
    assert all("Needs" in r["error"] for r in reports)
    saved = json.loads((tmp_path / "aligned_splices_report.json").read_text())
    assert {r["task_id"] for r in saved["splices"]} == {"N011_SPLICE_ALIGNED", "N013_SPLICE_ALIGNED"}
    build_aligned_splices(tmp_path, ["nirvana"], base="standin")  # a second base adds, not replaces
    saved = json.loads((tmp_path / "aligned_splices_report.json").read_text())
    assert len(saved["splices"]) == 3


def test_build_rejects_unknown_terms(tmp_path):
    with pytest.raises(ValueError, match="Unknown term"):
        build_aligned_splices(tmp_path, ["Nope"])
