"""Aligned splicing: replace one English word inside a single-pass English sentence.

The first retest rendered the English before and after a term as separate
fragments. A text-to-speech model treats the last word of a fragment as the end
of an utterance, so prefixes ended in stretched words ("by" about 850 ms against
240 ms inside the full sentence) and the joins sounded like awkward pauses.

This module avoids that. It takes the whole sentence rendered in one pass by
the narrator (the EN_FULL clip), finds where the English term word sits using
forced alignment against the known text, and swaps in the Hindi-bridged term clip
after matching its pitch, pace and level to the English word it replaces. The
surrounding English keeps its natural flow, accent and rhythm.

Forced alignment uses pocketsphinx (its English model ships inside the package,
no download). Pitch and duration changes use Praat's PSOLA via parselmouth.
Everything heavy is imported lazily so the pure parts run in CI.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

SAMPLE_RATE = 24000
ALIGN_RATE = 16000

# English-approximation pronunciations used only to anchor word boundaries.
# They need not match the Hindi-bridged pronunciation.
ALIGNMENT_PRONUNCIATIONS: dict[str, str] = {
    "avatamsaka": "AA V AH T AA M S AH K AH",
    "sutra": "S UW T R AH",
    "shambhala": "SH AA M B AA L AH",
    "supramundane": "S UW P R AH M AH N D EY N",
    "gautama": "G AW T AH M AH",
    "sandhinirmochana": "S AA N D IH N IH R M OW CH AH N AH",
    "subhuti": "S UW B UW T IY",
    "vijnanavada": "V IH JH N AA N AA V AA D AH",
    "yogachara": "Y OW G AA CH AA R AH",
    "nonduality": "N AA N D UW AE L IH T IY",
    "manjushri": "M AA N JH UW SH R IY",
    "avalokiteshvara": "AA V AH L OW K IH T EY SH V AA R AH",
    "maitreya": "M AY T R EY AH",
    "defilement": "D IH F AY L M AH N T",
    "immovability": "IH M UW V AH B IH L IH T IY",
    "thusness": "DH AH S N AH S",
}

# Spelled-out transcripts for sentences containing digits (how the narrator reads them).
ALIGNMENT_TEXT_OVERRIDES: dict[str, str] = {
    "Avatamsaka": (
        "thomas cleary translator the flower ornament scripture a translation of the avatamsaka sutra "
        "boston and london shambhala publications nineteen ninety three book twenty six"
    ),
}

DEFAULT_SETTINGS: dict[str, float] = {
    "crossfade_ms": 12.0,
    "boundary_snap_ms": 30.0,
    "term_trim_db": -45.0,
    "term_trim_keep_ms": 15.0,
    "max_pitch_shift_semitones": 4.0,
    "min_duration_factor": 0.75,
    "max_duration_factor": 1.25,
    "max_level_change_db": 4.0,
}


class AlignmentError(RuntimeError):
    """Raised when the audio cannot be aligned to the expected words."""


def normalize_words(text: str) -> list[str]:
    """Lowercase words for alignment: punctuation removed, hyphens split, digits rejected."""
    cleaned = re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("-", " "))
    words = cleaned.split()
    bad = [w for w in words if any(ch.isdigit() for ch in w)]
    if bad:
        raise AlignmentError(f"Spell out numbers for alignment (use an alignment text override): {bad}")
    return words


def term_token_range(sentence: str, surface: str, term_words: Sequence[str] | None = None) -> tuple[int, int]:
    """Return (first, last) word indices of the term inside the normalized sentence.

    `surface` is the exact spelling in the sentence. Found by counting the words before it.
    """
    index = sentence.find(surface)
    if index < 0 or sentence.find(surface, index + 1) >= 0:
        raise AlignmentError(f"{surface!r} must occur exactly once in the sentence.")
    before = normalize_words(sentence[:index])
    inside = normalize_words(surface)
    if not inside:
        raise AlignmentError("Term has no alignable words.")
    return len(before), len(before) + len(inside) - 1


def align_words(
    audio: Any,
    sample_rate: int,
    words: Sequence[str],
    extra_pronunciations: Mapping[str, str] | None = None,
) -> list[tuple[str, float, float]]:
    """Forced-align `words` to `audio`. Returns (word, start_s, end_s) per word, in order."""
    import librosa
    import numpy as np
    from pocketsphinx import Decoder

    mono = np.asarray(audio, dtype=np.float32).squeeze()
    if sample_rate != ALIGN_RATE:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=ALIGN_RATE)
    pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    decoder = Decoder(samprate=ALIGN_RATE, loglevel="FATAL")
    pronunciations = {**ALIGNMENT_PRONUNCIATIONS, **dict(extra_pronunciations or {})}
    for word, phones in pronunciations.items():
        if word in set(words):
            decoder.add_word(word, phones, False)
    missing = [w for w in dict.fromkeys(words) if decoder.lookup_word(w) is None]
    if missing:
        raise AlignmentError(f"No pronunciation for: {missing}. Add them to ALIGNMENT_PRONUNCIATIONS.")
    decoder.set_align_text(" ".join(words))
    decoder.start_utt()
    decoder.process_raw(pcm, full_utt=True)
    decoder.end_utt()

    spans: list[tuple[str, float, float]] = []
    for seg in decoder.seg():
        if seg.word.startswith(("<", "[", "+")):
            continue
        spans.append((re.sub(r"\(\d+\)$", "", seg.word), seg.start_frame / 100.0, (seg.end_frame + 1) / 100.0))
    if [w for w, _, _ in spans] != list(words):
        raise AlignmentError(f"Aligner returned {len(spans)} words for {len(words)} expected.")
    return spans


def _snap_to_quiet(x: Any, sample_rate: int, position: int, window_ms: float) -> int:
    """Move a cut point to the quietest 5 ms frame within +/- window_ms."""
    import numpy as np

    frame = max(1, int(sample_rate * 0.005))
    reach = int(sample_rate * window_ms / 1000)
    lo, hi = max(0, position - reach), min(len(x) - frame, position + reach)
    if hi <= lo:
        return position
    best, best_energy = position, None
    for start in range(lo, hi + 1, frame):
        energy = float(np.mean(np.square(x[start:start + frame])))
        if best_energy is None or energy < best_energy:
            best, best_energy = start + frame // 2, energy
    return best


def median_f0(audio: Any, sample_rate: int) -> float | None:
    """Median voiced pitch in Hz via Praat, or None when there is no voiced speech."""
    import numpy as np
    import parselmouth

    if len(audio) < int(sample_rate * 0.05):
        return None
    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=sample_rate)
    values = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=450).selected_array["frequency"]
    voiced = values[values > 0]
    return float(np.median(voiced)) if voiced.size else None


def change_prosody(audio: Any, sample_rate: int, pitch_factor: float, duration_factor: float) -> Any:
    """Scale pitch and duration with Praat's PSOLA (factor 1.0 leaves a dimension alone)."""
    import numpy as np
    import parselmouth
    from parselmouth.praat import call

    data = np.asarray(audio, dtype=np.float64)
    if abs(pitch_factor - 1) < 1e-3 and abs(duration_factor - 1) < 1e-3:
        return data.astype(np.float32)
    sound = parselmouth.Sound(data, sampling_frequency=sample_rate)
    manipulation = call(sound, "To Manipulation", 0.01, 75, 450)
    if abs(pitch_factor - 1) >= 1e-3:
        tier = call(manipulation, "Extract pitch tier")
        call(tier, "Multiply frequencies", sound.xmin, sound.xmax, pitch_factor)
        call([tier, manipulation], "Replace pitch tier")
    if abs(duration_factor - 1) >= 1e-3:
        duration = call("Create DurationTier", "d", sound.xmin, sound.xmax)
        call(duration, "Add point", sound.xmin, duration_factor)
        call(duration, "Add point", sound.xmax, duration_factor)
        call([duration, manipulation], "Replace duration tier")
    return np.asarray(call(manipulation, "Get resynthesis (overlap-add)").values[0], dtype=np.float32)


def _rms_db(x: Any) -> float:
    import numpy as np

    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9))


def match_term_to_word(term_audio: Any, english_word: Any, sample_rate: int, settings: Mapping[str, float]) -> tuple[Any, dict[str, Any]]:
    """Adjust the bridged term toward the pitch, pace and level of the English word it replaces."""
    import numpy as np
    from tools.narrator_retest import trim_silence

    term = trim_silence(term_audio, sample_rate, settings["term_trim_db"], int(settings["term_trim_keep_ms"]))
    report: dict[str, Any] = {"term_seconds_before": round(len(term) / sample_rate, 3),
                              "english_word_seconds": round(len(english_word) / sample_rate, 3)}

    target_f0, term_f0 = median_f0(english_word, sample_rate), median_f0(term, sample_rate)
    pitch_factor = 1.0
    if target_f0 and term_f0:
        limit = 2 ** (settings["max_pitch_shift_semitones"] / 12)
        pitch_factor = min(max(target_f0 / term_f0, 1 / limit), limit)
    duration_factor = min(
        max(len(english_word) / len(term), settings["min_duration_factor"]), settings["max_duration_factor"]
    )
    adjusted = change_prosody(term, sample_rate, pitch_factor, duration_factor)

    gain_db = 0.0
    if len(english_word) > 0 and len(adjusted) > 0:
        gain_db = min(max(_rms_db(english_word) - _rms_db(adjusted), -settings["max_level_change_db"]),
                      settings["max_level_change_db"])
        adjusted = adjusted * (10 ** (gain_db / 20))
    peak = float(np.max(np.abs(adjusted))) if len(adjusted) else 0.0
    if peak > 0.98:
        adjusted = adjusted * (0.98 / peak)
    report.update({
        "english_word_f0_hz": round(target_f0, 1) if target_f0 else None,
        "term_f0_hz_before": round(term_f0, 1) if term_f0 else None,
        "pitch_factor": round(pitch_factor, 3),
        "duration_factor": round(duration_factor, 3),
        "level_change_db": round(gain_db, 2),
        "term_seconds_after": round(len(adjusted) / sample_rate, 3),
    })
    return adjusted, report


def splice_replacing_word(
    full_audio: Any,
    sample_rate: int,
    spans: Sequence[tuple[str, float, float]],
    first_index: int,
    last_index: int,
    term_audio: Any,
    settings: Mapping[str, float] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Replace words first_index..last_index of the aligned sentence with `term_audio`."""
    import numpy as np

    cfg = {**DEFAULT_SETTINGS, **dict(settings or {})}
    if not 0 <= first_index <= last_index < len(spans):
        raise AlignmentError("Term word range is outside the aligned sentence.")
    full = np.asarray(full_audio, dtype=np.float32).squeeze()
    start = _snap_to_quiet(full, sample_rate, int(spans[first_index][1] * sample_rate), cfg["boundary_snap_ms"])
    end = _snap_to_quiet(full, sample_rate, int(spans[last_index][2] * sample_rate), cfg["boundary_snap_ms"])
    if end - start < int(sample_rate * 0.08):
        raise AlignmentError("Aligned English term span is implausibly short.")

    english_word = full[start:end]
    term, report = match_term_to_word(term_audio, english_word, sample_rate, cfg)

    fade = int(sample_rate * cfg["crossfade_ms"] / 1000)
    if len(term) < 2 * fade or start < fade or len(full) - end < fade:
        raise AlignmentError("Not enough audio around the term for a crossfade.")
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    left = full[: start - fade]
    join1 = full[start - fade:start] * ramp[::-1] + term[:fade] * ramp
    middle = term[fade:-fade]
    join2 = term[-fade:] * ramp[::-1] + full[end:end + fade] * ramp
    right = full[end + fade:]
    out = np.concatenate([left, join1, middle, join2, right]).astype(np.float32)

    report.update({
        "replaced_words": [w for w, _, _ in spans[first_index:last_index + 1]],
        "cut_start_seconds": round(start / sample_rate, 3),
        "cut_end_seconds": round(end / sample_rate, 3),
        "output_seconds": round(len(out) / sample_rate, 3),
        "full_sentence_seconds": round(len(full) / sample_rate, 3),
    })
    return out, report


def locate_term(name: str, info: Mapping[str, str], base: str) -> tuple[list[str], int, int]:
    """Words of the sentence as rendered in `base` ("full" or "standin"), and the term's word range."""
    from tools.narrator_retest import STAND_INS

    surface = str(info.get("surface") or name)
    sentence = str(info["sentence"])
    term_words = normalize_words(surface)
    override = ALIGNMENT_TEXT_OVERRIDES.get(name)
    if override:  # numbers spelled out, so locate the term in the override words
        words = override.split()
        first = words.index(term_words[0])
        last = first + len(term_words) - 1
    else:
        words = normalize_words(sentence)
        first, last = term_token_range(sentence, surface)
    if base == "standin":
        if name not in STAND_INS:
            raise AlignmentError(f"{name!r} has no stand-in word.")
        replacement = normalize_words(STAND_INS[name])
        words = words[:first] + replacement + words[last + 1:]
        last = first + len(replacement) - 1
    elif base != "full":
        raise ValueError("base must be 'full' or 'standin'.")
    return words, first, last


def build_aligned_splices(
    retest_dir: Any,
    term_names: Sequence[str],
    settings: Mapping[str, float] | None = None,
    route: str = "HYBRID",
    base: str = "full",
) -> list[dict[str, Any]]:
    """Make <ID>_SPLICE_ALIGNED[_STANDIN].wav for each term that has its base English clip.

    `base="full"` uses <ID>_EN_FULL.wav (the sentence with the term spelled out);
    `base="standin"` uses <ID>_EN_STANDIN.wav (a plain English word in the term's slot).
    Reads clips from `retest_dir`, writes the splices, and merges the results into
    `aligned_splices_report.json` there. Returns one report dict per term (failures
    carry the reason instead of raising).
    """
    import json
    from pathlib import Path

    import soundfile as sf

    from tools.narrator_retest import ALL_TERMS, SAMPLE_RATE as RETEST_RATE

    clip, suffix = ("EN_FULL", "SPLICE_ALIGNED") if base == "full" else ("EN_STANDIN", "SPLICE_ALIGNED_STANDIN")
    folder = Path(retest_dir)
    numbering = {name: i for i, name in enumerate(ALL_TERMS, start=1)}
    reports: list[dict[str, Any]] = []
    for name in term_names:
        if name not in ALL_TERMS:
            raise ValueError(f"Unknown term {name!r}.")
        prefix = f"N{numbering[name]:03d}"
        record: dict[str, Any] = {"term": name, "task_id": f"{prefix}_{suffix}", "base": base}
        try:
            base_path, term_path = folder / f"{prefix}_{clip}.wav", folder / f"{prefix}_{route}.wav"
            if not base_path.is_file() or not term_path.is_file():
                raise AlignmentError(f"Needs {base_path.name} and {term_path.name}.")
            full, rate = sf.read(str(base_path), dtype="float32")
            term, term_rate = sf.read(str(term_path), dtype="float32")
            if rate != RETEST_RATE or term_rate != RETEST_RATE:
                raise AlignmentError(f"Expected {RETEST_RATE} Hz clips.")
            words, first, last = locate_term(name, ALL_TERMS[name], base)
            spans = align_words(full, rate, words)
            spliced, report = splice_replacing_word(full, rate, spans, first, last, term, settings)
            out_path = folder / f"{record['task_id']}.wav"
            sf.write(str(out_path), spliced, rate, subtype="PCM_16")
            record.update({"status": "generated", "output_file": out_path.name, **report})
        except (AlignmentError, ValueError) as exc:
            record.update({"status": "failed", "error": str(exc)})
        reports.append(record)

    report_path = folder / "aligned_splices_report.json"
    previous: dict[str, Any] = {}
    if report_path.is_file():
        previous = {r["task_id"]: r for r in json.loads(report_path.read_text(encoding="utf-8")).get("splices", [])}
    previous.update({r["task_id"]: r for r in reports})
    report_path.write_text(
        json.dumps({"settings": {**DEFAULT_SETTINGS, **dict(settings or {})}, "route": route,
                    "splices": list(previous.values())}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return reports
