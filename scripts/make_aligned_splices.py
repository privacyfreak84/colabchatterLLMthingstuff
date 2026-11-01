#!/usr/bin/env python3
"""Build aligned splices from clips a retest run already produced (no model needed).

For each term, takes the single-pass English sentence (EN_FULL, or EN_STANDIN with a
plain English stand-in word) and replaces the English term word with the approved
HYBRID clip, matched to the English word's pitch, pace and level. Fixes the awkward
pauses, stretched final words and pitch/pace mismatches of the fragment-based splices.

    python scripts/make_aligned_splices.py --retest-dir ~/audiobook-narrator-retest/v3-narrator-retest

Needs: pip install pocketsphinx praat-parselmouth soundfile librosa numpy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.narrator_retest import REMAINING_TERM_NAMES, STAND_INS  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--retest-dir", required=True, type=Path)
    p.add_argument("--terms", nargs="*", help="Terms to splice (default: the nine remaining terms).")
    p.add_argument("--base", choices=["full", "standin", "both"], default="both",
                   help="Which English base clip to use (default: both, where the clips exist).")
    args = p.parse_args()

    try:
        from tools.aligned_splice import build_aligned_splices
    except ImportError as exc:  # pragma: no cover
        print(f"Missing dependency: {exc}\nInstall: pip install pocketsphinx praat-parselmouth soundfile librosa numpy")
        return 3

    terms = args.terms or list(REMAINING_TERM_NAMES)
    bases = ["full", "standin"] if args.base == "both" else [args.base]
    exit_code = 0
    for base in bases:
        names = [t for t in terms if base == "full" or t in STAND_INS]
        for r in build_aligned_splices(args.retest_dir.expanduser(), names, base=base):
            if r["status"] == "generated":
                print(f"{r['task_id']}: replaced {'+'.join(r['replaced_words'])} "
                      f"(pitch x{r['pitch_factor']}, pace x{r['duration_factor']}, level {r['level_change_db']:+.1f} dB)")
            else:
                print(f"{r['task_id']}: skipped - {r['error']}")
                exit_code = 0 if "Needs" in r["error"] else 1
    print(f"\nReport: {args.retest_dir.expanduser() / 'aligned_splices_report.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
