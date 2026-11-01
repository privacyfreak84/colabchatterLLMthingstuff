#!/usr/bin/env python3
"""Run the narrator-voice retest on a local machine (CPU or CUDA), outside Colab.

Typical first run on a CPU-only laptop (two short terms, resumable):

    python scripts/run_narrator_retest_local.py --narrator ~/narrator_reference.wav

Check only the recording (needs nothing but Python):

    python scripts/run_narrator_retest_local.py --narrator ~/narrator_reference.wav --check-only

Setup instructions: docs/local_cpu_setup.md. Nothing here edits the manuscript or
authorizes full-book synthesis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.narrator_reference import validate_narrator_reference  # noqa: E402
from tools.narrator_retest import (  # noqa: E402
    ALL_TERMS,
    CPU_SMOKE_TERMS,
    DEFAULT_GENERATION_PARAMETERS,
    REMAINING_TERM_NAMES,
    make_aligned_reference,
    make_transfer_bundle,
    run_narrator_retest,
    select_terms,
)

CHATTERBOX_COMMIT = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
MODEL_REPO = "ResembleAI/chatterbox"
MODEL_FILES = [
    "ve.pt", "s3gen.pt", "t3_mtl23ls_v3.safetensors",
    "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json",
]
MINIMUM_SIZES = {
    "ve.pt": 5_000_000, "s3gen.pt": 1_000_000_000, "t3_mtl23ls_v3.safetensors": 2_000_000_000,
    "grapheme_mtl_merged_expanded_v1.json": 60_000, "conds.pt": 50_000, "Cangjie5_TC.json": 1_000_000,
}
HINDI_DEMO_URL = "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--narrator", required=True, type=Path, help="Narrator reference WAV (8-10 s of clean speech).")
    p.add_argument("--terms", nargs="*", metavar="TERM",
                   help=f"Terms to run. Default: {' '.join(CPU_SMOKE_TERMS)} on CPU, the first four on CUDA. "
                        f"Valid: {' '.join(ALL_TERMS)}")
    p.add_argument("--remaining-terms", action="store_true",
                   help=f"Run the nine terms not yet heard: {' '.join(REMAINING_TERM_NAMES)}.")
    p.add_argument("--all-terms", action="store_true", help="Run all 13 inventory terms (slow on CPU).")
    p.add_argument("--routes", choices=["both", "hybrid", "vc"], default="both",
                   help="Which bridge route(s) to render. The approved route is hybrid.")
    p.add_argument("--english-control", action="store_true",
                   help="Also render each whole sentence in plain English by the narrator, to compare "
                        "plain English against the Hindi-bridged splice.")
    p.add_argument("--output-dir", type=Path, default=Path.home() / "audiobook-narrator-retest")
    p.add_argument("--model-dir", type=Path, default=Path.home() / ".cache" / "audiobook" / "chatterbox-v3")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch (default: PyTorch's choice).")
    p.add_argument("--check-only", action="store_true", help="Only check the recording, then exit.")
    return p.parse_args()


def report_recording(path: Path) -> bool:
    check = validate_narrator_reference(path)
    print(json.dumps(check["metrics"], indent=2))
    print("BLOCKING PROBLEMS:", check["problems"] or "none")
    print("WARNINGS:", check["warnings"] or "none")
    print("This only measures the recording. Whether the voice suits the book is your decision.\n")
    return check["ok"]


def ensure_model_snapshot(model_dir: Path) -> str:
    """Download the six V3 files once; later runs work offline. Returns the model revision."""
    from huggingface_hub import HfApi, snapshot_download

    revision_file = model_dir / "REVISION.txt"

    def problems() -> list[str]:
        found = []
        for name in MODEL_FILES:
            f = model_dir / name
            if not f.is_file():
                found.append(f"{name}: missing")
            elif f.stat().st_size < MINIMUM_SIZES[name]:
                found.append(f"{name}: suspiciously small ({f.stat().st_size:,} bytes)")
        return found

    if not problems() and revision_file.is_file():
        return revision_file.read_text(encoding="utf-8").strip()
    model_dir.mkdir(parents=True, exist_ok=True)
    revision = HfApi().model_info(MODEL_REPO, revision="main").sha
    print(f"Downloading the selected V3 files (several GB) to {model_dir} ...")
    snapshot_download(repo_id=MODEL_REPO, repo_type="model", revision=revision,
                      allow_patterns=MODEL_FILES, local_dir=str(model_dir))
    remaining = problems()
    if remaining:
        raise RuntimeError("Model download incomplete:\n" + "\n".join(remaining))
    revision_file.write_text(revision + "\n", encoding="utf-8")
    return revision


def main() -> int:
    args = parse_args()
    narrator = args.narrator.expanduser()
    if not narrator.is_file():
        print(f"Narrator recording not found: {narrator}")
        return 2
    if not report_recording(narrator):
        print("Fix the blocking problems above and run again.")
        return 2
    if args.check_only:
        return 0

    try:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as exc:
        print(f"Missing dependency: {exc}\nSee docs/local_cpu_setup.md for the install steps.")
        return 3

    if args.device == "cuda" and not torch.cuda.is_available():
        print("--device cuda was requested but CUDA is not available. Use --device cpu or auto.")
        return 3
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.terms:
        terms = select_terms(args.terms)
    elif args.all_terms:
        terms = select_terms(list(ALL_TERMS))
    elif args.remaining_terms:
        terms = select_terms(list(REMAINING_TERM_NAMES))
    elif device == "cuda":
        terms = select_terms(None)
    else:
        terms = select_terms(CPU_SMOKE_TERMS)
    routes = {"both": ("HYBRID", "VC"), "hybrid": ("HYBRID",), "vc": ("VC",)}[args.routes]
    print(f"Device: {device} | threads: {torch.get_num_threads()} | terms: {', '.join(terms)}")
    if device == "cpu":
        print("CPU generation is slow. The run is resumable: press Ctrl-C any time and rerun the same command.\n")

    out = args.output_dir.expanduser()
    refs = out / "references"
    refs.mkdir(parents=True, exist_ok=True)
    narrator_aligned = refs / "narrator_reference_padded.wav"
    seconds = make_aligned_reference(narrator, narrator_aligned)
    print(f"Narrator reference prepared: first {seconds:.2f} s, padded to a 40 ms multiple.")
    hindi_source = refs / "official_demo_hi_f1.flac"
    if not hindi_source.is_file() or hindi_source.stat().st_size < 1024:
        urllib.request.urlretrieve(HINDI_DEMO_URL, hindi_source)
    hindi_aligned = refs / "official_demo_hi_f1_padded.wav"
    make_aligned_reference(hindi_source, hindi_aligned)

    revision = ensure_model_snapshot(args.model_dir.expanduser())
    print("Loading Chatterbox Multilingual V3 ...")
    started = time.perf_counter()
    model = ChatterboxMultilingualTTS.from_local(str(args.model_dir.expanduser()), device=device, t3_model="v3")
    print(f"Model loaded in {time.perf_counter() - started:.0f} s.\n")

    result = run_narrator_retest(
        model=model,
        narrator_reference_audio=narrator_aligned,
        hindi_reference_audio=hindi_aligned,
        output_dir=out / "v3-narrator-retest",
        model_revision=revision,
        code_revision=CHATTERBOX_COMMIT,
        base_generation_parameters=DEFAULT_GENERATION_PARAMETERS,
        terms=terms,
        routes=routes,
        include_english_control=args.english_control,
    )
    print(json.dumps({k: result[k] for k in ("task_count", "generated_count", "failed_count", "runtime", "output_dir")},
                     indent=2))
    bundle = make_transfer_bundle(
        result["output_dir"], out / "narrator_retest_bundle.zip", out / "narrator_retest_bundle.txt",
        only_files={t["output_file"] for t in result["tasks"] if t.get("output_file")},
    )
    print(f"\nListen to the *_SPLICE_*.wav files in {result['output_dir']}"
          + (" and compare each with its *_EN_FULL.wav." if args.english_control else "."))
    print(f"Bundle for upload: {bundle['text']}  ({bundle['wav_count']} clips from this run)")
    print("Upload it to the repo root under a new name (for example nextrecon7.txt); do not overwrite earlier bundles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
