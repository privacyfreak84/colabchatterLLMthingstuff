# Buddhist Yoga PDF-to-Audiobook Pipeline

A document-first, zero-cost-inference project for turning the 122-page PDF of *Buddhist Yoga: A Comprehensive Course* into a complete, faithful audiobook, using **Chatterbox Multilingual V3** for reference-conditioned narration and Google Colab for GPU work where available.

The central idea is simple: **get the book right before generating the book**. The workflow first reconstructs and reviews the manuscript, then auditions specialist pronunciations, then tests how those terms sound in the selected narrator voice. Expensive model work is kept separate from document inspection, and every experiment is checkpointed so it can be resumed and audited.

> **Project status: this is a workbench, not a finished audiobook generator.** Manuscript preparation and the main pronunciation/narrator experiments are implemented. Full-book chunking, resumable chapter synthesis, end-to-end audio verification, mastering, and export have not been implemented. Full-book synthesis is deliberately locked. See [PROJECT_STATUS.md](PROJECT_STATUS.md) before running anything.

[**Open Stage 1 in Google Colab: PDF reconnaissance and manuscript approval**](https://colab.research.google.com/github/privacyfreak84/colabchatterLLMthingstuff/blob/main/notebooks/01_pdf_reconnaissance.ipynb)

[**Open Stage 2 in Google Colab: Chatterbox V3 pronunciation auditions**](https://colab.research.google.com/github/privacyfreak84/colabchatterLLMthingstuff/blob/main/notebooks/02_pronunciation_auditions.ipynb)

[**Open the standalone custom-text TTS playground in Google Colab**](https://colab.research.google.com/github/privacyfreak84/colabchatterLLMthingstuff/blob/main/notebooks/03_chatterbox_custom_text_gradio.ipynb)

The third notebook launches a password-protected Gradio web UI around the same pinned Chatterbox Multilingual V3 runtime. It accepts arbitrary text, lets you choose a supported language and reference voice (built-in English/Hindi demo or your own uploaded/recorded clip), exposes generation settings, and saves WAV output to Drive. It is a separate TTS playground: it does not use or change the audiobook manuscript, approval records, or full-book synthesis gate.

---

## What this project is trying to produce

The target is a full audiobook of the supplied *Buddhist Yoga* PDF, with:

- Every approved section narrated in the correct order, without omissions, duplication, or truncation.
- The source author's wording and meaning preserved. Extraction repairs and spoken-form substitutions must be explicit, reversible, and traceable.
- Natural, intelligible, consistent English narration with comfortable pacing and clean joins.
- Consistent, listener-approved pronunciation for Sanskrit, Buddhist, and philosophical vocabulary.
- One coherent narrator identity throughout the book, using a private narrator reference recording rather than model fine-tuning.
- Separate chapter/section audio files, plus a lossless master and a manifest recording chunk order, provenance, timing, and hashes.
- A reproducible workflow that can resume after a Colab reset and regenerate only missing or invalid work.

The intended speech engine is **Chatterbox Multilingual V3**. The project uses reference conditioning; it does not train or fine-tune a new voice model. The narrator recording is a local/private input and is intentionally not stored in this public repository.

The cost constraint is no paid inference API, subscription, hosted GPU, or export credits. The notebooks are designed around free Colab availability and downloadable model assets, but free GPU availability and runtime limits are not guaranteed.

## How the pipeline works

The pipeline is intentionally staged. A later stage must not silently bypass an earlier approval gate.

### 1. PDF reconnaissance and extraction

[Stage 1 notebook](https://colab.research.google.com/github/privacyfreak84/colabchatterLLMthingstuff/blob/main/notebooks/01_pdf_reconnaissance.ipynb) runs the document work on CPU before any Chatterbox model is loaded.

It reconstructs the bundled document-engine archive, verifies its size and SHA-256 checksum, extracts the PDF's native text and layout, and checks the book structure. It is designed for a digitally generated PDF; OCR is not used unless native extraction proves inadequate.

The extraction produces page/block-level provenance and separates the preface, eight chapters, main manuscript, and Notes. It also reports structural findings and builds a first-pass inventory of likely specialist terms.

### 2. Reversible manuscript-repair proposals

The repair helper uses page and block order to identify possible sentence continuations, especially at page boundaries.

- Likely cross-page joins are applied only to separate *proposed* copies, and only when the exact adjacent source-block pair occurs uniquely.
- Same-page candidates are listed for manual review, not automatically joined.
- The source PDF, extracted text, original section exports, and original main/Notes tracks are left untouched.
- A review bundle records the candidate, source blocks, pages, proposed join, confidence, and whether it was applied to a proposal.

For the current book, the approval checkpoint accepted **48 cross-page joins**. One same-page glossary boundary, **J048**, was deliberately excluded and left for manual review.

### 3. Manuscript approval and pronunciation inventory

The approval helper checks that the summary, detailed JSON review, and CSV review agree. It then copies the reviewed proposed tracks into separate approved main and Notes tracks and writes an approval record.

It also creates source-context audition prompts and an editable pronunciation sheet for 13 terms. A source spelling is not treated as an approved spoken form; the spoken form and pronunciation strategy remain explicit human decisions.

### 4. Chatterbox pronunciation auditions

[Stage 2 notebook](https://colab.research.google.com/github/privacyfreak84/colabchatterLLMthingstuff/blob/main/notebooks/02_pronunciation_auditions.ipynb) installs a pinned upstream Chatterbox source revision and selects the V3 model explicitly. The notebook does not rely on the stale PyPI 0.1.7 interface for this workflow.

The initial audition plan compares source English spelling, IAST transliteration, and provisional Devanagari forms in Hindi. It begins with one English and one Hindi smoke-test clip. The 36-clip batch is opt-in and remains off by default. The tokenizer is preflighted for unknown tokens, and generated clips are checkpointed with task IDs, seeds, settings, and hashes.

The later English-refinement experiment produced eight clips, but listening feedback rejected that route because of slow delivery, pronunciation problems, and audible artefacts. Those results are retained as historical evidence; the same failed approach should not be repeated as though it were an untested option.

### 5. Hindi-to-English pronunciation bridges

The project then tested two ways of generating a Hindi-origin pronunciation while retaining an English target voice:

- **HYBRID:** Hindi T3 pronunciation conditioning combined with English S3Gen voice conditioning.
- **VC:** generate a Hindi source clip, then convert it through ChatterboxVC using the English target voice.

A controlled four-term experiment produced 24 clips on GPU. Both routes sounded good and nearly identical in the demo-voice comparison. The project then retested the routes inside real book sentences using the selected narrator recording. The user preferred HYBRID for its pronunciation and clarity and explicitly approved the narrator voice and HYBRID route on 2026-10-31.

That approval is deliberately narrow: it approves the narrator and the route, not every term's final spoken form and not full-book generation.

### 6. Remaining-term retests and aligned splicing

The nine remaining inventory terms were subsequently tested with the narrator: 45 clips were generated and their hashes verified. The user chose to keep **Buddha** in its ordinary English pronunciation. Accent drift, awkward pauses, stretched words, and pitch/pace mismatch were also identified in some of the fragment-based comparisons.

The repository now includes an aligned-splicing approach:

1. Render the complete English sentence in one pass.
2. Use PocketSphinx forced alignment to locate the English word or words corresponding to the target term.
3. Replace only that word span with the HYBRID term clip.
4. Use Praat/Parselmouth PSOLA to bring the inserted term's pitch and duration closer to the replaced English word, adjust its level within limits, and crossfade the joins.

On the saved test material, this reduced measured pre-term silence from roughly 140–340 ms to 0–60 ms in most cases and brought the unusually slow Subhuti clip closer to the surrounding word's pace. These are objective measurements, **not yet a human listening approval**. Optional English stand-in renders were added to investigate whether spelling in the sentence contributes to accent drift; that remains a hypothesis, not a proven cause.

### 7. Full-book synthesis and mastering

This stage has **not started**. The remaining planned work includes:

- Final, reversible text-to-spoken-form decisions and repeated-term consistency checks.
- Approval of a representative narration passage.
- Token-aware chunking under the model's real limits.
- Per-chunk checkpointing, hash verification, and retry of only failed/invalid chunks.
- Reconciliation of audio coverage against the approved manuscript.
- QA for missing or duplicate chunks, truncation, duration anomalies, silence, clipping, and discontinuities.
- Human listening review of chapter boundaries, joins, flagged clips, and representative samples.
- Chapter exports, a lossless master, and the final listener-friendly format.

**Do not treat a successful short audition as permission to synthesize the book.** The current code and notebooks keep full-book synthesis explicitly unauthorized.

## Current status

The most recent commit on the default branch when this README was written is `ad1ab21` (2026-11-01), which added aligned splicing and the optional stand-in experiment.

| Area | Current state |
|---|---|
| PDF reconnaissance and structural extraction | Implemented in the bundled starter engine and Stage 1 notebook; source-specific extraction has been run |
| Manuscript repair proposals | Implemented; provenance-aware and non-destructive |
| Manuscript approval | Completed for 48 cross-page joins; J048 remains excluded for manual review |
| Pronunciation inventory | 13-term inventory and audition workflow implemented |
| Baseline pronunciation auditions | 36/36 clips generated; English/IAST variants were not accepted, Hindi samples were more promising |
| English refinement route | Tested and rejected after listening |
| HYBRID and VC bridge experiments | Implemented; 24/24 demo-reference clips generated and reviewed |
| Narrator reference and HYBRID route | Real-narrator retest completed; narrator and HYBRID route explicitly approved |
| Remaining terms | Nine-term retest completed; Buddha stays in English; final spoken-form decisions for the other terms remain open |
| Aligned splicing | Implemented and measured on the saved clips; awaiting listening approval |
| Token-aware full-book synthesis | Not implemented |
| End-to-end audio QA, mastering, and export | Not implemented |

The immediate checkpoint recorded in [PROJECT_STATUS.md](PROJECT_STATUS.md) is to listen to the aligned splices, check whether the pause/pitch/pace improvements sound natural, and determine whether the accent issue remains. If it does, the optional stand-in comparison can help isolate the cause. The spoken-form mapping, repeated-term consistency check, and representative passage approval come before full-book synthesis.

### Latest automated test result

GitHub Actions on the latest main commit ran the Python helper test suite successfully: **114 passed** (workflow run 32, 2026-11-01). These tests cover the helper code, deterministic task planning, validation, checkpoint/resume behaviour, and audio-processing logic. They do **not** validate real-model audio quality, run a full GPU synthesis, or prove that the complete audiobook pipeline is finished.

---

## Getting started

### Stage 1: prepare and approve the manuscript

1. Open [Stage 1 in Colab](https://colab.research.google.com/github/privacyfreak84/colabchatterLLMthingstuff/blob/main/notebooks/01_pdf_reconnaissance.ipynb).
2. Run the notebook from the top. It clones/updates this repository, reconstructs and verifies the starter-engine archive, installs the document dependencies, mounts Google Drive, and writes the extraction and review outputs under `MyDrive/Audiobook/Buddhist Yoga/reconnaissance`.
3. Inspect the extraction report, page/block samples, chapter boundaries, proposed joins, and pronunciation inventory.
4. Review the proposed and approved manuscript tracks. The approval helper records the 48 accepted cross-page joins and the excluded same-page candidate.

Stage 1 is CPU/document work. It does not load Chatterbox or synthesize speech. Keep the outputs in Drive so they survive a Colab runtime reset.

### Stage 2: run short pronunciation experiments

1. Start with a fresh Colab runtime, as the notebook instructs.
2. Open [Stage 2](https://colab.research.google.com/github/privacyfreak84/colabchatterLLMthingstuff/blob/main/notebooks/02_pronunciation_auditions.ipynb) and run the setup cells in order.
3. The notebook installs the pinned Chatterbox source and downloads/verifies the V3 model files. The model snapshot is several gigabytes; it is stored as ordinary files in a dedicated Drive folder so it can be reused.
4. Generate and listen to the two smoke-test clips before enabling any larger audition.
5. Leave experiment switches off unless you explicitly intend to run that specific short test. The notebook's audition and bridge sections do not authorize full-book synthesis.

A GPU is useful for the larger auditions, but the notebook can fall back to CPU. CPU inference is much slower.

### Retest the narrator on a local CPU machine

The local runner is useful when Colab GPU access is unavailable. It checks a candidate narrator WAV and can run the same small retest on CPU or CUDA. The first inference run downloads several gigabytes of model files; later runs reuse the local snapshot. Outputs and manifests are resumable.

See [docs/local_cpu_setup.md](docs/local_cpu_setup.md) for the tested Python 3.11 setup and dependency pins.

Check a narrator recording without loading a model:

```bash
git clone https://github.com/privacyfreak84/colabchatterLLMthingstuff.git
cd colabchatterLLMthingstuff
python3 scripts/run_narrator_retest_local.py \
  --narrator ~/narrator_reference.wav --check-only
```

Run the nine later terms using the HYBRID route and include a plain-English control:

```bash
python scripts/run_narrator_retest_local.py \
  --narrator ~/narrator_reference.wav \
  --remaining-terms --routes hybrid --english-control
```

Build aligned splices from clips that already exist (this step does not load Chatterbox or download model weights):

```bash
python scripts/make_aligned_splices.py \
  --retest-dir ~/audiobook-narrator-retest/v3-narrator-retest \
  --base full
```

The aligned-splice step needs `pocketsphinx` and `praat-parselmouth`. The runner writes WAVs, a JSON manifest, and a transfer bundle. Listen to the actual WAVs; a clean manifest or a passing file-integrity check is not a listening-quality judgement.

**Keep your narrator recording private.** Supply it as a local file; do not commit it or upload it into the public repository.

---

## Repository map

| Path | Purpose |
|---|---|
| `notebooks/01_pdf_reconnaissance.ipynb` | Stage 1: PDF extraction, structure validation, section exports, repair proposals, manuscript approval |
| `notebooks/02_pronunciation_auditions.ipynb` | Stage 2: pinned Chatterbox V3 setup, smoke tests, pronunciation auditions, bridge tests, narrator retests |
| `notebooks/03_chatterbox_custom_text_gradio.ipynb` | Standalone custom-text Chatterbox V3 TTS app with a password-protected Gradio UI; independent of the audiobook pipeline |
| `tools/manuscript_repair.py` | Builds provenance-aware, reversible repair proposals and a pronunciation inventory draft |
| `tools/manuscript_approval.py` | Validates the review bundle, creates approved manuscript copies, and prepares source-context pronunciation prompts |
| `tools/pronunciation_audition.py` | Builds and runs checkpointed English/IAST/Devanagari audition tasks with tokenizer preflight |
| `tools/chatterbox_voice_bridge.py` | Runs/resumes the HYBRID and ChatterboxVC comparison matrix |
| `tools/narrator_reference.py` | Standard-library WAV checks for a candidate narrator reference |
| `tools/narrator_retest.py` | Builds and runs narrator-conditioned term tests, splices, manifests, and transfer bundles |
| `tools/aligned_splice.py` | Forced alignment and pitch/pace/level-matched replacement of a term inside a one-pass English sentence |
| `scripts/run_narrator_retest_local.py` | Local CPU/CUDA entry point for narrator retests |
| `scripts/make_aligned_splices.py` | Builds aligned splices from existing clips without running the model |
| `docs/local_cpu_setup.md` | Local Fedora/CPU installation and retest instructions |
| `tests/` | CPU-side tests for the Python helper modules |
| `.github/workflows/tests.yml` | Runs the helper test suite on pull requests and pushes to `main` |
| `Buddhist_Yoga_TTS_clean.pdf` | Source PDF used by the project |
| `assets/audiobook-engine-starter.tar.xz.part-*` | Six pieces of the compressed PDF-ingestion starter engine, reassembled and checksum-verified by Stage 1 |
| `Buddhist_Yoga_reconnaissance.zip`, `recon.txt` | Saved reconnaissance material |
| `nextrecon*.txt` | Historical experiment handoff bundles; several are base64-encoded ZIPs containing manifests and generated WAVs, not ordinary prose documentation |
| `conversations/` | Project conversation record |
| `PROJECT_STATUS.md` | Detailed evidence-backed current status and next action |
| `ROADMAP.md` | Stages, dependencies, and approval gates |
| `END_PRODUCT_GOAL.md` | Desired deliverables and audiobook acceptance criteria |

### About the bundled starter engine

The PDF-ingestion engine is not checked in as an already assembled root-level archive. It is stored as six parts under `assets/`. Stage 1 concatenates them into `audiobook-engine-starter.tar.xz`, verifies the expected size (**22,116 bytes**) and SHA-256:

```text
69ae62a5ec94d2784b910595ddf642a8acdc02d19029c5660dc3c66026615463
```

It then extracts the engine into the Colab runtime's `engine/` directory. This is why the Stage 1 notebook is the supported starting point for document extraction; simply looking for a ready-made `audiobook-engine-starter.tar.xz` at the repository root will not work.

---

## Engineering principles and guardrails

- **Document first:** inspect and approve the manuscript before spending GPU time.
- **No silent rewriting:** source text is preserved; repairs and spoken forms are separate, reviewable outputs.
- **Provenance matters:** source page/block IDs and hashes travel with the preparation and experiment records.
- **Human listening is authoritative for speech quality:** generation success, duration, spectral measurements, and valid WAV files are not enough.
- **Checkpoint expensive work:** task identities, seeds, settings, manifests, and SHA-256 hashes make runs reproducible and resumable.
- **Do not regenerate good evidence unnecessarily:** completed clips are reused only when their manifest identity and hashes still match.
- **Keep experiments opt-in:** smoke tests come first; larger audition batches require an explicit switch.
- **Keep private audio private:** the narrator reference recording is not part of this public repository.
- **No full-book synthesis before approval:** manuscript, spoken forms, narrator delivery, and a representative passage must pass their review gates first.

## License and reuse

No `LICENSE` file is currently present in the repository. Do not assume that the code, source PDF, or bundled experiment data has been released under a particular reuse license.
