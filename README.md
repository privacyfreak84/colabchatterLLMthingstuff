# Free PDF-to-audiobook engine

A document-first, resumable audiobook pipeline centred on **Chatterbox Multilingual V3** and Google Colab. The first stages are CPU-only: reconstruct the PDF's structure, extract the manuscript with page/block provenance, validate it, prepare reversible narration repairs, and assemble a human-reviewed pronunciation inventory before spending GPU time.

## Project documents

- [Current project status](PROJECT_STATUS.md) — updated at the end of every engineering/research pass.
- [Roadmap](ROADMAP.md) — staged plan, completion gates, and next work.
- [End-product goal](END_PRODUCT_GOAL.md) — required audiobook qualities and acceptance criteria.

## Start in Google Colab

[**Open Stage 1: PDF reconnaissance and manuscript approval**](https://colab.research.google.com/github/privacyfreak84/colabchatterLLMthingstuff/blob/main/notebooks/01_pdf_reconnaissance.ipynb)

[**Open Stage 2: repaired Chatterbox V3 pronunciation auditions**](https://colab.research.google.com/github/privacyfreak84/colabchatterLLMthingstuff/blob/main/notebooks/02_pronunciation_auditions.ipynb)

The notebook checks out the latest main branch, expands the starter engine archive into a separate `engine/` directory, and runs extraction against `Buddhist_Yoga_TTS_clean.pdf`. It then runs the lightweight repair-review helper in `tools/`. Results are saved to Google Drive and survive Colab resets.

## Stage 1 outputs

- PDF metadata, page statistics, bookmarks, and source provenance
- Reading-order manuscript plus source-level JSONL text blocks
- Structural validation against the known eight-chapter book layout
- Separate preface, chapter, main-manuscript, and Notes exports
- Pronunciation candidate inventory and editable review CSV
- Warnings for suspicious line-break hyphenation and repeated page-edge text
- **Proposed** narration copies with provenance-linked cross-page join suggestions
- Explicit approval record and separate approved main/Notes narration masters
- Pronunciation audition CSV and listening prompts built from exact source-context sentences
- Same-page continuation candidates listed for manual review, never auto-merged
- Optional title-page opening proposal
- A focused pronunciation dictionary draft with blank spoken forms pending audition

## Engineering status

| Component | Status |
|---|---|
| PDF extraction and structural reconnaissance | Implemented in the starter engine archive; 17 foundation tests (in the archive, not run by repository CI) |
| Provenance-aware narration repair proposals | Implemented in review helper; covered by the repository helper tests (32 in total across the helpers, run by CI) |
| Approved manuscript tracks and pronunciation audition prompts | Implemented; approval is scoped to 48 applied cross-page joins, with same-page candidate excluded |
| Pronunciation candidate inventory | First-pass heuristic plus curated draft; human approval required |
| Chatterbox V3 adapter | Initial adapter; validated on GPU through the 36-clip baseline audition, the 8-clip English probe, and the 24-clip Hindi-English bridge audition. Validated with the approved narrator voice on four terms (CPU) |
| Narrator-voice retest (reference checker, sentence-splice test) | Tooling built; runs on GPU or CPU (Colab notebook or `scripts/run_narrator_retest_local.py`, see `docs/local_cpu_setup.md`); run on the real model for four terms; narrator voice and HYBRID route approved by the user; remaining nine terms pending |
| Token-aware chunking and resumable synthesis | Not started |
| ASR verification and audiobook mastering | Planned |

The repair helper never overwrites the source PDF, extracted manuscript, original main/Notes tracks, or section files. It applies high-heuristic cross-page joins only to separate proposed copies, and only when the exact adjacent block pair appears uniquely in the source track. All same-page candidates remain review-only. The approval helper copies the proposed tracks into separate approved files only after validating the complete review bundle; it records accepted candidate IDs and excluded same-page items. Pronunciation auditions use source spellings and exact source-context sentences, with spoken forms left blank until listening and approval. Stage 2 first generates one English and one Hindi smoke-test clip, then stops by default. The user must listen and set `RUN_FULL_AUDITIONS = True` to opt into the 36 pronunciation samples. It compares source English spelling, IAST transliteration, and Devanagari Hindi through Chatterbox Multilingual V3. It installs a pinned upstream Git commit rather than the stale PyPI 0.1.7 wheel, verifies that the V3 selector is present and the legacy alignment analyser is absent, preflights tokenizer unknowns and checkpoint files, stores the selected model files as ordinary Drive files rather than in the global Hugging Face cache, runs an English/Hindi smoke test first, and checkpoints every generated sample with its seed and hash. No `model.generate` monkey-patching or `torch.load(weights_only=False)` workaround is used.

The optional English-refinement probe in Stage 2 tested provisional English phonetic respellings and Devanagari pronunciation using an English reference voice. The user rejected those English outputs for poor pronunciation, slow delivery, and artifacts; preserve the bundle as evidence, but do not keep tuning that same route. The current Stage 2 notebook adds an opt-in Hindi-to-English bridge audition: (1) separate Hindi T3 and English S3Gen conditioning (HYBRID), and (2) Hindi-source speech converted with ChatterboxVC to the English target voice (VC). It creates 24 short clips across four terms, records settings and audio diagnostics, and resumes from hash-verified outputs. The experiment has been run on GPU (24/24 clips generated, hashes verified) and reviewed by listening: the user finds HYBRID and VC almost identical and both good. These auditions used public demo reference voices, so nothing is approved yet; the next step is a retest with the actual narrator recording. The experiment does not edit the manuscript or authorize full-book synthesis.

## Local use

```bash
mkdir -p engine
tar -xJf audiobook-engine-starter.tar.xz -C engine
python -m pip install -e 'engine[dev]'
python -m audiobook.cli inspect \
  --pdf Buddhist_Yoga_TTS_clean.pdf \
  --out outputs/buddhist-yoga/recon \
  --config engine/config/buddhist_yoga.yaml
pytest -q
```

## Design constraints

- Free-only workflow; no paid inference APIs.
- Preserve source wording and page provenance; never silently rewrite meaning.
- Do not OCR digitally-generated text unless native extraction proves inadequate.
- Respect the checkpoint's real tokenizer limits, not character-count guesses.
- Persist every expensive synthesis result and make chunk-level retries reproducible.
- Keep pronunciation corrections explicit, reviewable, and reversible.
- No full-book audio synthesis until the user approves the repaired manuscript and pronunciation dictionary.
- Pronunciation auditions are a limited test run and do not authorize full-book synthesis.
- Stage 2 uses a pinned Chatterbox Git revision because PyPI 0.1.7 lacks the optional `t3_model` selector and contains a legacy alignment analyser that is absent from the pinned upstream source.
- Short pronunciation auditions are a separate, reversible review activity; they do not authorize full-book synthesis.
