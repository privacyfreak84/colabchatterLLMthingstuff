# Free PDF-to-audiobook engine

A document-first, resumable audiobook pipeline centred on **Chatterbox Multilingual V3** and Google Colab. The first stages are CPU-only: reconstruct the PDF's structure, extract the manuscript with page/block provenance, validate it, prepare reversible narration repairs, and assemble a human-reviewed pronunciation inventory before spending GPU time.

## Start in Google Colab

[**Open the PDF reconnaissance and repair-review notebook**](https://colab.research.google.com/github/privacyfreak84/colabchatterLLMthingstuff/blob/feat/manuscript-repair-review/notebooks/01_pdf_reconnaissance.ipynb)

The notebook checks out the review branch, expands the starter engine archive into a separate `engine/` directory, and runs extraction against `Buddhist_Yoga_TTS_clean.pdf`. It then runs the lightweight repair-review helper in `tools/`. Results are saved to Google Drive and survive Colab resets.

## Stage 1 outputs

- PDF metadata, page statistics, bookmarks, and source provenance
- Reading-order manuscript plus source-level JSONL text blocks
- Structural validation against the known eight-chapter book layout
- Separate preface, chapter, main-manuscript, and Notes exports
- Pronunciation candidate inventory and editable review CSV
- Warnings for suspicious line-break hyphenation and repeated page-edge text
- **Proposed** narration copies with provenance-linked cross-page join suggestions
- Same-page continuation candidates listed for manual review, never auto-merged
- Optional title-page opening proposal
- A focused pronunciation dictionary draft with blank spoken forms pending audition

## Engineering status

| Component | Status |
|---|---|
| PDF extraction and structural reconnaissance | Implemented; 17 foundation tests passed |
| Provenance-aware narration repair proposals | Implemented in review helper; 8 helper tests passed |
| Pronunciation candidate inventory | First-pass heuristic plus curated draft; human approval required |
| Chatterbox V3 adapter | Initial adapter; GPU integration not yet validated |
| Token-aware chunking and resumable synthesis | Not started |
| ASR verification and audiobook mastering | Planned |

The repair helper never overwrites the source PDF, extracted manuscript, original main/Notes tracks, or section files. It applies high-heuristic cross-page joins only to separate proposed copies, and only when the exact adjacent block pair appears uniquely in the source track. All same-page candidates remain review-only.

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
- No audio synthesis until the user approves the repaired manuscript and pronunciation dictionary.
