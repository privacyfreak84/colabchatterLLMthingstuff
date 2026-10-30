# Free PDF-to-audiobook engine

A document-first, resumable audiobook pipeline centred on **Chatterbox Multilingual V3** and Google Colab. The first committed stage is CPU-only: reconstruct the PDF's structure, extract the manuscript with page/block provenance, validate it, and prepare a reviewable pronunciation watchlist before spending GPU time.

## Start in Google Colab

[**Open the PDF reconnaissance notebook**](https://colab.research.google.com/github/privacyfreak84/colabchatterLLMthingstuff/blob/feat/audiobook-engine-foundation/notebooks/01_pdf_reconnaissance.ipynb)

The notebook checks out the project branch, expands `audiobook-engine-starter.tar.xz` into a separate `engine/` directory, and runs the extraction against `Buddhist_Yoga_TTS_clean.pdf`. Results are saved to Google Drive and survive Colab resets.

## Stage 1 outputs

- PDF metadata, page statistics, bookmarks, and source provenance
- Reading-order manuscript plus source-level JSONL text blocks
- Structural validation against the known eight-chapter book layout
- Separate preface, chapter, main-manuscript, and notes exports
- Pronunciation candidate inventory and editable review CSV
- Warnings for suspicious line-break hyphenation and repeated page-edge text

## Engineering status

| Component | Status |
|---|---|
| PDF extraction and structural reconnaissance | Implemented; 17 automated tests pass |
| Buddhist Yoga project configuration | Implemented |
| Pronunciation candidate inventory | First-pass heuristic; human review required |
| Chatterbox V3 adapter | Initial adapter; GPU integration not yet validated |
| Token-aware chunking and resumable synthesis | Next implementation stage |
| ASR verification and audiobook mastering | Planned |

The source package is stored as a compact archive so the notebook can bootstrap it without dozens of separate setup steps. The archive extracts into `engine/`; it does not overwrite the PDF or conversation history.

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
