# End-product goal: a complete, faithful Buddhist Yoga audiobook

## Mission

Turn the supplied 122-page Buddhist Yoga PDF into a complete, high-quality audiobook using a reproducible, zero-cost workflow. The result should be pleasant to listen to for the entire book, not merely a technically valid collection of generated WAV files.

The intended engine is Chatterbox Multilingual V3 unless controlled auditions demonstrate that another free/open model or a hybrid strategy is materially better. Chatterbox remains the preferred route because its audio quality was promising in the user's initial testing and its source, model assets, and inference workflow can be run without paid API credits.

## Required qualities

1. **Completeness:** Every approved part of the book is narrated in the correct order, with no missing, duplicated, or truncated passages.
2. **Text fidelity:** Preserve the author's wording and meaning. Repairs to extraction artefacts, punctuation, page joins, and pronunciation must be explicit, reversible, and traceable to the source.
3. **Narration quality:** Natural, intelligible, consistent English narration with comfortable pacing, clean joins, no conspicuous synthesis artefacts, and no unwanted repetition or hallucinated text.
4. **Terminology:** Sanskrit, Buddhist, philosophical, and other specialist terms must be pronounced acceptably and consistently. A written spelling is not automatically an approved spoken form.
5. **Voice consistency:** Keep a coherent narrator identity across the book. Any non-English or converted insertions must blend closely enough not to sound like a different audiobook.
6. **Structure:** Preserve the book's preface, chapters, section order, and Notes. Keep useful chapter boundaries in the audio deliverables.
7. **Reproducibility:** Every expensive generation step must be checkpointed, resumable, hash-verifiable, and safely retryable without regenerating successful work.
8. **Cost:** Spend zero money. Use free Colab/runtime availability and downloadable model assets; never rely on a paid inference endpoint or paid export credit.

## Expected deliverables

The final deliverables are expected to include:

- A complete mastered audiobook in a widely playable format.
- Separate chapter/section audio files so a listener can resume or navigate easily.
- A lossless intermediate master for quality control and future remastering.
- A cue/manifest file recording source sections, chunk order, timing, and file hashes.
- A pronunciation decision record documenting approved spoken forms and any exceptions.
- A short README describing how the final audio was assembled and verified.

The exact lossy format, bitrate, loudness target, cover art, and whether a single M4B file is also produced will be selected during mastering. These are delivery decisions, not reasons to delay the core pipeline.

## Acceptance gates

The audiobook is **not complete** merely because synthesis finished. Completion requires:

- Manuscript approval and explicit pronunciation decisions.
- A small representative narration test approved for voice, intelligibility, pace, and sound quality.
- Full-book chunk coverage reconciled against the approved manuscript.
- Automated checks for missing/duplicate chunks, unexpected durations, silence, clipping, malformed audio, and discontinuities.
- Human listening review of chapter openings, endings, joins, all flagged chunks, and a representative sample throughout the book.
- Final assembly and playback verification of the complete master and chapter files.

## Non-goals

- Silently rewriting the source manuscript to make a TTS model pronounce a word.
- Treating zero failed Python tasks as proof of good narration.
- Accepting slow, distorted, accented, or artifact-ridden output merely to keep one model.
- Starting full-book synthesis before the voice and pronunciation strategy is approved.
- Paying for credits, subscriptions, hosted GPU inference, or export services.

**North star:** a complete audiobook that sounds intentionally narrated, faithfully represents the book, and costs nothing to produce.
