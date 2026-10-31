# Audiobook pipeline roadmap

This roadmap records the intended sequence and decision gates. The current, evidence-backed state is in PROJECT_STATUS.md; the end-product acceptance criteria are in END_PRODUCT_GOAL.md.

## Pass 0 — Reconnaissance and source inventory

**Complete**

- Inspect the source PDF, its structure, page order, text extraction, and available free runtimes/models.
- Preserve source provenance and identify page-boundary and reading-order problems.
- Store reconnaissance artefacts in the repository.

## Pass 1 — Manuscript reconstruction and approval

**Complete**

- Produce separate preface, main-book, and Notes tracks.
- Validate the known chapter structure and provenance.
- Generate explicit, reversible cross-page join proposals.
- Apply only reviewed joins to proposed copies.
- Approve the main/Notes manuscript checkpoints and record exclusions.
- Keep same-page uncertain joins as manual-review candidates.

## Pass 2 — Pronunciation inventory and baseline auditions

**Complete; decisions pending**

- Build the curated 13-term pronunciation sheet.
- Run the 36-task Chatterbox V3 baseline audition batch.
- Preserve the 36 WAVs and manifest; do not regenerate them.
- Treat Hindi Devanagari samples as promising candidates, not automatic approvals.
- Reject the tested source-English and IAST variants where listening showed poor pronunciations.

## Pass 3 — English-path refinement

**Tested and rejected**

- PR #8 added four-term English phonetic respelling and Hindi-script / English-reference experiments.
- All eight clips generated successfully, but the user found the English results slow, improperly pronounced, and artifact-ridden.
- Keep nextrecon3.txt as historical evidence. Do not spend more time merely sweeping the same English TTS sampling settings.

## Pass 4 — Chatterbox Hindi-to-English voice bridges

**Merged to main (PR #9); 32 CPU tests passed; GPU audition completed (24/24) and reviewed; HYBRID and VC both sound good and almost identical to the user; narrator retest pending**

This pass added two controlled ways to retain Chatterbox while separating pronunciation generation from English narrator rendering:

1. **Split T3/S3Gen conditioning:** prepare T3 from a Hindi reference and independently prepare S3Gen from the English target-voice reference. Generate Hindi text with Hindi language ID while decoding through the English-conditioned S3Gen.
2. **ChatterboxVC:** generate a correctly pronounced Hindi source clip with Hindi/Hindi conditioning, then pass its speech through Chatterbox's S3Gen-based voice-conversion component using the English target-voice reference.

For four representative terms, create six comparisons per term:

- Hindi source, baseline settings.
- Hindi source, pace-oriented settings.
- Hindi T3 + English S3Gen, baseline settings.
- Hindi T3 + English S3Gen, pace-oriented settings.
- Hindi source converted to English target voice, baseline source.
- Hindi source converted to English target voice, pace-oriented source.

The pace-oriented preset (exaggeration=0.7, cfg_weight=0.3) is an experiment informed by upstream guidance, not an assumed improvement. Both references are currently public Chatterbox demo voices, not the final narrator. The GPU run and human listening review are still required.

**Outcome so far:** the user judged HYBRID and VC almost identical and both good on the public demo references. The pace preset showed no demonstrated benefit. No route is approved yet.

**Next decision gate:** retest with the intended audiobook narrator recording and contextual sentences. Compare HYBRID and VC again for pronunciation, natural pacing, voice similarity, and artifacts. If they remain equivalent, choose on practical grounds (complexity, speed, resumability). Reject a route that sounds like a language/voice splice.

## Pass 5 — Improve Hindi pronunciation source, only if needed

**Conditional**

- Compare the general multilingual V3 Hindi output with the dedicated Chatterbox Hindi single-language T3 fine-tune.
- Confirm checkpoint compatibility and licensing/usage details before integration.
- Retain the same voice-bridge comparison so the only changed variable is the Hindi pronunciation generator.
- Do not download another multi-gigabyte checkpoint unless the current Hindi source is the limiting factor.

## Pass 6 — Final narrator and pronunciation approval

**Not started**

- Select the actual audiobook narrator reference recording, preferably a clean, representative 8–10 second speech sample.
- Re-audition only the chosen pronunciation route with that target voice.
- Decide each candidate spoken form and record the approved reversible text-to-spoken-form mapping.
- Verify English/Hindi/converted insertions are consistent across repeated terms.
- Explicitly approve the voice and a representative narration passage.

## Pass 7 — Chunking and resumable full-book synthesis

**Locked pending approval**

- Split the approved manuscript at natural structural boundaries and then into model-safe chunks.
- Preserve chapter/section IDs, source offsets, text hashes, and exact chunk order.
- Generate and checkpoint each chunk independently.
- Reuse verified outputs; retry only failed or invalid chunks.
- Never allow an experimental pronunciation change to mutate the approved source manuscript.

## Pass 8 — QA, mastering, and export

**Not started**

- Reconcile every audio chunk against the approved manuscript.
- Run duration, file-integrity, silence, clipping, discontinuity, duplicate, and missing-chunk checks.
- Listen to chapter transitions, flagged chunks, and representative samples.
- Assemble chapter masters and the complete audiobook.
- Export the agreed listening format and verify playback from beginning to end.

## Operating rule: update status after every pass

At the end of each meaningful engineering/research pass, update PROJECT_STATUS.md with:

- What changed (files, commits, PRs, model/code revisions).
- What was actually executed and verified.
- What the user listened to and accepted/rejected.
- What remains untested or blocked.
- The single immediate next action.

Do not mark a pass complete because its code was written or CI passed. GPU execution and listening-based quality gates must be reported separately.
