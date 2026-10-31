# Project status

**Last updated:** 2026-10-31  
**Current pass:** Narrator-voice retest tooling built; waiting on the narrator recording  
**Merged to:** main via PR #9, merge commit 047ac7f40e6e539da13d7b3fcf1d5381ada9b175  
**Full-book synthesis:** LOCKED

## Executive summary

The project is pursuing a complete, high-quality, zero-cost audiobook of the supplied 122-page Buddhist Yoga PDF. Chatterbox Multilingual V3 remains the preferred engine, but no model or pronunciation route is approved merely because it generates audio successfully.

The manuscript approval checkpoint is complete. The original 36 pronunciation auditions succeeded, but the English spelling/IAST variants were not good enough. A subsequent eight-clip English refinement experiment was also rejected after listening because the speech was slow, improperly pronounced, and artifact-ridden.

The bridge experiment has been run successfully on GPU (24/24 clips) and reviewed by listening. The user reports that the HYBRID (Hindi T3 + English S3Gen) and VC (ChatterboxVC) routes sound almost identical and that both sound good, a major positive signal after rejecting the English-refinement batch. The user has not reported a preference between baseline and pace settings, so those remain undecided. Because these auditions used public demo reference voices, nothing is approved yet: the next step is to retest the bridge with the actual narrator recording. PR #9 is merged to main.

## Verified completed work

### Source and manuscript

- Source PDF: Buddhist Yoga, 122 pages.
- Reconnaissance artefacts are in the repository, including the reconnaissance ZIP and recon text.
- Approved main and Notes manuscript checkpoints exist in the Google Drive workflow.
- Approval record accepts 48 cross-page joins and excludes J048 for manual review.
- Full-book synthesis remains explicitly unauthorized.

### Original Chatterbox V3 auditions

- 36/36 original pronunciation audition WAVs generated.
- Hindi samples exist for all 13 curated terms.
- No failed tasks in the completed bundle.
- Preserve the bundle and do not regenerate successful recordings.
- Hindi/Devanagari samples were more promising than the English and IAST variants, but still require a final human pronunciation decision.

### English refinement experiment

- PR #8 merged at commit ca883843763389b7e4742401b884d8b15e53d88b.
- The user uploaded nextrecon3.txt at commit cb0442fccde2011658b259a764f1918b810e6f1a.
- Its manifest records all eight probes as generated with no generation errors.
- The user listened and rejected the English results: slow delivery, inaccurate pronunciations, and audible artifacts.
- Do not repeat the same English respelling / cross-language CFG experiment or try to rescue it using only temperature and repetition-penalty sweeps.

## Current pass: Hindi-to-English Chatterbox bridges

**Implementation state:** merged to main in PR #9.  
**Merge commit:** 047ac7f40e6e539da13d7b3fcf1d5381ada9b175.  
**Static notebook check:** passed; the updated notebook parses as valid nbformat 4 JSON and contains a default-OFF experiment gate.  
**Automated CI:** passed, 32 tests in 0.06 seconds (GitHub Actions run 41135587975).  
**GPU execution:** complete; 24/24 clips generated, zero failed tasks.  
**Bundle uploaded:** nextrecon4.txt (commit d5cb8ac).  
**User listening feedback:** HYBRID and VC sound almost identical and both sound good. No baseline-versus-pace preference reported.  
**Pronunciation/voice approval:** pending. Both routes are acceptable on the demo references; approval needs a retest with the actual narrator recording.

Files added or changed in this pass (now on main):

- tools/chatterbox_voice_bridge.py — resumable, hash-verified bridge runner.
- tests/test_chatterbox_voice_bridge.py — CPU-only tests for the deterministic 24-task plan and its settings.
- notebooks/02_pronunciation_auditions.ipynb — new opt-in section 11.
- README.md — current experimental branch and project document links.
- ROADMAP.md — staged work and gates.
- END_PRODUCT_GOAL.md — end-product requirements and acceptance gates.
- PROJECT_STATUS.md — this living status record.

### What section 11 tests

Four representative terms, six clips per term (24 total):

1. Hindi TTS source, existing baseline settings.
2. Hindi TTS source, pace-oriented settings.
3. Hindi T3 conditioning + English S3Gen reference, baseline settings.
4. Hindi T3 conditioning + English S3Gen reference, pace-oriented settings.
5. Hindi source passed through ChatterboxVC to the English target voice.
6. Pace-oriented Hindi source passed through ChatterboxVC to the English target voice.

The pace-oriented setting is exaggeration=0.7 and cfg_weight=0.3. It is not a presumed improvement: manifest durations show that it did not consistently shorten the Hindi-source or hybrid clips and made several longer. Retain it only if the user prefers its perceived delivery.

The English and Hindi target references are the public Chatterbox demo clips already prepared by Stage 2. They are diagnostic references, not the final audiobook narrator. A passing route must later be retested with the actual selected narrator recording.

### Technical design

- The hybrid route prepares T3 conditionals from the Hindi reference, independently creates an English S3Gen reference dictionary, and swaps only the S3Gen conditioning before generation.
- The voice-conversion route uses the already-loaded V3 S3Gen through the official ChatterboxVC class. It does not download a second decoder checkpoint.
- Each result is saved atomically, with a SHA-256 hash, duration, peak, RMS, clipping fraction, seed, and exact settings in the manifest.
- Successful results are reused only when the manifest identity and WAV hash match.
- The helper restores the in-memory model to its original Hindi T3/Hindi S3Gen state before returning.
- The notebook packages the manifest and WAVs into hindi_english_bridge_bundle.txt. That transfer file was uploaded to the repo root as nextrecon4.txt (a base64-encoded zip containing the manifest and 24 WAVs).
- The experiment never edits the approved manuscript or authorizes full-book synthesis.

## Bundle verification (nextrecon4.txt)

Checked on 2026-10-31 by decoding nextrecon4.txt (base64 zip) and reading its manifest:

- 24 tasks, all with status `generated`; 4 terms (Avalokiteshvara, Maitreya, Manjushri, Vijnanavada) x 6 variants.
- All 24 WAV SHA-256 hashes match the manifest; manifest durations match the actual WAV lengths (2.56-3.28 s per clip).
- No clipping in any clip; sample rate 24 kHz.
- `full_book_synthesis_authorized` is `False`.
- This verifies file integrity only. Audio quality is judged by listening, not by these checks.
- The repo test suite passed locally on main at da2d9d2 (32 tests before the narrator-retest tooling was added).

## Listening result

- HYBRID and VC sound almost identical to the user, and both sound good.
- The pace preset (exaggeration=0.7, cfg_weight=0.3) was not reported as better or worse; it did not consistently shorten clips, so it has no demonstrated benefit yet.
- No per-term winner was identified, and none is needed yet: the routes are interchangeable on the demo voices.

## Narrator retest tooling (built, not yet run on GPU)

Files added in this pass:

- tools/narrator_reference.py: stdlib-only checker for a candidate narrator WAV (length, sample rate, clipping, noise floor, silence in the first 10 s that Chatterbox actually uses). It reports blocking problems and warnings; it cannot judge whether the voice suits the book.
- tools/narrator_retest.py: for four terms, splits a real manuscript sentence at the term, renders the English before/after in the narrator voice, renders the term through both bridge routes (HYBRID and VC), and splices each route between the English parts. 28 tasks per run (20 generated clips, 8 CPU splices), resumable and hash-verified, with the join settings recorded in the manifest.
- tests/test_narrator_reference.py and tests/test_narrator_retest.py: CPU-only tests. The full suite is now 59 tests and CI installs numpy for them.
- notebooks/02_pronunciation_auditions.ipynb: new section 12 with a checker cell and a gated (default OFF) retest cell. After listening, the bundle is uploaded as nextrecon5.txt.

What was verified: the 59 tests pass locally, and the GPU runner's control flow (conditioning per task, resume, identity guard, blocking-recording guard) was exercised end-to-end with a stubbed model. What was not verified: any real Chatterbox generation. Open assumptions the first GPU run will test: that a bare Devanagari term (no carrier sentence) renders cleanly, and that the 60 ms join pause (180 ms before a comma) sounds natural. The pace preset is dropped from this retest because it showed no benefit.

## Immediate next action

1. Record the narrator reference: a clean 8-10 s WAV in a quiet room, representative of the delivery wanted for the whole book. Save it as narrator_reference.wav in the references folder on Drive.
2. Run the Step 1 checker cell in section 12 and fix any blocking problems.
3. Enable RUN_NARRATOR_RETEST in the Step 2 cell, listen to the eight splices (two per term), and upload the bundle as nextrecon5.txt.
4. Update this file with the counts, the verified hashes, and your listening decision. Keep full-book synthesis locked until the voice and pronunciation decisions are explicitly approved.

## Decision policy

- Code added is not the same as a completed experiment.
- CI passing is not GPU validation.
- A valid WAV is not automatically good speech.
- Hindi source quality, English voice similarity, pacing, and audible artifacts must be judged separately.
- Do not approve any pronunciation or start full-book synthesis without explicit human review.
- Update this file at the end of every meaningful engineering/research pass, including what was executed, what was listened to, what passed/failed, and the single next action.
