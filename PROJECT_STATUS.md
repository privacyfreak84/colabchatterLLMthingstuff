# Project status

**Last updated:** 2026-10-31  
**Current pass:** Chatterbox Hindi-to-English voice bridge implementation  
**Working branch:** experiments/chatterbox-hindi-english-bridges  
**Full-book synthesis:** LOCKED

## Executive summary

The project is pursuing a complete, high-quality, zero-cost audiobook of the supplied 122-page Buddhist Yoga PDF. Chatterbox Multilingual V3 remains the preferred engine, but no model or pronunciation route is approved merely because it generates audio successfully.

The manuscript approval checkpoint is complete. The original 36 pronunciation auditions succeeded, but the English spelling/IAST variants were not good enough. A subsequent eight-clip English refinement experiment was also rejected after listening because the speech was slow, improperly pronounced, and artifact-ridden.

The current pass implements two more promising Chatterbox-native routes that keep Hindi pronunciation generation separate from English narrator rendering. Code and CPU-only task-builder tests are in the working branch. The new experiment has **not** yet been run on a GPU and has **not** been listened to.

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

**Implementation state:** written in the working branch.  
**Static notebook check:** passed; the updated notebook parses as valid nbformat 4 JSON and contains a default-OFF experiment gate.  
**Automated CI:** pending pull-request run at the time of this update.  
**GPU execution:** not run.  
**Listening approval:** not run.

Files added or changed in this pass:

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

The pace-oriented setting is exaggeration=0.7 and cfg_weight=0.3. It is an experiment, not a presumed improvement.

The English and Hindi target references are the public Chatterbox demo clips already prepared by Stage 2. They are diagnostic references, not the final audiobook narrator. A passing route must later be retested with the actual selected narrator recording.

### Technical design

- The hybrid route prepares T3 conditionals from the Hindi reference, independently creates an English S3Gen reference dictionary, and swaps only the S3Gen conditioning before generation.
- The voice-conversion route uses the already-loaded V3 S3Gen through the official ChatterboxVC class. It does not download a second decoder checkpoint.
- Each result is saved atomically, with a SHA-256 hash, duration, peak, RMS, clipping fraction, seed, and exact settings in the manifest.
- Successful results are reused only when the manifest identity and WAV hash match.
- The helper restores the in-memory model to its original Hindi T3/Hindi S3Gen state before returning.
- The notebook packages the manifest and WAVs into hindi_english_bridge_bundle.txt. After listening, upload that file to the repo root as nextrecon4.txt.
- The experiment never edits the approved manuscript or authorizes full-book synthesis.

## Immediate next action

1. Complete the PR/CI checks for this branch.
2. Once the notebook/helper changes are merged to main, open the updated Stage 2 Colab notebook, run from the top, leave the earlier full-audition and English-probe gates OFF, and enable only section 11.
3. Listen to all 24 clips, especially whether Hindi pronunciation survives while the English target voice remains natural and consistent.
4. Upload the generated bridge bundle as nextrecon4.txt.
5. Update this status file with the actual generated counts, manifest/checkpoint hashes, listening decisions, and next experiment.

## Decision policy

- Code added is not the same as a completed experiment.
- CI passing is not GPU validation.
- A valid WAV is not automatically good speech.
- Hindi source quality, English voice similarity, pacing, and audible artifacts must be judged separately.
- Do not approve any pronunciation or start full-book synthesis without explicit human review.
- Update this file at the end of every meaningful engineering/research pass, including what was executed, what was listened to, what passed/failed, and the single next action.
