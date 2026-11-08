# Project status

**Last updated:** 2026-11-08  
**Current pass:** Pass 6: narrator voice and HYBRID route approved by the user (2026-10-31); spoken-form mapping for the remaining nine terms pending  
**Merged to:** main via PR #9, merge commit 047ac7f40e6e539da13d7b3fcf1d5381ada9b175  
**Full-book synthesis:** LOCKED

## Executive summary

The project is pursuing a complete, high-quality, zero-cost audiobook of the supplied 122-page Buddhist Yoga PDF. Chatterbox Multilingual V3 remains the preferred engine, but no model or pronunciation route is approved merely because it generates audio successfully.

The manuscript approval checkpoint is complete. The original 36 pronunciation auditions succeeded, but the English spelling/IAST variants were not good enough. A subsequent eight-clip English refinement experiment was also rejected after listening because the speech was slow, improperly pronounced, and artifact-ridden.

The bridge experiment has been run successfully on GPU (24/24 clips) and reviewed by listening. The user reports that the HYBRID (Hindi T3 + English S3Gen) and VC (ChatterboxVC) routes sound almost identical and that both sound good, a major positive signal after rejecting the English-refinement batch. The user has not reported a preference between baseline and pace settings, so those remain undecided. These auditions used public demo reference voices. The bridge was then retested with the actual narrator recording on four real manuscript sentences (see the narrator retest results below): the user found the results very good and prefers HYBRID for every term. The user then explicitly approved the narrator voice and the HYBRID route (see the approval record below). PR #9 is merged to main.

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
**Pronunciation/voice approval:** the user approved the narrator voice and the HYBRID route on 2026-10-31 (see the approval record below). Terms other than the four heard so far are not yet approved.

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

## Narrator retest tooling (built and run)

Files added in this pass:

- tools/narrator_reference.py: stdlib-only checker for a candidate narrator WAV (length, sample rate, clipping, noise floor, silence in the first 10 s that Chatterbox actually uses). It reports blocking problems and warnings; it cannot judge whether the voice suits the book.
- tools/narrator_retest.py: for four terms, splits a real manuscript sentence at the term, renders the English before/after in the narrator voice, renders the term through both bridge routes (HYBRID and VC), and splices each route between the English parts. 28 tasks per run (20 generated clips, 8 CPU splices), resumable and hash-verified, with the join settings recorded in the manifest.
- tests/test_narrator_reference.py, tests/test_narrator_retest.py and tests/test_narrator_retest_runner.py: CPU-only tests, including a stand-in-model run of the whole retest flow. The full suite is now 69 tests and CI installs numpy and soundfile for them.
- notebooks/02_pronunciation_auditions.ipynb: section 12 with a checker cell and a gated (default OFF) retest cell. The model-loading cell no longer requires a GPU and falls back to CPU. After listening, the bundle is uploaded as nextrecon5.txt.
- scripts/run_narrator_retest_local.py and docs/local_cpu_setup.md: run the same retest outside Colab on a CPU-only machine (written for a Fedora laptop with 16 GB RAM and no usable GPU). On CPU it runs the two shortest terms by default; runs are resumable and can be extended term by term. The manifest records seconds per clip.

Before the real run, the 69 tests passed and the runner's control flow was exercised with a stand-in model; the pinned dependency set resolves together and the recording checker runs on a bare Python. The real run then confirmed the two open assumptions in the user's listening: a bare Devanagari term (no carrier sentence) renders cleanly, and the join pauses (60 ms, or 180 ms before a comma) sound natural. The pace preset was dropped from this retest because it showed no benefit.

## Narrator retest results

**Narrator reference:** about 19.6 s, 24 kHz mono WAV, passes the checker with no blocking problems (speech about 50 dB above the noise floor, no clipping). Its silences are digital zeros, so some noise gating or suppression was applied upstream. The file is deliberately not in this public repository. Chatterbox uses roughly the first 10 s.

**Execution:** on the user's laptop (Fedora, Ryzen 5 5625U, 16 GB RAM, CPU only, Python 3.11, torch 2.6.0+cpu) with scripts/run_narrator_retest_local.py. Four real manuscript sentences, the term split out and rendered through both routes, each route spliced between English parts rendered in the narrator voice.

- Run 1, nextrecon5.txt: Maitreya (N002) and Vijnanavada (N004), 14/14 tasks, 227 s of generation.
- Run 2, nextrecon6.txt: added Manjushri (N003) and Avalokiteshvara (N001), 28/28 tasks in the manifest, 308 s of new generation. The Avalokiteshvara suffix is 23.5 s of audio generated in 150 s, under the 40 s token cap, so it was not truncated.
- CPU speed: about 7 s of generation per second of audio on long clips, about 10-20 s fixed cost per short clip. A full four-term run is roughly 9 minutes of generation.

**Verification of the bundles (2026-10-31):** all 28 SHA-256 hashes match the manifests, manifest durations match the WAVs, all clips are 24 kHz with status generated, the run 1 clips are byte-identical in run 2, and `full_book_synthesis_authorized` is False. Six clips (N003_EN_PREFIX, N001_EN_SUFFIX and the four splices containing them) have exactly one full-scale sample each: the model's raw output overshot by about 1-4% before 16-bit conversion. That is inaudible, but a peak limiter belongs in the full-book assembly step.

**Objective hints (first two terms only, crude, not a quality judgement):** the bridged term sat within about 2 dB of the surrounding English in level. Pitch medians were 206-222 Hz for the English parts against 220 Hz for the narrator reference, 187-193 Hz for HYBRID terms and 197/212 Hz for VC. Maitreya VC scored well below the other clips on a rough spectral comparison with the narrator, consistent with the user's preference for HYBRID, but the measure is crude and the clips are about a second long.

**User listening result:**

- Maitreya and Vijnanavada: everything sounds very good; HYBRID and VC sound almost identical.
- Overall, across all terms, the user prefers HYBRID: a little better pronunciation and clarity.
- This was first a stated preference; the user then gave an explicit approval (next section).

## Approval record

On 2026-10-31 the user explicitly approved **the narrator voice** and **the HYBRID route**, after listening to four terms in real manuscript sentences. The approved configuration is exactly what was run:

- Narrator reference: the user's recording (first 10 s, padded to a 40 ms multiple). The file is kept out of this public repository.
- HYBRID: Hindi T3 conditioned on the public Hindi demo reference (hi_f1) with the narrator's S3Gen conditioning, bare Devanagari term text with no carrier sentence.
- English parts rendered in the narrator voice; generation parameters exaggeration 0.5, cfg_weight 0.5, temperature 0.7, repetition_penalty 1.2, min_p 0.05, top_p 0.95.
- Join: edges trimmed below -45 dB with a 30 ms margin, 10 ms edge fades, 60 ms pause (180 ms when a comma follows the term).

Scope: this approves the voice and route. It does not approve the spoken form of the nine terms not yet heard with the narrator, and it does not authorize full-book synthesis.

## Remaining-terms listening results (nextrecon7, 2026-11-01)

nextrecon7.txt: 45 new clips (nine terms, HYBRID plus a plain-English control sentence each), 45/45 SHA-256 hashes match, all generated on the user's CPU in about 20 minutes. Clip numbers below are read as N005-N013 in the order Avatamsaka, Buddha, Gautama, Sandhinirmochana-sutra, Subhuti, Yogachara, nirvana, samsara, sutra.

**Decision made:** Buddha keeps its normal English pronunciation (no Hindi-bridged form). The user likes it this way.

**Problems reported by the user:**

1. Clip 5 and the second half of clip 6 have a British accent instead of the American accent of the reference. Clip 8 has a very slight British blend. Clip 10 maybe slightly. The reference has enough r-sounds and other cues to identify an American accent. It is not yet known whether this is in the plain-English sentence, the splice's English parts, or both.
2. Clips 7 and 8: awkward pause before the term.
3. Clip 9 (Subhuti): said too slowly, which also makes the surrounding sentence awkward.
4. Clip 11 (nirvana): different pitch from the rest of the sentence; unnatural.
5. Clip 13 (sutra): the prefix has a very long trailing "the" before the term.

**Measurements (from the clips, not by ear):**

- Subhuti is the slowest term at 0.47 s per syllable (most terms 0.25-0.38). Bridged terms sit about 2-4 semitones below the English around them (nirvana -2.5, Yogachara -3.9, sutra -3.3).
- Silence before the term in the fragment-based splices was 140-340 ms against a nominal 60 ms. The prefixes end in stretched function words: the last voiced stretch is about 850 ms for "by" (Gautama), 740 ms for "of" (nirvana) and 520 ms for "of the" (Avatamsaka), against 240 ms for "by" inside the single-pass sentence. Cause: rendering the prefix as a separate fragment makes the model treat its last word as the end of an utterance.

**Fix built (aligned splicing):** instead of three separate renders, take the single-pass English sentence, locate the English term word with forced alignment (pocketsphinx), and swap in the HYBRID term clip after matching its pitch (within +/-4 semitones), pace (0.75-1.25x) and level (+/-4 dB) to the English word it replaces, with 12 ms crossfades. Measured on the nextrecon7 clips: silence before the term fell from 140-340 ms to 0-60 ms (samsara 120 ms), and term pace follows the English word (Subhuti 0.47 -> 0.23 s per syllable). Silences after the term are the narrator's own comma pauses. Not yet judged by ear.

**Accent, not fixed by splicing:** a stand-in option renders the sentence with a plain English word in the term's slot (Alexandria, Montana, Mediterranean manuscript, Emily, Alabama, banana, Sahara, chapter), so a Sanskrit spelling in the text cannot cue the accent; the HYBRID term is swapped in afterwards. This is a hypothesis to test, not an established cause. The aligned-splice report is saved as aligned_splices_report.json in the run folder.

## Standalone custom-text notebook extension (2026-11-08)

The third notebook now has an optional **Accent Check** tab. It uses the open-source AccentCL regional English accent classifier over overlapping 15-second windows with a 10-second hop. It accepts saved audiobook WAV masters/sections or uploaded WAV/MP3 files, streams decoding through FFmpeg, and writes timestamped JSON/CSV reports without speech transcription or sentence text. The UI temporarily unloads Chatterbox to free GPU memory and reloads it after the scan.

- New helper: \`tools/accent_analysis.py\`; pinned upstream AccentCL source commit \`a37811a5b772b1056cd90d44666585e35aedd657\`.
- Model: AccentCL seven-class checkpoint, with Whisper Large v3. Both model assets are cached beneath the selected Accent Cache storage path; when Drive persistence is enabled, they remain available across Colab runtime resets. First use requires multi-gigabyte downloads; GPU strongly recommended.
- CPU-side CI: **160 passed** on commit \`be19176eedd45bb348f36340f3059c0a08a6f4ce\`, workflow run [113](https://github.com/privacyfreak84/colabchatterLLMthingstuff/actions/runs/42462049087).
- Not yet tested: downloading/loading the actual AccentCL checkpoint in Colab, running GPU inference, or evaluating how accurately it classifies this user's Chatterbox narrator. The published benchmark is not a guarantee on synthetic speech.
- This is a broad regional triage aid, not phoneme-level pronunciation QA. Scores are uncalibrated, and human spot-checking remains the authority. This extension does not change the document approval gate or authorize full-book synthesis.

## Immediate next action

**Single next action:** the user listens to the aligned splices (eight files, built from the existing nextrecon7 clips, no model needed: `python scripts/make_aligned_splices.py --retest-dir ~/audiobook-narrator-retest/v3-narrator-retest --base full`) and says whether the pauses, stretched words, pitch and pace are fixed, and which clips still have a British accent.

If the accent persists, test the stand-in renders: `python scripts/run_narrator_retest_local.py --narrator ~/narrator_reference.wav --remaining-terms --routes hybrid --english-standin --aligned-splices` (see docs/local_cpu_setup.md). Per-term choices (plain English or Hindi-bridged) and the approved spoken forms stay open for the eight terms other than Buddha until the user is satisfied with the result.

Then the rest of Pass 6: the reversible text-to-spoken-form mapping for all 13 terms, a repeated-term consistency check, and an approved representative narration passage. Pass 7 and full-book synthesis stay locked until those are approved.

## Decision policy

- Code added is not the same as a completed experiment.
- CI passing is not GPU validation.
- A valid WAV is not automatically good speech.
- Hindi source quality, English voice similarity, pacing, and audible artifacts must be judged separately.
- Do not approve any pronunciation or start full-book synthesis without explicit human review.
- Update this file at the end of every meaningful engineering/research pass, including what was executed, what was listened to, what passed/failed, and the single next action.
