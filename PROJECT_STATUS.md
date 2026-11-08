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

## Standalone custom-text notebook extensions (2026-11-08)

### Spoiler-free accent analysis

The third notebook has an optional **Accent Check** tab. It uses the open-source AccentCL regional English accent classifier over overlapping 15-second windows with a 10-second hop. It accepts saved audiobook WAV masters/sections or uploaded WAV/MP3 files, streams decoding through FFmpeg, and writes timestamped JSON/CSV reports without speech transcription or sentence text. The UI temporarily unloads Chatterbox to free GPU memory and reloads it after the scan.

- Helper: `tools/accent_analysis.py`; pinned upstream AccentCL source commit `a37811a5b772b1056cd90d44666585e35aedd657`.
- Model: AccentCL seven-class checkpoint, with Whisper Large v3. Both model assets are cached beneath the selected Accent Cache storage path; Drive persistence keeps them across Colab resets. First use requires multi-gigabyte downloads; a GPU is strongly recommended.
- The scanner is a broad regional triage aid, not phoneme-level pronunciation QA. Scores are uncalibrated, and human spot-checking remains the authority.
- Not yet tested: downloading/loading the actual AccentCL checkpoint in Colab, running GPU inference, or evaluating its accuracy on this synthetic Chatterbox narrator.

### Source selection, selective rerendering, and master assembly

The Audiobook Jobs workflow now supports selecting exactly what is spoken and replacing only defective audio:

- **PDF:** one-based inclusive page range. Set first and last page to the same number to speak one page; last page `0` means the final PDF page. Selectable text extraction only, no OCR.
- **TXT/Markdown/pasted text:** one-based inclusive blank-line paragraph range. These formats do not have intrinsic pages, so the notebook does not pretend that they do. Visual line-wrap cleanup runs before paragraph numbering.
- **Detected sections:** preview the source to populate a multi-select list of headings/chapters. Choose one or more sections; an empty selection means all sections. The same control works with PDF, TXT, Markdown, and pasted text.
- **Repair:** enter one or more chunk IDs such as `C00007` or section IDs such as `S002`. A section ID rerenders all chunks in that section. Reuse the same Job ID and identical source/range/voice/synthesis settings.
- **Replacement policy:** by default a successful rerender atomically replaces the active chunk WAV, without keeping an old revision. An opt-in checkbox copies the previous WAV to `revisions/` as a backup. The old valid WAV remains active if synthesis fails before replacement.
- **Master outputs:** completed jobs automatically create separate section WAVs and one combined WAV. Full-source jobs use `whole_book_master.wav`; jobs limited to selected sections use `selected_sections_master.wav`. The delivery ZIP includes the appropriate combined master. Rerendering rebuilds the section and combined masters from the latest active chunk files.
- Helper: `tools/longform_audiobook.py`; selected-section plan filtering, text paragraph-range selection, atomic replacement/optional backups, and partial/full master assembly are CPU-tested.
- Existing jobs remain resumable after helper-only version changes when their source, selected ranges, chunk plan, narrator reference, and synthesis settings still match. The default all-sections and full-text paragraph ranges are normalized as defaults, so jobs created before these selector controls remain resumable.
- CI: **172 passed** on commit `70cc6a4e8f610237c4581308a9e5b971f6ac199d`, workflow run [144](https://github.com/privacyfreak84/colabchatterLLMthingstuff/actions/runs/42469206864). The suite includes legacy-job compatibility, all-section defaults, and rejection of changed partial-selection identities.
- Not yet tested: launching the updated notebook in a live Colab session, generating a pilot, scanning the resulting narrator audio, rerendering a selected chunk/section on GPU, and listening to the rebuilt master.
- These additions do not change the separate document-first manuscript approval gate or authorize full-book synthesis.

### Optional custom output naming (2026-11-08)

The Audiobook Jobs tab now has an optional **Custom output-name prefix**:

- The prefix is applied to every generated chunk WAV, section/chapter WAV, combined master WAV, and delivery ZIP. For example, `Buddhist Yoga_c00001.wav`, `Buddhist Yoga_s001_chapter-one.wav`, `Buddhist Yoga_whole_book_master.wav`, and `Buddhist Yoga_audiobook_delivery_bundle.zip`.
- The default Job ID and storage folder remain unchanged. The Job ID is still the resume/recovery key; the custom prefix names the deliverable files.
- The name is validated as a filename stem only (no folder paths, reserved filename characters, or unsafe reserved device names), up to 96 characters. Do not include an extension.
- A custom name is fixed for that Job ID. On resume, leave the field blank to reuse the saved name or enter the same name; a different name is rejected rather than silently moving active WAV files.
- Accent Check's saved-audio list also discovers custom-named full and selected-section masters.
- Automated verification: **175 tests passed** on the notebook/helper implementation, including unsafe-name rejection, chunk/section/master/ZIP filenames, resume-name preservation, and notebook/UI wiring. Workflow run [152](https://github.com/privacyfreak84/colabchatterLLMthingstuff/actions/runs/42471448428).
- Not yet tested: a live Colab session generating WAVs with a custom name, resuming after a runtime reset, rerendering, and confirming that Accent Check lists the custom-named masters. CI uses synthetic WAVs and static notebook checks, not live Chatterbox synthesis.

### Chatterbox English accent drift research and source patch (2026-11-08)

The third notebook now applies a pinned source overlay to the loaded Chatterbox Multilingual V3 class. The overlay gives the model two independent conditioning inputs: a narrator identity recording and an optional same-language accent/style guide. The narrator supplies the VoiceEncoder and S3Gen speaker embeddings; the guide supplies T3 speech-prompt tokens and S3Gen speech-token/mel context. The option is wired into both Quick TTS and Audiobook Jobs. A prompt-start offset selects a later slice from the guide. With the guide blank and offset zero, the standard one-reference path is retained. The user's 19.6-second narrator recording can be uploaded in both fields, with a 6-second guide offset, to use the later 6-second T3 and 10-second S3Gen slices without introducing another speaker.

**Root-cause findings are documented in** [docs/chatterbox-accent-root-cause.md](docs/chatterbox-accent-root-cause.md):

- The exact pinned source (resemble-ai/chatterbox@5de7a54...) supports one `en` language tag; it has no `en-US` or `en-GB` locale, dialect embedding, or regional English selector.
- The same reference recording feeds speaker identity embeddings and acoustic speech prompts in both T3 and S3Gen. Those streams therefore do not cleanly separate speaker identity from accent/pronunciation style.
- T3 samples speech tokens with `torch.multinomial`; generation can vary between learned English realizations. This is a plausible contributor to intermittency, not a proven isolated cause.
- Resemble's June 2026 V3 release article explicitly lists accent drift among the problems targeted during training. Its dialect-specific packs are for Portuguese and Spanish variants, not US/UK English.
- Upstream issue [#267](https://github.com/resemble-ai/chatterbox/issues/267) reports intermittent British-accent generations, including a roughly one-in-five report. It remains without a recorded fix.

**Source patch:** `tools/chatterbox_accent_factorization.py`, version 1.1.0, source commit `9ea38a1d6708bf632f3de185eda1b51617843b0a`, pinned by notebook 03. It changes inference conditioning, not the weights. A second same-language guide can steer the acoustic prompts while retaining the narrator's identity embeddings. A guide offset supports later slices of the same narrator recording, and the patch rejects slices that are too short for the full configured T3/S3Gen prompt windows. It is a source-level factorization, not post-processing. A different guide speaker can still affect delivery/timbre through the acoustic prompt, so use the narrator's own later segment first and evaluate by listening.

**Verification:** **181 tests passed** in workflow [182](https://github.com/privacyfreak84/colabchatterLLMthingstuff/actions/runs/42477966755), commit `1ed5e893da452787b24229966fdcf44b0ce9e360`. Tests cover distinct identity/style streams, later-segment slicing, full prompt-window validation, negative/incomplete offset rejection, Quick TTS and Audiobook Jobs callback wiring, and existing pipeline regressions. The tests use synthetic/fake model components and static notebook checks. They do not load the real V3 checkpoint or synthesize audio.

**Not resolved / not claimed:** there is no evidence yet that British-sounding output has been eliminated. The source patch has not had a live Colab GPU listening evaluation. The general checkpoint has no dialect control token. A hard US-English guarantee would require region-conditioned US/UK checkpoints or suitable retraining/fine-tuning with region-labelled speech and objective/human evaluation. No retraining has been started.


## Accent Check checkpoint compatibility and narration rate (2026-11-08)

### AccentCL deserialization failure

A live Colab Accent Check attempt failed in PyTorch's restricted unpickler with `IndexError: pop from empty list` while loading `accentcl_spanish_chinese.pt` with `weights_only=True`. The Diffusers `LoRACompatibleLinear` deprecation and optional `pkuseg`-missing tokenizer warning were separate and did not cause that exception.

**Root cause:** the official AccentCL inference loader uses the legacy `torch.load(..., weights_only=False)` path for this historical checkpoint. Forcing the newer restricted weights-only unpickler onto that file is incompatible with its serialization format.

**Fix:** `tools/accent_analysis.py` now pins the checkpoint to immutable Hugging Face revision `af925d811ff95be41a2cacc3988cd3df410e762f`, checks the exact published size (**2,563,884,656 bytes**) and Git-LFS SHA-256 (**f4db857f696323d8a6c71073fff734c81dee33267a393b50c795f12ca9f86c93**), and only then uses the legacy loader. If size or hash differs, it raises before invoking `torch.load`. This compatibility path is restricted to the exact pinned official artifact; it is not a general loader for arbitrary checkpoints.

Regression tests exercise both the verified-load path and rejection of a tampered artifact. The LoRA deprecation warning and missing optional Chinese segmentation package remain non-fatal environment warnings; they are not patched because they are unrelated to English AccentCL inference.

### Pitch-preserving narration-rate control

New helper: `tools/speech_rate.py`. It uses FFmpeg's `atempo` WSOLA filter on the model's float waveform. It does not change the WAV sample rate, lower pitch by naïvely changing playback rate, or insert echo/reverb. Processing is per speech chunk and runs **before** the explicitly configured inter-chunk / paragraph / section pauses are assembled.

- Rate range: **0.75x to 1.15x**; default: **0.85x**; **1.00x** is the original model tempo.
- Both Quick TTS and Audiobook Jobs expose a narration-pace slider.
- Quick-TTS metadata and audiobook generation settings record the selected rate and processor ID. Each chunk records both original model duration and processed duration.
- A different speed changes the audiobook generation fingerprint. Use a fresh Job ID for a different-rate version; do not mix old-rate chunks with new-rate chunks.
- FFmpeg processing is checked for finite mono output and plausible duration. Missing FFmpeg, invalid rates, malformed output, and processing errors are reported rather than silently falling back to a pitch-shifting method.

**Verification:** workflow [195](https://github.com/privacyfreak84/colabchatterLLMthingstuff/actions/runs/42499453277) passed **197 tests** on commit `27ce126cf4d7682564166d26a679bfc53f4ee2aa`. The test suite includes checkpoint digest enforcement, rate-helper behaviour and error handling, and both notebook callback/input integrations. The FFmpeg helper was also exercised locally against synthetic tones at several rates. This does not substitute for listening to actual Chatterbox speech.

**Not yet verified:** reloading the updated notebook in Colab, downloading/verifying/loading the AccentCL checkpoint on the Colab runtime, scanning a reference accent clip, and listening to a representative page rendered at 0.85x and another chosen rate. The speech-rate DSP may still create content-dependent artefacts, so a short human-audited pilot is required before a long batch.



## Audiobook workspace redesign, playback history and manual master assembly (2026-11-08)

### Interface redesign

The third notebook's former single crowded Audiobook Jobs form has been replaced by two tabs:

- **Create / Resume:** five collapsible steps group the source/range, preview and section selection, narrator/accent guide/pace, advanced voice settings, and job identity/repair controls.
- **Listen & Assemble:** select a saved Job ID, browse every valid WAV inside that job, load a prior WAV into a player, or assemble selected chunks / all completed chunks.

The generation screen now includes a **Latest spoken chunk** audio player. It updates as each new chunk finishes and is initialized with the final valid chunk already in the saved job when resuming an otherwise-complete job.

The saved WAV browser includes chunk files, replacement backups, manually assembled files, and legacy section/master files from older jobs. The browser is intentionally scoped to the selected Job ID so audio from separate books is not mixed. Playback and assembly use the sample rate recorded in the job manifest, so these tools do not need the Chatterbox model resident in GPU memory. Accent Check also now lists saved chunk and assembly WAVs. Drive persistence is required if saved audio must survive a Colab runtime reset.

### No automatic master WAV

New audiobook jobs no longer call the old automatic master-assembly step and no longer create combined whole-book or per-section master WAVs during generation. Chunk WAVs remain the primary synthesis artifacts. Technical QA, manifest, logs, and delivery ZIP are still produced after generation completes.

The old helper remains in the repository for backward compatibility/testing, but the new notebook generation callback does not invoke it. Existing legacy masters are not deleted; the playback browser can still find them.

### User-controlled master assembly

New helper: tools/longform_audiobook.py, function assemble_selected_chunk_wavs.

- The user selects one or more completed chunk WAVs, or checks **Use all completed chunks in this job**.
- Selection is resolved against manifest order, not the order the user clicked the multi-select list.
- Every selected WAV must be a complete, hash-valid, mono file at the expected sample rate. Unknown, missing, modified, or incomplete WAVs are rejected.
- Recorded chunk/paragraph pauses are preserved between the selected chunks.
- Output WAVs are stored in assemblies/, with source chunk IDs, selected/all mode, duration, hash, and creation timestamp recorded in the manifest.
- Assemblies are included in a refreshed delivery ZIP for completed jobs. For an unfinished/paused job, the selected WAV is still saved and playable; a complete delivery ZIP is generated only after the job itself is complete.
- The saved-audio browser discovers the newly created assembly so it can be played again later.

### Verification

GitHub Actions workflow [210](https://github.com/privacyfreak84/colabchatterLLMthingstuff/actions/runs/42501313822) passed **199 tests** on commit `d2b0d7fa1c809d8bd669f5e61e522649a3c0f739`. The suite covers selected/all assembly, source order, invalid/unknown WAV rejection, ZIP inclusion, notebook syntax, generator output arity, current/previous audio playback wiring, model-independent saved-audio handling, chunk Accent Check discovery, and UI callback/input wiring.

**Not yet tested in live Colab:** the rebuilt Gradio layout in a running session, playback of a freshly generated chunk, browsing previous job audio after a runtime reset with Drive enabled, assembling a subset and all completed chunks, or listening to the resulting master WAV. CI uses synthetic WAVs and static notebook checks, not real Chatterbox narration.

## Immediate next action

**Single next action:** open the updated third notebook in Colab and smoke-test the redesigned audiobook workspace with a short reviewed page: confirm the latest chunk player updates, use Listen & Assemble to play a saved chunk, assemble two selected chunks, then assemble all completed chunks and listen for order, pauses, and audio artefacts. Use Drive persistence if checking audio history after a runtime reset. In the same session, run Accent Check on a known North American or British English clip and render a short comparison at **0.85x** and **1.00x**. Do not start a long audiobook batch until these live checks pass.

After that smoke test, validate Accent Check on known North American and British Isles clips plus one narrator retest clip. Then return to the existing Pass 6 listening checkpoint: build/listen to the aligned splices (eight files, built from the existing nextrecon7 clips, no model needed: `python scripts/make_aligned_splices.py --retest-dir ~/audiobook-narrator-retest/v3-narrator-retest --base full`) and judge pauses, stretched words, pitch, pace, and remaining accent drift.

If the accent persists, test the stand-in renders: `python scripts/run_narrator_retest_local.py --narrator ~/narrator_reference.wav --remaining-terms --routes hybrid --english-standin --aligned-splices` (see docs/local_cpu_setup.md). Per-term choices and spoken forms remain open for the eight terms other than Buddha until the user is satisfied.

Then complete Pass 6: the reversible text-to-spoken-form mapping for all 13 terms, a repeated-term consistency check, and an approved representative narration passage. Pass 7 and full-book synthesis stay locked until those are approved.
## Decision policy

- Code added is not the same as a completed experiment.
- CI passing is not GPU validation.
- A valid WAV is not automatically good speech.
- Hindi source quality, English voice similarity, pacing, and audible artifacts must be judged separately.
- Do not approve any pronunciation or start full-book synthesis without explicit human review.
- Update this file at the end of every meaningful engineering/research pass, including what was executed, what was listened to, what passed/failed, and the single next action.

## Initial audiobook stream-update fix (2026-11-08)

A live audiobook job failed immediately after job creation with `UnboundLocalError: latest_audio_path`, before Chatterbox generated the first chunk.

**Root cause:** the generator's initial Gradio update yielded `latest_audio_path` before the variable was assigned. The lookup for an already-completed chunk was below that first yield, so a new job failed at startup.

**Fix:** moved the completed-chunk count and verified latest-WAV lookup above the first yield. A new job now sends `None` to the player until its first chunk completes; a resumed job initializes the player with its latest valid completed chunk. No synthesis or audio-processing logic changed.

**Regression coverage:** added a notebook AST test asserting that the first assignment to `latest_audio_path` precedes the first generator yield.

**Verification:** CI workflow [213](https://github.com/privacyfreak84/colabchatterLLMthingstuff/actions/runs/42505847558) passed **200 tests** on commit `9c4c81aec21d6d5a91d149ba18a10aee27dece77`.

**Not yet verified:** retrying the job in a live Colab runtime and confirming playback of the first generated chunk. The failed run stopped before the first chunk was synthesized. Reuse the same Job ID only with the same manuscript, page/section selection, narrator/accent references, and generation settings; the resume helper checks these identities before continuing.

**Next action:** reload the corrected notebook code in Colab and retry the short pilot with the same Job ID and unchanged settings. If Drive persistence was off and the runtime was reset, the old job folder is gone; recreate it with the same Job ID and inputs.

