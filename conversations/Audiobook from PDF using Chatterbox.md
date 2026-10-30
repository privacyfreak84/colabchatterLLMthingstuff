# Audiobook from PDF using Chatterbox

> Exported from Claude.ai | Model: claude-sonnet-5-5 | Messages: 4

---

## Human

> 30/10/2026, 7:10:28 pm

Help man, look at this: https://github.com/resemble-ai/chatterbox

I want to create an audiobook out of this one PDF i have. I gave it to elevanlabs and it was absolutely mindblowing. The quality was unmatched. It even gave a list of words that may need special pronunciations and allowed me to see those and add more if I wanted, which I did by the way, and it worked fantastically. It was absolutely incredible. But the problem was that full export of audiobook costed 180k tokens, and I had like 4XXX something left. So obviously, with me undesirous of paying anything, I couldn't get my audiobook.

I came across this, it seems pretty cool, tested on huggingface. I think it may theoretically be able to run on my computer, but it would be abysmally slow. I am instead thinking of using a google colab space, which I think should be able to run this better.
Plus, I don't want just another chatterbox colab notebook. Elevenlabs took its good time before presenting any options at all, in which it understood the entire structure and everything of the text, and worked its way around any imperfections or artifacts or other issues, and got the best possible text, only after which it did the actual TTSing.

I think this step is also crucial.

Anyway, take all this info I've given you, probably clone the repo, I think it would be better than attempting random fetches from websites, and see what you can do.

---

## Claude

> 30/10/2026, 7:12:04 pm

<details>
<summary>🔧 memory_write — Updating memory</summary>

```json
{
  "path": "/areas/audiobook-pipeline.md",
  "if_version": "new",
  "content": "---\nname: audiobook-pipeline\ndescription: F's free audiobook-from-PDF project: Chatterbox (resemble-ai) TTS on Google Colab, modeled on ElevenLabs' analyze-then-synthesize workflow\nsources: [chat]\naliases: [audiobook, chatterbox audiobook, pdf to audiobook, chatterbox colab]\n---\n\n- [stated] Wants to turn one specific PDF into an audiobook without paying anything\n- [stated] Tried ElevenLabs first: quality was \"unmatched\"; it analyzed the whole text structure, handled artifacts/imperfections, then proposed a list of words needing special pronunciation that F could review and extend (F added their own and it worked great)\n- [stated] ElevenLabs blocker: full export cost 180k credits, F had only ~4k left\n- [stated] Found github.com/resemble-ai/chatterbox, tested it on Hugging Face; thinks it could run locally but would be abysmally slow, so plans to use Google Colab\n- [stated] Does not want \"just another Chatterbox Colab notebook\": the text-analysis/cleanup/pronunciation-review stage before TTS is crucial and must be part of it\n- [stated] Asked Claude to clone the repo and work from it rather than fetching random web pages\n"
}
```

</details>

<details>
<summary>Tool result</summary>

```
Accepted (1136 of 49152 bytes), pending a content check that completes before this reply ends — you will hear back only if it is refused. [version: 0ad1cda7d98b] (pass as if_version on your next write or edit of this path)
```

</details>

<details>
<summary>🔧 bash_tool — Clone Chatterbox repo and inspect structure, dependencies, version</summary>

```json
{
  "command": "cd /home/claude && git clone --depth 1 https://github.com/resemble-ai/chatterbox.git 2>&1 | tail -3 && cd chatterbox && ls -la && echo \"-----\" && cat pyproject.toml && echo \"-----\" && git log -1 --format='%H %ad %s' && find src -type f | head -80 && wc -l README.md",
  "description": "Clone Chatterbox repo and inspect structure, dependencies, version"
}
```

</details>

<details>
<summary>Tool result</summary>

```

```

</details>

<details>
<summary>🔧 bash_tool — Read the README for model variants, API, and usage tips</summary>

```json
{
  "command": "cd /home/claude/chatterbox && cat README.md",
  "description": "Read the README for model variants, API, and usage tips"
}
```

</details>

<details>
<summary>Tool result</summary>

```

```

</details>

<details>
<summary>🔧 bash_tool — Read original Chatterbox TTS API: text normalization, generate signature, limits</summary>

```json
{
  "command": "cd /home/claude/chatterbox/src/chatterbox && echo \"=== tts.py ===\" && cat tts.py | sed -n 1,400p",
  "description": "Read original Chatterbox TTS API: text normalization, generate signature, limits"
}
```

</details>

<details>
<summary>Tool result</summary>

```

```

</details>

<details>
<summary>🔧 bash_tool — Inspect Turbo generate path, limits, and the English tokenizer</summary>

```json
{
  "command": "cd /home/claude/chatterbox/src/chatterbox && echo \"=== tts_turbo.py (generate + helpers) ===\" && grep -n \"def \\|max_new_tokens\\|norm_loudness\\|pyloudnorm\\|tokenizer\\|AutoTokenizer\\|REPO_ID\\|nano\\|\\[\" tts_turbo.py | sed -n 1,120p && echo && echo \"=== tokenizer.py ===\" && cat models/tokenizers/tokenizer.py | sed -n 1,80p",
  "description": "Inspect Turbo generate path, limits, and the English tokenizer"
}
```

</details>

<details>
<summary>Tool result</summary>

```

```

</details>

<details>
<summary>🔧 bash_tool — Inspect Turbo generate, token limits, and sample rates</summary>

```json
{
  "command": "cd /home/claude/chatterbox/src/chatterbox && sed -n 28,75p tts_turbo.py && echo \"......\" && sed -n 270,330p tts_turbo.py && echo \"=== T3 inference limits ===\" && grep -n \"max_new_tokens\\|max_text_tokens\\|max_speech_tokens\\|start_speech\\|stop_speech\\|speech_tokens_dict_size\" models/t3/t3.py models/t3/modules/t3_config.py | head -30 && echo \"=== s3gen sample rates ===\" && grep -n \"S3GEN_SR\\|S3_SR\\|S3_TOKEN_RATE\" models/s3gen/const.py models/s3tokenizer/__init__.py | head",
  "description": "Inspect Turbo generate, token limits, and sample rates"
}
```

</details>

<details>
<summary>Tool result</summary>

```

```

</details>

One quick check before I give you the plan: I want to confirm what I can and can't actually run in my sandbox, so I don't promise something I can't test.
<details>
<summary>🔧 bash_tool — Check GPU, Python, HF network access, license, PDF libs in sandbox</summary>

```json
{
  "command": "(nvidia-smi 2>&1 | head -3) ; python3 --version ; nproc ; free -g | head -2 ; echo \"--- HF reachability ---\" ; curl -sI -m 8 https://huggingface.co/ResembleAI/chatterbox 2>&1 | head -5 ; echo \"--- LICENSE ---\" ; head -3 /home/claude/chatterbox/LICENSE ; python3 -c \"import fitz; print('pymupdf', fitz.__doc__[:40])\" 2>&1 | tail -1",
  "description": "Check GPU, Python, HF network access, license, PDF libs in sandbox"
}
```

</details>

<details>
<summary>Tool result</summary>

```

```

</details>

I cloned the repo and read through the code. Here's what matters for your project, then how I'd build it.

**What the repo gives you**

Chatterbox is only the voice engine. There are four variants:
- **Original (500M):** English, with exaggeration and CFG controls.
- **Turbo (350M):** faster, but it ignores those controls.
- **Nano (110M):** the README claims 3x realtime on 8 CPU cores, so running locally might not be hopeless, at some quality cost.
- **Multilingual V3:** 23 languages, including Hindi.

It's MIT licensed. Three things in the code shape the design:

- **No long-form handling.** `generate()` takes one string, and the original model caps each call at 1000 speech tokens, roughly 40 seconds of audio. A book means thousands of calls, so chunking, retrying and stitching are all ours to build.
- **No phoneme or SSML input.** Text goes straight to a tokenizer, so an ElevenLabs-style pronunciation dictionary has to be respelling (e.g. "Nguyen" written as "win"), found by trial and error. The original model's `punc_norm` also silently turns colons, semicolons and ellipses into commas, so the cleanup has to account for that.
- **Watermarking.** Every output carries an inaudible Perth watermark. It doesn't affect listening, but you should know it's there.

**What I can and can't test**

My sandbox has no GPU, one CPU core, and no access to Hugging Face, so I can't run the model. I can build and fully test everything around it. The Colab synthesis stage you'd run and I'd debug from your output. It will fail loudly, starting with a smoke-test cell that also measures real speed, so we get an actual number for how long your book takes.

**Proposed build, bottom to top**

1. **Ingest and clean (CPU).** PDF to structure (chapters), strip running headers, footers and page numbers, repair hyphenation and line-break damage, decide how to treat footnotes, tables and captions, expand numbers and abbreviations, and split into sentence-aware chunks. The output is clean text plus a report of what it changed and what it was unsure about. This is the "understand the whole text before speaking" step you liked.
2. **Pronunciation pass.** Flag names, acronyms and foreign terms, propose respellings, and audition each one by synthesizing it in a sentence on Colab so you hear it before accepting. Everything is saved to an editable file.
3. **Synthesis on Colab.** One fixed reference voice, with every chunk saved to Drive as it finishes, so a disconnect costs minutes rather than the whole run. Each chunk is also transcribed back with Whisper and compared to the source text, and mismatches (skipped words, repeated phrases, babble) are regenerated automatically. I think this check is the highest-value part of the project, since it's the main defense against the model's random failures.
4. **Assembly.** Paragraph- and chapter-aware pauses, loudness normalization, chapter markers, and export to M4B or MP3.

The UI comes last: a pronunciation review table and a progress view. To test the pipeline before touching a GPU, I'll make the synthesizer a swappable backend with a stub that outputs placeholder tones. That way stages 1, 3 and 4 can run end to end here first.

**Honest expectations**

Chatterbox is strong for open source, but expect it to sit a notch below ElevenLabs on prosody over long stretches, with occasional glitches that the verifier catches most of. Free Colab GPUs aren't guaranteed and sessions get cut, so checkpointing is essential rather than optional. Whether the book takes hours or a day depends on its length and the measured speed.

**What I need from you**

Upload the PDF here. Stage 1 depends heavily on what it looks like (text or scanned, columns, footnotes, language), and I'd rather design against the real thing than guess. If you still have the pronunciation list you built in ElevenLabs, send that too, since it's a free head start.

---

## Human

> 30/10/2026, 7:21:33 pm

One minute, I want to dig deeper into what you said about:

1. No phoneme or SSML input. Is there any way we can work around this? BTW, I want to use the latest v3 model, which does support hindi. This means it should support certain pronunciations not found in english but that are present in my text.
2. No long form handling. I want to see if this can also be worked around.



Is there anything we can tweak or something to bypass these?

---

## Claude

> 30/10/2026, 7:22:51 pm

<details>
<summary>🔧 memory_append — Adding to memory</summary>

```json
{
  "path": "/areas/audiobook-pipeline.md",
  "content": "- [stated] Wants to use the latest Chatterbox Multilingual V3 model (supports Hindi) because the text contains pronunciations not found in English\n- [stated] Wants to find workarounds for Chatterbox's lack of phoneme/SSML input and its lack of long-form handling",
  "if_version": "0ad1cda7d98b"
}
```

</details>

<details>
<summary>Tool result</summary>

```
Append accepted (262 bytes, file is now 1398 of 49152 bytes), pending a content check that completes before this reply ends — you will hear back only if it is refused. [version: 9a2acbb428ef] (pass as if_version on your next edit of this path)
```

</details>

<details>
<summary>🔧 bash_tool — Inspect multilingual TTS class: V3 loading, generate signature, limits</summary>

```json
{
  "command": "cd /home/claude/chatterbox/src/chatterbox && wc -l mtl_tts.py models/tokenizers/tokenizer.py && echo \"=== mtl_tts.py: key parts ===\" && grep -n \"SUPPORTED_LANGUAGES\\|def \\|t3_model\\|v3\\|v2\\|max_new_tokens\\|punc_norm\\|REPO_ID\\|language_id\\|cfg_weight=\\|exaggeration=\\|temperature=\\|repetition_penalty=\\|min_p=\\|top_p=\" mtl_tts.py | sed -n 1,140p",
  "description": "Inspect multilingual TTS class: V3 loading, generate signature, limits"
}
```

</details>

<details>
<summary>Tool result</summary>

```

```

</details>

<details>
<summary>🔧 bash_tool — Read multilingual punc_norm and generate body</summary>

```json
{
  "command": "cd /home/claude/chatterbox/src/chatterbox && sed -n 71,125p mtl_tts.py && echo \"=== generate body ===\" && sed -n 280,355p mtl_tts.py",
  "description": "Read multilingual punc_norm and generate body"
}
```

</details>

<details>
<summary>Tool result</summary>

```

```

</details>

<details>
<summary>🔧 bash_tool — Read the multilingual tokenizer: language handling, per-language preprocessing</summary>

```json
{
  "command": "cd /home/claude/chatterbox/src/chatterbox/models/tokenizers && sed -n 80,313p tokenizer.py",
  "description": "Read the multilingual tokenizer: language handling, per-language preprocessing"
}
```

</details>

<details>
<summary>Tool result</summary>

```

```

</details>

<details>
<summary>🔧 bash_tool — Inspect T3 inference loop, hallucination guards, and how multilingual weights are loaded</summary>

```json
{
  "command": "cd /home/claude/chatterbox/src/chatterbox && grep -rn \"AlignmentStreamAnalyzer\\|alignment_stream\\|class .*Analyzer\\|force_eos\\|repetition\" --include=*.py . | grep -v \"^./models/s3gen\" | head -20; echo \"=== t3.inference (loop part) ===\"; sed -n 232,262p models/t3/t3.py; echo ...; sed -n 300,385p models/t3/t3.py; echo \"=== mtl from_local / from_pretrained ===\"; sed -n 183,252p mtl_tts.py",
  "description": "Inspect T3 inference loop, hallucination guards, and how multilingual weights are loaded"
}
```

</details>

<details>
<summary>Tool result</summary>

```

```

</details>

I've now read the multilingual model's tokenizer, `generate()` and the T3 sampling loop, so these answers come from the code. Short version: neither limitation can be removed, but both can be worked around well.

## 1. Pronunciation without phoneme input

There is no hidden phoneme channel. The V3 tokenizer is grapheme-based, with a language token like `[hi]` at the front. Before encoding, it lowercases the text and applies Unicode NFKD normalization. Only zh, ja, he, ko and ru get special preprocessing, and Hindi and English don't. So the workarounds all happen at the text level:

- **Respelling in the model's own script.** This is the main tool. Devanagari is close to phonetic, so Hindi respellings should be far more predictable than English ones. Because of the lowercasing, the model can't tell "US" from "us", or "IT" from "it", so those always need respelling or spacing (e.g. "U S").
- **Per-span language routing.** `language_id` applies to the whole call. For a Hindi word inside English text (or the reverse), I can split at the span, generate it under its own language token, and crossfade the pieces together. The cost is seams and prosody breaks, and very short calls may be less stable. I'd measure that rather than assume.
- **Mixed-script sentences are a real unknown.** Hindi words in Devanagari inside an `[en]` sentence, Latin transliteration, or span-splitting could each win in different cases. I can't predict which, so the audition tool will render the same sentence all three ways for you to pick a policy.
- **Tokenizer coverage check.** Characters missing from the vocab become `[UNK]` and are silently dropped, so a rare mark could vanish from the audio without any error. A pre-flight cell on Colab can encode every unique character in your book and flag them. It has to run there, since I can't reach Hugging Face from here.
- **Respelling search (later).** Generate candidate respellings, synthesize each, transcribe with Whisper, and rank by which one gets heard as the target word. This is only a proxy for gross errors, not for stress or accent, so your ear stays the judge.

Real IPA or SSML input would require retraining the model, which isn't worth it for one audiobook. If the book is mostly Hindi, the dedicated Hindi pack is worth A/B testing against V3. It lives in a separate Hugging Face repo and the loader hardcodes the main repo, so we'd load it through `from_local`. I haven't verified its file layout.

## 2. Long-form

- **The 1000-token cap is a hardcoded constant** in `generate()` (with a TODO comment), while the config allows 4096. I can wrap the function and expose it. I wouldn't simply raise it, though: long generations are where autoregressive TTS tends to drift, repeat or skip. That's my expectation, and I haven't measured it here.
- **Use the cap as a runaway detector instead.** I found no repetition or alignment guard in this version of the repo, only the repetition penalty. If a chunk never hits the end-of-speech token, or its duration is implausible for its text length, it's a runaway, so we reject it and retry with a new seed. That check is free, and Whisper verification then catches the subtler failures.
- **Continuing from the previous chunk isn't possible.** `prepend_prompt_speech_tokens` literally asserts "not implemented". Hacking in previous tokens would break the format the model was trained on, so I won't.

Long-form therefore lives in our layer:
1. Chunk at sentence and paragraph boundaries.
2. Prepare the voice conditioning once and save it to Drive, since `Conditionals.save` exists.
3. Fix the sampling parameters.
4. Seed each chunk and record the seed, so any chunk can be reproduced or re-rolled.
5. Insert our own pauses at joins rather than trusting the model's trailing silence.
6. Normalize loudness per chunk.

The residual limit is prosody across chunk boundaries, which no wrapper fully fixes. Paragraph-aligned chunking keeps most joins where a pause is natural. The right chunk length is empirical, so we'll tune it on one real chapter.

I'll wrap the model in our own thin class instead of patching the repo, so library updates won't break us. Whenever you're ready, upload the PDF, since the language mix inside it decides which of these tools matter most.

---
