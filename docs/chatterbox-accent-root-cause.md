# Chatterbox Multilingual V3: intermittent British-accent drift

**Research date:** 2026-11-08  
**Exact inference source revision:** `resemble-ai/chatterbox@5de7a54aa4e5e2baadb0182dde554908b48b85c2`  
**Checkpoint family:** `ResembleAI/chatterbox`, Multilingual V3  
**Project patch:** `tools/chatterbox_accent_factorization.py`, version `1.1.0`

## Executive finding

The British-accent episodes are not explained by the PDF reader, manuscript normalizer, Buddhist vocabulary, or WAV assembly. They are a model-level accent-control limitation in the general multilingual English path.

The strongest concrete finding is that **English is a single language ID in this model**. Text is tagged `[en]`; there is no `en-US`, `en-GB`, dialect embedding, or regional pronunciation selector in the pinned Multilingual V3 source. The English text/language representation therefore does not tell the model which English phonology to use.

At the same time, the model's zero-shot conditioning path combines several reference-derived signals:

1. A 256-dimensional VoiceEncoder speaker embedding enters T3.
2. A sequence of speech tokens from the reference enters T3 as a prompt prefix.
3. S3Gen receives reference speech tokens and mel features as acoustic prompts.
4. S3Gen also receives a CAMPPlus speaker x-vector.

The stock implementation obtains all of these from one reference recording. These representations are not explicitly disentangled into speaker identity versus regional accent. The learned speaker and acoustic representations can carry correlated pronunciation/accent traits, while the prompt prefix directly exposes the model to the reference's phonetic realization.

This architecture explains why reference accent can influence the result. It does **not by itself prove why a clean American reference sometimes produces British output**. The more specific intermittent behavior is consistent with the stochastic T3 speech-token sampler choosing different learned English realizations under a single broad `[en]` condition. The public upstream tracker contains a matching report: one user described British-sounding generations in roughly one in five attempts, with another user confirming the problem. No fix is recorded there. That is corroborating evidence of a model-level failure, not a controlled reproduction on this user's audio.

Resemble's own V3 release article explicitly says that accent drift was one of the reliability problems they targeted during V3 training. The same article lists the four dialect-specific packs as Brazilian/European Portuguese and Latin American/European Spanish. It lists English only as `English (en)`, with no US-English or UK-English pack. Thus the published release does not expose a regional English control for the base English model.

## Source-level trace

All paths below refer to the pinned upstream revision above.

### 1. Text carries language, not English region

- `src/chatterbox/mtl_tts.py`: `SUPPORTED_LANGUAGES` contains `"en": "English"`; no separate regional English identifiers.
- `src/chatterbox/models/tokenizers/tokenizer.py`: `MTLTokenizer.encode` normalizes the text and prepends one language token, e.g. `[en]`.
- `src/chatterbox/models/t3/modules/t3_config.py`: the multilingual T3 configuration specifies a multilingual vocabulary, but no dialect or locale field.

This is the primary control gap. There is no supported API argument that means “speak American English”.

### 2. T3 consumes both identity and reference speech

- `src/chatterbox/mtl_tts.py::prepare_conditionals` derives the VoiceEncoder vector and T3 speech-prompt tokens from the same reference waveform.
- `src/chatterbox/models/t3/modules/cond_enc.py::T3CondEnc.forward` projects the speaker vector and appends the Perceiver-resampled prompt-speech embeddings and emotion conditioning.
- `src/chatterbox/models/t3/t3.py::prepare_input_embeds` places these conditioning embeddings alongside the text embeddings before autoregressive speech-token generation.

The VoiceEncoder is trained to represent speaker identity, but the public inference path does not perform an accent-invariance transformation or remove regional pronunciation information from it. More importantly, the speech-prompt tokens are explicitly acoustic/language context, not a pure identity representation.

### 3. S3Gen consumes reference phonetics, mels, and speaker vector

- `src/chatterbox/models/s3gen/s3gen.py::S3Token2Mel.embed_ref` builds a reference dictionary containing `prompt_token`, `prompt_feat`, and a CAMPPlus `embedding`.
- `src/chatterbox/models/s3gen/flow.py::CausalMaskedDiffWithXvec.inference` concatenates reference speech tokens with newly generated speech tokens, carries reference mel features into the conditioning tensor, and supplies the speaker vector to the decoder.

So the reference is not just a face-like identity vector. Its speech-token and mel context are part of the acoustic synthesis path. Those cues can influence pronunciation, timing, vowel realization, and accent.

### 4. T3 generation samples speech tokens

- `src/chatterbox/models/t3/t3.py::T3.inference` combines conditional and unconditional logits using CFG and then samples each next speech token using `torch.multinomial`.
- The `cfg_weight` argument is a classifier-free-guidance scale, not a dialect selector.
- The upstream README's suggestion to set `cfg_weight=0` is specifically a mitigation for **reference-language versus output-language mismatch**. In this project both reference and requested language are English, so that documented mismatch does not exist. Changing CFG may be worth testing as one controlled variable, but it is not a principled US/UK fix.

Sampling provides a plausible route for intermittent variation between learned English realizations. The source proves that sampling is stochastic; it does not expose a British-accent logit, label, or deterministic switch that could simply be disabled.

## Evidence versus inference

| Finding | Status |
|---|---|
| The pinned source has one `en` language token and no US/UK selector | Directly verified in source |
| T3 receives reference speaker identity and reference speech tokens | Directly verified in source |
| S3Gen receives reference speech tokens, mel features, and speaker x-vector | Directly verified in source |
| T3 samples the next speech token with `torch.multinomial` | Directly verified in source |
| Accent drift is a known V3 reliability issue according to Resemble's release notes | Directly stated by model publisher |
| The upstream GitHub tracker has an unresolved report of intermittent British-accent output | Directly visible in issue #267 |
| Stochastic sampling is the precise cause of this user's US-to-UK switches | Plausible hypothesis; not yet isolated |
| A separate style/accent prompt will eliminate British output | Not established; requires listening/evaluation |
| A software-only inference patch can guarantee zero British accent with this checkpoint | Not supported by the architecture or evidence |

## Source-level modification now in the project

The project adds `tools/chatterbox_accent_factorization.py`, which patches the loaded `ChatterboxMultilingualTTS` class before inference.

It adds an optional second input called `accent_prompt_path`:

- **Narrator identity reference:** supplies the target speaker's VoiceEncoder embedding and S3Gen CAMPPlus x-vector.
- **Accent/style reference:** supplies the T3 speech-token prompt and S3Gen reference speech-token/mel prompts.

The patch explicitly replaces the S3Gen prompt dictionary's speaker x-vector with one computed from the narrator identity recording. In other words, the second recording provides pronunciation/acoustic style context without deliberately replacing the target speaker embedding.

The patch applies to both Quick TTS and Audiobook Jobs in notebook 03. If the new field is left blank, it uses the narrator reference for both streams, preserving the normal single-reference workflow. The model checkpoint weights are unchanged; this is an inference-architecture modification, not a fine-tune.

### What to use for an American-English test

For the optional accent/style field, use a clean English recording with unmistakably American pronunciation. A neutral, steady 8–15 second sample with normal connected speech is preferable to a dramatic performance, a single word, or an accent imitation. It can be a second recording of the narrator if one exists. A different speaker may guide accent but can also influence rhythm, vowel quality, and timbre through the acoustic prompt, so speaker similarity must be listened to separately.

### Controlled pilot protocol

Before using this for a long audiobook, compare the following on 10–20 representative English sentences, including sentences containing rhotics, words such as *water*, *lot*, *can't*, *bath*, and unstressed vowels:

1. **Baseline:** current narrator reference; accent/style field blank.
2. **Factorized:** same narrator identity reference plus an American-English accent/style guide.
3. **Repeat:** repeat both conditions with fixed, recorded seeds so the effect can be separated from sampling variability.

Hold the text, `language_id="en"`, chunking, conditioning lengths, `cfg_weight`, temperature, and other generation parameters constant. Compare:
- human listening for rhoticity, vowel quality, lexical stress, and overall regional consistency;
- Accent Check's broad regional estimates as a triage aid only;
- speaker similarity to the narrator, since the second acoustic prompt must not make the output sound like the guide speaker.

A successful pilot means the factorized path consistently sounds American while retaining the narrator identity over repeated seeds. One good sample is not sufficient.

## What is required for a true hard guarantee

The base checkpoint has no explicit American-English locale condition. A fully reliable source-level locale control would require a model trained or fine-tuned with an explicit regional English condition (for example, distinct `en-US` and `en-GB` labels), with balanced and audited regional speech data and evaluation that checks both pronunciation and speaker identity.

Resemble's published V3 materials describe language-specific/dialect-specific specialization for Portuguese and Spanish, but not separate American and British English checkpoints. The archived open-source repository does not include the V3 training recipe, regional training labels, or the dataset needed to create a validated US-English checkpoint. The upstream project also states that it was retired commercially in August 2026 and remains an archived research release.

No retraining or fine-tuning has been started. The present source overlay is the strongest maintainable inference-time separation that can be made without changing the weights, but it is a mitigation/control mechanism, not proof that the latent British mode has been removed.

## References

- Pinned source: https://github.com/resemble-ai/chatterbox/tree/5de7a54aa4e5e2baadb0182dde554908b48b85c2
- Upstream issue #267, “TTS keeps inserting british accent”: https://github.com/resemble-ai/chatterbox/issues/267
- Resemble AI, “Chatterbox Multilingual v3” (June 10, 2026; archive/retirement update and discussion of accent drift, dialect packs, and training focus): https://www.resemble.ai/resources/chatterbox-multilingual-v3-tts-with-embedded-watermarking-for-25-languages
