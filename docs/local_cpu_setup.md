# Running the narrator retest on your own machine (CPU)

This runs the same narrator-voice retest as notebook section 12, outside Colab, on a CPU-only laptop. It was written for Fedora on an AMD Ryzen laptop with integrated Radeon graphics and 16 GB of RAM.

**Status:** the scripts are tested with a stand-in model, not with the real Chatterbox weights, so the first real run is also the first real test. Expect to report one or two errors.

## 0. Check the recording first (needs nothing installed)

```bash
git clone https://github.com/privacyfreak84/colabchatterLLMthingstuff.git
cd colabchatterLLMthingstuff
python3 scripts/run_narrator_retest_local.py --narrator ~/narrator_reference.wav --check-only
```

It prints length, level, clipping, noise and silence, with blocking problems and warnings. It only measures the recording; whether the voice suits the book is your decision.

## 1. Install (once)

Fedora's default Python is newer than the pinned Chatterbox stack expects, so use Python 3.11 in a virtual environment. `uv` is the easiest way to get it.

```bash
sudo dnf install -y uv git      # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 .venv
source .venv/bin/activate

# CPU-only PyTorch (the default PyPI wheel pulls in gigabytes of CUDA libraries you cannot use)
uv pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# The pinned Chatterbox source, then its inference dependencies (same list as the Colab notebook)
uv pip install --no-deps "git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2"
uv pip install "numpy<2" librosa==0.11.0 s3tokenizer transformers==5.2.0 diffusers==0.29.0 \
    "git+https://github.com/resemble-ai/Perth.git@master" conformer==0.3.2 safetensors==0.5.3 \
    omegaconf soundfile huggingface_hub
```

Your AMD integrated graphics are not used. PyTorch's GPU support for AMD (ROCm) does not officially cover this kind of laptop chip, so this setup is deliberately CPU-only.

## 2. Run

```bash
source .venv/bin/activate
python scripts/run_narrator_retest_local.py --narrator ~/narrator_reference.wav
```

- The first run downloads the six V3 model files (several GB) into `~/.cache/audiobook/chatterbox-v3`. Later runs work offline.
- On CPU it runs only the two shortest terms (Maitreya and Vijnanavada): 14 tasks, 10 generated clips and 4 splices. Add `--all-terms` or `--terms Manjushri Avalokiteshvara` for more. Avalokiteshvara has a very long sentence and will be the slowest.
- Output goes to `~/audiobook-narrator-retest/v3-narrator-retest`. Listen to the `*_SPLICE_HYBRID.wav` and `*_SPLICE_VC.wav` files first.
- It is resumable. Press Ctrl-C any time and rerun the same command; finished clips are reused. Running more terms later adds to the same folder without redoing finished clips.
- The manifest records seconds per clip, so the first run tells us how slow your CPU really is. `--threads N` lets you experiment with PyTorch's thread count.

## 3. Send the result back

When it finishes it writes `~/audiobook-narrator-retest/narrator_retest_bundle.txt`. After listening, upload that file to the repository root as `nextrecon5.txt`, as with the earlier bundles.

## What to listen for

Hindi-origin pronunciation of the term, whether the voice changes at the join, pace and pause length, and artifacts. Nothing here approves a voice or authorizes full-book synthesis.
