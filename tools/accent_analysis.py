"""Streaming, spoiler-free regional accent analysis for long WAV/MP3 files."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

SAMPLE_RATE = 16_000
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3"}
ACCENTCL_REPO_URL = "https://github.com/PSI-TAMU/AccentCL.git"
ACCENTCL_COMMIT = "a37811a5b772b1056cd90d44666585e35aedd657"
ACCENTCL_MODEL_REPO = "Morris88826/accentcl"
ACCENTCL_CHECKPOINT = "accentcl_spanish_chinese.pt"
ACCENTCL_MIN_CHECKPOINT_BYTES = 1_000_000_000
ACCENTCL_MODEL_NAME = "AccentCL with Whisper Large v3, seven regional English categories"

DISPLAY_LABELS = {
    "north_american": "North American English",
    "british_isles": "British Isles English",
    "australasian": "Australasian English",
    "south_asian": "South Asian English",
    "southeast_asian": "Southeast Asian English",
    "spanish": "Spanish-accented English",
    "chinese": "Chinese-accented English",
}


def validate_audio_path(audio_path: str | Path) -> Path:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError("Select an existing audio file.")
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError("Accent analysis accepts WAV and MP3 files.")
    if path.stat().st_size == 0:
        raise ValueError("The selected audio file is empty.")
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_timestamp(seconds: float) -> str:
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def probe_audio_duration(audio_path: str | Path) -> float | None:
    path = validate_audio_path(audio_path)
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True, text=True, timeout=60,
        )
        value = float(result.stdout.strip())
        return value if math.isfinite(value) and value >= 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _rms_dbfs(samples) -> float:
    import numpy as np
    if len(samples) == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-12))


def iter_audio_windows(
    audio_path: str | Path,
    *,
    window_seconds: float = 15.0,
    hop_seconds: float = 10.0,
    minimum_segment_seconds: float = 3.0,
    sample_rate: int = SAMPLE_RATE,
) -> Iterator[dict]:
    """Decode sequentially and yield overlapping windows without loading the book into RAM."""
    import numpy as np

    path = validate_audio_path(audio_path)
    if not 5 <= float(window_seconds) <= 30:
        raise ValueError("Window length must be between 5 and 30 seconds.")
    if not 1 <= float(hop_seconds) <= float(window_seconds):
        raise ValueError("Hop length must be between 1 second and the window length.")
    if not 3 <= float(minimum_segment_seconds) <= float(window_seconds):
        raise ValueError("Minimum segment length must be between 3 seconds and the window length.")
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError("AccentCL expects mono audio at 16,000 Hz.")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found. Install FFmpeg and retry.")

    window_samples = int(round(float(window_seconds) * sample_rate))
    hop_samples = int(round(float(hop_seconds) * sample_rate))
    minimum_samples = int(round(float(minimum_segment_seconds) * sample_rate))
    read_bytes = sample_rate * 5 * 4
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1",
        "-ar", str(sample_rate), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = np.empty(0, dtype=np.float32)
    buffer_start_sample = 0
    last_emitted_end_sample = 0
    completed = False
    try:
        assert process.stdout is not None
        while True:
            payload = process.stdout.read(read_bytes)
            if not payload:
                break
            if len(payload) % 4:
                raise RuntimeError("FFmpeg returned an incomplete float32 sample.")
            block = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=False)
            if block.size:
                buffer = np.concatenate((buffer, block))
            while buffer.size >= window_samples:
                segment = buffer[:window_samples].copy()
                end_sample = buffer_start_sample + window_samples
                yield {
                    "audio": segment,
                    "start_seconds": round(buffer_start_sample / sample_rate, 3),
                    "end_seconds": round(end_sample / sample_rate, 3),
                    "rms_dbfs": round(_rms_dbfs(segment), 2),
                }
                last_emitted_end_sample = end_sample
                buffer = buffer[hop_samples:].copy()
                buffer_start_sample += hop_samples

        return_code = process.wait()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if return_code != 0:
            raise RuntimeError("FFmpeg could not decode the audio file: " + (stderr.strip()[-1200:] or str(return_code)))

        total_end_sample = buffer_start_sample + buffer.size
        newly_uncovered_samples = total_end_sample - last_emitted_end_sample
        if buffer.size >= minimum_samples and newly_uncovered_samples >= minimum_samples:
            yield {
                "audio": buffer.copy(),
                "start_seconds": round(buffer_start_sample / sample_rate, 3),
                "end_seconds": round(total_end_sample / sample_rate, 3),
                "rms_dbfs": round(_rms_dbfs(buffer), 2),
            }
        completed = True
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()


def _label_display(label: str) -> str:
    return DISPLAY_LABELS.get(label, label.replace("_", " ").title())


def build_accent_report(
    segments: list[dict],
    *,
    audio_path: str | Path,
    audio_duration_seconds: float | None,
    skipped_quiet_windows: int = 0,
    window_seconds: float = 15.0,
    hop_seconds: float = 10.0,
) -> dict:
    if not segments:
        raise ValueError("No usable speech windows were found. The audio may be too short, silent, or undecodable.")

    labels = sorted({label for row in segments for label in row["scores"]})
    means = {
        label: sum(float(row["scores"].get(label, 0.0)) for row in segments) / len(segments)
        for label in labels
    }
    top_counts: dict[str, int] = {}
    for row in segments:
        top_counts[row["accent"]] = top_counts.get(row["accent"], 0) + 1
    dominant = max(means, key=means.get)

    regions: list[dict] = []
    for row in segments:
        if not regions or regions[-1]["accent"] != row["accent"]:
            regions.append({
                "accent": row["accent"],
                "start_seconds": row["start_seconds"],
                "end_seconds": row["end_seconds"],
                "window_count": 1,
                "score_sum": float(row["score"]),
            })
        else:
            regions[-1]["end_seconds"] = row["end_seconds"]
            regions[-1]["window_count"] += 1
            regions[-1]["score_sum"] += float(row["score"])
    for region in regions:
        region["mean_model_score"] = round(region.pop("score_sum") / region["window_count"], 4)
        region["accent_display"] = _label_display(region["accent"])

    changes = []
    for index in range(1, len(regions)):
        previous, current = regions[index - 1], regions[index]
        if previous["window_count"] >= 2 and current["window_count"] >= 2:
            changes.append({
                "around_seconds": round(
                    (previous["end_seconds"] + current["start_seconds"]) / 2.0, 3
                ),
                "from": previous["accent_display"],
                "to": current["accent_display"],
                "note": "Sustained model-label change; inspect this timestamp if it matters.",
            })

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": ACCENTCL_MODEL_NAME,
        "source_audio_name": Path(audio_path).name,
        "source_audio_sha256": sha256_file(audio_path),
        "audio_duration_seconds": round(float(audio_duration_seconds), 3) if audio_duration_seconds is not None else None,
        "sample_rate_hz": SAMPLE_RATE,
        "window_seconds": float(window_seconds),
        "hop_seconds": float(hop_seconds),
        "analysed_window_count": len(segments),
        "skipped_quiet_window_count": int(skipped_quiet_windows),
        "dominant_accent": dominant,
        "dominant_accent_display": _label_display(dominant),
        "dominant_top_label_share": round(top_counts.get(dominant, 0) / len(segments), 4),
        "mean_model_scores": {label: round(score, 6) for label, score in means.items()},
        "regions": regions,
        "possible_sustained_changes": changes,
        "segments": segments,
        "transcription_performed": False,
        "score_warning": (
            "Softmax scores are not calibrated probabilities of correctness. "
            "A regional prediction is a review hint, not proof of a speaker accent."
        ),
    }


def analyse_audio_file(
    audio_path: str | Path,
    model,
    *,
    window_seconds: float = 15.0,
    hop_seconds: float = 10.0,
    silence_threshold_dbfs: float = -55.0,
    progress_callback: Callable[[int, int | None, float, float], None] | None = None,
) -> dict:
    import numpy as np
    import torch

    path = validate_audio_path(audio_path)
    duration = probe_audio_duration(path)
    estimated_windows = (
        1 if duration is not None and duration <= window_seconds
        else int(math.ceil((duration - window_seconds) / hop_seconds)) + 1
        if duration is not None else None
    )
    id2label = getattr(model, "id2label", None)
    if not id2label:
        label2id = getattr(model, "label2id", None)
        if not label2id:
            raise ValueError("The loaded accent model does not expose its label mapping.")
        id2label = {int(index): label for label, index in label2id.items()}
    id2label = {int(index): str(label) for index, label in id2label.items()}
    device = next(model.parameters()).device
    rows = []
    quiet_count = 0
    for window in iter_audio_windows(path, window_seconds=window_seconds, hop_seconds=hop_seconds):
        if window["rms_dbfs"] <= silence_threshold_dbfs:
            quiet_count += 1
        else:
            waveform = torch.from_numpy(np.asarray(window["audio"], dtype=np.float32)).unsqueeze(0).to(device)
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        _, logits, _ = model.predict(waveform, return_feature=False)
                else:
                    _, logits, _ = model.predict(waveform, return_feature=False)
            probabilities = torch.softmax(logits[0].float(), dim=-1).detach().cpu().tolist()
            scores = {
                id2label[index]: float(probability)
                for index, probability in enumerate(probabilities)
                if index in id2label
            }
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if not ranked:
                raise RuntimeError("AccentCL returned no class scores.")
            rows.append({
                "start_seconds": window["start_seconds"],
                "end_seconds": window["end_seconds"],
                "start": format_timestamp(window["start_seconds"]),
                "end": format_timestamp(window["end_seconds"]),
                "accent": ranked[0][0],
                "accent_display": _label_display(ranked[0][0]),
                "score": round(ranked[0][1], 6),
                "runner_up": ranked[1][0] if len(ranked) > 1 else "",
                "runner_up_display": _label_display(ranked[1][0]) if len(ranked) > 1 else "",
                "runner_up_score": round(ranked[1][1], 6) if len(ranked) > 1 else 0.0,
                "margin": round(ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0), 6),
                "scores": {label: round(score, 6) for label, score in ranked},
            })
        if progress_callback:
            progress_callback(
                len(rows) + quiet_count, estimated_windows,
                window["start_seconds"], window["end_seconds"],
            )

    return build_accent_report(
        rows, audio_path=path, audio_duration_seconds=duration,
        skipped_quiet_windows=quiet_count,
        window_seconds=window_seconds, hop_seconds=hop_seconds,
    )


def format_report_markdown(report: dict) -> str:
    duration = report.get("audio_duration_seconds")
    duration_text = format_timestamp(duration) if duration is not None else "unknown"
    lines = [
        "## Accent scan complete",
        "",
        f"**File:** {report['source_audio_name']}  ",
        f"**Duration:** {duration_text}  ",
        f"**Windows analysed:** {report['analysed_window_count']}  ",
        f"**Overall leading category:** **{report['dominant_accent_display']}** "
        f"({report['dominant_top_label_share'] * 100:.1f}% of windows had this top label)",
        "",
        "### Segment regions",
        "",
        "| Time range | Leading accent | Windows | Mean model score |",
        "|---|---|---:|---:|",
    ]
    for region in report["regions"]:
        lines.append(
            f"| {format_timestamp(region['start_seconds'])} to {format_timestamp(region['end_seconds'])} "
            f"| {region['accent_display']} | {region['window_count']} "
            f"| {region['mean_model_score']:.3f} |"
        )
    if report["possible_sustained_changes"]:
        lines.extend(["", "### Possible sustained changes", ""])
        for change in report["possible_sustained_changes"]:
            lines.append(
                f"- Around {format_timestamp(change['around_seconds'])}: "
                f"{change['from']} to {change['to']}"
            )
    else:
        lines.extend(["", "No sustained regional-category change was supported by two consecutive windows per side."])
    lines.extend([
        "",
        "> **Interpretation:** this estimates broad regional accent categories. It is not phoneme-level pronunciation QA. "
        "Scores are not calibrated correctness probabilities. No transcription is performed, so this report contains no sentence text.",
    ])
    return "\n".join(lines)


def write_accent_report(report: dict, output_dir: str | Path) -> tuple[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"accent_report_{stamp}"
    json_path = root / f"{base}.json"
    csv_path = root / f"{base}.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "start", "end", "start_seconds", "end_seconds", "accent_display",
                "score", "runner_up_display", "runner_up_score", "margin", "scores",
            ],
        )
        writer.writeheader()
        for row in report["segments"]:
            item = {key: row.get(key, "") for key in writer.fieldnames}
            item["scores"] = json.dumps(row.get("scores", {}), ensure_ascii=False)
            writer.writerow(item)
    return str(json_path), str(csv_path)


def load_accentcl_model(
    *,
    cache_dir: str | Path,
    repo_dir: str | Path,
    device: str = "cuda",
    progress_callback: Callable[[str], None] | None = None,
):
    """Download pinned model source and load weights using PyTorch safe mode only."""
    import gc
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    if progress_callback:
        progress_callback("Preparing pinned AccentCL source code...")
    repo_path = Path(repo_dir)
    git = shutil.which("git")
    if not git:
        raise RuntimeError("Git is required to download the AccentCL source code.")
    if repo_path.exists() and not (repo_path / ".git").exists():
        shutil.rmtree(repo_path)
    if not repo_path.exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [git, "clone", "--quiet", "--filter=blob:none", "--no-checkout", ACCENTCL_REPO_URL, str(repo_path)],
            check=True, capture_output=True, text=True,
        )
    subprocess.run(
        [git, "-C", str(repo_path), "fetch", "--quiet", "--depth", "1", "origin", ACCENTCL_COMMIT],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [git, "-C", str(repo_path), "checkout", "--quiet", "--detach", ACCENTCL_COMMIT],
        check=True, capture_output=True, text=True,
    )
    actual_commit = subprocess.run(
        [git, "-C", str(repo_path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if actual_commit != ACCENTCL_COMMIT:
        raise RuntimeError("AccentCL source checkout did not match the pinned commit.")

    sys.path.insert(0, str(repo_path))
    from src.accentcl.models.accentcl import WhisperCLAccentModel

    if progress_callback:
        progress_callback("Downloading the AccentCL checkpoint (about 2.56 GB)...")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = hf_hub_download(
        repo_id=ACCENTCL_MODEL_REPO,
        filename=ACCENTCL_CHECKPOINT,
        cache_dir=str(cache_path),
    )
    if Path(checkpoint_path).stat().st_size < ACCENTCL_MIN_CHECKPOINT_BYTES:
        raise RuntimeError("Downloaded AccentCL checkpoint is unexpectedly small or incomplete.")

    if progress_callback:
        progress_callback("Downloading Whisper Large v3 files if they are not cached...")
    whisper_path = snapshot_download(
        repo_id="openai/whisper-large-v3",
        cache_dir=str(cache_path / "whisper-hub-cache"),
        local_dir=str(cache_path / "whisper-large-v3"),
        allow_patterns=[
            "*.safetensors",
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "normalizer.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "generation_config.json",
        ],
    )

    if progress_callback:
        progress_callback("Loading Whisper Large v3 and the accent classifier...")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True,
    )
    label2id = checkpoint.get("label2id")
    state_dict = checkpoint.get("model")
    if not isinstance(label2id, dict) or not isinstance(state_dict, dict):
        raise RuntimeError("AccentCL checkpoint is missing its label mapping or model weights.")

    # Instantiate the upstream architecture, but bypass its unsafe convenience
    # loader, which currently uses unrestricted pickle deserialization.
    model = WhisperCLAccentModel(
        num_classes=len(label2id),
        label2id=label2id,
        model_name=str(whisper_path),
    )
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict
    gc.collect()
    model.to(torch.device(device)).eval()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return model
