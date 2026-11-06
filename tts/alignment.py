"""Acoustic phoneme alignment via BFA 2.0.1 / CUPE-3.

Telugu is ZERO-SHOT for CUPE-3-mls7 (European-language training). It must be
manually audited before full training; successful computation is not proof of
Telugu boundary accuracy. No uniform or text-proportional durations are used.
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from .frontend import configure_espeak, language_code, normalize_text, phoneme_detail, fold_phone_groups, alignment_text, configure_phonemizer


@lru_cache(maxsize=4)
def _aligner(language, device):
    configure_espeak()
    from bournemouth_aligner import BFA
    from huggingface_hub import snapshot_download
    code = language_code(language)
    model, revision = (("Tabahi/CUPE-3-en", "cf87be29456f00f71bcfcd0f1c0e37d48e719e33") if code == "en-us" else
                       ("Tabahi/CUPE-3-mls7", "7c1cfc128fdd9799b247b912c264822d8efcf993"))
    model_path = snapshot_download(model, revision=revision, allow_patterns=["config.json", "model.safetensors"])
    b = BFA(preset=code, lang=code, ckpt_path=model_path, phoneme_set="ph66", device=device,
            chunk_audio=True, warn_level=1)
    configure_phonemizer(b.phonemizer)
    return b


def align(wav, sr, text, language, *, device="cpu"):
    """Return IPA phones, acoustic intervals (seconds), confidence and provenance.

    `wav` can be a path or mono/multichannel ndarray. Leading/trailing silence
    absorbs only the uncovered boundary audio. Internal boundaries come from
    the acoustic decoder. Low-quality/missing speech targets raise.
    """
    import torch
    import librosa
    import soundfile as sf
    if isinstance(wav, (str, Path)):
        samples, sample_rate = sf.read(wav, dtype="float32", always_2d=False)
        if sr is not None and sample_rate != sr:
            raise ValueError(f"Requested sr={sr}, actual sr={sample_rate}")
        sr = sample_rate
    else:
        samples = np.asarray(wav, dtype=np.float32)
    if sr is None:
        raise ValueError("sr required for array audio")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Invalid waveform")
    original_duration = len(samples) / sr
    samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
    b = _aligner(language_code(language), device)
    normalized = normalize_text(text, language)
    detail = phoneme_detail(normalized, language)
    raw = b.process_sentence(alignment_text(normalized, language), torch.from_numpy(samples).float().unsqueeze(0))
    segments = raw["segments"]
    if len(segments) != 1:
        raise ValueError("Expected one aligned segment")
    segment = segments[0]
    # Guard against any future BFA update changing its text frontend.
    if segment["ipa"] != detail["eipa"]:
        raise ValueError("Alignment/frontend phone sequence mismatch")
    targets = {int(x["target_seq_idx"]): x for x in segment["phoneme_ts"]}
    groups = fold_phone_groups(detail)
    intervals, confidence, phones = [], [], []
    for phone, indices in groups:
        matched = [targets[i] for i in indices if i in targets]
        if phone != "SIL" and len(matched) != len(indices):
            raise ValueError(f"Missing acoustic target for {phone}: {indices}")
        if matched:
            start = min(original_duration, max(0.0, min(x["start_ms"] for x in matched) / 1000))
            end = min(original_duration, max(0.0, max(x["end_ms"] for x in matched) / 1000))
            conf = float(np.mean([x["confidence"] for x in matched]))
        else:
            start = end = 0.0
            conf = None
        phones.append(phone); intervals.append([start, end]); confidence.append(conf)
    # BFA does not emit uncovered boundary silence. Make it explicit, never assign
    # that silence to the first/last speech phone.
    for i, phone in enumerate(phones):
        if phone == "SIL":
            left = intervals[i-1][1] if i else 0.0
            right = intervals[i+1][0] if i+1 < len(intervals) else original_duration
            intervals[i] = [left, right]
    confidence[0] = confidence[-1] = None
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1] - 1e-5:
            raise ValueError("Overlapping decoder boundaries")
        # Decoder intervals are contiguous except potential skipped silence. Use
        # a midpoint for sub-frame numeric gaps, never large unexplained gaps.
        gap = intervals[i][0] - intervals[i-1][1]
        if gap > .030 and phones[i] != "SIL" and phones[i-1] != "SIL":
            raise ValueError(f"Unexplained {gap:.3f}s gap in alignment")
        midpoint = (intervals[i][0] + intervals[i-1][1]) / 2
        intervals[i][0] = midpoint; intervals[i-1][1] = midpoint
    if any(end < start for start, end in intervals):
        raise ValueError("Negative-duration alignment")
    return {"phones": phones, "intervals_seconds": intervals, "confidence": confidence,
            "source_text": text, "normalized_text": normalized,
            "duration_seconds": original_duration, "raw_alignment": raw,
            "backend": {"package": "bournemouth-forced-aligner", "version": "2.0.1",
                        "phoneme_set": "ph66", "model": "Tabahi/CUPE-3-en" if language_code(language) == "en-us" else "Tabahi/CUPE-3-mls7",
                        "telugu_zero_shot": language_code(language) == "te",
                        "boundary_audit_required": True}}


def frame_durations(result, total_frames, sample_rate=22050, hop_length=256):
    """Quantize acoustic boundaries; exact total, no invented phone timing."""
    ends = np.array([x[1] for x in result["intervals_seconds"]])
    bounds = np.rint(ends * sample_rate / hop_length).astype(np.int64)
    bounds = np.clip(bounds, 0, total_frames)
    bounds[-1] = total_frames
    durations = np.diff(np.concatenate([[0], bounds]))
    if np.any(durations < 0) or durations.sum() != total_frames:
        raise ValueError("Invalid quantized durations")
    return durations


def save_alignment_plot(result, wav, sr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    samples = np.asarray(wav)
    fig, ax = plt.subplots(figsize=(max(12, len(result["phones"]) * .4), 3.5))
    ax.plot(np.arange(len(samples))/sr, samples, color="slategray", linewidth=.5)
    for i, (phone, (start, end), conf) in enumerate(zip(result["phones"], result["intervals_seconds"], result["confidence"])):
        ax.axvspan(start, end, color="steelblue" if i%2 else "goldenrod", alpha=.13)
        ax.axvline(start, color="gray", linewidth=.5)
        label = phone if conf is None else f"{phone}\n{conf:.2f}"
        ax.text((start+end)/2, .94, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax.set(xlabel="Seconds", ylabel="Amplitude", title=f"Acoustic alignment — {result['backend']['model']} (Telugu zero-shot: {result['backend']['telugu_zero_shot']})")
    fig.tight_layout();fig.savefig(path, dpi=160);plt.close(fig)
