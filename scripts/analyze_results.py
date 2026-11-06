#!/usr/bin/env python3
"""Measure final inference artifacts; never substitute estimates for missing WAVs.

Run from any directory: python scripts/analyze_results.py
Development fixtures require --development --result-dir PATH and are never
written to Results. This audit measures artifacts; it does not assess perceived
speech quality, pronunciation correctness, or intelligibility.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tts.checkpoint_selection import resolve_selected_checkpoint
LANGUAGE_IDS = {
    "English": {"E1", "E2a", "E2b", "E3", "E4"},
    "Telugu": {"T1", "T2a", "T2b", "T3", "T4"},
}


class AnalysisError(ValueError):
    """An artifact is absent, malformed, or inconsistent with its provenance."""


def require(condition, message):
    if not condition:
        raise AnalysisError(message)


def sha256(path):
    path = Path(path)
    require(path.is_file(), f"Missing required file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    sha256(path)  # Give missing files the same explicit error as other artifacts.
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Invalid JSON: {path}: {exc}") from exc


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and np.isfinite(values).all(), "Invalid summary values")
    if not len(values):
        return {"count": 0, **{key: None for key in ("min", "p05", "mean", "median", "p95", "max", "std")}}
    return {"count": int(len(values)), "min": float(values.min()),
            "p05": float(np.percentile(values, 5)), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(values.max()), "std": float(values.std())}


def acoustic_f0(audio, sample_rate, hop_length, skip=False):
    settings = {"algorithm": "WORLD DIO + StoneMask", "f0_floor_hz": 60.0,
                "f0_ceiling_hz": 600.0, "frame_period_ms": 1000 * hop_length / sample_rate,
                "source": "decoded generated WAV", "scope": "entire waveform, including pauses"}
    if skip:
        return {"status": "skipped", "reason": "--skip-acoustic-f0", **settings}
    try:
        import pyworld
    except ImportError:
        return {"status": "unavailable", "reason": "Optional pyworld package is not installed", **settings}
    samples = np.ascontiguousarray(audio, dtype=np.float64)
    initial, times = pyworld.dio(samples, sample_rate, f0_floor=settings["f0_floor_hz"],
                                f0_ceil=settings["f0_ceiling_hz"],
                                frame_period=settings["frame_period_ms"])
    f0 = pyworld.stonemask(samples, initial, times, sample_rate)
    require(f0.ndim == 1 and len(f0) > 0 and np.isfinite(f0).all() and np.all(f0 >= 0),
            "WORLD returned invalid acoustic F0")
    voiced = f0 > 0
    return {"status": "measured", **settings, "frame_count": int(len(f0)),
            "voiced_frame_count": int(voiced.sum()), "voiced_fraction": float(voiced.mean()),
            "voiced_f0_hz": summary(f0[voiced]),
            "interpretation": "Extractor voicing decisions and F0 estimates; not a quality or intelligibility score."}


def wav_measurements(path, expected_rate):
    digest = sha256(path)
    info = sf.info(path)
    audio, rate = sf.read(path, dtype="float64", always_2d=False)
    require(rate == expected_rate, f"{path}: expected {expected_rate} Hz, got {rate}")
    require(audio.ndim == 1 and len(audio) > 0, f"{path}: WAV must be nonempty mono audio")
    require(np.isfinite(audio).all(), f"{path}: WAV contains non-finite samples")
    bits = {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}.get(info.subtype)
    positive_rail = 1 - 2 ** (1 - bits) if bits else 1.0
    rail_mask = (audio >= positive_rail) | (audio <= -1.0)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return audio, {"sha256": digest, "sample_rate_hz": int(rate), "channels": 1,
                   "encoding": info.subtype, "sample_count": int(len(audio)),
                   "duration_seconds": len(audio) / rate, "rms": rms,
                   "rms_dbfs": float(20 * np.log10(rms)) if rms else None,
                   "peak": float(np.max(np.abs(audio))), "dc_mean": float(audio.mean()),
                   "full_scale_sample_count": int(rail_mask.sum()),
                   "clipping_fraction_at_pcm_rails": float(rail_mask.mean()),
                   "clipping_definition": "Samples at or beyond the positive/negative encoding rails. A rail hit alone does not establish pre-encoding clipping.",
                   "positive_rail": positive_rail, "negative_rail": -1.0}


def prediction_measurements(path, manifest_phones, sample_rate, hop_length):
    digest = sha256(path)
    needed = {"phones", "duration_frames", "duration_seconds", "pitch_hz", "energy", "mel"}
    try:
        with np.load(path, allow_pickle=False) as data:
            require(needed <= set(data.files), f"{path}: missing prediction arrays: {sorted(needed - set(data.files))}")
            phones = data["phones"]
            require(phones.ndim == 1 and len(phones) > 0 and phones.dtype.kind in "US",
                    f"{path}: phones must be a nonempty string vector")
            phones = phones.astype(str)
            require(phones.tolist() == manifest_phones, f"{path}: prediction phones differ from manifest")
            require(all(phones), f"{path}: empty phoneme label")
            arrays = {}
            for name in ("duration_frames", "duration_seconds", "pitch_hz", "energy"):
                value = data[name]
                require(value.dtype.kind in "iuf" and value.ndim == 1 and len(value) == len(phones),
                        f"{path}: {name} must be a numeric vector matching phones")
                require(np.isfinite(value).all(), f"{path}: {name} contains non-finite values")
                arrays[name] = value.astype(np.float64)
            durations = arrays["duration_frames"]
            require(np.all(durations >= 0) and np.all(durations == np.floor(durations)),
                    f"{path}: duration frames must be nonnegative integers")
            total_frames = int(durations.sum())
            require(total_frames > 0, f"{path}: all duration frames are zero")
            seconds = durations * hop_length / sample_rate
            require(np.allclose(arrays["duration_seconds"], seconds, rtol=1e-5, atol=1e-7),
                    f"{path}: duration_seconds does not match duration_frames and config")
            mel = data["mel"]
            require(mel.dtype.kind in "iuf" and mel.shape == (total_frames, 80) and np.isfinite(mel).all(),
                    f"{path}: mel must be finite [sum(duration_frames), 80]")
    except (OSError, TypeError) as exc:
        raise AnalysisError(f"Cannot read prediction file {path}: {exc}") from exc
    non_sil = phones != "SIL"
    require(non_sil.any(), f"{path}: no non-SIL phones")
    active = non_sil & (durations > 0)
    require(active.any(), f"{path}: no non-SIL phone has positive duration")
    phone_indices = np.flatnonzero(non_sil)
    maximum = int(phone_indices[np.argmax(durations[non_sil])])
    pauses = []
    index = 0
    while index < len(phones):
        if non_sil[index]:
            index += 1
            continue
        end = index + 1
        while end < len(phones) and not non_sil[end]:
            end += 1
        frames = int(durations[index:end].sum())
        pauses.append({"start_phone_index": index, "end_phone_index_inclusive": end - 1,
                       "duration_frames": frames, "duration_seconds": frames * hop_length / sample_rate})
        index = end
    variance = {}
    for name in ("pitch_hz", "energy"):
        values = arrays[name]
        variance[name + "_non_sil"] = summary(values[non_sil])
        variance[name + "_non_sil"]["nonpositive_count"] = int((values[non_sil] <= 0).sum())
        variance[name + "_non_sil"]["duration_weighted_mean"] = float(np.average(values[active], weights=durations[active]))
    return {"sha256": digest, "phone_count": len(phones), "non_sil_phone_count": int(non_sil.sum()),
            "zero_duration_phone_count": int((durations == 0).sum()),
            "zero_duration_non_sil_phone_count": int((durations[non_sil] == 0).sum()),
            "mel_frame_count": total_frames, "duration_seconds": total_frames * hop_length / sample_rate,
            "non_sil_duration_seconds": float(seconds[non_sil].sum()),
            "sil_duration_seconds": float(seconds[~non_sil].sum()),
            "max_phone": {"phone_index": maximum, "phone": str(phones[maximum]),
                          "duration_frames": int(durations[maximum]), "duration_seconds": float(seconds[maximum])},
            "pause_run_count": len(pauses), "max_pause": max(pauses, key=lambda x: x["duration_frames"]) if pauses else None,
            "predicted_variance": {"source": "acoustic-model phone-level predictions, denormalized at inference",
                                   "selection": "All non-SIL phones; duration-weighted means exclude zero-duration phones",
                                   "energy_units": "linear spectral energy (not waveform RMS)", **variance}}


def analyze_language(root, language, *, result_dir=None, checkpoint=None,
                     samples_path=None, config_path=None, development=False, skip_acoustic_f0=False):
    root = Path(root).resolve()
    require(language in LANGUAGE_IDS, f"Unsupported language: {language}")
    if development:
        require(result_dir is not None, "Development analysis requires an explicit result directory")
        result_dir = Path(result_dir).resolve()
        require("Results" not in result_dir.parts,
                "Development evidence cannot be written anywhere under Results")
    else:
        require(all(x is None for x in (result_dir, checkpoint, samples_path, config_path)),
                "Artifact overrides require --development")
        result_dir = root / "Results" / language
    selection = None
    if not development:
        checkpoint = resolve_selected_checkpoint(root, language)
        declaration = root / language / "checkpoints" / "selection.json"
        selection = read_json(declaration) if declaration.exists() else None
    else:
        checkpoint = Path(checkpoint) if checkpoint else root / language / "checkpoints" / "best.pt"
    samples_path = Path(samples_path) if samples_path else root / language / "test_samples.json"
    config_path = Path(config_path) if config_path else root / language / "config.yaml"
    checkpoint_hash = sha256(checkpoint)
    config_hash = sha256(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        sample_rate = int(config["preprocess"]["preprocessing"]["audio"]["sampling_rate"])
        hop_length = int(config["preprocess"]["preprocessing"]["stft"]["hop_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f"{config_path}: missing audio sampling rate or hop length") from exc
    require(sample_rate > 0 and hop_length > 0, "Sampling rate and hop length must be positive")
    samples = read_json(samples_path)
    require(isinstance(samples, list) and samples, f"{samples_path}: expected nonempty sample list")
    source = {}
    for item in samples:
        require(isinstance(item, dict) and isinstance(item.get("id"), str) and
                isinstance(item.get("text"), str), f"{samples_path}: invalid sample record")
        identifier = item["id"]
        require(re.fullmatch(r"[A-Za-z0-9_-]+", identifier) and identifier not in source,
                f"{samples_path}: duplicate or unsafe sample ID")
        source[identifier] = item["text"]
    if not development:
        require(set(source) == LANGUAGE_IDS[language], f"{samples_path}: wrong assignment sample IDs")
    manifest_path = result_dir / "inference_manifest.json"
    records = read_json(manifest_path)
    require(isinstance(records, list) and all(isinstance(row, dict) for row in records),
            f"{manifest_path}: expected a list of inference records")
    record_ids = [row.get("id") for row in records]
    require(all(isinstance(identifier, str) for identifier in record_ids) and
            len(record_ids) == len(set(record_ids)) and set(record_ids) == set(source),
            f"{manifest_path}: missing, duplicate, or unexpected sample IDs")
    require({p.stem for p in (result_dir / "predictions").glob("*.npz")} == set(source),
            f"{result_dir}: prediction file IDs differ from source samples")
    require({p.stem for p in (result_dir / "test_generated_audios").glob("*.wav")} == set(source),
            f"{result_dir}: WAV file IDs differ from source samples")
    by_id = {row["id"]: row for row in records}
    analyses = []
    steps = set()
    for identifier, text in source.items():
        record = by_id[identifier]
        require(record.get("text") == text, f"{identifier}: inference text differs from exact original sample")
        require(record.get("checkpoint_sha256") == checkpoint_hash,
                f"{identifier}: checkpoint hash mismatch")
        step = record.get("checkpoint_step")
        require(type(step) is int and step > 0, f"{identifier}: invalid checkpoint step")
        if selection is not None:
            require(step == selection["selected_step"], f"{identifier}: step differs from explicit checkpoint selection")
        steps.add(step)
        require(isinstance(record.get("phonemes"), list), f"{identifier}: missing phonemes")
        prediction = prediction_measurements(result_dir / "predictions" / f"{identifier}.npz",
                                             record["phonemes"], sample_rate, hop_length)
        audio, measured = wav_measurements(result_dir / "test_generated_audios" / f"{identifier}.wav", sample_rate)
        manifest_audio = record.get("audio", {})
        require(isinstance(manifest_audio, dict) and manifest_audio.get("sha256") == measured["sha256"],
                f"{identifier}: WAV hash mismatch")
        require(manifest_audio.get("frames") == measured["sample_count"], f"{identifier}: manifest WAV sample count mismatch")
        recorded_seconds = manifest_audio.get("seconds")
        require(type(recorded_seconds) in (int, float) and np.isfinite(recorded_seconds) and
                np.isclose(recorded_seconds, measured["duration_seconds"], rtol=0, atol=1 / sample_rate),
                f"{identifier}: manifest WAV duration mismatch")
        require(measured["sample_count"] == prediction["mel_frame_count"] * hop_length,
                f"{identifier}: WAV length differs from mel length times hop length")
        analyses.append({"id": identifier, "text": text, "checkpoint_step": step,
                         "prediction": prediction, "wav": measured,
                         "acoustic_f0": acoustic_f0(audio, sample_rate, hop_length, skip_acoustic_f0)})
    require(len(steps) == 1, "Inference records disagree about checkpoint step")
    return {"schema_version": 1, "scope": "development fixture evidence" if development else "assignment inference evidence",
            "language": language, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {"test_samples_sha256": sha256(samples_path), "config_sha256": config_hash,
                       "checkpoint_sha256": checkpoint_hash, "checkpoint_step": next(iter(steps)),
                       "checkpoint_file": checkpoint.name,
                       "checkpoint_selection": selection if selection is not None else {"basis": "explicit_development_checkpoint" if development else "minimum_total_validation_loss", "file": checkpoint.name},
                       "inference_manifest_sha256": sha256(manifest_path),
                       "sample_rate_hz": sample_rate, "hop_length": hop_length},
            "limitations": ["These are objective artifact measurements, not subjective speech-quality scores.",
                            "Predicted pitch and spectral energy are not measured waveform F0 and RMS.",
                            "WORLD uses 60–600 Hz search bounds; its voicing/F0 estimates can be wrong on synthesized audio.",
                            "Full-scale PCM samples indicate rail hits; lost peaks cannot be recovered from the written WAV.",
                            "Silence maxima refer to predicted SIL runs, not silence independently detected in audio."],
            "samples": analyses}


def analysis_csv(analysis):
    rows = []
    for item in analysis["samples"]:
        wav, prediction, f0 = item["wav"], item["prediction"], item["acoustic_f0"]
        variance = prediction["predicted_variance"]
        rows.append({"scope": analysis["scope"], "language": analysis["language"], "id": item["id"],
                     "wav_duration_seconds": wav["duration_seconds"], "wav_rms": wav["rms"],
                     "wav_rms_dbfs": wav["rms_dbfs"], "wav_peak": wav["peak"],
                     "wav_clipping_fraction_at_pcm_rails": wav["clipping_fraction_at_pcm_rails"],
                     "predicted_pitch_hz_non_sil_mean": variance["pitch_hz_non_sil"]["mean"],
                     "predicted_energy_non_sil_mean": variance["energy_non_sil"]["mean"],
                     "predicted_max_non_sil_phone": prediction["max_phone"]["phone"],
                     "predicted_max_non_sil_phone_seconds": prediction["max_phone"]["duration_seconds"],
                     "predicted_max_sil_run_seconds": prediction["max_pause"]["duration_seconds"] if prediction["max_pause"] else 0.0,
                     "predicted_zero_duration_non_sil_phone_count": prediction["zero_duration_non_sil_phone_count"],
                     "acoustic_f0_status": f0["status"], "acoustic_voiced_fraction": f0.get("voiced_fraction"),
                     "acoustic_voiced_f0_hz_mean": f0.get("voiced_f0_hz", {}).get("mean"),
                     "wav_sha256": wav["sha256"], "prediction_sha256": prediction["sha256"],
                     "checkpoint_sha256": analysis["source"]["checkpoint_sha256"], "text": item["text"]})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_analysis(result_dir, analysis):
    """Publish only a completely validated analysis; no inputs are modified."""
    result_dir = Path(result_dir)
    contents = {"analysis.json": json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                "analysis.csv": analysis_csv(analysis)}
    for name, content in contents.items():
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=result_dir,
                                         prefix=".analysis-", delete=False) as temporary:
            temporary.write(content)
            path = Path(temporary.name)
        try:
            path.replace(result_dir / name)
        finally:
            path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--language", choices=["all", *LANGUAGE_IDS], default="all")
    parser.add_argument("--skip-acoustic-f0", action="store_true", help="Skip optional WORLD waveform pitch extraction; record it as skipped")
    parser.add_argument("--development", action="store_true", help="Analyze explicit development fixtures outside Results")
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if args.development and args.language == "all":
        parser.error("--development requires one --language")
    languages = list(LANGUAGE_IDS) if args.language == "all" else [args.language]
    pending = []
    try:
        for language in languages:
            analysis = analyze_language(args.root, language, result_dir=args.result_dir,
                                        checkpoint=args.checkpoint, samples_path=args.samples, config_path=args.config,
                                        development=args.development, skip_acoustic_f0=args.skip_acoustic_f0)
            destination = args.result_dir if args.development else args.root / "Results" / language
            pending.append((destination, analysis))
        for destination, analysis in pending:
            write_analysis(destination, analysis)
    except (AnalysisError, OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"complete": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"complete": True, "outputs": [str(Path(path) / "analysis.json") for path, _ in pending],
                      "note": "Artifact measurements do not establish perceived speech quality."}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
