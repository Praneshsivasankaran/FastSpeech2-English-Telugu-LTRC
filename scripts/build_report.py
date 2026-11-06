#!/usr/bin/env python3
"""Build the final, evidence-bound FastSpeech 2 technical report.

This is deliberately not a progress-report generator. It writes no PDF until
both production runs, every requested output, current objective analyses, and
explicit observations for the exact final WAVs have passed validation.

Usage from the project directory:
  python scripts/build_report.py --check-only
  python scripts/build_report.py

Copy Report/report_observations.template.json to Report/report_observations.json
and complete it after inspecting the final artifacts. objective_only is a valid
assessment mode; it must not contain fabricated listening claims. A report in
that mode explicitly says that pronunciation and perceived quality were not
assessed. Final PDF authoring requires reportlab and the artifact-operation
marker required by the host PDF workflow; --check-only never creates a PDF.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sys
import unicodedata
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {"English": ["E1", "E2a", "E2b", "E3", "E4"],
             "Telugu": ["T1", "T2a", "T2b", "T3", "T4"]}
LOSS_NAMES = ("total", "mel", "postnet_mel", "pitch", "energy", "duration")
REFERENCE_COMMIT = "d4e79eb52e8b01d24703b2dfc0385544092958f3"


class EvidenceError(ValueError):
    """The report cannot truthfully describe the supplied artifacts."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, UnicodeError, ValueError) as exc:
        raise EvidenceError(f"Cannot read valid finite JSON from {path}: {exc}") from exc


def read_jsonl(path):
    try:
        rows = [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, ValueError) as exc:
        raise EvidenceError(f"Cannot read JSONL from {path}: {exc}") from exc
    require(all(isinstance(row, dict) for row in rows), f"Expected JSON objects: {path}")
    return rows


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).casefold()).strip()


def require_text(value, label, maximum=4000):
    require(isinstance(value, str) and value.strip(), f"Missing explicit observation: {label}")
    require(len(value) <= maximum, f"Observation is too long ({maximum} characters maximum): {label}")
    require(not re.search(r"\b(TODO|TBD|PLACEHOLDER|FILL\s+ME|REPLACE\s+ME)\b", value, re.I),
            f"Unfinished observation: {label}")
    # The PDF intentionally uses Roman-script assessment prose. Complex Telugu
    # text remains available, unmodified, in the hashed source JSON files.
    require(all(ord(char) < 128 or char in "\u2018\u2019\u201c\u201d\u2013\u2014\u2026" for char in value),
            f"Use Roman-script prose for {label}; exact Telugu text belongs in test_samples.json")


def validate_observations(observations, analyses):
    require(isinstance(observations, dict) and observations.get("schema_version") == 1,
            "Observations must use schema_version 1")
    require(observations.get("finalized") is True, "Observations are not finalized")
    for key in ("scope_summary", "cross_language_comparison", "failure_cases", "next_steps"):
        require_text(observations.get(key), key)
    require(isinstance(observations.get("language_assessments"), dict), "Missing language_assessments")
    require(set(observations["language_assessments"]) == set(LANGUAGES), "Wrong language assessment keys")
    for language in LANGUAGES:
        notes = observations["language_assessments"][language]
        require(isinstance(notes, dict), f"Invalid {language} assessment")
        for key in ("training_interpretation", "instability_or_overfitting", "limitations"):
            require_text(notes.get(key), f"{language}.{key}")
    samples = observations.get("samples")
    expected = {identifier for ids in LANGUAGES.values() for identifier in ids}
    require(isinstance(samples, dict) and set(samples) == expected,
            "Observations must contain exactly the ten prescribed sample IDs")
    for language, identifiers in LANGUAGES.items():
        measured = {row["id"]: row for row in analyses[language]["samples"]}
        for identifier in identifiers:
            item = samples[identifier]
            require(isinstance(item, dict), f"Invalid observation: {identifier}")
            mode = item.get("assessment_mode")
            require(mode in ("objective_only", "listening"), f"Choose an assessment_mode for {identifier}")
            require(item.get("wav_sha256") == measured[identifier]["wav"]["sha256"],
                    f"{identifier}: observation does not identify the final WAV SHA256")
            notes = item.get("objective_review")
            require(isinstance(notes, dict), f"{identifier}: missing objective_review")
            for key in ("duration", "pitch", "energy", "limitations"):
                require_text(notes.get(key), f"{identifier}.objective_review.{key}", maximum=1400)
            listening = item.get("listening")
            if mode == "objective_only":
                require(listening is None, f"{identifier}: objective_only must have listening=null")
            else:
                require(isinstance(listening, dict), f"{identifier}: missing listening assessment")
                require(listening.get("listened_to_final_wav") is True,
                        f"{identifier}: listening must explicitly confirm the final WAV was heard")
                for key in ("reviewer", "reviewed_at", "pronunciation", "intelligibility", "naturalness", "artifacts"):
                    require_text(listening.get(key), f"{identifier}.listening.{key}", maximum=1400)
                try:
                    datetime.fromisoformat(listening["reviewed_at"].replace("Z", "+00:00"))
                except ValueError as exc:
                    raise EvidenceError(f"{identifier}: reviewed_at must be an ISO 8601 timestamp") from exc


def required_paths(root, observations_path):
    paths = [root / filename for filename in
             ("README.md", "requirements.txt", "requirements-lock-macos.txt", "THIRD_PARTY_NOTICES.md",
              "provenance/REQUIREMENTS.md", "provenance/DATASETS.md", "provenance/FRONTEND_AUDIT.md",
              "provenance/ALIGNMENT_QUALITY.md", "provenance/dataset_sources.json")]
    paths.append(observations_path)
    for language, ids in LANGUAGES.items():
        base, result = root / language, root / "Results" / language
        paths.extend(base / name for name in
                     ("config.yaml", "test_samples.json", "preprocessed/preparation_summary.json",
                      "preprocessed/train.jsonl", "preprocessed/val.jsonl", "preprocessed/stats.json",
                      "preprocessed/vocab.json", "preprocessed/speakers.json", "preprocessed/rejected.jsonl",
                      "checkpoints/best.pt", "checkpoints/latest.pt", "checkpoints/split_audit.json",
                      "checkpoints/run_summary.json", "checkpoints/progress.json",
                      "checkpoints/logs/training.jsonl", "checkpoints/logs/validation.jsonl"))
        if (base / "checkpoints/selection.json").exists():
            paths.extend([base / "checkpoints/selection.json", base / "checkpoints/selected.pt",
                          root / "provenance" / f"{language}_validation_probes.json"])
        paths.extend(result / name for name in
                     ("training_loss.png", "validation_loss.png", "inference_manifest.json", "analysis.json"))
        for identifier in ids:
            paths.extend([result / "test_generated_audios" / f"{identifier}.wav",
                          result / "predictions" / f"{identifier}.npz"])
            paths.extend(result / "variance_adaptor_plots" / kind / f"{identifier}.png"
                         for kind in ("duration", "pitch", "energy"))
    return paths


def verify_png(path):
    # Validate the PNG header without importing the renderer during --check-only.
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    require(len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR",
            f"Invalid PNG: {path}")
    require(int.from_bytes(header[16:20], "big") >= 200 and int.from_bytes(header[20:24], "big") >= 100,
            f"PNG is too small to review: {path}")


def verify_logs(rows, label):
    require(rows, f"Empty production log: {label}")
    steps = [row.get("step") for row in rows]
    require(all(type(step) is int and step > 0 for step in steps), f"Invalid optimizer steps: {label}")
    require(all(a < b for a, b in zip(steps, steps[1:])), f"Unordered/repeated steps: {label}")
    for row in rows:
        require(all(finite_number(row.get(key)) for key in LOSS_NAMES), f"Missing/non-finite loss: {label}")
        require(all(row[key] >= 0 for key in LOSS_NAMES), f"Negative loss: {label}")
    return steps


def effective_training_settings(config):
    """Settings that must stay fixed across one resumed optimizer trajectory.

    Paths, device, log interval and the target stop step may change on resume.
    The legacy learning_rate field is inactive when peak_learning_rate exists.
    """
    training = config["training"]
    return {"seed": training.get("seed", 20261105),
            "batch_size": training.get("batch_size", 8),
            "peak_learning_rate": training.get("peak_learning_rate", training.get("learning_rate", 1e-4)),
            "warmup_steps": training.get("warmup_steps", 0),
            "precision": training.get("precision", "float32"),
            "weight_decay": training.get("weight_decay", 0),
            "grad_clip": training.get("grad_clip", 1.0),
            "length_bucket_size": training.get("length_bucket_size", 0)}


def verify_training_settings(actual, expected, label):
    for key, value in effective_training_settings(expected).items():
        require(effective_training_settings(actual)[key] == value,
                f"{label}: {key} differs from the selected production checkpoint; "
                "retain the actual cloud training settings when making paths portable")


def verify_progress(progress, validation, selected_step, latest_step, label):
    """Progress describes latest.pt; the inference checkpoint can be earlier."""
    require(progress.get("step") == latest_step, f"{label}: progress/latest step mismatch")
    require(type(progress.get("total_target_steps")) is int and progress["total_target_steps"] >= latest_step,
            f"{label}: invalid final invocation target step")
    require(selected_step <= latest_step, f"{label}: selected checkpoint is later than latest")
    latest_metrics = next(row for row in validation if row["step"] == latest_step)
    for key in LOSS_NAMES:
        require(finite_number(progress.get("validation", {}).get(key))
                and math.isclose(progress["validation"][key], latest_metrics[key], rel_tol=1e-6, abs_tol=1e-7),
                f"{label}: progress metrics must describe latest.pt, not an earlier inference checkpoint")
    require(finite_number(progress.get("best_validation")) and
            math.isclose(progress["best_validation"], min(row["total"] for row in validation), rel_tol=1e-6),
            f"{label}: progress best_validation disagrees with the complete validation history")


def verify_selection_for_report(selection, probes, probe_hash, validation_ids,
                                best_step, best_hash, selected_step, selected_hash, language):
    """Validate report-specific evidence in addition to the shared resolver gate."""
    require_text(selection.get("rationale"), f"{language}.selection.rationale")
    require_text(selection.get("protocol_note"), f"{language}.selection.protocol_note")
    require(selection.get("selected_step") == selected_step and
            selection.get("selected_sha256", "").lower() == selected_hash,
            f"{language}: selection declaration differs from the inference checkpoint")
    require(selection.get("validation_probe_manifest_sha256", "").lower() == probe_hash,
            f"{language}: published validation-probe manifest hash mismatch")
    require(isinstance(probes, list) and len(probes) == 3 and all(isinstance(row, dict) for row in probes),
            f"{language}: selection requires exactly three fixed validation prompts")
    ids = [row.get("id") for row in probes]
    require(all(isinstance(identifier, str) for identifier in ids) and len(set(ids)) == 3 and
            selection.get("probe_ids") == ids and set(ids) <= validation_ids,
            f"{language}: selection probes must match the published held-out validation IDs")
    require(all(isinstance(row.get("text"), str) and row["text"] for row in probes),
            f"{language}: missing source text in validation probes")
    candidates = selection.get("candidates")
    labels = [row.get("label") for row in candidates if isinstance(row, dict)] if isinstance(candidates, list) else []
    posthoc = selection.get('decision') == 'posthoc_validation_improvement'
    require(isinstance(candidates, list) and len(candidates) == 2 and len(labels) == 2 and
            labels.count('minimum_total') == 1 and all(isinstance(label, str) and label for label in labels) and
            (posthoc or set(labels) == {"minimum_total", "later_fixed"}),
            f"{language}: selection must identify the baseline and one declared comparison candidate")
    for candidate in candidates:
        require(type(candidate.get("step")) is int and candidate['step'] > 0,
                f"{language}: candidate step is missing")
        require(isinstance(candidate.get("sha256"), str) and re.fullmatch(r"[0-9a-fA-F]{64}", candidate['sha256']),
                f"{language}: candidate hash is missing")
        for metric in ("wer", "cer", "duration_ratio_min", "duration_ratio_max"):
            require(finite_number(candidate.get(metric)) and candidate[metric] >= 0,
                    f"{language}: invalid candidate {metric}")
        require(candidate['duration_ratio_min'] <= candidate['duration_ratio_max'] and
                type(candidate.get('valid_audio')) is bool,
                f"{language}: invalid candidate artifact/duration evidence")
    baseline = next(row for row in candidates if row['label'] == 'minimum_total')
    require(baseline['step'] == best_step and baseline['sha256'].lower() == best_hash,
            f"{language}: minimum_total candidate must identify the separately audited best.pt")
    chosen = [row for row in candidates if row['step'] == selected_step and row['sha256'].lower() == selected_hash]
    require(len(chosen) == 1, f"{language}: selected checkpoint must be one of the two fixed candidates")
    controls = selection.get("source_controls")
    require(isinstance(controls, dict) and all(finite_number(controls.get(key)) and controls[key] >= 0
                                              for key in ('wer', 'cer')),
            f"{language}: source-recording ASR controls are required")
    require(selection.get("decision") in ('lower_validation_wer_and_nonincreasing_cer', 'baseline_retained_inconclusive',
                                           'later_best_available_failed_quality_gate', 'posthoc_validation_improvement'),
            f"{language}: unknown checkpoint-selection decision")
    if selection['decision'] == 'baseline_retained_inconclusive':
        require(chosen[0]['label'] == 'minimum_total', f"{language}: inconclusive decision must retain the baseline")
    elif selection['decision'] == 'later_best_available_failed_quality_gate':
        require(chosen[0]['label'] == 'later_fixed' and selection.get('quality_gate_passed') is False,
                f"{language}: a best-available deviation must explicitly record the failed quality gate")
        require(chosen[0]['valid_audio'], f"{language}: best-available artifact is invalid audio")
    elif posthoc:
        require(chosen[0]['label'] != 'minimum_total' and selection.get('quality_gate_passed') is False,
                f"{language}: post-hoc preference must preserve disclosure of the failed quality gate")
        require(chosen[0]['valid_audio'], f"{language}: post-hoc selected artifact is invalid audio")
    else:
        require(chosen[0]['valid_audio'], f"{language}: selected comparison candidate has invalid audio")


def inspect_inference_recipe(path, record, stats, config):
    """Validate optional raw/conditioning arrays without changing raw analysis."""
    import numpy as np
    controls = record.get('inference_controls', {})
    frontend = record.get('frontend', {})
    require(isinstance(controls, dict) and isinstance(frontend, dict), f"{path}: malformed inference metadata")
    detail = {'controls': controls, 'frontend': frontend, 'zero_sil_energy_applied': False,
              'has_conditioning_evidence': False}
    required = {'raw_energy_normalized', 'conditioning_energy_normalized', 'conditioning_energy',
                'silence_mask', 'zero_sil_energy_applied'}
    with np.load(path, allow_pickle=False) as values:
        present = required & set(values.files)
        if not present:
            require(not controls.get('zero_sil_energy'), f"{path}: declared SIL override has no array evidence")
            return detail
        require(present == required, f"{path}: incomplete conditioning energy evidence")
        phones = values['phones']; raw = values['energy']; silence = values['silence_mask']
        require(silence.dtype.kind == 'b' and silence.shape == phones.shape and
                np.array_equal(silence, phones == 'SIL'), f"{path}: silence mask differs from actual phones")
        flag = values['zero_sil_energy_applied']
        require(flag.shape == () and flag.dtype.kind == 'b', f"{path}: invalid SIL override flag")
        applied = bool(flag.item())
        require(bool(controls.get('zero_sil_energy', False)) == applied,
                f"{path}: inference-control flag differs from the saved prediction")
        arrays = {name: values[name] for name in required if name not in ('silence_mask', 'zero_sil_energy_applied')}
        require(all(value.shape == raw.shape and value.dtype.kind in 'iuf' and np.isfinite(value).all()
                    for value in arrays.values()), f"{path}: invalid conditioning arrays")
        normalized = config['preprocess']['preprocessing']['energy'].get('normalization', False)
        scale, mean = (stats['energy'][3], stats['energy'][2]) if normalized else (1.0, 0.0)
        require(np.allclose(arrays['raw_energy_normalized'] * scale + mean, raw, rtol=1e-5, atol=1e-5),
                f"{path}: raw energy was changed or incorrectly denormalized")
        conditioned = arrays['conditioning_energy']
        require(np.allclose(arrays['conditioning_energy_normalized'] * scale + mean, conditioned,
                            rtol=1e-5, atol=1e-4), f"{path}: conditioning energy units disagree")
        require(np.allclose(conditioned[~silence], raw[~silence], rtol=1e-5, atol=1e-5),
                f"{path}: SIL-only conditioning unexpectedly changes non-SIL energy")
        if applied:
            require(np.allclose(conditioned[silence], 0.0, rtol=0, atol=1e-4),
                    f"{path}: declared zero-SIL energy is not zero in linear units")
        else:
            require(np.allclose(conditioned, raw, rtol=1e-5, atol=1e-5),
                    f"{path}: undeclared conditioning-energy change")
        detail.update(has_conditioning_evidence=True, zero_sil_energy_applied=applied,
                      raw_sil_energy_min=float(raw[silence].min()) if silence.any() else None,
                      conditioning_sil_energy_min=float(conditioned[silence].min()) if silence.any() else None,
                      conditioning_sil_energy_max=float(conditioned[silence].max()) if silence.any() else None)
    return detail


def verify_manifest(root, rows, label):
    require(rows, f"Empty split: {label}")
    for key in ("id", "audio_sha256"):
        values = [row.get(key) for row in rows]
        require(all(isinstance(value, str) and value for value in values) and len(values) == len(set(values)),
                f"Missing/duplicate {key}: {label}")
    for row in rows:
        require(isinstance(row.get("text"), str) and normalized(row["text"]), f"Missing text: {label}")
        require(finite_number(row.get("duration_seconds")) and 1 <= row["duration_seconds"] <= 30,
                f"Invalid source duration: {label}/{row['id']}")
        require(type(row.get("speaker_id")) is int, f"Missing speaker/profile ID: {label}")
    # Same portable content fingerprint as tts.training.audit_splits; raw audio
    # is not needed, but the actual processed targets must still be available.
    portable = []
    for row in rows:
        feature = Path(row.get("features", ""))
        if not feature.is_absolute():
            feature = root / feature
        require(feature.is_file(), f"Missing processed target: {feature}")
        record = {key: value for key, value in row.items()
                  if key not in ("features", "audio_path", "wav_path", "raw_features")}
        record["feature_content_sha256"] = sha256(feature)
        portable.append(record)
    return hashlib.sha256(json.dumps(portable, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def same_measurements(actual, expected, label):
    """Compare the stored audit with fresh measurements of its hashed artifacts."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"Stale analysis fields: {label}")
        for key, value in expected.items():
            same_measurements(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Stale analysis list: {label}")
        for index, value in enumerate(expected):
            same_measurements(actual[index], value, f"{label}[{index}]")
    elif finite_number(expected):
        require(finite_number(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10),
                f"Stale measured value: {label}")
    else:
        require(actual == expected, f"Stale analysis value: {label}")


@dataclass
class LanguageEvidence:
    language: str
    config: dict
    latest_config: dict
    preparation: dict
    audit: dict
    run: dict
    progress: dict
    samples: list
    analysis: dict
    training: list
    validation: list
    rejected: list
    stats: dict
    checkpoint_hash: str
    checkpoint_file: str
    selected_step: int
    best_hash: str
    best_step: int
    latest_step: int
    vocabulary_size: int
    selection: dict | None
    selection_probes: list
    selection_history: dict | None
    selection_history_v2: dict | None
    inference_details: dict


def collect_evidence(root, observations_path):
    paths = required_paths(root, observations_path)
    missing = [str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
               for path in paths if not path.is_file()
               or (path.stat().st_size == 0 and path.name != "rejected.jsonl")]
    require(not missing, "Final report prerequisites are missing or empty:\n  " + "\n  ".join(missing))
    # A rejection log is legitimately empty if no candidates were rejected.
    import yaml
    import torch
    import analyze_results
    from tts.checkpoint_selection import resolve_selected_checkpoint

    sources = read_json(root / "provenance/dataset_sources.json")
    observations = read_json(observations_path)
    languages = {}
    for language, ids in LANGUAGES.items():
        base, results = root / language, root / "Results" / language
        code = "en" if language == "English" else "te"
        config = yaml.safe_load((base / "config.yaml").read_text(encoding="utf-8"))
        require(config.get("language") == code, f"Wrong config language: {language}")
        prep = read_json(base / "preprocessed/preparation_summary.json")
        audit = read_json(base / "checkpoints/split_audit.json")
        run = read_json(base / "checkpoints/run_summary.json")
        progress = read_json(base / "checkpoints/progress.json")
        training = read_jsonl(base / "checkpoints/logs/training.jsonl")
        validation = read_jsonl(base / "checkpoints/logs/validation.jsonl")
        training_steps = verify_logs(training, f"{language} training")
        validation_steps = verify_logs(validation, f"{language} validation")
        require(training_steps == list(range(1, training_steps[-1] + 1)),
                f"{language}: production log must contain every optimizer step from 1")
        require(set(validation_steps) <= set(training_steps), f"{language}: validation references missing steps")
        require(validation_steps[-1] == training_steps[-1], f"{language}: no validation at the final step")
        require(audit.get("production_duration_gate") is True and audit.get("overlap_checks") == "passed",
                f"{language}: no successful production split audit")
        require(run.get("benchmark_only") is False and progress.get("benchmark_only") is False,
                f"{language}: a benchmark run cannot support the final report")
        require(run.get("split_audit") == audit and progress.get("split_audit") == audit,
                f"{language}: inconsistent split evidence")
        require(progress.get("step") == training_steps[-1], f"{language}: progress/log step mismatch")
        require(prep.get("language") == code and prep.get("statistics_fit_on") == "train only",
                f"{language}: missing language-specific train-only preparation evidence")
        train, val = read_jsonl(base / "preprocessed/train.jsonl"), read_jsonl(base / "preprocessed/val.jsonl")
        for name, rows, count_key, seconds_key, lower, upper in (
                ("train", train, "train_examples", "train_seconds", 16200, 19800),
                ("validation", val, "validation_examples", "validation_seconds", 3240, 3960)):
            fingerprint = verify_manifest(root, rows, f"{language}/{name}")
            require(fingerprint == audit[f"{name}_manifest_sha256"], f"{language}: changed {name} targets")
            seconds = sum(row["duration_seconds"] for row in rows)
            require(lower <= seconds <= upper and math.isclose(seconds, audit[seconds_key], abs_tol=.001),
                    f"{language}: incorrect {name} duration")
            run_count_key = "training_examples" if name == "train" else "validation_examples"
            require(len(rows) == audit[count_key] == run[run_count_key], f"{language}: incorrect {name} count")
            preparation_key = "val" if name == "validation" else "train"
            require(math.isclose(seconds, prep["duration_seconds"][preparation_key], abs_tol=.001),
                    f"{language}: preparation/{name} duration mismatch")
        for key in ("id", "audio_sha256"):
            require(not ({row[key] for row in train} & {row[key] for row in val}), f"{language}: {key} leakage")
        require(not ({normalized(row['text']) for row in train} & {normalized(row['text']) for row in val}),
                f"{language}: normalized transcript leakage")
        require({row['speaker_id'] for row in val} <= {row['speaker_id'] for row in train},
                f"{language}: validation contains an untrained conditioning ID")
        vocab, stats = read_json(base / "preprocessed/vocab.json"), read_json(base / "preprocessed/stats.json")
        speakers = read_json(base / "preprocessed/speakers.json")
        selected_path = resolve_selected_checkpoint(root, language)
        selection_path = base / "checkpoints/selection.json"
        selection = read_json(selection_path) if selection_path.exists() else None
        selection_probes = []
        history_path = base / 'checkpoints/selection_v1.json'
        selection_history = read_json(history_path) if history_path.is_file() else None
        history_v2_path = base / 'checkpoints/selection_v2.json'
        selection_history_v2 = read_json(history_v2_path) if history_v2_path.is_file() else None
        if selection is not None and selection.get('decision') == 'posthoc_validation_improvement':
            require(selection_history is not None, f"{language}: post-hoc selection requires preserved selection_v1.json")
        best_hash = sha256(base / "checkpoints/best.pt")
        checkpoint_hash = sha256(selected_path)
        best_step, latest_step, selected_step = None, None, None
        actual_config, latest_config, best_config = None, None, None
        checkpoint_names = ("best", "latest", "selected") if selection is not None else ("best", "latest")
        for name in checkpoint_names:
            # Only load trusted, locally produced checkpoints. torch checkpoints
            # are pickle containers and are not safe inputs from strangers.
            state = torch.load(base / f"checkpoints/{name}.pt", map_location="cpu", weights_only=False)
            require(state.get("benchmark_only") is False and state.get("language") == code,
                    f"{language}/{name}: not a production language checkpoint")
            require(state.get("split_audit") == audit, f"{language}/{name}: split mismatch")
            require(state.get("vocab") == vocab, f"{language}/{name}: vocabulary mismatch")
            contract = {"language": code, "preprocessing": config['preprocess']['preprocessing'],
                        "model": config['model'], "stats": stats, "speakers": speakers}
            digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            require(state.get("contract_sha256") == digest, f"{language}/{name}: acoustic contract mismatch")
            step = state.get("step")
            require(type(step) is int and step in validation_steps, f"{language}/{name}: unlogged checkpoint step")
            at_step = next(row for row in validation if row['step'] == step)
            for metric in LOSS_NAMES:
                require(math.isclose(state['validation'][metric], at_step[metric], rel_tol=1e-6, abs_tol=1e-7),
                        f"{language}/{name}: checkpoint validation metric mismatch")
            require(isinstance(state.get("model"), dict) and state['model'], f"{language}/{name}: missing model weights")
            if name == "best":
                best_step = step
                best_config = state['config']
                if selection is None:
                    selected_step, actual_config = step, state['config']
                require(math.isclose(at_step['total'], min(row['total'] for row in validation), rel_tol=1e-6),
                        f"{language}: best checkpoint is not the logged validation minimum")
            elif name == "latest":
                latest_step = step
                latest_config = state['config']
                require(step == training_steps[-1], f"{language}: latest checkpoint does not match final logged step")
            else:
                selected_step, actual_config = step, state['config']
            del state
        # A best checkpoint may precede the last saved step. Its optimizer
        # contract must match the trajectory, but its validation metrics must
        # never be compared to progress.json's final-step validation metrics.
        verify_training_settings(latest_config, actual_config, f"{language}/latest.pt")
        verify_training_settings(best_config, actual_config, f"{language}/best.pt")
        verify_training_settings(config, actual_config, f"{language}/config.yaml")
        settings = effective_training_settings(actual_config)
        require(run.get("language") == code, f"{language}: wrong run-summary language")
        for key in ("batch_size", "precision", "peak_learning_rate", "warmup_steps", "length_bucket_size"):
            require(run.get(key) == settings[key], f"{language}: run-summary {key} differs from the checkpoint")
        require(type(run.get("steps")) is int and 0 < run["steps"] <= latest_step,
                f"{language}: invalid last-invocation optimizer-step count")
        for row in training:
            require(row.get("precision") == settings["precision"], f"{language}: mixed training precision in logs")
            step = row['step']
            warmup = settings['warmup_steps']
            expected_lr = settings['peak_learning_rate'] * (min(step / warmup, math.sqrt(warmup / step)) if warmup else 1)
            require(finite_number(row.get('learning_rate')) and
                    math.isclose(row['learning_rate'], expected_lr, rel_tol=1e-8, abs_tol=1e-12),
                    f"{language}: logged learning-rate schedule differs from the selected checkpoint")
        verify_progress(progress, validation, selected_step, latest_step, language)
        if selection is not None:
            probe_path = root / "provenance" / f"{language}_validation_probes.json"
            selection_probes = read_json(probe_path)
            verify_selection_for_report(selection, selection_probes, sha256(probe_path),
                                        {row['id'] for row in val}, best_step, best_hash,
                                        selected_step, checkpoint_hash, language)
        samples = read_json(base / "test_samples.json")
        require(isinstance(samples, list) and [row.get("id") for row in samples] == ids,
                f"{language}: wrong source sample IDs/order")
        analysis = read_json(results / "analysis.json")
        require(analysis.get("schema_version") == 1 and analysis.get("scope") == "assignment inference evidence",
                f"{language}: final objective analysis is missing or is development evidence")
        require(analysis.get("language") == language, f"{language}: wrong analysis language")
        current = analyze_results.analyze_language(root, language, skip_acoustic_f0=True)
        same_measurements(analysis['source'], current['source'], f"{language}.analysis.source")
        require([row.get("id") for row in analysis['samples']] == ids, f"{language}: wrong analysis sample order")
        records = {row['id']: row for row in read_json(results / 'inference_manifest.json')}
        inference_details = {}
        for saved, fresh in zip(analysis['samples'], current['samples']):
            for key in ("id", "text", "checkpoint_step", "prediction", "wav"):
                same_measurements(saved[key], fresh[key], f"{language}.{saved['id']}.{key}")
            require(saved['checkpoint_step'] == selected_step, f"{language}: inference is not from {selected_path.name}")
            require(saved.get('acoustic_f0', {}).get('status') in ('measured', 'skipped', 'unavailable'),
                    f"{language}: unclear waveform F0 measurement status")
            inference_details[saved['id']] = inspect_inference_recipe(
                results / 'predictions' / f"{saved['id']}.npz", records[saved['id']], stats, config)
        for kind in ("duration", "pitch", "energy"):
            require({path.stem for path in (results / "variance_adaptor_plots" / kind).glob("*.png")} == set(ids),
                    f"{language}: incomplete or unexpected {kind} plot IDs")
        for path in results.rglob("*.png"):
            verify_png(path)
        languages[language] = LanguageEvidence(
            language=language, config=actual_config, latest_config=latest_config, preparation=prep,
            audit=audit, run=run, progress=progress, samples=samples, analysis=analysis, training=training,
            validation=validation, rejected=read_jsonl(base / "preprocessed/rejected.jsonl"), stats=stats,
            checkpoint_hash=checkpoint_hash, checkpoint_file=selected_path.name, selected_step=selected_step,
            best_hash=best_hash, best_step=best_step, latest_step=latest_step, vocabulary_size=len(vocab),
            selection=selection, selection_probes=selection_probes, selection_history=selection_history,
            selection_history_v2=selection_history_v2,
            inference_details=inference_details)
    validate_observations(observations, {language: data.analysis for language, data in languages.items()})
    return languages, sources, observations


def numeric(value, digits=3):
    return "not measured" if value is None else f"{value:,.{digits}f}"


def prose(value):
    replacements = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                    "\u2013": "-", "\u2014": " - ", "\u2026": "..."}
    return escape(str(value).translate(str.maketrans(replacements)))


def build_pdf(root, languages, sources, observations, output):
    """Render only previously validated evidence. The caller owns visual QA."""
    try:
        import reportlab
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak,
                                       PageTemplate, Paragraph, Spacer, Table, TableStyle)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise EvidenceError("PDF rendering needs reportlab, matplotlib, and numpy in this environment. "
                            "Install the missing renderer package, then rerun the unchanged evidence gate. "
                            f"Original error: {exc}") from exc

    page_width, page_height = A4
    margin, width = 43, page_width - 86
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=26, leading=30,
                              textColor=colors.HexColor("#18354A"), spaceAfter=17))
    styles.add(ParagraphStyle(name="ReportSubtitle", fontName="Helvetica", fontSize=12, leading=17,
                              textColor=colors.HexColor("#456579"), spaceAfter=16))
    styles.add(ParagraphStyle(name="Section", fontName="Helvetica-Bold", fontSize=17, leading=21,
                              textColor=colors.HexColor("#18354A"), spaceAfter=12))
    styles.add(ParagraphStyle(name="Subsection", fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                              textColor=colors.HexColor("#18354A"), spaceBefore=9, spaceAfter=5))
    styles.add(ParagraphStyle(name="BodyTextReport", fontName="Helvetica", fontSize=8.8, leading=12.2,
                              spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallReport", fontName="Helvetica", fontSize=7.1, leading=9.5,
                              spaceAfter=4, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="TableReport", fontName="Helvetica", fontSize=7.4, leading=9.4,
                              spaceAfter=0, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="CodeReport", fontName="Courier", fontSize=7.3, leading=10.2,
                              leftIndent=8, rightIndent=4, spaceAfter=6, wordWrap="CJK"))
    story, image_buffers = [], []

    def p(text, style="BodyTextReport"):
        return Paragraph(prose(text), styles[style])

    def add(text, style="BodyTextReport"):
        story.append(p(text, style))

    def heading(text):
        if story:
            story.append(PageBreak())
        add(text, "Section")

    def table(rows, widths=None, header=True):
        cell_rows = [[p(cell, "TableReport") for cell in row] for row in rows]
        result = Table(cell_rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
        rules = [('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 6),
                 ('RIGHTPADDING', (0, 0), (-1, -1), 6), ('TOPPADDING', (0, 0), (-1, -1), 4),
                 ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                 ('LINEBELOW', (0, 0), (-1, 0), .5, colors.HexColor('#93AAB6')),
                 ('ROWBACKGROUNDS', (0, 1 if header else 0), (-1, -1),
                  [colors.HexColor('#F1F5F7'), colors.white])]
        if header:
            rules.append(('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E1EBEF')))
        result.setStyle(TableStyle(rules))
        return result

    def chart(figure, chart_width, chart_height):
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=220, facecolor="white")
        plt.close(figure)
        buffer.seek(0)
        image_buffers.append(buffer)
        return Image(buffer, width=chart_width, height=chart_height)

    def loss_chart(evidence, split):
        rows = getattr(evidence, split)
        fig, ax = plt.subplots(figsize=(3.35, 1.72))
        for key in LOSS_NAMES:
            ax.plot([row['step'] for row in rows], [row[key] for row in rows],
                    label=key.replace('_', ' '), linewidth=.9 if key != 'total' else 1.5)
        ax.set(title=f"{evidence.language}: {split}", xlabel="Optimizer step", ylabel="Loss")
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=.2)
        ax.legend(fontsize=5.3, ncol=3, loc='upper right', framealpha=.8)
        fig.tight_layout(pad=.9)
        return chart(fig, (width - 10) / 2, 129)

    def sample_charts(language, identifier):
        path = root / "Results" / language / "predictions" / f"{identifier}.npz"
        charts = []
        with np.load(path, allow_pickle=False) as values:
            for key, title, units in (("duration_seconds", "Duration", "Seconds / phone"),
                                      ("pitch_hz", "Predicted F0", "Hz"),
                                      ("energy", "Predicted energy", "Linear spectral energy")):
                fig, ax = plt.subplots(figsize=(2.22, 1.34))
                ax.plot(np.arange(len(values[key])), values[key], linewidth=.9, color='#216A8E', label='Raw head')
                if key == 'energy' and languages[language].inference_details[identifier]['zero_sil_energy_applied']:
                    ax.plot(np.arange(len(values[key])), values['conditioning_energy'], linewidth=.9,
                            linestyle='--', color='#AD6A16', label='Decoder conditioning')
                    ax.legend(fontsize=5, loc='best', framealpha=.8)
                ax.set(title=title, xlabel="Phoneme index", ylabel=units)
                ax.tick_params(labelsize=6)
                ax.xaxis.label.set_size(6.5)
                ax.yaxis.label.set_size(6.3)
                ax.title.set_size(8)
                ax.grid(alpha=.2)
                fig.tight_layout(pad=.8)
                charts.append(chart(fig, (width - 8) / 3, 102))
        result = Table([charts], colWidths=[width / 3] * 3)
        result.setStyle(TableStyle([('LEFTPADDING', (0, 0), (-1, -1), 0),
                                   ('RIGHTPADDING', (0, 0), (-1, -1), 0)]))
        return result

    def sample_block(evidence, sample):
        identifier = sample['id']
        measured = next(row for row in evidence.analysis['samples'] if row['id'] == identifier)
        wav, prediction = measured['wav'], measured['prediction']
        variance = prediction['predicted_variance']
        note = observations['samples'][identifier]
        block = [p(f"{identifier} | {evidence.language} | {note['assessment_mode'].replace('_', ' ')}", "Subsection")]
        if evidence.language == "English":
            block.append(p('Source text: "' + sample['text'] + '"', "SmallReport"))
        else:
            block.append(p(f"Exact Telugu source text: Telugu/test_samples.json, entry {identifier}. "
                           "The unchanged UTF-8 source is authoritative; complex-script shaping was not "
                           "validated in this Roman-script PDF.", "SmallReport"))
        pause = prediction['max_pause']
        block.append(p(f"WAV {numeric(wav['duration_seconds'])} s; RMS {numeric(wav['rms_dbfs'], 2)} dBFS; "
                       f"peak {numeric(wav['peak'])}; PCM rail fraction {numeric(wav['clipping_fraction_at_pcm_rails'], 6)}. "
                       f"Non-SIL zero-duration phones: {prediction['zero_duration_non_sil_phone_count']}/"
                       f"{prediction['non_sil_phone_count']}; longest non-SIL duration "
                       f"{numeric(prediction['max_phone']['duration_seconds'])} s at phone index "
                       f"{prediction['max_phone']['phone_index']}; longest SIL run "
                       f"{numeric(pause['duration_seconds'] if pause else 0)} s.", "SmallReport"))
        block.append(sample_charts(evidence.language, identifier))
        recipe = evidence.inference_details[identifier]
        if recipe['zero_sil_energy_applied']:
            block.append(p("Energy: solid line is the unchanged raw head; dashed line is the actual decoder "
                           "conditioning after setting SIL energy to zero. This is an inference control, "
                           "not a change to training losses or raw predictions.", "SmallReport"))
        block.append(p(f"Non-SIL predicted F0 mean {numeric(variance['pitch_hz_non_sil']['mean'], 1)} Hz; "
                       f"range {numeric(variance['pitch_hz_non_sil']['min'], 1)}-"
                       f"{numeric(variance['pitch_hz_non_sil']['max'], 1)} Hz. Non-SIL predicted energy mean "
                       f"{numeric(variance['energy_non_sil']['mean'])}. Waveform F0 status: "
                       f"{measured['acoustic_f0']['status']}.", "SmallReport"))
        review = note['objective_review']
        for key in ('duration', 'pitch', 'energy', 'limitations'):
            block.append(p(f"{key.capitalize()}: {review[key]}", "SmallReport"))
        if note['assessment_mode'] == 'objective_only':
            block.append(p("Listening assessment: not performed. Pronunciation correctness, intelligibility, "
                           "naturalness, and perceived artifacts are unassessed; tensor and WAV statistics "
                           "do not establish these properties.", "SmallReport"))
        else:
            listening = note['listening']
            block.append(p(f"Listening by {listening['reviewer']} ({listening['reviewed_at']}): "
                           f"pronunciation - {listening['pronunciation']}; intelligibility - {listening['intelligibility']}; "
                           f"naturalness - {listening['naturalness']}; artifacts - {listening['artifacts']}.", "SmallReport"))
        block.append(p(f"Artifacts: Results/{evidence.language}/test_generated_audios/{identifier}.wav; "
                       f"predictions/{identifier}.npz; variance_adaptor_plots/{{duration,pitch,energy}}/{identifier}.png.",
                       "SmallReport"))
        return KeepTogether(block)

    heading("FastSpeech 2\nEnglish and Telugu")
    add("Technical report | Separate assignment-trained acoustic models", "ReportSubtitle")
    add("Evidence cut-off: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    add(observations['scope_summary'])
    add("Evidence boundary", "Subsection")
    add("This report is generated from production split manifests, saved training and validation logs, "
        "the selected checkpoints used for inference, and objective measurements of the ten final WAV files. "
        "The builder verifies current file hashes before rendering. It does not substitute development "
        "smoke runs or benchmark checkpoints for production evidence.")
    rows = [["Model", "Train / validation", "Latest / selected step", "Final samples"]]
    for name, data in languages.items():
        rows.append([name, f"{data.audit['train_seconds']/3600:.4f} h / {data.audit['validation_seconds']/3600:.4f} h",
                     f"{data.latest_step:,} / {data.selected_step:,}", ", ".join(LANGUAGES[name])])
    story.append(table(rows, [62, 145, 130, width - 337]))
    modes = Counter(value['assessment_mode'] for value in observations['samples'].values())
    add(f"Assessment coverage: {modes['listening']} WAVs with explicitly recorded listening; "
        f"{modes['objective_only']} objective-only reviews. Objective-only entries leave perceived speech quality "
        "and pronunciation unassessed. No MOS, intelligibility score, or native-speaker verdict is inferred.")
    add("Deliverables checked", "Subsection")
    add("Two separate acoustic checkpoints; ten generated WAVs; thirty duration/F0/energy plots; four loss plots; "
        "ten raw prediction archives; two current analysis.json files; exact source sentence JSON files; "
        "and this report. All reported timing and signal values are measurements of those artifacts.")
    add("Reading guide", "Subsection")
    add("Architecture and adaptations; data, splits and hyperparameters; production loss curves; individual "
        "sample reviews; cross-language comparison and limitations; reproduction; source and license records.")

    heading("Architecture and reference adaptations")
    add(f"Reference: ming024/FastSpeech2 at commit {REFERENCE_COMMIT} [1]. The vendored encoder, variance "
        "adaptor, length regulator, decoder, postnet and reference loss are retained. The acoustic models "
        "start without pretrained TTS acoustic weights. The separate pretrained universal HiFi-GAN decodes "
        "predicted mel spectrograms to waveforms [2,3].")
    pipeline = [["Stage", "Implemented behavior"],
                ["Text front end", "NFC and whitespace normalization; eSpeak NG IPA tokens; English AM/PM time-suffix expansion; original assignment text retained."],
                ["Encoder", "Learned IPA embedding and positional encoding, followed by FFT blocks. Independent language vocabularies and weights."],
                ["Variance adaptor", "Phone-level log-duration, pitch and energy predictors; pitch/energy embeddings; duration-based length regulation."],
                ["Decoder / postnet", "FFT decoder predicts 80-bin mel frames; postnet adds a mel residual. Losses: mel, postnet mel, duration, pitch, energy."],
                ["Waveform synthesis", "Bundled pretrained universal HiFi-GAN at 22,050 Hz. Its weights are not an assignment-trained acoustic model."]]
    story.append(table(pipeline, [95, width - 95]))
    add("Project adaptations", "Subsection")
    add("tts/modeling.py replaces the reference symbol embedding with each prepared IPA vocabulary and sets "
        "device-aware helper allocations. Shared modules expose separate English and Telugu entry points, "
        "configurations, statistics, speaker maps, manifests, checkpoints and outputs. The training loop "
        "adds split-integrity gates, checkpoint contract hashes, reproducible batch ordering, interruption "
        "resume, gradient clipping and separate validation. Validation averages per-utterance reference "
        "losses; training uses the reference masked batch reduction.")
    add("Language-specific alignment and conditioning", "Subsection")
    add("English uses CUPE-3-en alignment. Svarah's available export lacks verified speaker IDs, so hashed "
        "demographic profiles condition the multi-speaker model. These are proxy groups, not verified people. "
        "Telugu uses one female voice and a single-speaker acoustic model. CUPE-3-mls7 has no Telugu "
        "training; its alignment is zero-shot. Telugu-only acoustic mappings approximate /palatalization/ "
        "and two aspirated consonants while preserving original IPA labels as model inputs. The acoustic "
        "mapping collapses some Telugu distinctions and does not validate their pronunciation [6-8].")
    add("Front-end and inference controls", "Subsection")
    english_modes = Counter(str(row['frontend'].get('used', 'legacy word-level'))
                            for row in languages['English'].inference_details.values())
    sil_overrides = {name: sum(row['zero_sil_energy_applied'] for row in evidence.inference_details.values())
                     for name, evidence in languages.items()}
    add("Training used the original word-level English front end. The sentence-context inference option "
        "can change the 'conduct' vowel by context; lexical stress is not encoded, so it is not a complete "
        "stress or homograph-quality solution. Requested/used modes and fallback status are recorded in "
        "each inference manifest. The visible 'rhyth m', 'andaroo', and Telugu internal-space challenge "
        "remain in the exact source text.")
    add(f"Recorded English modes: {dict(english_modes)}. Zero-SIL-energy conditioning was applied to "
        f"{sil_overrides['English']}/5 English and {sil_overrides['Telugu']}/5 Telugu samples. When enabled, "
        "only the energy values fed to the decoder are changed at SIL positions; raw energy predictions "
        "and checkpoint training/validation metrics remain unchanged.", "SmallReport")

    heading("Data preparation, splits and hyperparameters")
    add("Svarah [4] is repurposed from its original ASR test partition into new TTS train/validation subsets; "
        "this is not the standard benchmark evaluation. The imported files measure 9.6144 h at 16 kHz. "
        "IndicTTS Telugu [5] uses the female subset of the first two training shards; the 1-30 s candidate "
        "pool measures 13.3541 h at 16 kHz. The Telugu card's 48 kHz/8.74 h summary contradicts the "
        "downloaded files. Candidate pools are distinct from the accepted splits below.")
    rows = [["Measured preparation", "English", "Telugu"]]
    for label, getter in (
            ("Train utterances / hours", lambda d: f"{d.audit['train_examples']} / {d.audit['train_seconds']/3600:.5f}"),
            ("Validation utterances / hours", lambda d: f"{d.audit['validation_examples']} / {d.audit['validation_seconds']/3600:.5f}"),
            ("Accepted / rejected candidates", lambda d: f"{d.preparation['accepted']} / {d.preparation['rejected']}"),
            ("Candidate prefix inspected", lambda d: f"{d.preparation.get('candidate_scan', {}).get('processed_rows', 'unrecorded')} rows"),
            ("Train / validation conditioning IDs", lambda d: f"{d.audit['train_speakers']} / {d.audit['validation_speakers']}"),
            ("Vocabulary incl. padding / seed", lambda d: f"{d.vocabulary_size} / {d.preparation['seed']}"),
            ("QA max phone s / min confidence", lambda d: f"{d.preparation['acoustic_qa']['maximum_speech_phone_seconds']} / {d.preparation['acoustic_qa']['minimum_mean_speech_confidence']}")):
        rows.append([label, getter(languages['English']), getter(languages['Telugu'])])
    story.append(table(rows, [width * .42, width * .29, width * .29]))
    add("Eligible 1-30 s source rows are sorted by ID and shuffled with seed 20261105. English takes "
        "whole recordings through the first 7.5 h; Telugu retains the shuffled female pool. The accepted "
        "scan stops at 6.5 h or candidate exhaustion. This is followed by seeded allocation of whole connected groups "
        "sharing normalized text or audio hashes. Utterances are not arbitrarily cropped. Train/validation "
        "IDs, audio, normalized text and feature identities are disjoint. Validation conditioning IDs must "
        "occur in training. Pitch and energy statistics are fitted only on training targets. Vocabulary "
        "coverage includes the prescribed test text, but no test recording or acoustic target.", "SmallReport")
    rows = [["Actual checkpoint setting", "English", "Telugu"]]
    getters = [("FFT encoder / decoder / hidden / heads", lambda c: f"{c['model']['transformer']['encoder_layer']} / {c['model']['transformer']['decoder_layer']} / {c['model']['transformer']['encoder_hidden']} / {c['model']['transformer']['encoder_head']}"),
               ("FFT filter / kernels / dropout", lambda c: f"{c['model']['transformer']['conv_filter_size']} / {c['model']['transformer']['conv_kernel_size']} / {c['model']['transformer']['encoder_dropout']}"),
               ("Variance filter / kernel / dropout", lambda c: f"{c['model']['variance_predictor']['filter_size']} / {c['model']['variance_predictor']['kernel_size']} / {c['model']['variance_predictor']['dropout']}"),
               ("Pitch / energy bins and quantization", lambda c: f"{c['model']['variance_embedding']['n_bins']}; {c['model']['variance_embedding']['pitch_quantization']} / {c['model']['variance_embedding']['energy_quantization']}"),
               ("Batch / precision / length bucket", lambda c: f"{c['training']['batch_size']} / {c['training'].get('precision', 'float32')} / {c['training'].get('length_bucket_size', 0)}"),
               ("Peak learning rate / warmup steps", lambda c: f"{c['training'].get('peak_learning_rate', c['training'].get('learning_rate', 1e-4))} / {c['training'].get('warmup_steps', 0)}"),
               ("Gradient clip / weight decay", lambda c: f"{c['training'].get('grad_clip', 1)} / {c['training'].get('weight_decay', 0)}"),
               ("Validation interval / saved config target", lambda c: f"{c['training']['validate_every']} / {c['training']['steps']}"),
               ("Multispeaker / maximum frames", lambda c: f"{c['model']['multi_speaker']} / {c['model']['max_seq_len']}")]
    for label, getter in getters:
        rows.append([label, getter(languages['English'].config), getter(languages['Telugu'].config)])
    story.append(table(rows, [width * .42, width * .29, width * .29]))
    add("AdamW uses betas (0.9, 0.98), epsilon 1e-9. If warmup is enabled, learning rate rises linearly then "
        "falls with inverse square-root step. Features: resample to 22,050 Hz; 1,024-sample FFT/window; "
        "256-sample hop; 80 natural-log magnitude Slaney mel bins, 0-8,000 Hz. WORLD DIO/StoneMask F0 is "
        "interpolated through unvoiced regions before phone averaging. Energy is the linear STFT magnitude "
        "norm, then phone averaged. Acoustic QA rejects backend flags, speech-phone spans above the recorded "
        "threshold and low mean confidence; these heuristic gates are not human alignment verification.", "SmallReport")

    heading("Training loss and held-out validation")
    for name in LANGUAGES:
        d = languages[name]
        story.append(Table([[loss_chart(d, 'training'), loss_chart(d, 'validation')]],
                           colWidths=[width/2, width/2], hAlign="LEFT"))
    add("Curves are redrawn from the complete production logs in checkpoint/logs; full-size original plots "
        "remain in Results/<language>/{training_loss,validation_loss}.png. Loss values are dimensionless "
        "criterion values, not a speech-quality scale. Training and validation aggregation differ.", "SmallReport")
    rows = [["Model", "First / last train total", "First / last validation total", "Minimum validation / step"]]
    for name, d in languages.items():
        best = min(d.validation, key=lambda row: row['total'])
        rows.append([name, f"{numeric(d.training[0]['total'])} / {numeric(d.training[-1]['total'])}",
                     f"{numeric(d.validation[0]['total'])} / {numeric(d.validation[-1]['total'])}",
                     f"{numeric(best['total'])} / {best['step']:,}"])
    story.append(table(rows, [57, 147, 157, width - 361]))
    for name in LANGUAGES:
        note = observations['language_assessments'][name]
        add(f"{name}: {note['training_interpretation']} Instability/overfitting: {note['instability_or_overfitting']}",
            "SmallReport")
    add("An endpoint reduction or validation minimum alone does not establish convergence. Quality, "
        "undertraining and overfitting conclusions remain bounded by the recorded trajectory and sample "
        "assessment coverage. best.pt is always retained as the minimum-total-validation-loss baseline. "
        "An explicit selection.json may designate selected.pt for inference; otherwise inference uses best.pt.",
        "SmallReport")
    for name, d in languages.items():
        add(f"{name}: inference {d.checkpoint_file} step {d.selected_step:,}; minimum-total best.pt step "
            f"{d.best_step:,}; latest.pt / final logged step {d.latest_step:,}; "
            f"last training invocation target {d.progress['total_target_steps']:,}, completed optimizer updates "
            f"during that invocation {d.run['steps']:,}. A saved configuration target is not a claim that "
            "all of its planned steps were run.", "SmallReport")

    if any(data.selection is not None for data in languages.values()):
        heading("Checkpoint selection and validation probes")
        add("Earlier ASR diagnostics on the assignment prompts motivated this review. The subsequent "
            "original frozen comparison used three held-out validation prompts and two supplied checkpoints "
            "per language. Further English inference/checkpoint experiments are explicitly post-hoc; "
            "their outcome is not covered by the original two-candidate protocol. None is a pristine blind "
            "final test, a final-test quality score, or a human listening assessment.")
        for name, d in languages.items():
            if d.selection is None:
                add(f"{name}: no explicit selection declared; inference retains best.pt at step {d.best_step:,}.")
                continue
            selected = d.selection
            if name == 'Telugu':
                heading("Checkpoint selection | Telugu")
            else:
                add(name, "Subsection")
            add(f"Checkpoint: {d.checkpoint_file}, step {d.selected_step:,}. {selected['rationale']}", "SmallReport")
            add(f"Protocol: {selected['protocol_note']}", "SmallReport")
            if selected.get('quality_gate_passed') is False:
                add("Frozen comparison outcome: inconclusive; no validated-quality winner was established.", "SmallReport")
            if selected['decision'] == 'later_best_available_failed_quality_gate':
                add("Explicit protocol deviation: the later checkpoint is the pragmatic best-available "
                    "artifact despite the failed quality gate. Its use does not turn the frozen "
                    "comparison into a successful quality selection.", "SmallReport")
            if selected['decision'] == 'posthoc_validation_improvement':
                add("Post-hoc preference: both the checkpoint and inference recipe may differ from the "
                    "earlier experiment. Any measured improvement cannot be attributed to additional "
                    "training alone. The failed duration gate remains a failure, and raw full-validation "
                    "losses are not recomputed or replaced by these synthesis diagnostics.", "SmallReport")
            if selected.get('protocol_deviation'):
                add(f"Recorded deviation: {selected['protocol_deviation']}", "SmallReport")
            rows = [["Reported comparison", "WER", "CER", "Generated/source duration", "Screening checks"]]
            for candidate in selected['candidates']:
                ratio_gate = candidate['duration_ratio_min'] >= .5 and candidate['duration_ratio_max'] <= 2.0
                rows.append([f"{candidate['label'].replace('_', ' ')}\nstep {candidate['step']:,}",
                             numeric(candidate['wer']), numeric(candidate['cer']),
                             f"{numeric(candidate['duration_ratio_min'])}-{numeric(candidate['duration_ratio_max'])}",
                             f"Audio integrity: {candidate['valid_audio']}; duration gate: {'pass' if ratio_gate else 'FAIL'}"])
            control = selected['source_controls']
            rows.append(["Source-recording ASR control", numeric(control['wer']), numeric(control['cer']),
                         "Reference audio", "Control"])
            story.append(table(rows, [145, 54, 54, 155, width - 408]))
            add("WER and CER above are proportions, micro-averaged across only these three validation "
                "prompts; duration ratios report their minimum and maximum. The predeclared duration "
                "gate requires every ratio in [0.5, 2.0]. Audio integrity is a separate flag and does not "
                "override a failed duration gate or certify speech quality. Source-recording controls "
                "help reveal recognizer errors and text-normalization effects. None of these values "
                "establish pronunciation correctness or perceived speech quality.", "SmallReport")
            for probe in d.selection_probes:
                if name == 'English':
                    add(f"{probe['id']}: {probe['text']}", "SmallReport")
                else:
                    add(f"{probe['id']}: exact Telugu source in provenance/{name}_validation_probes.json.",
                        "SmallReport")
            add(f"Decision: {selected['decision'].replace('_', ' ')}. Full record: "
                f"{name}/checkpoints/selection.json. Frozen prompt SHA256: "
                f"{selected['validation_probe_manifest_sha256']}", "SmallReport")
            if d.selection_history is not None:
                old = d.selection_history
                add(f"Preserved original decision: {name}/checkpoints/selection_v1.json selected step "
                    f"{old.get('selected_step')} with decision {str(old.get('decision')).replace('_', ' ')}. "
                    "Its original metrics and failed/inconclusive gate outcome remain historical evidence; "
                    "the new synthesis recipe does not rewrite them.", "SmallReport")
            if d.selection_history_v2 is not None:
                old = d.selection_history_v2
                add(f"The later experimental recipe is also preserved in {name}/checkpoints/selection_v2.json "
                    f"(step {old.get('selected_step')}). Its metrics and decision are historical; they do not "
                    "replace the final recipe or turn rejected conditioning changes into improvements.", "SmallReport")

    all_samples = [(languages[name], sample) for name in LANGUAGES for sample in languages[name].samples]
    for index in range(0, len(all_samples), 2):
        heading(f"Individual samples | {all_samples[index][1]['id']} and {all_samples[index+1][1]['id']}")
        for evidence, sample in all_samples[index:index+2]:
            story.append(sample_block(evidence, sample))
            story.append(Spacer(1, 8))
        add("Plots use the actual phone-level predictions, indexed from zero; each includes SIL entries. "
            "Full phoneme labels are available in the corresponding PNG and .npz. Predicted F0 and energy "
            "are denormalized with training statistics. The energy panel retains the raw prediction and "
            "adds a dashed decoder-conditioning overlay only when the saved SIL control was active. "
            "These are not waveform F0 or RMS.", "SmallReport")

    heading("Variance behavior, comparison and limitations")
    add(observations['cross_language_comparison'])
    rows = [["Across five requested samples", "English", "Telugu"]]
    for label, getter in (
            ("Mean generated duration (s)", lambda d: sum(row['wav']['duration_seconds'] for row in d.analysis['samples'])/5),
            ("Non-SIL phones rounded to zero (count)", lambda d: sum(row['prediction']['zero_duration_non_sil_phone_count'] for row in d.analysis['samples'])),
            ("Longest predicted non-SIL phone (s)", lambda d: max(row['prediction']['max_phone']['duration_seconds'] for row in d.analysis['samples'])),
            ("Maximum PCM rail fraction", lambda d: max(row['wav']['clipping_fraction_at_pcm_rails'] for row in d.analysis['samples']))):
        rows.append([label, numeric(getter(languages['English']), 4), numeric(getter(languages['Telugu']), 4)])
    story.append(table(rows, [width * .5, width * .25, width * .25]))
    add("Interpretation limits", "Subsection")
    add("The sample sets contain different text and only five sentences per language. Cross-language "
        "duration and pitch differences are descriptive, not a controlled language comparison. There is "
        "no reference audio for these sentences and no reference-based pronunciation, F0 or duration error. "
        "Zero rounded phone durations can omit material, but audibility needs listening. Predicted SIL "
        "runs are model decisions, not independently detected waveform silence. PCM rail hits indicate "
        "full-scale samples; they cannot recover or prove earlier clipping.")
    add("Dataset and supervision limits", "Subsection")
    for name in LANGUAGES:
        add(f"{name}: {observations['language_assessments'][name]['limitations']}")
    add("Svarah conditioning groups can merge voices and are not a verified speaker inventory. Telugu "
        "alignment is zero-shot and its acoustic phone mappings collapse distinctions. Confidence and "
        "maximum-phone-length filters can reject correct speech or retain incorrect boundaries. WORLD "
        "F0 estimates can be wrong; interpolated training pitch is not a voicing target. Native-listener "
        "review, improved speaker labels and a Telugu-trained aligner remain separate evidence needs.")
    add("Observed failures", "Subsection")
    add(observations['failure_cases'])
    add("Next improvements", "Subsection")
    add(observations['next_steps'])

    heading("Reproduction and software environment")
    add("Run from the project root with Python 3.11. Use an isolated environment and the pinned requirements. "
        "Select the official PyTorch build for the execution host. Obtain gated data using your own "
        "authorized account; do not put credentials or raw corpus audio into the public repository.")
    commands = ["python3.11 -m venv .venv", ". .venv/bin/activate",
                "python -m pip install -r requirements.txt", "hf auth login",
                "python scripts/download_data.py --help", "python scripts/import_parquet.py --help",
                "python scripts/select_candidates.py --language en --manifest data/en/metadata.jsonl --output data/en/candidates.jsonl",
                "python scripts/select_candidates.py --language te --manifest data/te/metadata.jsonl --output data/te/candidates.jsonl",
                "python English/preprocess.py --config English/config.yaml --manifest data/en/candidates.jsonl --device mps --accepted-hours 6.5",
                "python Telugu/preprocess.py --config Telugu/config.yaml --manifest data/te/candidates.jsonl --device mps --accepted-hours 6.5",
                "python English/train.py --config English/config.yaml",
                "python Telugu/train.py --config Telugu/config.yaml",
                "python English/train.py --config English/config.yaml --resume English/checkpoints/latest.pt",
                f"python English/inference.py --config English/config.yaml --checkpoint English/checkpoints/{languages['English'].checkpoint_file} --samples English/test_samples.json --speaker-id <recorded-id>",
                f"python Telugu/inference.py --config Telugu/config.yaml --checkpoint Telugu/checkpoints/{languages['Telugu'].checkpoint_file} --samples Telugu/test_samples.json --speaker-id 0",
                "python scripts/analyze_results.py", "python scripts/build_report.py --check-only",
                "python scripts/build_report.py", "python scripts/validate_submission.py"]
    for command in commands:
        add(command, "CodeReport")
    selected_ids = {}
    for name in LANGUAGES:
        manifest = read_json(root / "Results" / name / "inference_manifest.json")
        selected_ids[name] = sorted({row['speaker_id'] for row in manifest})
    add(f"Recorded inference conditioning IDs: English {selected_ids['English']}; Telugu {selected_ids['Telugu']}. "
        "Replace <recorded-id> with the recorded English ID. These are prepared dense IDs. The configuration "
        "tables describe checkpoint settings; if a resumed run used a different host or a changed target step "
        "count, keep its saved evidence and logs. Use --device cuda or cpu in preprocessing where appropriate.",
        "SmallReport")
    rows = [["Recorded environment", "English", "Telugu"]]
    for key in ('device', 'torch_version', 'precision', 'parameters', 'batch_size', 'median_step_seconds'):
        rows.append([key, str(languages['English'].run.get(key, 'not recorded')),
                     str(languages['Telugu'].run.get(key, 'not recorded'))])
    story.append(table(rows, [width * .4, width * .3, width * .3]))
    add("requirements-lock-macos.txt and preparation_summary.json/extraction_versions record the macOS "
        "feature-extraction environment. They do not describe the separate CUDA training host. The table "
        "above uses the actual final training invocation's run_summary.json; saved checkpoint configuration "
        "supplies the optimizer settings. Package versions absent from those records are not inferred from "
        "the local lock. Retain a verified cloud package inventory with the run artifacts for exact "
        "environment reconstruction. Cross-device bitwise equality is not promised. Report rendering "
        f"uses reportlab {reportlab.Version}; waveform analysis runs in the analysis environment.", "SmallReport")

    heading("Sources, licenses and artifact traceability")
    references = [
        ("1", "ming024/FastSpeech2, MIT reference implementation, pinned commit " + REFERENCE_COMMIT,
         "https://github.com/ming024/FastSpeech2/tree/" + REFERENCE_COMMIT),
        ("2", "Ren et al. FastSpeech 2: Fast and High-Quality End-to-End Text to Speech. ICLR 2021.", "https://arxiv.org/abs/2006.04558"),
        ("3", "Kong et al. HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis. NeurIPS 2020. MIT bundled code.", "https://arxiv.org/abs/2010.05646"),
        ("4", "Javed et al. Svarah: Evaluating English ASR Systems on Indian Accents. INTERSPEECH 2023, pp. 5087-5091. Dataset: CC BY 4.0.", "https://huggingface.co/datasets/ai4bharat/Svarah"),
        ("5", "IIT Madras IndicTTS / SPRINGLab Telugu export. Custom IITM EULA, not generically MIT or CC BY.", "https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu"),
        ("6", "Bournemouth Forced Aligner 2.0.1. GPLv3 package.", "https://github.com/tabahi/bournemouth-forced-aligner"),
        ("7", "CUPE acoustic alignment models. Preserve each model's declared license, including AGPL-3.0 for CUPE-3-mls7.", "https://huggingface.co/Tabahi/CUPE-3-mls7"),
        ("8", "eSpeak NG and Phonemizer pronunciation dependencies. GPL-family licensing.", "https://github.com/espeak-ng/espeak-ng"),
        ("9", "Official IITM IndicTTS EULA URL; official server did not yield the document during the initial audit.", "https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf"),
        ("10", "Reviewed mirror of the primary-authored IITM EULA. Byte identity with the current official file is unverified.", "https://huggingface.co/k-m-irfan/Fastspeech2_HS_Flask_API/blob/main/license.pdf")]
    for number, label, url in references:
        add(f"[{number}] {label}", "SmallReport")
        story.append(Paragraph(f'<link href="{escape(url)}" color="#216A8E">{escape(url)}</link>', styles['SmallReport']))
    add("Required corpus notice", "Subsection")
    add("COPYRIGHT 2016 TTS Consortium, TDIL, Meity represented by Hema A Murthy & S Umesh, DEPARTMENT OF "
        "Computer Science and Engineering and Electrical Engineering, IIT Madras. ALL RIGHTS RESERVED. "
        "Acknowledge the Speech Technology Consortium, IIT Madras, Bhashini/MeitY, Prof. Hema A Murthy and "
        "Prof. S Umesh. Retain the corpus EULA and downstream restrictions. See THIRD_PARTY_NOTICES.md "
        "for the complete project notice record and dependency license distinctions.", "SmallReport")
    add("Hash-bound evidence", "Subsection")
    for name, data in languages.items():
        add(f"{name} inference {data.checkpoint_file} SHA256: {data.checkpoint_hash}", "SmallReport")
        if data.selection is not None:
            add(f"{name} minimum-total best.pt SHA256: {data.best_hash}", "SmallReport")
        add(f"{name} exact test_samples.json SHA256: {data.analysis['source']['test_samples_sha256']}", "SmallReport")
        add(f"{name} dataset revision: {sources['datasets']['en' if name == 'English' else 'te']['revision']}", "SmallReport")
    add("Each Results/<language>/analysis.json binds the checkpoint, configuration, exact test text, "
        "inference manifest, individual WAVs and raw predictions to hashes. The builder remeasures the "
        "stored WAV/prediction statistics and checks current processed-target fingerprints against the "
        "production split audit. Rejected cases and threshold settings are preserved in each preparation "
        "folder. No private source paths, gated recordings, credentials or assignment PDF are embedded here.", "SmallReport")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#D0DCE2'))
        canvas.line(margin, 37, page_width - margin, 37)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor('#456579'))
        canvas.drawString(margin, 25, "FastSpeech 2 | English and Telugu | Evidence-bound technical report")
        canvas.drawRightString(page_width - margin, 25, str(document.page))
        if document.page > 1:
            canvas.setFont("Helvetica", 7)
            canvas.drawString(margin, page_height - 28, "Separate acoustic models - actual production artifacts")
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".pending")
    try:
        doc = BaseDocTemplate(str(temporary), pagesize=A4, leftMargin=margin, rightMargin=margin,
                              topMargin=45, bottomMargin=48, title="FastSpeech 2: English and Telugu",
                              author="Assignment project", subject="Production training and inference evidence")
        frame = Frame(margin, 48, width, page_height - 93, leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        doc.addPageTemplates(PageTemplate(id='report', frames=frame, onPage=footer))
        doc.build(story)
        require(temporary.is_file() and temporary.stat().st_size > 1000, "Renderer produced an empty PDF")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--observations", type=Path, help="Defaults to <root>/Report/report_observations.json")
    parser.add_argument("--check-only", action="store_true", help="Validate all evidence; do not create any PDF")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    observations = (args.observations or root / "Report/report_observations.json").resolve()
    try:
        # Empty rejection logs are valid, unlike empty metric logs or manifests.
        # required_paths includes them to require existence, handled separately.
        missing = []
        for path in required_paths(root, observations):
            if not path.is_file() or (path.stat().st_size == 0 and path.name != 'rejected.jsonl'):
                missing.append(str(path.relative_to(root)) if path.is_relative_to(root) else str(path))
        require(not missing, "Final report prerequisites are missing or empty:\n  " + "\n  ".join(missing))
        languages, sources, notes = collect_evidence(root, observations)
        if args.check_only:
            print(json.dumps({"ready": True, "pdf_created": False, "samples": 10,
                              "languages": {name: {"selected_checkpoint_step": data.selected_step,
                                                    "selected_checkpoint_file": data.checkpoint_file,
                                                    "minimum_total_checkpoint_step": data.best_step,
                                                    "last_optimizer_step": data.latest_step}
                                            for name, data in languages.items()}}, indent=2))
            return 0
        output = root / "Report/Technical_Report.pdf"
        build_pdf(root, languages, sources, notes, output)
        print(json.dumps({"pdf_created": True, "output": str(output), "sha256": sha256(output),
                          "visual_review_required": True,
                          "note": "Render every page and inspect it before treating the report as final."}, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, ImportError, OSError) as exc:
        print(json.dumps({"ready": False, "pdf_created": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
