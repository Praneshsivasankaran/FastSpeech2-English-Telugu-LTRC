#!/usr/bin/env python3
"""Optional local ASR comparison of the exact, hash-verified assignment WAVs.

Install requirements-asr.txt separately, then run e.g.:
  python scripts/transcribe_results.py --language all --check-only
  python scripts/transcribe_results.py --language all --allow-model-download

The default is cached/local models only. The download flag permits fetching a
public model from Hugging Face; it never uploads audio or uses an inference API.
No report is written until every requested WAV has been validated and decoded.
WER/CER are recognizer comparisons, not proof of TTS pronunciation correctness.

API references:
https://github.com/SYSTRAN/faster-whisper/tree/v1.2.1
https://huggingface.co/docs/huggingface_hub/package_reference/file_download
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import io
import json
import math
from pathlib import Path
import platform
import re
import sys
import tempfile
import unicodedata
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {"en": ("English", "E"), "te": ("Telugu", "T")}
# Public aliases from faster-whisper's model registry. Explicit Hub IDs and
# local CTranslate2 model directories are also accepted.
MODEL_ALIASES = {
    **{name: f"Systran/faster-whisper-{name}" for name in (
        "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
        "large-v1", "large-v2", "large-v3")},
    "large": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    **{name: f"Systran/faster-{name}" for name in (
        "distil-large-v2", "distil-medium.en", "distil-small.en", "distil-large-v3")},
}
CAVEAT = (
    "These are ASR hypotheses and text-comparison error rates, not ground-truth "
    "transcripts or a pronunciation score. Errors can arise from the recognizer, "
    "the synthesized speech, or text conventions. A low WER/CER does not establish "
    "correct stress, homograph sense, pronunciation, naturalness, or timing. "
    "A high WER/CER does not by itself prove a TTS error. Human listening is required."
)
NORMALIZATION = (
    "Unicode NFC; casefold; remove apostrophes and zero-width joiners; replace "
    "other punctuation/symbols with spaces; collapse whitespace; preserve Unicode "
    "letters, combining marks and digits. No spelling repairs, number/date/time "
    "expansion, transliteration, or correction of the deliberate input anomalies. "
    "WER uses whitespace-separated words; CER uses Unicode code points excluding "
    "whitespace (not grapheme clusters, so Telugu vowel signs count separately)."
)


class EvaluationError(ValueError):
    """Missing, stale, or invalid input; no evaluation should be published."""


def require(condition, message):
    if not condition:
        raise EvaluationError(message)


def digest_bytes(content):
    return hashlib.sha256(content).hexdigest()


def file_digest(path):
    require(Path(path).is_file(), f"Missing required file: {path}")
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    require(Path(path).is_file(), f"Missing required file: {path}")
    content = Path(path).read_bytes()
    try:
        return json.loads(content.decode("utf-8")), digest_bytes(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Invalid JSON: {path}: {exc}") from exc


def normalize_text(text):
    text = unicodedata.normalize("NFC", text.casefold())
    normalized = []
    for character in text:
        if character in "'\u2019\u02bc\u200c\u200d":
            continue
        normalized.append(character if unicodedata.category(character)[0] in "LMN" else " ")
    return " ".join("".join(normalized).split())


def edit_distance(reference, hypothesis):
    """Unit-cost Levenshtein distance for words or Unicode code points."""
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, observed in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (expected != observed)))
        previous = current
    return previous[-1]


def error_metrics(reference, hypothesis):
    ref_normalized, hyp_normalized = normalize_text(reference), normalize_text(hypothesis)
    ref_words, hyp_words = ref_normalized.split(), hyp_normalized.split()
    ref_chars, hyp_chars = ref_normalized.replace(" ", ""), hyp_normalized.replace(" ", "")
    word_errors = edit_distance(ref_words, hyp_words)
    char_errors = edit_distance(ref_chars, hyp_chars)
    return {"reference_normalized": ref_normalized, "hypothesis_normalized": hyp_normalized,
            "word_edit_distance": word_errors, "reference_word_count": len(ref_words),
            "hypothesis_word_count": len(hyp_words),
            "wer": word_errors / len(ref_words) if ref_words else None,
            "character_edit_distance": char_errors, "reference_character_count": len(ref_chars),
            "hypothesis_character_count": len(hyp_chars),
            "cer": char_errors / len(ref_chars) if ref_chars else None}


def indexed_rows(rows, expected_ids, name):
    require(isinstance(rows, list), f"{name}: expected a JSON list")
    require(all(isinstance(row, dict) for row in rows), f"{name}: invalid row")
    ids = [row.get("id") for row in rows]
    require(all(isinstance(identifier, str) for identifier in ids), f"{name}: missing ID")
    require(len(ids) == len(set(ids)), f"{name}: duplicate IDs")
    require(set(ids) == expected_ids, f"{name}: requires exactly {sorted(expected_ids)}")
    return {row["id"]: row for row in rows}


def load_inputs(root, language):
    """Read and bind reference text, manifest and audio bytes before model loading."""
    name, prefix = LANGUAGES[language]
    expected_ids = {prefix + suffix for suffix in ("1", "2a", "2b", "3", "4")}
    sample_path = root / name / "test_samples.json"
    result_dir = root / "Results" / name
    manifest_path = result_dir / "inference_manifest.json"
    samples, sample_hash = read_json(sample_path)
    manifest, manifest_hash = read_json(manifest_path)
    sample_rows = indexed_rows(samples, expected_ids, str(sample_path))
    manifest_rows = indexed_rows(manifest, expected_ids, str(manifest_path))
    checkpoint_hashes = set()
    records = []
    for identifier in sorted(expected_ids):
        sample, row = sample_rows[identifier], manifest_rows[identifier]
        reference = sample.get("text")
        require(isinstance(reference, str) and bool(normalize_text(reference)),
                f"{identifier}: empty or invalid reference text")
        require(sample.get("language") == language, f"{identifier}: wrong reference language")
        require(row.get("text") == reference, f"{identifier}: manifest text differs from exact test sample")
        checkpoint_hash = row.get("checkpoint_sha256")
        require(isinstance(checkpoint_hash, str) and bool(re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash)),
                f"{identifier}: missing/invalid checkpoint SHA-256")
        require(isinstance(row.get("checkpoint_step"), int) and row["checkpoint_step"] > 0,
                f"{identifier}: no trained checkpoint step in manifest")
        checkpoint_hashes.add(checkpoint_hash)
        audio_path = result_dir / "test_generated_audios" / f"{identifier}.wav"
        require(audio_path.is_file(), f"Missing required WAV: {audio_path}")
        content = audio_path.read_bytes()
        actual_hash = digest_bytes(content)
        audio_manifest = row.get("audio", {})
        require(isinstance(audio_manifest, dict) and audio_manifest.get("sha256") == actual_hash,
                f"{identifier}: WAV SHA-256 differs from inference manifest")
        try:
            with wave.open(io.BytesIO(content), "rb") as wav:
                frames, rate, channels = wav.getnframes(), wav.getframerate(), wav.getnchannels()
                require(rate == 22050 and channels == 1 and wav.getsampwidth() == 2
                        and wav.getcomptype() == "NONE" and frames > 0,
                        f"{identifier}: expected nonempty mono 22050 Hz PCM_16 WAV")
                require(len(wav.readframes(frames)) == frames * 2, f"{identifier}: truncated WAV")
        except (wave.Error, EOFError) as exc:
            raise EvaluationError(f"{identifier}: invalid WAV: {exc}") from exc
        require(audio_manifest.get("frames") == frames, f"{identifier}: manifest frame count differs")
        records.append({"id": identifier, "text": reference, "language": language,
                        "wav_path": str(audio_path.relative_to(root)), "wav_sha256": actual_hash,
                        "duration_seconds": frames / rate, "audio_bytes": content})
    require(len(checkpoint_hashes) == 1, f"{name}: manifest mixes different checkpoints")
    return {"language": language, "language_name": name, "result_dir": result_dir,
            "source": {"samples_path": str(sample_path.relative_to(root)), "samples_sha256": sample_hash,
                       "inference_manifest_path": str(manifest_path.relative_to(root)),
                       "inference_manifest_sha256": manifest_hash,
                       "checkpoint_sha256": next(iter(checkpoint_hashes))}, "records": records}


def package_versions():
    versions = {"python": platform.python_version()}
    for name in ("faster-whisper", "ctranslate2", "huggingface-hub", "av", "tokenizers", "numpy"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def load_model(args):
    try:
        from faster_whisper import WhisperModel
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise EvaluationError("Optional ASR packages are missing; install requirements-asr.txt separately") from exc
    supplied_path = Path(args.model).expanduser()
    repo_id, revision, revision_source = None, None, None
    if supplied_path.is_dir():
        model_path = supplied_path.resolve()
        revision_source = "Local directory; content hashes identify this model, upstream revision unavailable"
    else:
        require(not args.model.startswith(("/", ".", "~")), f"Missing local model directory: {supplied_path}")
        repo_id = MODEL_ALIASES.get(args.model, args.model)
        require(bool(re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id)),
                "Unknown model alias; supply a public Hugging Face repo ID or local CTranslate2 directory")
        requested_revision = args.revision or "main"
        if args.allow_model_download:
            # Resolve and pin a commit before downloading. Only public model metadata
            # and weights are fetched; reference text and audio never leave the process.
            revision = HfApi(token=False).model_info(repo_id, revision=requested_revision, token=False).sha
            require(bool(re.fullmatch(r"[0-9a-f]{40,64}", revision or "")), "Hub returned no immutable model revision")
            requested_revision = revision
            revision_source = "Hugging Face model_info response, pinned for snapshot_download"
        model_path = Path(snapshot_download(
            repo_id, revision=requested_revision, local_files_only=not args.allow_model_download,
            cache_dir=str(args.cache_dir) if args.cache_dir else None, token=False,
            allow_patterns=["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"],
        ))
        if revision is None:
            require(model_path.parent.name == "snapshots" and bool(re.fullmatch(r"[0-9a-f]{40,64}", model_path.name)),
                    "Cannot identify the immutable cached model revision")
            revision = model_path.name
            revision_source = "Hugging Face cached snapshot directory; no network query"
    # A complete local tokenizer prevents WhisperModel from fetching a fallback.
    for filename in ("config.json", "model.bin", "tokenizer.json"):
        require((model_path / filename).is_file(), f"Incomplete local CTranslate2 model: missing {filename}")
    hashes = {path.name: file_digest(path) for path in sorted(model_path.iterdir())
              if path.is_file() and (path.name in {"config.json", "preprocessor_config.json", "model.bin", "tokenizer.json"}
                                     or path.name.startswith("vocabulary."))}
    model = WhisperModel(str(model_path), device=args.device, compute_type=args.compute_type,
                         cpu_threads=args.cpu_threads, num_workers=1, local_files_only=True)
    metadata = {"engine": "faster-whisper / CTranslate2 local inference", "requested_model": args.model,
                "repository_id": repo_id, "requested_revision": args.revision,
                "resolved_revision": revision, "revision_source": revision_source,
                "model_file_sha256": hashes, "requested_device": args.device,
                "requested_compute_type": args.compute_type, "cpu_threads": args.cpu_threads,
                "versions": package_versions(), "audio_uploaded": False}
    return model, metadata


def transcribe_language(inputs, model, model_metadata, beam_size):
    records = []
    for source in inputs["records"]:
        # Decode the very bytes that passed the manifest hash check, not a path
        # that could have been changed since validation.
        segments, info = model.transcribe(io.BytesIO(source["audio_bytes"]), language=inputs["language"],
                                          task="transcribe", beam_size=beam_size, temperature=0.0,
                                          condition_on_previous_text=False, vad_filter=False,
                                          initial_prompt=None, hotwords=None, word_timestamps=False)
        decoded = []
        for segment in segments:  # Iteration performs faster-whisper inference.
            require(math.isfinite(segment.start) and math.isfinite(segment.end), "ASR returned invalid segment timestamps")
            decoded.append({"start_seconds": float(segment.start), "end_seconds": float(segment.end),
                            "text": segment.text})
        require(info.language == inputs["language"], "ASR decoder language differs from explicitly requested language")
        hypothesis = "".join(segment["text"] for segment in decoded).strip()
        record = {key: value for key, value in source.items() if key != "audio_bytes"}
        record.update({"asr_transcript": hypothesis, "segments": decoded,
                       "metrics": error_metrics(source["text"], hypothesis)})
        records.append(record)
        print(f"Transcribed {inputs['language_name']}/{source['id']}", file=sys.stderr, flush=True)
    total_words = sum(row["metrics"]["reference_word_count"] for row in records)
    total_chars = sum(row["metrics"]["reference_character_count"] for row in records)
    word_errors = sum(row["metrics"]["word_edit_distance"] for row in records)
    char_errors = sum(row["metrics"]["character_edit_distance"] for row in records)
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "language": inputs["language"], "language_name": inputs["language_name"],
            "interpretation": CAVEAT, "normalization": NORMALIZATION, "source": inputs["source"],
            "model": model_metadata,
            "decoding": {"language": inputs["language"], "language_is_forced": True, "task": "transcribe",
                         "beam_size": beam_size, "temperature": 0.0, "condition_on_previous_text": False,
                         "vad_filter": False, "initial_prompt": None, "hotwords": None,
                         "note": "Reference text is not supplied as a decoding prompt."},
            "aggregate": {"sample_count": len(records), "word_edit_distance": word_errors,
                          "reference_word_count": total_words, "wer": word_errors / total_words,
                          "character_edit_distance": char_errors, "reference_character_count": total_chars,
                          "cer": char_errors / total_chars, "aggregation": "micro-average within this language"},
            "samples": records}


def write_evaluation(result_dir, evaluation):
    csv_stream = io.StringIO(newline="")
    fields = ["id", "language", "text", "asr_transcript", "wer", "cer", "word_edit_distance",
              "reference_word_count", "character_edit_distance", "reference_character_count", "wav_sha256"]
    writer = csv.DictWriter(csv_stream, fieldnames=fields)
    writer.writeheader()
    for row in evaluation["samples"]:
        writer.writerow({key: row.get(key, row["metrics"].get(key)) for key in fields})
    payloads = {"asr_evaluation.json": json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                "asr_transcripts.csv": csv_stream.getvalue()}
    for filename, content in payloads.items():
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=result_dir,
                                         prefix=".asr-", delete=False) as stream:
            stream.write(content)
            temporary = Path(stream.name)
        try:
            temporary.replace(result_dir / filename)
        finally:
            temporary.unlink(missing_ok=True)


def self_test():
    class MetricsTests(unittest.TestCase):
        def test_distance(self):
            self.assertEqual(edit_distance("kitten", "sitting"), 3)
            self.assertEqual(edit_distance([], ["new", "words"]), 2)
            self.assertEqual(edit_distance(["a", "b"], []), 2)
            self.assertEqual(edit_distance("same", "same"), 0)

        def test_metrics(self):
            self.assertAlmostEqual(error_metrics("one two three", "one four three")["wer"], 1 / 3)
            self.assertEqual(error_metrics("a", "a b c")["wer"], 2)
            self.assertEqual(error_metrics("hello", "")["wer"], 1)
            self.assertIsNone(error_metrics("", "hello")["wer"])

        def test_normalization_preserves_assignment_anomalies(self):
            self.assertEqual(normalize_text("The orchestra’s rhyth m at 10:45 AM."), "the orchestras rhyth m at 10 45 am")
            self.assertEqual(normalize_text("పులి అడ విలో నడిచింది."), "పులి అడ విలో నడిచింది")
            self.assertEqual(error_metrics("పట్టు", "పట్టు")["cer"], 0)
            self.assertEqual(normalize_text("andaroo"), "andaroo")

        def test_manifest_ids(self):
            with self.assertRaises(EvaluationError):
                indexed_rows([{"id": "E1"}, {"id": "E1"}], {"E1"}, "fixture")
            with self.assertRaises(EvaluationError):
                indexed_rows([{"id": "E1"}], {"E1", "E2a"}, "fixture")

    return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MetricsTests)).wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--language", choices=["all", "en", "te"], default="all")
    parser.add_argument("--model", default="large-v3-turbo", help="Model alias, public Hub repo ID, or local CTranslate2 directory")
    parser.add_argument("--revision", help="Hub commit, tag or branch; resolved to immutable commit before download")
    parser.add_argument("--allow-model-download", action="store_true", help="Permit public model metadata/weight downloads; never uploads audio")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--check-only", action="store_true", help="Validate source hashes/WAVs only; do not load/download a model or write files")
    parser.add_argument("--self-test", action="store_true", help="Run standard-library metric and ID validation tests without models or audio")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.cpu_threads < 1 or args.beam_size < 1:
        parser.error("--cpu-threads and --beam-size must be positive")
    try:
        languages = list(LANGUAGES) if args.language == "all" else [args.language]
        inputs = [load_inputs(args.root.resolve(), language) for language in languages]
        if args.check_only:
            print(json.dumps({"complete": True, "check_only": True,
                              "validated_wavs": sum(len(item["records"]) for item in inputs),
                              "language_codes": languages, "outputs_written": []}))
            return 0
        model, metadata = load_model(args)
        evaluations = [transcribe_language(item, model, metadata, args.beam_size) for item in inputs]
        # Fail closed if inference/reference files changed during transcription.
        for previous in inputs:
            current = load_inputs(args.root.resolve(), previous["language"])
            require(current["source"] == previous["source"] and
                    [row["wav_sha256"] for row in current["records"]] ==
                    [row["wav_sha256"] for row in previous["records"]],
                    "Input artifacts changed during ASR evaluation; no reports written")
        for item, evaluation in zip(inputs, evaluations):
            write_evaluation(item["result_dir"], evaluation)
        print(json.dumps({"complete": True, "outputs": [str(item["result_dir"] / "asr_evaluation.json") for item in inputs],
                          "note": CAVEAT}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"complete": False, "error": str(exc), "error_type": type(exc).__name__,
                          "note": "No new ASR report is published unless all requested inputs pass validation and transcription."},
                         ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
