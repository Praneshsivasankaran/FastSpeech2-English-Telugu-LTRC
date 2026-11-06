#!/usr/bin/env python3
"""Reproduce the assignment run's ordered preprocessing candidate manifests.

Input is import_parquet.py's private metadata.jsonl, not downloaded Parquet.
No audio is changed. The output preserves the chosen source rows verbatim in
JSON form and adds a separate audit. English: first >=7.5 h after the seeded
shuffle. Telugu: all eligible female rows after the same seeded shuffle.
Both use the inclusive 1-30 second filter and seed 20261105.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

SOURCES = {
    "en": ("ai4bharat/Svarah", "ebbf7777fe771490696a3f7b007097606fa8c924"),
    "te": ("SPRINGLab/IndicTTS_Telugu", "7ab3c18491bef3f5d4e820c2701b5f78f54031d6"),
}
IDENTITY_FIELDS = (
    "id", "language", "dataset", "dataset_revision", "source_split", "source_file", "source_row",
    "audio_sha256", "text", "duration_seconds", "sample_rate", "channels", "speaker_id", "speaker",
    "speaker_identity_verified", "speaker_identity_basis",
)
REFERENCE = {
    "en": {"rows": 4354, "seconds": 27003.187,
           "identity_sha256": "6a46e477af04317fb3171d2163276ff7f156ada93fb32ab9fcf497ac95c75d93"},
    "te": {"rows": 3139, "seconds": 48074.6324375,
           "identity_sha256": "46bd0f85d3ffcad2cb48233c3f92fe1817893736e7a65b8c6d3f136b9c5d337c"},
}


def identity_sha256(rows):
    """Portable sample identity; deliberately excludes absolute audio locators."""
    records = [{key: row[key] for key in IDENTITY_FIELDS} for row in rows]
    value = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def select(rows, language):
    if language not in SOURCES:
        raise ValueError("Expected en or te")
    if not rows or len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("Source rows must be nonempty with unique IDs")
    dataset, revision = SOURCES[language]
    for row in rows:
        if row.get("dataset") != dataset or row.get("dataset_revision") != revision or row.get("language") != language:
            raise ValueError("Source manifest has the wrong dataset, revision or language")
        if any(key not in row for key in IDENTITY_FIELDS):
            raise ValueError("Source manifest is missing reproducibility fields; rerun import_parquet.py")
        if not isinstance(row["text"], str) or type(row["speaker_id"]) is not int:
            raise ValueError("Source rows need text and integer speaker/profile IDs")
        seconds = row["duration_seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid measured source duration")
        if language == "te" and (row["speaker"] != "te_female" or row["speaker_id"] != 0):
            raise ValueError("Telugu reproduction requires the female-only import (--gender 0)")
    eligible = sorted((row for row in rows if 1 <= row["duration_seconds"] <= 30), key=lambda row: row["id"])
    random.Random(20261105).shuffle(eligible)
    selected, seconds = [], 0.0
    for row in eligible:
        selected.append(row)
        seconds += row["duration_seconds"]
        if language == "en" and seconds >= 7.5 * 3600:
            break
    identity = identity_sha256(selected)
    expected = REFERENCE[language]
    if (len(selected) != expected["rows"] or not math.isclose(seconds, expected["seconds"], abs_tol=1e-6)
            or identity != expected["identity_sha256"]):
        raise ValueError("Candidate identities differ from the recorded assignment run. "
                         "Check pinned source shards, import options, speaker mapping and source metadata. "
                         f"Observed rows={len(selected)}, seconds={seconds}, identity_sha256={identity}")
    return selected, {"schema_version": 1, "language": language, "seed": 20261105,
                      "eligible_duration_filter_seconds_inclusive": [1, 30],
                      "ordering": "sort source rows by ID, then random.Random(seed).shuffle",
                      "english_candidate_stop_hours": 7.5 if language == "en" else None,
                      "source_rows": len(rows), "eligible_rows": len(eligible), "selected_rows": len(selected),
                      "selected_seconds": seconds, "portable_identity_sha256": identity,
                      "matches_recorded_run": True,
                      "identity_fields": list(IDENTITY_FIELDS),
                      "note": "Byte-level manifest SHA256 includes machine audio paths; portable identity does not."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, choices=SOURCES)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true", help="Verify exact candidate identity without writing")
    args = parser.parse_args(argv)
    try:
        source = args.manifest.resolve()
        output = args.output.resolve()
        if output == source:
            raise ValueError("Output must not overwrite the imported source manifest")
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        selected, audit = select(rows, args.language)
        audit["source_manifest_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
        audit["candidate_manifest_sha256"] = hashlib.sha256(content.encode()).hexdigest()
        if not args.check_only:
            audit_path = output.with_name(output.stem + ".audit.json")
            if output.exists() or audit_path.exists():
                raise FileExistsError("Candidate output or audit already exists; choose a fresh private directory")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(audit | {"written": not args.check_only}, indent=2))
        return 0
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"complete": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
