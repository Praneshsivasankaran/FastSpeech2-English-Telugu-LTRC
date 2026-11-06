#!/usr/bin/env python3
"""Prepare a new, sanitized local project copy; never publish or change the source.

--check-only reports the export plan without writing anything. A real export
requires validate_submission.py to pass, copies to a new sibling directory, and
excludes private audio/features/alignments, caches, development files and Git
history. Known machine locators are removed from split manifests without changing
the portable metadata used by tts.training.audit_splits. Checkpoint bytes and all
final result bytes are copied unchanged. Review the new copy before publishing.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {"README.md", "THIRD_PARTY_NOTICES.md", "LICENSE", ".gitignore", ".gitattributes"}
TOP_DIRS = {"English", "Telugu", "Results", "Report", "provenance", "scripts", "tts", "tests", "vendor"}
LOCATOR_FIELDS = {"features", "audio_path", "wav_path", "raw_features"}
PRIVATE_PATH = re.compile(r"/(?:(?:Users|home)/[A-Za-z0-9_. -]+|root)/[A-Za-z0-9_. -]+(?:/|$)")
TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".jsonl", ".yaml", ".yml", ".csv", ".sh"}


def portable_metadata(rows):
    """Exactly the metadata exclusion set used by the production fingerprint."""
    return [{key: value for key, value in row.items() if key not in LOCATOR_FIELDS} for row in rows]


def sanitize_manifest(rows, source_root):
    cleaned = []
    for row in rows:
        result = {key: value for key, value in row.items() if key not in {"audio_path", "wav_path", "raw_features"}}
        if "features" in result:
            feature = Path(result["features"])
            if feature.is_absolute():
                try:
                    result["features"] = feature.relative_to(source_root).as_posix()
                except ValueError as exc:
                    raise ValueError("Feature locator is outside the source project; resolve it before exporting") from exc
        cleaned.append(result)
    if portable_metadata(rows) != portable_metadata(cleaned):
        raise ValueError("Sanitization changed portable training metadata")
    return cleaned


def sanitize_smoke(value):
    if isinstance(value, list):
        return [sanitize_smoke(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: sanitize_smoke(item) for key, item in value.items()}
    if isinstance(result.get("wav"), str) and Path(result["wav"]).is_absolute():
        result["wav"] = "private-development/" + Path(result["wav"]).name
        result["wav_location_note"] = "Development audio is not distributed; original machine path removed."
    return result


def sanitize_cloud_run(value):
    """Omit private account/billing metadata while retaining execution evidence."""
    if not isinstance(value, dict):
        raise ValueError("Cloud-run evidence must be a JSON object")
    financial = re.compile(r"balance|budget|billing|cost|charge|price|(?:^|_)usd(?:_|$)", re.I)
    removed = sorted(key for key in value if financial.search(key))
    return {key: item for key, item in value.items() if key not in removed}, removed


def included(relative):
    parts = relative.parts
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0] in TOP_FILES or (parts[0].startswith("requirements") and relative.suffix == ".txt")
    if parts[0] not in TOP_DIRS:
        return False
    if any(part in {".git", "__pycache__", ".venv", ".cache", ".pytest_cache", "smoke_inference"}
           or part.startswith(".pending-") for part in parts):
        return False
    if len(parts) >= 3 and parts[0] in {"English", "Telugu"} and parts[1] == "preprocessed" and parts[2] in {"features", "alignments"}:
        return False
    if parts[:3] == ("vendor", "FastSpeech2", "config"):
        # Unused upstream dataset presets contain their author's home paths.
        # The assignment has its own English/config.yaml and Telugu/config.yaml.
        return False
    if relative.suffix in {".parquet", ".pyc", ".pyo", ".tmp", ".log"} or ".inprogress." in relative.name:
        return False
    if relative.suffix == ".wav" and not (len(parts) == 4 and parts[0] == "Results" and parts[2] == "test_generated_audios"):
        return False
    if relative.suffix == ".npz" and not (len(parts) == 4 and parts[0] == "Results" and parts[2] == "predictions"):
        return False
    if relative.suffix == ".pt" and not (len(parts) == 3 and parts[0] in {"English", "Telugu"}
                                          and parts[1] == "checkpoints" and parts[2] in {"best.pt", "latest.pt", "selected.pt"}):
        return False
    if relative.name.endswith(".pth.tar") or relative.name.startswith("generator_LJSpeech"):
        return False
    return True


def absolute_strings(value, location="$"):
    if isinstance(value, str):
        return [location] if value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value) else []
    if isinstance(value, list):
        return [found for i, item in enumerate(value) for found in absolute_strings(item, f"{location}[{i}]")]
    if isinstance(value, dict):
        return [found for key, item in value.items() for found in absolute_strings(item, f"{location}.{key}")]
    return []


def transform(source, relative, source_root):
    """Return replacement UTF-8 bytes where necessary, plus a change record."""
    parts = relative.parts
    if len(parts) == 3 and parts[0] in {"English", "Telugu"} and parts[1] == "preprocessed" and relative.name in {"train.jsonl", "val.jsonl", "unused.jsonl"}:
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        cleaned = sanitize_manifest(rows, source_root)
        contract = json.dumps(portable_metadata(rows), ensure_ascii=False, sort_keys=True).encode()
        change = {"path": relative.as_posix(), "rows": len(rows), "portable_metadata_unchanged": True,
                  "portable_metadata_sha256": hashlib.sha256(contract).hexdigest(),
                  "removed_locators": ["audio_path", "wav_path", "raw_features"]}
        return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cleaned).encode(), change
    if relative.as_posix() == "provenance/smoke_training_evidence.json":
        value = sanitize_smoke(json.loads(source.read_text(encoding="utf-8")))
        return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(), {
            "path": relative.as_posix(), "change": "Replace private development WAV locators; preserve measurements and SHA256"}
    if relative.as_posix() == "provenance/cloud_run.json":
        value, removed = sanitize_cloud_run(json.loads(source.read_text(encoding="utf-8")))
        return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(), {
            "path": relative.as_posix(), "change": "Omit private account and billing fields; preserve execution and cleanup evidence",
            "removed_fields": removed}
    return None, None


def plan(source_root):
    entries, transformations, problems = [], [], []
    for source in sorted(source_root.rglob("*")):
        if source.is_symlink():
            if included(source.relative_to(source_root)):
                problems.append(f"Symlink not exported: {source.relative_to(source_root)}")
            continue
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        if not included(relative):
            continue
        replacement, change = transform(source, relative, source_root)
        if change:
            transformations.append(change)
        if relative.suffix in TEXT_SUFFIXES:
            content = replacement.decode("utf-8") if replacement is not None else source.read_text(encoding="utf-8")
            if PRIVATE_PATH.search(content):
                problems.append(f"Private machine path remains in {relative}")
            if relative.suffix in {".json", ".jsonl"}:
                values = [json.loads(line) for line in content.splitlines() if line.strip()] if relative.suffix == ".jsonl" else [json.loads(content)]
                for value in values:
                    for location in absolute_strings(value):
                        problems.append(f"Absolute JSON locator remains in {relative}: {location}")
            if relative.name == "config.yaml" and parts_language(relative):
                import yaml
                config = yaml.safe_load(content)
                locators = dict(config.get("paths", {})) | dict(config.get("preprocess", {}).get("path", {}))
                for key, value in locators.items():
                    if isinstance(value, str) and Path(value).is_absolute():
                        problems.append(f"Nonportable configuration locator: {relative}:{key}")
        entries.append((source, relative, replacement))
    return entries, transformations, problems


def parts_language(relative):
    return len(relative.parts) == 2 and relative.parts[0] in {"English", "Telugu"}


def require_complete(source_root):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_submission_export_audit", source_root / "scripts/validate_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with contextlib.redirect_stdout(io.StringIO()) as output:
        result = module.validate()
    if result:
        value = json.loads(output.getvalue())
        raise ValueError("Source submission is incomplete: " + "; ".join(value.get("failures", [])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--destination", type=Path, required=True, help="New sibling directory outside the source project")
    parser.add_argument("--check-only", action="store_true", help="Inspect the sanitized export plan; write nothing")
    args = parser.parse_args(argv)
    source_root, destination = args.source_root.resolve(), args.destination.resolve()
    try:
        if destination == source_root or destination.is_relative_to(source_root) or source_root.is_relative_to(destination):
            raise ValueError("Destination must be a separate directory, not the source, its parent or its child")
        if destination.exists():
            raise FileExistsError("Destination already exists; a public copy must use a fresh directory")
        entries, transformations, problems = plan(source_root)
        if problems:
            raise ValueError("Export privacy/portability checks failed: " + "; ".join(problems))
        audit = {"schema_version": 1, "file_count": len(entries), "bytes_before_sanitization": sum(source.stat().st_size for source, _, _ in entries),
                 "transformations": transformations, "source_modified": False,
                 "excluded": ["raw datasets/audio", "preprocessed feature and alignment caches", "Git history", "development outputs", "runtime caches", "unused upstream dataset presets"],
                 "note": "Portable metadata equality preserves the production fingerprint inputs; no fingerprint or checkpoint is rewritten. Existing split_audit hashes are retained."}
        if args.check_only:
            print(json.dumps(audit | {"written": False, "submission_completeness_checked": False}, indent=2))
            return 0
        require_complete(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".public-copy-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "project"
            stage.mkdir()
            for source, relative, replacement in entries:
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if replacement is None:
                    shutil.copy2(source, target)
                else:
                    target.write_bytes(replacement)
            (stage / "provenance/public_copy_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            os.replace(stage, destination)
        print(json.dumps(audit | {"written": True, "submission_completeness_checked": True, "destination": str(destination)}, indent=2))
        return 0
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"complete": False, "written": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
