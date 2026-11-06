"""Resolve an explicitly documented final checkpoint without replacing best.pt.

best.pt remains the minimum-total-validation-loss baseline. An optional
selection.json may designate only the sibling selected.pt. This module verifies
that declaration and file bytes; it never changes weights or recorded metrics.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re


class CheckpointSelectionError(ValueError):
    """The explicit selection is malformed, missing, or fails integrity checks."""


def _checkpoint_directory(root, language):
    names={'en':'English','english':'English','te':'Telugu','telugu':'Telugu'}
    try: name=names[language.lower()]
    except (AttributeError,KeyError) as exc:
        raise CheckpointSelectionError(f'Unsupported checkpoint language: {language!r}') from exc
    return Path(root).resolve()/name/'checkpoints'


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load_selection_metadata(root, language):
    """Return validated explicit metadata, or None when no selection is declared."""
    directory=_checkpoint_directory(root,language); declaration=directory/'selection.json'
    if not declaration.exists():
        if declaration.is_symlink(): raise CheckpointSelectionError(f"Broken selection declaration: {declaration}")
        return None
    try: metadata=json.loads(declaration.read_text(encoding='utf-8'))
    except (OSError,UnicodeError,json.JSONDecodeError) as exc:
        raise CheckpointSelectionError(f'Cannot read checkpoint selection: {declaration}: {exc}') from exc
    if not isinstance(metadata,dict): raise CheckpointSelectionError('selection.json must contain an object')
    if type(metadata.get('schema_version')) is not int or metadata['schema_version']!=1:
        raise CheckpointSelectionError('selection.json requires schema_version: 1')
    if metadata.get('selected_file')!='selected.pt':
        raise CheckpointSelectionError('selection.json may designate only selected_file: selected.pt')
    if type(metadata.get('selected_step')) is not int or metadata['selected_step']<=0:
        raise CheckpointSelectionError('selection.json requires a positive integer selected_step')
    if not isinstance(metadata.get('rationale'),str) or not metadata['rationale'].strip():
        raise CheckpointSelectionError('selection.json requires a nonempty rationale')
    if metadata.get('selection_basis')!='fixed_validation_comparison':
        raise CheckpointSelectionError('selection.json requires selection_basis: fixed_validation_comparison')
    if not isinstance(metadata.get('candidates'),list) or not metadata['candidates']:
        raise CheckpointSelectionError('selection.json requires a nonempty candidates list')
    for field in ('selected_sha256','validation_probe_manifest_sha256'):
        if not isinstance(metadata.get(field),str) or not re.fullmatch('[0-9a-fA-F]{64}',metadata[field]):
            raise CheckpointSelectionError(f'selection.json requires a 64-digit hexadecimal {field}')
    selected=directory/'selected.pt'
    if not selected.is_file(): raise CheckpointSelectionError(f'Explicit selected checkpoint is missing: {selected}')
    if selected.resolve().parent!=directory.resolve():
        raise CheckpointSelectionError('selected.pt must remain within its checkpoints directory')
    actual=_sha256(selected)
    if actual!=metadata['selected_sha256'].lower():
        raise CheckpointSelectionError('selected.pt SHA256 differs from selection.json')
    return metadata


def resolve_selected_checkpoint(root, language):
    """Return selected.pt after validation, otherwise the unchanged best.pt path."""
    directory=_checkpoint_directory(root,language)
    metadata=load_selection_metadata(root,language)
    return directory/('selected.pt' if metadata is not None else 'best.pt')
