#!/usr/bin/env python3
"""Import the exact assignment corpora without decoding/resampling embedded WAVs.

Dependencies: pyarrow, soundfile. Emits metadata.jsonl plus wavs/ and import_audit.json.
Svarah demographic groups are explicitly NOT verified speaker identities. Supply a
reviewed original-path-to-speaker mapping with --speaker-map to resolve identity.
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATASETS = {
    'en': ('ai4bharat/Svarah', 'ebbf7777fe771490696a3f7b007097606fa8c924', 'test', 'audio_filepath'),
    'te': ('SPRINGLab/IndicTTS_Telugu', '7ab3c18491bef3f5d4e820c2701b5f78f54031d6', 'train', 'audio'),
}
DEMO = ('gender', 'age-group', 'primary_language', 'native_place_state',
        'native_place_district', 'highest_qualification', 'job_category', 'occupation_domain')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main(argv=None):
    import pyarrow.parquet as pq
    import soundfile as sf
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--language', required=True, choices=DATASETS)
    p.add_argument('--parquet', required=True, nargs='+', help='One or more paths or quoted globs')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--gender', type=int, choices=(0,1), help='Telugu: female=0, male=1')
    p.add_argument('--min-seconds', type=float, default=0)
    p.add_argument('--max-seconds', type=float, default=float('inf'))
    p.add_argument('--allow-profile-speaker-ids', action='store_true', help='Assign dense embedding IDs to unverified demographic groups, explicitly labelled as proxies')
    p.add_argument('--speaker-map', type=Path, help='JSON mapping original embedded path to verified speaker name')
    args = p.parse_args(argv)
    paths = sorted({Path(f).resolve() for spec in args.parquet for f in glob.glob(spec)})
    if not paths:
        p.error('No parquet files matched')
    out = args.output.resolve(); (out / 'wavs').mkdir(parents=True, exist_ok=True)
    speaker_map = json.loads(args.speaker_map.read_text()) if args.speaker_map else {}
    dataset, revision, split, audio_col = DATASETS[args.language]
    profile_ids = {}
    if args.language == 'en' and (args.allow_profile_speaker_ids or speaker_map):
        groups = set()
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=list(DEMO) + ([audio_col] if speaker_map else [])):
                for row in batch.to_pylist():
                    demo = json.dumps([row.get(k) for k in DEMO], ensure_ascii=False, separators=(',',':'))
                    original = row.get(audio_col, {}).get('path', '')
                    if speaker_map and original not in speaker_map and not args.allow_profile_speaker_ids:
                        raise ValueError(f'No verified speaker mapping for {original}')
                    groups.add(str(speaker_map[original]) if original in speaker_map else 'en_demographic_group_' + sha(demo.encode())[:16])
        profile_ids = {group: idx for idx,group in enumerate(sorted(groups))}
        (out / 'profile_groups.json').write_text(json.dumps(profile_ids,indent=2)+'\n')
    counts = Counter(); seconds = Counter(); recording_groups = defaultdict(set)
    files = []; ids = set(); rows_total = 0; source_rows = 0; skipped = 0
    temporary = out / 'metadata.jsonl.tmp'
    with temporary.open('w', encoding='utf8') as manifest:
        for path in paths:
            pf = pq.ParquetFile(path)
            required = {audio_col, 'text', 'gender'}
            if not required <= set(pf.schema_arrow.names):
                raise ValueError(f'{path}: incorrect schema: missing {required-set(pf.schema_arrow.names)}')
            files.append({'name':path.name, 'bytes':path.stat().st_size, 'sha256':file_sha(path), 'rows':pf.metadata.num_rows})
            local_row = 0
            for batch in pf.iter_batches(batch_size=64):
                for r in batch.to_pylist():
                    row_index = local_row; local_row += 1
                    audio = r[audio_col]; raw = audio.get('bytes'); original = audio.get('path') or ''
                    if not raw:
                        raise ValueError(f'{path.name}:{local_row}: no embedded audio bytes')
                    info = sf.info(io.BytesIO(raw))
                    if info.frames <= 0 or info.samplerate <= 0:
                        raise ValueError(f'{path.name}:{local_row}: invalid audio')
                    if not args.min_seconds <= info.duration <= args.max_seconds or (args.gender is not None and int(r['gender']) != args.gender):
                        skipped += 1
                        continue
                    audio_hash = sha(raw)
                    uid = f'{args.language}_{path.stem}_{row_index:06d}'
                    if uid in ids: raise ValueError(f'Duplicate ID: {uid}')
                    ids.add(uid)
                    wav = out / 'wavs' / (uid + '.wav')
                    if not wav.exists() or sha(wav.read_bytes()) != audio_hash:
                        wav.write_bytes(raw)
                    if args.language == 'te':
                        gender = int(r['gender'])
                        if gender not in (0,1): raise ValueError(f'Unexpected gender: {gender}')
                        group = ['te_female','te_male'][gender]
                        verified = True
                        identity_basis = 'dataset card states exactly one female and one male speaker'
                        speaker_id = gender
                    else:
                        # Deliberately avoid emitting raw personal/demographic fields.
                        demo = json.dumps([r.get(k) for k in DEMO], ensure_ascii=False, separators=(',',':'))
                        group = 'en_demographic_group_' + sha(demo.encode())[:16]
                        speaker_id = profile_ids.get(group)
                        verified = original in speaker_map
                        identity_basis = 'verified supplied mapping' if verified else 'unverified demographic grouping, not a speaker identity'
                        if verified:
                            group = str(speaker_map[original])
                            speaker_id = profile_ids[group]
                        recording = re.sub(r'_chunk_\d+\.wav$', '', Path(original).name)
                        recording_groups[recording].add(group)
                    result = {'id':uid, 'language':args.language, 'dataset':dataset, 'dataset_revision':revision,
                        'source_split':split, 'source_file':path.name, 'source_row':row_index,
                        'source_audio_path':original, 'recording_id':re.sub(r'_chunk_\d+\.wav$', '', Path(original).name), 'audio_path':str(wav), 'audio_sha256':audio_hash,
                        'text':r['text'], 'duration_seconds':info.duration, 'sample_rate':info.samplerate,
                        'channels':info.channels, 'speaker_id':speaker_id, 'speaker':group,
                        'speaker_identity_verified':verified, 'speaker_identity_basis':identity_basis}
                    manifest.write(json.dumps(result, ensure_ascii=False)+'\n')
                    counts[group] += 1; seconds[group] += info.duration
                    rows_total += 1
            source_rows += pf.metadata.num_rows
    if rows_total + skipped != source_rows: raise RuntimeError('Source row count mismatch')
    temporary.replace(out/'metadata.jsonl')
    audit = {'dataset':dataset,'revision':revision,'original_split':split,'rows':rows_total,
        'source_rows':source_rows,'skipped_rows':skipped,'seconds':sum(seconds.values()),'hours':sum(seconds.values())/3600,'files':files,
        'speaker_or_proxy_groups':dict(counts),'seconds_by_group':dict(seconds),
        'recording_group_conflicts':{k:sorted(v) for k,v in recording_groups.items() if len(v)>1},
        'warning': 'English demographic groups are NOT verified speaker identities' if args.language=='en' and not speaker_map else None}
    (out/'import_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':rows_total,'hours':audit['hours'],'groups':len(counts),'manifest':str(out/'metadata.jsonl'),'warning':audit['warning']},ensure_ascii=False))

if __name__ == '__main__': main()
