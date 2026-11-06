#!/usr/bin/env python3
"""Build a private training-only archive after the final split audit passes.

Does not upload, provision a GPU, authenticate, or spend money.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    from tts.modeling import read_config,read_vocab
    from tts.training import Features,audit_splits
    p=argparse.ArgumentParser();p.add_argument('--language',choices=['English','Telugu'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--steps',type=int,default=50000);a=p.parse_args()
    c=read_config(ROOT/a.language/'config.yaml');v=read_vocab(c)
    train=Features(c['paths']['train_manifest'],v,c['model']['max_seq_len'])
    val=Features(c['paths']['val_manifest'],v,c['model']['max_seq_len'])
    audit=audit_splits(train,val,production=True)
    dest=a.output.resolve()/a.language/'project'
    if dest.exists(): raise FileExistsError(dest)
    dest.mkdir(parents=True)
    for folder in ['tts','scripts']:
        shutil.copytree(ROOT/folder,dest/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    vendor=dest/'vendor/FastSpeech2'
    vendor.mkdir(parents=True)
    for folder in ['model','transformer','utils','text','hifigan']:
        shutil.copytree(ROOT/'vendor/FastSpeech2'/folder,vendor/folder,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pth.tar','generator_LJSpeech*'))
    shutil.copy(ROOT/'vendor/FastSpeech2/LICENSE',vendor/'LICENSE')
    lang=dest/a.language;lang.mkdir()
    for name in ['train.py','inference.py','preprocess.py','test_samples.json']:
        shutil.copy(ROOT/a.language/name,lang/name)
    prepared=lang/'preprocessed';(prepared/'features').mkdir(parents=True)
    source=ROOT/a.language/'preprocessed'
    for name in ['stats.json','vocab.json','speakers.json','preparation_summary.json']:
        shutil.copy(source/name,prepared/name)
    for split,rows in [('train',train.rows),('val',val.rows)]:
        clean=[]
        for r in rows:
            fp=ROOT/r['features']
            shutil.copy(fp,prepared/'features'/fp.name)
            clean.append({k:value for k,value in r.items() if k not in ['audio_path','wav_path','raw_features']})
        (prepared/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in clean))
    c['preprocess']['path']['preprocessed_path']=f'{a.language}/preprocessed'
    c['training'].update(device='cuda',precision='bf16',batch_size=a.batch_size,steps=a.steps,
                         warmup_steps=4000,peak_learning_rate=.001,length_bucket_size=64,cpu_threads=4)
    (lang/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    # Use the recorded CUDA environment, not the macOS extraction NumPy version.
    shutil.copy(ROOT/'requirements-training.txt',dest/'requirements-training.txt')
    (dest/'split_audit.json').write_text(json.dumps(audit,indent=2))
    files={str(f.relative_to(dest)):hashlib.file_digest(f.open('rb'),'sha256').hexdigest()
           for f in sorted(dest.rglob('*')) if f.is_file()}
    (dest/'SHA256SUMS.json').write_text(json.dumps(files,indent=2))
    archive=dest.parent/(a.language+'.tar')
    with tarfile.open(archive,'w') as tar:tar.add(dest,arcname='project')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,
                     'sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),'audit':audit},indent=2))

if __name__=='__main__':main()
