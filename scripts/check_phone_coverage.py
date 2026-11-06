#!/usr/bin/env python3
"""Audit actual training occurrences of every prescribed test phoneme."""
import collections
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tts.inference import phonemize_for_inference

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    report={}
    for language,code in [('English','en'),('Telugu','te')]:
        config_path=ROOT/language/'config.yaml'
        settings=yaml.safe_load(config_path.read_text()).get('inference',{})
        d=ROOT/language/'preprocessed';vocab=json.loads((d/'vocab.json').read_text())
        counts=collections.Counter()
        manifest=d/'train.jsonl'
        for row in map(json.loads,manifest.read_text().splitlines()):
            with np.load(ROOT/row['features'],allow_pickle=False) as features:
                counts.update(features['phones'].tolist())
        samples=[]
        for row in json.loads((ROOT/language/'test_samples.json').read_text()):
            actual_phones,frontend=phonemize_for_inference(row['text'],code,settings)
            phones=sorted(set(actual_phones))
            frequencies={p:counts[vocab[p]] for p in phones}
            samples.append({'id':row['id'],'frontend':frontend,'training_occurrences':frequencies,
                            'untrained_phones':[p for p,n in frequencies.items() if n==0]})
        report[language]={'train_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                          'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'samples':samples}
    out=ROOT/'provenance/phone_coverage.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if any(s['untrained_phones'] for language in report.values() for s in language['samples']):
        raise SystemExit('Untrained test phonemes found; inspect '+str(out))
    print('All prescribed test phones occur in the corresponding training split.')

if __name__=='__main__':main()
