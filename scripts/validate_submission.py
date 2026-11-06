#!/usr/bin/env python3
"""Fail closed until the assignment's actual deliverables exist."""
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tts.checkpoint_selection import resolve_selected_checkpoint


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def validate(root=ROOT):
    root=Path(root).resolve();failures=[]
    for language,prefix in [('English','E'),('Telugu','T')]:
        expected={prefix+'1',prefix+'2a',prefix+'2b',prefix+'3',prefix+'4'}
        try:
            samples=json.loads((root/language/'test_samples.json').read_text())
            if not isinstance(samples,list) or len(samples)!=len(expected) or {r['id'] for r in samples}!=expected:
                failures.append(language+': wrong test IDs')
            source={r['id']:r['text'] for r in samples}
        except (OSError,ValueError,TypeError,KeyError) as exc:
            failures.append(language+': invalid/missing test samples: '+str(exc));source={}
        # best.pt is always retained as the original minimum-total baseline.
        for name in ['train.py','preprocess.py','inference.py','config.yaml','checkpoints/best.pt']:
            if not (root/language/name).is_file():failures.append(f'{language}/{name}: missing')
        results=root/'Results'/language
        for name in ['training_loss.png','validation_loss.png','inference_manifest.json']:
            if not (results/name).is_file():failures.append(f'Results/{language}/{name}: missing')
        for identifier in sorted(expected):
            wav=results/'test_generated_audios'/(identifier+'.wav')
            if not wav.exists(): failures.append(str(wav.relative_to(root))+': missing')
            else:
                try:
                    audio,sr=sf.read(wav)
                    if audio.ndim!=1 or sr!=22050 or not np.isfinite(audio).all() or len(audio)<sr*.1 or np.sqrt(np.mean(audio**2))<1e-5:
                        failures.append(f'{language}/{identifier}: invalid/silent WAV')
                except (OSError,RuntimeError,ValueError) as exc:
                    failures.append(f'{language}/{identifier}: unreadable WAV: {exc}')
            for kind in ['duration','pitch','energy']:
                if not (results/'variance_adaptor_plots'/kind/(identifier+'.png')).is_file():
                    failures.append(f'{language}/{identifier}/{kind}: missing')
        selection=None;chosen=None
        try:
            chosen=resolve_selected_checkpoint(root,language)
            # Resolver has validated schema, filename and selected bytes. Avoid
            # rehashing a large checkpoint just to expose its recorded metadata.
            declaration=root/language/'checkpoints/selection.json'
            if declaration.exists():selection=json.loads(declaration.read_text())
        except (OSError,ValueError) as exc:
            failures.append(language+': invalid checkpoint selection: '+str(exc))
        baseline=root/language/'checkpoints/best.pt'
        candidates=[baseline]
        if chosen is not None and chosen!=baseline:candidates.append(chosen)
        audited={}
        for ckpt in candidates:
            if not ckpt.is_file():
                if ckpt!=baseline:failures.append(language+'/'+ckpt.name+': missing')
                continue
            try:
                import torch
                state=torch.load(ckpt,map_location='cpu',weights_only=False)
                label=language+'/'+ckpt.name
                if state.get('benchmark_only',True):failures.append(label+': checkpoint is a development benchmark')
                if type(state.get('step')) is not int or state['step']<1:failures.append(label+': no optimizer steps')
                if state.get('language')!={'English':'en','Telugu':'te'}[language]:failures.append(label+': wrong checkpoint language')
                audit=state.get('split_audit',{})
                if not audit.get('production_duration_gate',False):failures.append(label+': no production split audit')
                for key,low,high in [('train_seconds',16200,19800),('validation_seconds',3240,3960)]:
                    if not low<=audit.get(key,0)<=high:failures.append(label+': invalid '+key)
                audited[ckpt]={'step':state.get('step'),'sha256':sha256(ckpt)}
                del state
            except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
                failures.append(language+'/'+ckpt.name+': invalid checkpoint: '+str(exc))
        if chosen is not None and chosen in audited:
            selected=audited[chosen]
            if selection is not None and selected['step']!=selection['selected_step']:
                failures.append(language+': selected checkpoint step differs from selection.json')
            manifest=results/'inference_manifest.json'
            if manifest.exists():
                try:
                    records=json.loads(manifest.read_text())
                    if not isinstance(records,list) or len(records)!=len(expected) or {r['id'] for r in records}!=expected:
                        failures.append(language+': incomplete inference manifest')
                    for row in records:
                        if row.get('checkpoint_sha256')!=selected['sha256']:failures.append(language+': stale selected-checkpoint provenance')
                        if row.get('checkpoint_step')!=selected['step']:failures.append(language+': stale selected-checkpoint step')
                        if source.get(row['id'])!=row.get('text'):failures.append(language+': test text changed')
                        audio_path=results/'test_generated_audios'/(row['id']+'.wav')
                        if audio_path.exists() and row.get('audio',{}).get('sha256')!=sha256(audio_path):
                            failures.append(language+': stale WAV provenance')
                except (OSError,ValueError,TypeError,KeyError) as exc:
                    failures.append(language+': invalid inference manifest: '+str(exc))
    for filename in ['README.md','requirements.txt','Report/Technical_Report.pdf']:
        if not (root/filename).is_file(): failures.append(filename+': missing')
    result={'complete':not failures,'failures':failures,'note':'File/integrity audit does not establish perceived speech quality.'}
    print(json.dumps(result,indent=2))
    return 0 if not failures else 1

if __name__=='__main__':sys.exit(validate())
