"""Deterministic corpus preparation with acoustic alignment and train-only statistics."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import inspect
import importlib.metadata
import json
from pathlib import Path
import random
import re
import unicodedata
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

def canonical(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text).casefold()).strip()

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def split_rows(rows,seed=20261105,train_seconds=18000,val_seconds=3600):
    """Group identical text AND audio before seeded allocation, without truncating WAVs."""
    rows=sorted(rows,key=lambda r:r['id'])
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate source IDs')
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    seen={}
    for i,r in enumerate(rows):
        for key in [('text',canonical(r['text'])),('audio',r['audio_sha256'])]:
            if key in seen: parent[find(i)]=find(seen[key])
            else: seen[key]=i
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[find(i)].append(r)
    groups=list(groups.values()); random.Random(seed).shuffle(groups)
    out={'train':[],'val':[],'unused':[]}; totals={'train':0.,'val':0.,'unused':0.}
    # Validation is assigned first at whole-group granularity; training retains
    # the rest. Unknown-speaker validation is checked below, never silently used.
    for group in groups:
        sec=sum(r['duration_seconds'] for r in group)
        if totals['val']<val_seconds and totals['val']+sec<=val_seconds*1.1:
            target='val'
        elif totals['train']<train_seconds and totals['train']+sec<=train_seconds*1.1:
            target='train'
        else:
            target='unused'
        out[target].extend(group); totals[target]+=sec
    known={r['speaker_id'] for r in out['train']}
    missing={r['speaker_id'] for r in out['val']}-known
    if missing:
        # A learnable speaker table cannot synthesize an unseen validation
        # identity. Move whole transcript/audio components, never individual
        # recordings, and retain the original duration tolerance.
        val_ids={r['id'] for r in out['val']}
        moved=[]
        for group in groups:
            if group[0]['id'] in val_ids and any(r['speaker_id'] in missing for r in group):
                moved.extend(group)
        moved_ids={r['id'] for r in moved}
        out['val']=[r for r in out['val'] if r['id'] not in moved_ids]
        out['train'].extend(moved)
        seconds=sum(r['duration_seconds'] for r in moved)
        totals['train']+=seconds;totals['val']-=seconds
        if totals['train']>train_seconds*1.1 or totals['val']<val_seconds*.9:
            raise ValueError(f'Validation speakers absent from training: {missing}; cannot repair within duration limits')
    for key,target in [('train',train_seconds),('val',val_seconds)]:
        if totals[key]<target*.9: raise ValueError(f'Insufficient {key} audio: {totals[key]/3600:.3f}h')
        if totals[key]>target*1.1: raise ValueError(f'Excessive {key} audio: {totals[key]/3600:.3f}h')
    return out,totals

def features(wav_path,text,language,device='cpu'):
    import librosa
    import soundfile as sf
    import pyworld
    from .alignment import align,frame_durations
    wave,rate=sf.read(wav_path,dtype='float32',always_2d=True)
    if wave.shape[1]!=1: raise ValueError('Expected mono recording')
    wave=wave[:,0]
    if not np.isfinite(wave).all() or np.max(np.abs(wave))<1e-5: raise ValueError('Empty/nonfinite audio')
    aligned=align(wave,rate,text,language,device=device)
    y=librosa.resample(wave,orig_sr=rate,target_sr=22050)
    magnitude=np.abs(librosa.stft(y,n_fft=1024,hop_length=256,win_length=1024,center=True,pad_mode='reflect'))
    basis=librosa.filters.mel(sr=22050,n_fft=1024,n_mels=80,fmin=0,fmax=8000)
    mel=np.log(np.maximum(basis@magnitude,1e-5)).T.astype('float32')
    energy=np.linalg.norm(magnitude,axis=0)
    f0,time=pyworld.dio(y.astype('float64'),22050,frame_period=256/22050*1000)
    f0=pyworld.stonemask(y.astype('float64'),f0,time,22050)[:len(mel)]
    f0=np.pad(f0,(0,max(0,len(mel)-len(f0))))
    voiced=np.flatnonzero(f0>0)
    if len(voiced)<2: raise ValueError('Too few voiced frames for pitch supervision')
    f0=np.interp(np.arange(len(mel)),voiced,f0[voiced])
    durations=np.array(frame_durations(aligned,len(mel),sample_rate=22050,hop_length=256),dtype=np.int64)
    if durations.sum()!=len(mel): raise ValueError('Alignment does not partition mel frames')
    phone_f0=[]; phone_energy=[]; cursor=0
    for n in durations:
        phone_f0.append(float(np.mean(f0[cursor:cursor+n])) if n else 0.)
        phone_energy.append(float(np.mean(energy[cursor:cursor+n])) if n else 0.)
        cursor+=n
    return {'durations':durations,'mel':mel,'pitch':np.array(phone_f0,dtype=np.float32),
            'energy':np.array(phone_energy,dtype=np.float32)},aligned

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--train-hours',type=float,default=5);p.add_argument('--val-hours',type=float,default=1)
    p.add_argument('--reuse-only',action='store_true',help='Finalize existing raw features after acoustic QA without recomputing')
    p.add_argument('--max-phone-seconds',type=float,default=1.0)
    p.add_argument('--min-mean-confidence',type=float,default=0.2)
    p.add_argument('--accepted-hours',type=float,help='Stop the seeded candidate scan once this many usable hours pass all QA gates')
    args=p.parse_args(argv)
    import yaml
    from .frontend import phonemize
    conf=yaml.safe_load(Path(args.config).read_text());language=conf['language']
    out=PROJECT/conf['preprocess']['path']['preprocessed_path'];out.mkdir(parents=True,exist_ok=True)
    (out/'features').mkdir(exist_ok=True);(out/'alignments').mkdir(exist_ok=True)
    rows=[json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    rejected=[]; accepted=[]; seen_audio=set();accepted_seconds=0.
    previous_rejections={r['id']:r['error'] for r in
        (json.loads(line) for line in (out/'rejected.jsonl').read_text().splitlines())} if (out/'rejected.jsonl').exists() else {}
    live_rejections=out/'rejections.inprogress.jsonl'
    live_rejections.write_text('')
    extraction_versions={name:importlib.metadata.version(name) for name in
                         ('torch','numpy','librosa','pyworld','bournemouth-forced-aligner','espeakng-loader')}
    frontend_hash=hashlib.sha256((Path(__file__).parent/'frontend.py').read_bytes()
                    +(Path(__file__).parent/'alignment.py').read_bytes()+inspect.getsource(features).encode()
                    +json.dumps(extraction_versions,sort_keys=True).encode()).hexdigest()
    for idx,row in enumerate(rows):
        try:
            if not 1<=row['duration_seconds']<=30: raise ValueError('Outside 1–30 second length filter')
            if not canonical(row['text']): raise ValueError('Empty transcript')
            if row['audio_sha256'] in seen_audio: raise ValueError('Duplicate audio SHA256')
            if not isinstance(row['speaker_id'],int): raise ValueError('Missing integer speaker/profile ID')
            fp=out/'features'/(row['id']+'.raw.npz')
            ap=out/'alignments'/(row['id']+'.json')
            cached=json.loads(ap.read_text()) if ap.exists() and fp.exists() else {}
            using_cache=False
            if (cached.get('audio_sha256')==row['audio_sha256'] and cached.get('preprocessing_sha256')==frontend_hash
                and cached.get('source_text')==row['text']):
                try:
                    with np.load(fp) as f: x={k:f[k].copy() for k in f.files}
                    a=cached;using_cache=True
                except (ValueError,OSError,EOFError):
                    using_cache=False
            if not using_cache:
                if args.reuse_only: raise ValueError(previous_rejections.get(row['id'],'Matching cached features unavailable'))
                x,a=features(row['audio_path'],row['text'],language,args.device)
                a.update(audio_sha256=row['audio_sha256'],preprocessing_sha256=frontend_hash)
            if len(x['mel'])>conf['model']['max_seq_len']: raise ValueError('Beyond decoder length limit')
            if not using_cache:
                expected=phonemize(row['text'],language)
                if a['phones']!=expected: raise ValueError('Training/inference phone sequence mismatch')
            if any(segment.get('coverage_analysis',{}).get('bad_alignment',False)
                   for segment in a.get('raw_alignment',{}).get('segments',[])):
                raise ValueError('Acoustic QA: backend flagged unreliable alignment')
            speech=[(interval[1]-interval[0],confidence) for phone,interval,confidence in
                    zip(a['phones'],a['intervals_seconds'],a['confidence']) if phone!='SIL']
            if max(duration for duration,_ in speech)>args.max_phone_seconds:
                raise ValueError('Acoustic QA: speech phone longer than threshold')
            confidence=[c for _,c in speech if c is not None]
            if not confidence or np.mean(confidence)<args.min_mean_confidence:
                raise ValueError('Acoustic QA: mean speech-phone confidence below threshold')
            if not using_cache:
                np.savez_compressed(fp,**x)
                ap.write_text(json.dumps(a,ensure_ascii=False))
            accepted.append(row|{'raw_features':str(fp),'phone_labels':a['phones']})
            seen_audio.add(row['audio_sha256'])
            accepted_seconds+=row['duration_seconds']
            if args.accepted_hours and accepted_seconds>=args.accepted_hours*3600: break
        except Exception as e:
            failure={'id':row['id'],'error':str(e)}
            rejected.append(failure)
            with live_rejections.open('a') as stream:stream.write(json.dumps(failure,ensure_ascii=False)+'\n')
        if (idx+1)%50==0: print(json.dumps({'processed':idx+1,'accepted':len(accepted),'rejected':len(rejected)}),flush=True)
    write_jsonl(out/'rejected.jsonl',rejected)
    seed=conf.get('training',{}).get('seed',20261105)
    splits,totals=split_rows(accepted,seed=seed,train_seconds=args.train_hours*3600,val_seconds=args.val_hours*3600)
    # Vocabulary is text-only. Include the fixed requested test texts for OOV
    # coverage; never use their audio or targets. Record this choice explicitly.
    sample_path=Path(args.config).parent/'test_samples.json'
    labels={s for r in accepted for s in r['phone_labels']}
    if sample_path.exists():
        for r in json.loads(sample_path.read_text()): labels.update(phonemize(r['text'],language))
    vocab={s:i+1 for i,s in enumerate(sorted(labels))};vocab['<pad>']=0
    (out/'vocab.json').write_text(json.dumps(vocab,ensure_ascii=False,indent=2))
    speakers={str(r['speaker_id']):r['speaker_id'] for r in accepted}
    # Speakers lost to filtering are excluded, but embedding IDs must stay dense.
    remap={old:i for i,old in enumerate(sorted(set(speakers.values())))}
    for rows_in_split in splits.values():
        for row in rows_in_split: row['speaker_id']=remap[row['speaker_id']]
    speakers={name:remap[old] for name,old in speakers.items()}
    (out/'speakers.json').write_text(json.dumps(speakers,indent=2))
    stats={}
    for kind in ['pitch','energy']:
        values=[]
        for r in splits['train']:
            with np.load(r['raw_features']) as f: values.extend(f[kind][f['durations']>0].tolist())
        v=np.asarray(values);mean=float(v.mean());std=float(v.std())
        if std<1e-8: raise ValueError(f'Degenerate {kind} statistics')
        stats[kind]=[float((v.min()-mean)/std),float((v.max()-mean)/std),mean,std]
    (out/'stats.json').write_text(json.dumps(stats,indent=2))
    for split,items in splits.items():
        manifest=[]
        for row in items:
            with np.load(row['raw_features']) as f: x={k:f[k].copy() for k in f.files}
            x['phones']=np.array([vocab[s] for s in row['phone_labels']],dtype=np.int64)
            for kind in ['pitch','energy']: x[kind]=(x[kind]-stats[kind][2])/stats[kind][3]
            fp=out/'features'/(row['id']+'.npz');np.savez_compressed(fp,**x)
            manifest.append({k:v for k,v in row.items() if k not in ['raw_features','phone_labels']}|{'features':str(fp.relative_to(PROJECT))})
        write_jsonl(out/(split+'.jsonl'),manifest)
    summary={'language':language,'source_manifest_sha256':sha(args.manifest),'seed':seed,'accepted':len(accepted),
             'rejected':len(rejected),'duration_seconds':totals,'statistics_fit_on':'train only','vocabulary':'corpus text and prescribed test text only',
             'mel':{'sr':22050,'n_fft':1024,'hop_length':256,'n_mels':80,'fmin':0,'fmax':8000,'scale':'natural log magnitude Slaney mel'},
             'split_policy':'seeded whole connected groups of NFC-casefolded whitespace-normalized text and audio SHA256'}
    summary['acoustic_qa']={'maximum_speech_phone_seconds':args.max_phone_seconds,'minimum_mean_speech_confidence':args.min_mean_confidence,
                            'reject_backend_bad_alignment':True,'note':'Heuristic screening, not ground-truth phone boundary accuracy'}
    summary['preprocessing_sha256']=frontend_hash
    summary['extraction_versions']=extraction_versions
    summary['candidate_scan']={'processed_rows':idx+1,'available_rows':len(rows),'accepted_hours_stop':args.accepted_hours}
    (out/'preparation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
