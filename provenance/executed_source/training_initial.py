"""Train and separately evaluate reference FastSpeech2 with auditable manifests."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import math
import json
import os
import random
import time
import hashlib
import re
import unicodedata
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .modeling import (LOSS_NAMES, FastSpeech2Loss, resolve, read_config, digest,
    seed_everything, device_for, read_vocab, build_model, load_checkpoint)

class Features(Dataset):
    def __init__(self, manifest, vocab=None, max_frames=None):
        self.vocab_ids = set(vocab.values()) - {0} if vocab is not None else None
        self.max_frames = max_frames
        self.path = resolve(manifest)
        self.rows = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
        if not self.rows: raise ValueError(f'Empty manifest: {manifest}')
        ids = [r['id'] for r in self.rows]
        if len(ids) != len(set(ids)): raise ValueError('Duplicate IDs')
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row = self.rows[i]
        p = resolve(row['features'])
        with np.load(p, allow_pickle=False) as f:
            x = {k:f[k].copy() for k in ('phones','durations','mel','pitch','energy')}
        n = len(x['phones']); t = len(x['mel'])
        if n < 1 or t < 2 or x['mel'].shape != (t,80): raise ValueError(f'Invalid shape: {p}')
        if any(len(x[k]) != n for k in ('durations','pitch','energy')): raise ValueError(f'Phone target mismatch: {p}')
        if x['phones'].ndim != 1 or x['durations'].ndim != 1: raise ValueError(f'Invalid phone/duration dimensions: {p}')
        if not np.issubdtype(x['phones'].dtype,np.integer) or not np.issubdtype(x['durations'].dtype,np.integer): raise ValueError(f'Phone IDs/durations must be integers: {p}')
        if np.any(x['phones']<=0) or np.any(x['durations']<0): raise ValueError(f'Invalid targets: {p}')
        if self.vocab_ids is not None and not set(x['phones'].tolist()) <= self.vocab_ids: raise ValueError(f'Unknown phone ID in {p}')
        if self.max_frames is not None and (t > self.max_frames or n > self.max_frames): raise ValueError(f'Features exceed positional limit: {p}')
        if int(x['durations'].sum()) != t: raise ValueError(f'Duration sum differs from mel frames: {p}')
        if not all(np.isfinite(a).all() for a in x.values()): raise ValueError(f'Non-finite features: {p}')
        return row,x

def audit_splits(train, val, production):
    """Reject leakage before constructing the model, with a persisted audit result."""
    def values(rows, key):
        return {str(r[key]) for r in rows if r.get(key) not in (None, '')}
    for rows in (train.rows,val.rows):
        for key in ('features','audio_sha256','wav_sha256'):
            entries=[str(r[key]) for r in rows if r.get(key)]
            if len(entries)!=len(set(entries)): raise ValueError(f'Duplicate {key} within a split')
    for key in ('id', 'audio_sha256', 'wav_sha256', 'normalized_text_sha256', 'text_sha256'):
        overlap=values(train.rows,key) & values(val.rows,key)
        if overlap: raise ValueError(f'Training/validation {key} overlap: {sorted(overlap)[:3]}')
    for key in ('features','audio_path','wav_path'):
        def paths(rows): return {str(resolve(r[key]).resolve()) for r in rows if r.get(key)}
        if paths(train.rows) & paths(val.rows): raise ValueError(f'Training/validation {key} overlap')
    def normalized(rows):
        return {re.sub(r'\s+',' ',unicodedata.normalize('NFC',str(r.get('normalized_text',r.get('text',''))))).strip().casefold()
                for r in rows if r.get('normalized_text',r.get('text',''))}
    if normalized(train.rows) & normalized(val.rows): raise ValueError('Training/validation normalized text overlap')
    train_speakers=values(train.rows,'speaker_id'); val_speakers=values(val.rows,'speaker_id')
    if not val_speakers <= train_speakers: raise ValueError('Validation contains speakers absent from training')
    def seconds(rows):
        durations=[r.get('duration_seconds',r.get('duration')) for r in rows]
        if any(d is None for d in durations): return None
        if any(not np.isfinite(float(d)) or float(d)<=0 for d in durations): raise ValueError('Invalid manifest audio duration')
        return float(sum(float(d) for d in durations))
    train_seconds=seconds(train.rows); val_seconds=seconds(val.rows)
    if production:
        if train_seconds is None or val_seconds is None: raise ValueError('Production training requires duration_seconds in each manifest row')
        if not 16200 <= train_seconds <= 19800: raise ValueError(f'Production training requires about 5h audio (+/-10%); got {train_seconds/3600:.3f}h')
        if not 3240 <= val_seconds <= 3960: raise ValueError(f'Production validation requires about 1h audio (+/-10%); got {val_seconds/3600:.3f}h')
    def fingerprint(rows):
        # Machine paths are locators, not data identity. Hash actual target bytes
        # so a relocated, unchanged dataset resumes but rewritten targets do not.
        records=[]
        for row in rows:
            portable={k:v for k,v in row.items() if k not in ('features','audio_path','wav_path','raw_features')}
            h=hashlib.sha256()
            with resolve(row['features']).open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
            portable['feature_content_sha256']=h.hexdigest(); records.append(portable)
        return hashlib.sha256(json.dumps(records,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return {'production_duration_gate':production, 'train_examples':len(train),'validation_examples':len(val),
            'train_seconds':train_seconds,'validation_seconds':val_seconds,'train_speakers':len(train_speakers),
            'validation_speakers':len(val_speakers),'train_speaker_ids':sorted(int(x) for x in train_speakers),
            'validation_speaker_ids':sorted(int(x) for x in val_speakers),'fingerprint_version':2,'overlap_checks':'passed',
            'train_manifest_sha256':fingerprint(train.rows),'validation_manifest_sha256':fingerprint(val.rows)}

def collate(items):
    rows,xs=zip(*items); b=len(xs)
    n=max(len(x['phones']) for x in xs); t=max(len(x['mel']) for x in xs)
    phones=np.zeros((b,n),np.int64); duration=np.zeros((b,n),np.int64)
    pitch=np.zeros((b,n),np.float32); energy=np.zeros((b,n),np.float32); mel=np.zeros((b,t,80),np.float32)
    for i,x in enumerate(xs):
        ni=len(x['phones']); ti=len(x['mel'])
        phones[i,:ni]=x['phones']; duration[i,:ni]=x['durations']; pitch[i,:ni]=x['pitch']; energy[i,:ni]=x['energy']; mel[i,:ti]=x['mel']
    return ([r['id'] for r in rows], [r.get('text','') for r in rows],
        torch.tensor([r['speaker_id'] for r in rows],dtype=torch.long), torch.from_numpy(phones),
        torch.tensor([len(x['phones']) for x in xs]),n,torch.from_numpy(mel),
        torch.tensor([len(x['mel']) for x in xs]),t,torch.from_numpy(pitch),torch.from_numpy(energy),torch.from_numpy(duration))

def move(batch,device):
    return tuple(x.to(device) if isinstance(x,torch.Tensor) else x for x in batch)

def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize()
    if device.type=='mps': torch.mps.synchronize()

def evaluate(model, loader, lossfn, device, precision="float32"):
    model.eval(); sums=np.zeros(len(LOSS_NAMES)); count=0
    with torch.no_grad():
        for b in loader:
            b=move(b,device)
            with precision_context(device,precision): predictions=model(*b[2:])
            # The reference criterion reduces over all valid elements. Evaluate
            # each item from one shared forward pass so long utterances do not
            # change the weight of other items through batch composition.
            for i in range(len(b[0])):
                single=tuple(x[i:i+1] if isinstance(x,(torch.Tensor,list)) else x for x in b)
                pred=tuple(x[i:i+1] for x in predictions)
                losses=lossfn(single,pred)
                values=np.array([l.item() for l in losses])
                if not np.isfinite(values).all(): raise FloatingPointError('Non-finite validation loss')
                sums+=values; count+=1
    model.train()
    return {k:float(v) for k,v in zip(LOSS_NAMES,sums/count)} | {'examples':count,'aggregation':'mean_per_utterance'}

def append(path, record):
    with path.open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n'); f.flush()

def plot_losses(logdir, result):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result.mkdir(parents=True,exist_ok=True)
    for split in ('training','validation'):
        p=logdir/(split+'.jsonl')
        if not p.exists(): continue
        records=[json.loads(l) for l in p.read_text().splitlines()]
        if not records: continue
        fig,ax=plt.subplots(figsize=(9,5))
        for name in LOSS_NAMES: ax.plot([r['step'] for r in records],[r[name] for r in records],label=name)
        ax.set(xlabel='Optimization step',ylabel='Loss',title=f'{split.capitalize()} loss'); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(result/(split+'_loss.png'),dpi=160); plt.close(fig)

def precision_context(device, precision):
    if precision=='float32': return nullcontext()
    if precision!='bf16': raise ValueError('precision must be float32 or bf16')
    if device.type!='cuda' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('bf16 requires an available CUDA device with BF16 support')
    return torch.autocast(device_type='cuda',dtype=torch.bfloat16)

def learning_rate_at_step(step, peak_learning_rate, warmup_steps):
    if step<1 or warmup_steps<0 or peak_learning_rate<=0:
        raise ValueError('Invalid step, warmup_steps, or peak_learning_rate')
    if warmup_steps==0: return peak_learning_rate
    return peak_learning_rate*min(step/warmup_steps,math.sqrt(warmup_steps/step))

def epoch_batches(rows, batch_size, seed, epoch, bucket_multiplier=0):
    """Step-addressable shuffled batches; length sorting is local to random pools."""
    if bucket_multiplier < 0: raise ValueError('length_bucket_size must be non-negative')
    generator=torch.Generator().manual_seed(seed+epoch)
    permutation=torch.randperm(len(rows),generator=generator).tolist()
    if not bucket_multiplier:
        return [permutation[i:i+batch_size] for i in range(0,len(permutation),batch_size)]
    lengths=[]
    for row in rows:
        value=row.get('duration_seconds',row.get('duration'))
        if value is None or not np.isfinite(float(value)) or float(value)<=0:
            raise ValueError('Length bucketing requires positive duration_seconds metadata')
        lengths.append(float(value))
    pool_size=batch_size*bucket_multiplier
    batches=[]
    for start in range(0,len(permutation),pool_size):
        pool=sorted(permutation[start:start+pool_size],key=lambda i:(lengths[i],i))
        batches.extend(pool[i:i+batch_size] for i in range(0,len(pool),batch_size))
    order=torch.randperm(len(batches),generator=generator).tolist()
    return [batches[i] for i in order]

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--steps',type=int)
    p.add_argument('--resume'); p.add_argument('--benchmark',action='store_true'); args=p.parse_args(argv)
    c=read_config(args.config); tr=c['training']; seed_everything(tr.get('seed',20261105)); device=device_for(tr.get('device','auto'))
    if 'cpu_threads' in tr: torch.set_num_threads(tr['cpu_threads'])
    precision=tr.get('precision','float32')
    precision_context(device,precision)  # Check capability before allocating model/data.
    peak_lr=tr.get('peak_learning_rate',tr.get('learning_rate',1e-4))
    warmup_steps=tr.get('warmup_steps',0)
    learning_rate_at_step(1,peak_lr,warmup_steps)
    vocab=read_vocab(c)
    train=Features(c['paths']['train_manifest'],vocab,c['model']['max_seq_len'])
    val=Features(c['paths']['val_manifest'],vocab,c['model']['max_seq_len'])
    audit=audit_splits(train,val,production=not args.benchmark)
    speakers=json.loads((Path(c['preprocess']['path']['preprocessed_path'])/'speakers.json').read_text())
    known_speaker_ids=set(speakers.values())
    if not {r['speaker_id'] for r in train.rows+val.rows} <= known_speaker_ids: raise ValueError('Manifest speaker ID absent from speakers.json')
    if known_speaker_ids != set(range(len(speakers))): raise ValueError('Speaker IDs must be contiguous from zero')
    batch=tr.get('batch_size',8)
    if batch<1 or tr.get('validate_every',500)<1 or tr.get('log_every',10)<1: raise ValueError('Batch/log/validation intervals must be positive')
    vl=DataLoader(val,batch_size=batch,shuffle=False,collate_fn=collate,num_workers=0)
    model=build_model(c,vocab,device); model.train(); lossfn=FastSpeech2Loss(c['preprocess'],c['model'])
    optim=torch.optim.AdamW(model.parameters(),lr=peak_lr,betas=(.9,.98),eps=1e-9,weight_decay=tr.get('weight_decay',0))
    start=0
    if args.resume:
        state=load_checkpoint(args.resume,c,vocab,model,device)
        if bool(state.get('benchmark_only')) != args.benchmark: raise ValueError('Cannot mix production and benchmark checkpoints')
        if 'split_audit' in state:
            if state['split_audit'].get('fingerprint_version')!=audit['fingerprint_version']:
                raise ValueError('Legacy checkpoint data fingerprint; use the source version that created it or restart this short development run')
            for key in ('train_manifest_sha256','validation_manifest_sha256'):
                if state['split_audit'].get(key)!=audit[key]: raise ValueError('Resume manifests differ from checkpoint')
        for key in ('seed','batch_size','learning_rate','peak_learning_rate','warmup_steps','precision','weight_decay','grad_clip','length_bucket_size'):
            if state['config']['training'].get(key)!=tr.get(key): raise ValueError(f'Resume training setting changed: {key}')
        optim.load_state_dict(state['optimizer']); start=state['step']
        if 'torch_rng' in state: torch.set_rng_state(state['torch_rng'].cpu())
        if 'numpy_rng' in state: np.random.set_state(state['numpy_rng'])
        if 'python_rng' in state: random.setstate(state['python_rng'])
        if device.type=='cuda' and 'cuda_rng' in state: torch.cuda.set_rng_state_all(state['cuda_rng'])
        if device.type=='mps' and 'mps_rng' in state: torch.mps.set_rng_state(state['mps_rng'].cpu())
    dest=resolve(c['paths']['checkpoint_dir']); dest.mkdir(parents=True,exist_ok=True)
    logdir=dest/'logs'; logdir.mkdir(exist_ok=True)
    (dest/'split_audit.json').write_text(json.dumps(audit,indent=2))
    if start==0 and any((logdir/(n+'.jsonl')).exists() for n in ('training','validation')):
        raise FileExistsError('Existing run logs; resume or choose a new checkpoint directory')
    stop=args.steps if args.steps is not None else tr.get('steps',100000)
    if stop<=start: raise ValueError('Target steps must exceed resumed step')
    timings=[]; began=time.perf_counter(); best=state.get('best_validation',float('inf')) if args.resume else float('inf')
    batches_per_epoch=(len(train)+batch-1)//batch
    current_epoch=-1; batches=None
    for step in range(start+1,stop+1):
        # Step-addressable ordering preserves the exact batch sequence after resume.
        epoch=(step-1)//batches_per_epoch; batch_index=(step-1)%batches_per_epoch
        if epoch!=current_epoch:
            batches=epoch_batches(train.rows,batch,tr.get('seed',20261105),epoch,tr.get('length_bucket_size',0))
            current_epoch=epoch
        indices=batches[batch_index]
        b=collate([train[i] for i in indices])
        if b[8] > c['model']['max_seq_len']:
            raise ValueError('Mel exceeds model.max_seq_len; increase configured positional limit or filter consistently before splitting')
        b=move(b,device); synchronize(device); t=time.perf_counter()
        lr=learning_rate_at_step(step,peak_lr,warmup_steps)
        for group in optim.param_groups: group['lr']=lr
        optim.zero_grad(set_to_none=True)
        with precision_context(device,precision):
            pred=model(*b[2:]); losses=lossfn(b,pred)
        if not torch.isfinite(losses[0]): raise FloatingPointError('Non-finite training loss')
        losses[0].backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tr.get('grad_clip',1.0))
        if not torch.isfinite(norm): raise FloatingPointError('Non-finite gradient norm')
        optim.step()
        synchronize(device); elapsed=time.perf_counter()-t; timings.append(elapsed)
        rec={k:l.item() for k,l in zip(LOSS_NAMES,losses)}|{'step':step,'seconds':elapsed,'examples':len(b[0]),'grad_norm':norm.item(),'learning_rate':lr,'precision':precision}
        append(logdir/'training.jsonl',rec)
        if step%tr.get('log_every',10)==0 or step==stop: print(json.dumps(rec),flush=True)
        if step%tr.get('validate_every',500)==0 or step==stop:
            metrics=evaluate(model,vl,lossfn,device,precision); append(logdir/'validation.jsonl',metrics|{'step':step})
            improved=metrics['total']<best
            if improved: best=metrics['total']
            state={'format_version':1,'language':c['language'],'step':step,'contract_sha256':digest(c),'config':c,'vocab':vocab,
                   'model':model.state_dict(),'optimizer':optim.state_dict(),'torch_rng':torch.get_rng_state(),
                   'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'validation':metrics,
                   'best_validation':best,'benchmark_only':args.benchmark,'split_audit':audit,
                   'run_elapsed_seconds':time.perf_counter()-began, 'median_step_seconds':float(np.median(timings))}
            if device.type=='cuda': state['cuda_rng']=torch.cuda.get_rng_state_all()
            if device.type=='mps': state['mps_rng']=torch.mps.get_rng_state()
            tmp=dest/'latest.tmp'; torch.save(state,tmp); os.replace(tmp,dest/'latest.pt')
            if improved: torch.save(state,dest/'best.pt')
            plot_losses(logdir,resolve(c['paths']['result_dir']))
            (dest/'progress.json').write_text(json.dumps({'step':step,'total_target_steps':stop,
                'elapsed_seconds':time.perf_counter()-began,'median_step_seconds':float(np.median(timings)),
                'validation':metrics,'best_validation':best,'benchmark_only':args.benchmark,'split_audit':audit},indent=2))
    steady=timings[min(5,len(timings)//2):]
    summary={'language':c['language'],'device':str(device),'torch_version':str(torch.__version__),'steps':stop-start,
             'parameters':sum(x.numel() for x in model.parameters()),'median_step_seconds':float(np.median(steady)),
             'mean_step_seconds':float(np.mean(steady)),'batch_size':batch,'wall_seconds':time.perf_counter()-began,
             'benchmark_only':args.benchmark,'training_examples':len(train),'validation_examples':len(val),
             'length_bucket_size':tr.get('length_bucket_size',0),'precision':precision,
             'peak_learning_rate':peak_lr,'warmup_steps':warmup_steps,'split_audit':audit}
    (dest/'run_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
