"""Generate assignment WAVs and plots from a trained language checkpoint."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from .modeling import (resolve, read_config, read_vocab, build_model, device_for,
                       load_checkpoint, load_vocoder)

def synthesize_mel(vocoder, mel, sample_rate, path):
    if sample_rate!=22050: raise ValueError('Bundled universal HiFi-GAN requires 22050 Hz mel features')
    if mel.ndim!=2 or mel.shape[1]!=80 or len(mel)<1 or not torch.isfinite(mel).all():
        raise ValueError('Invalid mel before vocoder')
    with torch.no_grad(): audio=vocoder(mel.transpose(0,1).unsqueeze(0)).squeeze().cpu().numpy()
    if not np.isfinite(audio).all() or audio.size==0: raise ValueError('Invalid vocoder audio')
    sf.write(path,np.clip(audio,-1,1),sample_rate,subtype='PCM_16')
    return {'frames':len(audio),'seconds':len(audio)/sample_rate,'rms':float(np.sqrt(np.mean(audio**2))),
            'peak':float(np.max(np.abs(audio))),'clipped_fraction':float(np.mean(np.abs(audio)>1)), 'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}

def safe_predict(model, speakers, ids, lengths, max_frames=3000):
    """Fail before duration expansion; never invent floors for unlearned durations."""
    def guard(module, inputs, output):
        if not torch.isfinite(output).all(): raise FloatingPointError('Non-finite predicted log durations')
        duration=torch.clamp(torch.round(torch.exp(output)-1),min=0)
        totals=duration.sum(dim=1)
        if not torch.isfinite(totals).all() or torch.any(totals>max_frames):
            raise ValueError(f'Predicted duration exceeds {max_frames} frames; inspect checkpoint convergence')
        if torch.any(totals<1): raise ValueError('All predicted durations round to zero; model needs further training')
    handle=model.variance_adaptor.duration_predictor.register_forward_hook(guard)
    try:
        with torch.no_grad(): prediction=model(speakers,ids,lengths,ids.shape[1])
    finally:
        handle.remove()
    if any(not torch.isfinite(t).all() for t in prediction if isinstance(t,torch.Tensor)):
        raise FloatingPointError('Non-finite acoustic-model prediction')
    return prediction

def phonemize_for_inference(text, language, settings):
    """Opt-in context frontend with explicit whole-utterance fallback metadata."""
    from .frontend import phonemize
    requested=settings.get('frontend','trained_word_level')
    if not isinstance(requested,str): raise ValueError('inference.frontend must be a string')
    english=language.lower() in ('en','english','en-us')
    if requested=='sentence_context_v1' and english:
        from .contextual_frontend import phonemize_with_metadata
        result=phonemize_with_metadata(text,language);details=result['metadata']
        fallback=details.get('status')=='fallback_word_level'
        metadata={'requested':requested,'used':'trained_word_level' if fallback else 'sentence_context_v1',
                  'status':details.get('status'),'fallback_reason':details.get('fallback_reason'),'details':details}
        return result['phones'],metadata
    reason=('sentence_context_v1_is_English_only' if requested=='sentence_context_v1' else
            None if requested=='trained_word_level' else 'unrecognized_frontend_option_uses_training_frontend')
    return phonemize(text,language),{'requested':requested,'used':'trained_word_level',
                                    'status':'fallback_word_level' if reason else 'trained_frontend','fallback_reason':reason}


def predict_with_energy_control(model,speakers,ids,lengths,max_frames=3000,*,silence_mask=None,zero_energy_value=0.):
    """Change only SIL energy conditioning; retain the finite raw head outputs.

    No waveform masking, duration floor, pitch change, or weight mutation occurs.
    A None mask is exactly the ordinary reference inference path.
    """
    if silence_mask is None:
        prediction=safe_predict(model,speakers,ids,lengths,max_frames)
        return prediction,prediction[3].detach().clone()
    if silence_mask.dtype!=torch.bool or silence_mask.shape!=ids.shape:
        raise ValueError('SIL energy mask must be boolean and match input phoneme IDs')
    if not np.isfinite(zero_energy_value):raise ValueError('Non-finite normalized zero-energy value')
    captured=[]
    def override(module,inputs,output):
        if output.shape!=silence_mask.shape:raise ValueError('SIL energy override requires phoneme-level energy')
        if not torch.isfinite(output).all():raise FloatingPointError('Non-finite raw energy prediction before SIL override')
        captured.append(output.detach().clone())
        return torch.where(silence_mask,torch.full_like(output,zero_energy_value),output)
    handle=model.variance_adaptor.energy_predictor.register_forward_hook(override)
    try:prediction=safe_predict(model,speakers,ids,lengths,max_frames)
    finally:handle.remove()
    if len(captured)!=1:raise ValueError('Expected exactly one phoneme-level energy prediction')
    return prediction,captured[0]

def plot(values, phones, kind, units, path, sample_id, conditioning_values=None):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    width=min(24,max(10,len(values)*.13)); fig,ax=plt.subplots(figsize=(width,4.5))
    ax.plot(np.arange(len(values)),values,marker='.',linewidth=1,label='Raw model prediction' if conditioning_values is not None else None)
    if conditioning_values is not None:
        ax.plot(np.arange(len(conditioning_values)),conditioning_values,linestyle='--',linewidth=1,label='Used energy conditioning (SIL override)')
        ax.legend()
    if len(phones)<=140:
        ax.set_xticks(np.arange(len(phones)));ax.set_xticklabels(phones,rotation=90,fontsize=7)
    ax.set(xlabel='Phoneme index',ylabel=units,title=f'{sample_id}: predicted {kind}')
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def read_samples(path,language):
    data=json.loads(Path(path).read_text())
    if isinstance(data,dict):
        data=data.get(language,data.get(language.capitalize(),data.get('samples',data)))
    if isinstance(data,dict): data=[{'id':k,'text':v} for k,v in data.items()]
    return [x for x in data if not x.get('language') or x['language'].lower() in (language.lower(), {'en':'english','te':'telugu'}.get(language,language))]

def publish_outputs(stage, destination):
    """Publish one complete prediction set, rolling back if a file move fails."""
    destination.mkdir(parents=True,exist_ok=True)
    names=('test_generated_audios','variance_adaptor_plots','predictions','inference_manifest.json')
    backup=stage/'previous';backup.mkdir()
    moved_old=[];moved_new=[]
    try:
        for name in names:
            dst=destination/name
            if dst.exists(): os.replace(dst,backup/name); moved_old.append(name)
            os.replace(stage/name,dst); moved_new.append(name)
    except Exception:
        for name in reversed(moved_new):
            dst=destination/name
            if dst.is_dir(): shutil.rmtree(dst)
            elif dst.exists(): dst.unlink()
        for name in reversed(moved_old): os.replace(backup/name,destination/name)
        raise

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--samples',required=True);parser.add_argument('--vocab');parser.add_argument('--speaker-id',type=int,default=0)
    parser.add_argument('--device',help='Inference device override, e.g. cpu, mps or cuda:0; does not change trained model configuration')
    parser.add_argument('--allow-smoke',action='store_true',help='Write development outputs only under smoke_inference, never assignment results')
    args=parser.parse_args(argv); c=read_config(args.config); device=device_for(args.device or c['training'].get('device','auto'))
    vocab=read_vocab(c,args.vocab); model=build_model(c,vocab,device); state=load_checkpoint(args.checkpoint,c,vocab,model,device)
    if state.get('benchmark_only') and not args.allow_smoke: raise ValueError('Benchmark checkpoint cannot produce submission results')
    if model.speaker_emb is not None and not 0<=args.speaker_id<model.speaker_emb.num_embeddings:
        raise ValueError('Requested speaker ID is absent from the checkpoint')
    trained_speakers=state.get('split_audit',{}).get('train_speaker_ids')
    if trained_speakers is not None and args.speaker_id not in trained_speakers:
        raise ValueError('Requested speaker has no training examples; choose a train_speaker_ids entry from split_audit.json')
    model.eval(); vocoder=load_vocoder(device)
    inference_settings=c.get('inference',{})
    if not isinstance(inference_settings,dict):raise ValueError('inference settings must be an object')
    zero_sil_energy=inference_settings.get('zero_sil_energy',False)
    if type(zero_sil_energy) is not bool:raise ValueError('inference.zero_sil_energy must be a boolean')
    result=resolve(c['paths']['result_dir'])
    if args.allow_smoke: result=result/'smoke_inference'
    final_result=result
    final_result.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pending-inference-',dir=final_result.parent) as temporary:
        result=Path(temporary)
        audio_dir=result/'test_generated_audios';audio_dir.mkdir(parents=True,exist_ok=True)
        for kind in ('duration','pitch','energy'): (result/'variance_adaptor_plots'/kind).mkdir(parents=True,exist_ok=True)
        (result/'predictions').mkdir(exist_ok=True)
        stats=json.loads((Path(c['preprocess']['path']['preprocessed_path'])/'stats.json').read_text())
        sr=c['preprocess']['preprocessing']['audio']['sampling_rate']; hop=c['preprocess']['preprocessing']['stft']['hop_length']
        energy_normalized=c['preprocess']['preprocessing']['energy'].get('normalization',False)
        if zero_sil_energy and c['preprocess']['preprocessing']['energy']['feature']!='phoneme_level':
            raise ValueError('zero_sil_energy requires phoneme-level energy')
        if energy_normalized:
            if not np.isfinite(stats['energy'][2:4]).all() or stats['energy'][3]<=0:raise ValueError('Invalid energy normalization statistics')
            zero_energy_value=-stats['energy'][2]/stats['energy'][3]
        else:zero_energy_value=0.0
        records=[]
        samples=read_samples(resolve(args.samples),c['language'])
        sample_ids=[x['id'] for x in samples]
        if len(sample_ids)!=len(set(sample_ids)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+',x) for x in sample_ids):
            raise ValueError('Sample IDs must be unique safe filenames')
        if not args.allow_smoke:
            prefix='E' if c['language'].lower() in ('en','english','en-us') else 'T'
            if set(sample_ids)!={prefix+'1',prefix+'2a',prefix+'2b',prefix+'3',prefix+'4'}:
                raise ValueError('Production inference requires the five assignment sentence IDs')
        checkpoint_sha256=hashlib.sha256(resolve(args.checkpoint).read_bytes()).hexdigest()
        for item in samples:
            identifier=item['id']; text=item['text']
            phones,frontend_metadata=phonemize_for_inference(text,c['language'],inference_settings)
            unknown=[p for p in phones if p not in vocab]
            if unknown: raise ValueError(f'{identifier}: unknown phonemes: {unknown}')
            ids=torch.tensor([[vocab[p] for p in phones]],dtype=torch.long,device=device)
            if not phones or torch.any(ids==0): raise ValueError('Empty or padding phoneme in input')
            silence_mask=np.array([phone=='SIL' for phone in phones],dtype=bool)
            pred,raw_energy=predict_with_energy_control(model,torch.tensor([args.speaker_id],device=device),ids,
                              torch.tensor([len(phones)],device=device),inference_settings.get('max_frames',3000),
                              silence_mask=torch.tensor(silence_mask[None,:],device=device) if zero_sil_energy else None,
                              zero_energy_value=zero_energy_value)
            durations=pred[5][0].cpu().numpy(); mel_len=int(pred[9][0]); mel=pred[1][0,:mel_len]
            if mel_len<1 or mel_len>20000: raise ValueError(f'Invalid predicted mel length {mel_len}; model may need more training')
            pitch=pred[2][0].cpu().numpy();raw_energy_normalized=raw_energy[0].cpu().numpy()
            conditioning_energy_normalized=pred[3][0].cpu().numpy()
            energy=raw_energy_normalized.copy();conditioning_energy=conditioning_energy_normalized.copy()
            # Training statistics are fitted only on the training split.
            if c['preprocess']['preprocessing']['pitch'].get('normalization',False): pitch=pitch*stats['pitch'][3]+stats['pitch'][2]
            if energy_normalized:
                energy=energy*stats['energy'][3]+stats['energy'][2]
                conditioning_energy=conditioning_energy*stats['energy'][3]+stats['energy'][2]
            np.savez_compressed(result/'predictions'/f'{identifier}.npz',phones=np.array(phones),duration_frames=durations,
                                duration_seconds=durations*hop/sr,pitch_hz=pitch,energy=energy,mel=mel.cpu().numpy(),
                                raw_energy_normalized=raw_energy_normalized,conditioning_energy_normalized=conditioning_energy_normalized,
                                conditioning_energy=conditioning_energy,silence_mask=silence_mask,zero_sil_energy_applied=np.array(zero_sil_energy))
            audio=synthesize_mel(vocoder,mel,sr,audio_dir/f'{identifier}.wav')
            for kind,values,unit in [('duration',durations*hop/sr,'Seconds'),('pitch',pitch,'F0 (Hz)'),('energy',energy,'Linear spectral energy')]:
                plot(values,phones,kind,unit,result/'variance_adaptor_plots'/kind/f'{identifier}.png',identifier,
                     conditioning_values=conditioning_energy if kind=='energy' and zero_sil_energy else None)
            records.append({'id':identifier,'text':text,'phonemes':phones,'speaker_id':args.speaker_id,'checkpoint_step':state['step'],
                            'checkpoint_sha256':checkpoint_sha256,'audio':audio,'frontend':frontend_metadata,
                            'inference_controls':{'frontend':inference_settings.get('frontend','trained_word_level'),
                              'zero_sil_energy':zero_sil_energy,'sil_energy_linear_value':0.0 if zero_sil_energy else None,
                              'duration_control':1.0,'pitch_control':1.0,'max_frames':inference_settings.get('max_frames',3000)}})
        (result/'inference_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
        publish_outputs(result,final_result)
    print(json.dumps({'samples':len(records),'result_dir':str(final_result),'checkpoint_step':state['step']}))

if __name__=='__main__': main()
