"""Run with actual dataset WAV/transcript paths; writes audit evidence, not scores."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tts.frontend import phonemize
from tts.alignment import align,frame_durations,save_alignment_plot
p=argparse.ArgumentParser();p.add_argument('wav');p.add_argument('transcript');p.add_argument('--language',required=True);p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
text=Path(a.transcript).read_text().strip();wave,sr=sf.read(a.wav,dtype='float32')
r=align(wave,sr,text,a.language)
assert r['phones']==phonemize(text,a.language)
assert len(r['phones'])==len(r['intervals_seconds'])
assert r['intervals_seconds'][0][0]==0
assert abs(r['intervals_seconds'][-1][1]-len(wave)/sr)<1e-8
frames=int(len(wave)/sr*22050)//256+1
d=frame_durations(r,frames)
assert sum(d)==frames and min(d)>=0
speech_d=[b-a for p,(a,b) in zip(r['phones'],r['intervals_seconds']) if p!='SIL']
assert np.std(speech_d)>0.001
r['quantized_frame_durations']=d.tolist();r['total_frames']=frames
stem=Path(a.wav).stem
(out/f'{stem}_alignment.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
save_alignment_plot(r,wave,sr,out/f'{stem}_alignment.png')
print(json.dumps({'wav':a.wav,'phones':len(r['phones']),'seconds':len(wave)/sr,'mean_phone_confidence':float(np.mean([x for x in r['confidence'] if x is not None])),'frame_sum':int(sum(d)),'passed':True},ensure_ascii=False))
