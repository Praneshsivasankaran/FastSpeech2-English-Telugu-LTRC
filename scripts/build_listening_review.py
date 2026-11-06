#!/usr/bin/env python3
"""Build a local-only, blank listening-review worksheet for the exact final WAVs.

This does not perform listening or change report_observations.json. Browser
exports are explicit reviewer input, not automatic pronunciation assessments.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LANGUAGES={'English':('E1','E2a','E2b','E3','E4'),'Telugu':('T1','T2a','T2b','T3','T4')}
FIELDS=('pronunciation','intelligibility','naturalness','artifacts')


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def script_json(value):
    # JSON is inert data. Escape HTML delimiters and JS line separators so an
    # arbitrary reviewer/source string cannot terminate its script element.
    return (json.dumps(value,ensure_ascii=False).replace('&','\\u0026').replace('<','\\u003c')
            .replace('>','\\u003e').replace('\u2028','\\u2028').replace('\u2029','\\u2029'))


def build(root=ROOT):
    root=Path(root).resolve();observation_path=root/'Report/report_observations.json'
    baseline=json.loads(observation_path.read_text(encoding='utf-8'))
    if baseline.get('schema_version')!=1:raise ValueError('Expected observation schema_version 1')
    expected={i for ids in LANGUAGES.values() for i in ids}
    if set(baseline.get('samples',{}))!=expected:raise ValueError('Observations must contain all ten exact sample IDs')
    clips=[];sources={'observations_sha256':sha256(observation_path),'test_samples_sha256':{},'inference_manifest_sha256':{}}
    cards=[]
    hints={'pronunciation':'Describe specific words/sounds and stress. State any uncertainty.',
           'intelligibility':'Describe what you could understand, including omissions or substitutions.',
           'naturalness':'Describe rhythm, pace, intonation and pauses.',
           'artifacts':'Describe noise, distortion, breaks or repetition; write “none heard” only if true.'}
    for language,ids in LANGUAGES.items():
        sample_path=root/language/'test_samples.json';manifest_path=root/'Results'/language/'inference_manifest.json'
        samples=json.loads(sample_path.read_text(encoding='utf-8'));manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if len(samples)!=5 or {r['id'] for r in samples}!=set(ids):raise ValueError('Invalid '+language+' test IDs')
        if len(manifest)!=5 or {r['id'] for r in manifest}!=set(ids):raise ValueError('Invalid '+language+' inference IDs')
        sample_map={r['id']:r for r in samples};record_map={r['id']:r for r in manifest}
        sources['test_samples_sha256'][language]=sha256(sample_path);sources['inference_manifest_sha256'][language]=sha256(manifest_path)
        for identifier in ids:
            sample=sample_map[identifier];record=record_map[identifier]
            if sample['text']!=record['text']:raise ValueError(identifier+': inference text differs from exact source')
            wav=root/'Results'/language/'test_generated_audios'/(identifier+'.wav');digest=sha256(wav)
            if digest!=record.get('audio',{}).get('sha256') or digest!=baseline['samples'][identifier].get('wav_sha256'):
                raise ValueError(identifier+': WAV/manifest/observation hash mismatch')
            relative='../Results/'+language+'/test_generated_audios/'+identifier+'.wav'
            clips.append({'id':identifier,'language':language,'text':sample['text'],'wav_sha256':digest,'wav':relative})
            fields=''.join('<label for="'+identifier+'-'+field+'">'+field.capitalize()+'</label>'
                           '<textarea id="'+identifier+'-'+field+'" data-note="'+field+'" rows="3" maxlength="1400" '
                           'placeholder="'+html.escape(hints[field],quote=True)+'"></textarea>' for field in FIELDS)
            cards.append('<section class="clip" data-clip="'+identifier+'" aria-labelledby="title-'+identifier+'">'
                         '<div class="clip-head"><h2 id="title-'+identifier+'">'+identifier+' <span>'+language+'</span></h2>'
                         '<span class="clip-status" id="status-'+identifier+'">Not yet confirmed</span></div>'
                         '<p class="sentence" lang="'+('te' if language=='Telugu' else 'en')+'">'+html.escape(sample['text'])+'</p>'
                         '<audio controls preload="none" aria-label="'+identifier+' final synthesized WAV"><source src="'+relative+'" type="audio/wav">Your browser cannot play this WAV.</audio>'
                         '<label class="attestation"><input type="checkbox" id="listened-'+identifier+'" data-listened="'+identifier+'"> '
                         'I listened to this exact WAV.</label><div class="notes">'+fields+'</div>'
                         '<details><summary>Audio identity</summary><code>SHA-256: '+digest+'</code></details></section>')
    payload={'baseline':baseline,'clips':clips,'sources':sources,'fields':FIELDS}
    document='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src 'self' file:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>FastSpeech 2 · Listening review</title>
<style>
:root{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172b35;background:#f1f5f5;line-height:1.55}*{box-sizing:border-box}body{margin:0}main{max-width:940px;margin:auto;padding:36px 20px 80px}h1{font-size:2rem;letter-spacing:-.03em;margin:0 0 12px}h2{margin:0;font-size:1.2rem}p{margin:10px 0}.eyebrow{text-transform:uppercase;letter-spacing:.13em;color:#366b69;font-size:.76rem;font-weight:700}.intro,.clip,.toolbar{background:#fff;border:1px solid #d6e2e1;border-radius:14px;padding:24px;margin:18px 0}.intro{border-top:5px solid #376f6b}.notice{background:#eff7f6;padding:14px 16px;border-radius:8px}.small{font-size:.88rem;color:#4d626b}label{display:block;font-size:.9rem;font-weight:650;margin:12px 0 5px}input[type=text],textarea{width:100%;border:1px solid #aebfbe;border-radius:7px;background:#fff;color:inherit;font:inherit;font-size:.95rem;padding:10px}textarea{resize:vertical}input:focus,textarea:focus,button:focus{outline:3px solid #91c7c3;outline-offset:2px}.clip-head{display:flex;align-items:center;justify-content:space-between;gap:12px}.clip-head h2 span{font-weight:450;color:#62757a;font-size:.88rem;margin-left:8px}.clip-status{font-size:.78rem;color:#586b73;border-radius:20px;background:#eef2f2;padding:4px 10px}.sentence{white-space:pre-wrap;font-size:1.12rem;line-height:1.8;margin:20px 0}audio{width:100%;margin:0 0 10px}.attestation{background:#f2f7f6;border-radius:7px;padding:12px;display:flex;gap:10px;align-items:center}.attestation input{width:18px;height:18px}.notes{display:grid;grid-template-columns:1fr 1fr;column-gap:20px}.notes label{align-self:end}.notes textarea{grid-row:auto}.notes label:nth-of-type(1){grid-column:1;grid-row:1}.notes textarea:nth-of-type(1){grid-column:1;grid-row:2}.notes label:nth-of-type(2){grid-column:2;grid-row:1}.notes textarea:nth-of-type(2){grid-column:2;grid-row:2}.notes label:nth-of-type(3){grid-column:1;grid-row:3}.notes textarea:nth-of-type(3){grid-column:1;grid-row:4}.notes label:nth-of-type(4){grid-column:2;grid-row:3}.notes textarea:nth-of-type(4){grid-column:2;grid-row:4}details{margin-top:16px;color:#587075;font-size:.8rem}code{display:block;overflow-wrap:anywhere;padding-top:8px}.toolbar{position:sticky;bottom:12px;box-shadow:0 6px 25px #142e3020;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:12px;padding:16px 20px}.buttons{display:flex;flex-wrap:wrap;gap:10px}button{border:1px solid #376f6b;border-radius:7px;background:#fff;color:#265c58;padding:10px 14px;font:inherit;font-size:.87rem;font-weight:600;cursor:pointer}button.primary{background:#376f6b;color:#fff}button:disabled{opacity:.4;cursor:not-allowed}#download-message{width:100%;margin:0;font-size:.85rem;color:#375e5b}@media(max-width:650px){main{padding:20px 12px 60px}.intro,.clip{padding:18px}.notes{display:block}.clip-head{align-items:flex-start}.toolbar{position:static}.buttons button{width:100%}.buttons{width:100%}}
</style></head><body><main>
<header><div class="eyebrow">Local listening worksheet · 10 final WAVs</div><h1>Listen first. Record your own assessment.</h1></header>
<div class="intro"><p>Play each recording and compare it with its exact displayed sentence. Then check <strong>“I listened”</strong> and write your own notes in all four fields. Audio never starts automatically.</p>
<p class="notice">No human pronunciation assessment is supplied or inferred by this page. Automated scores and previous assessments are not displayed, so they cannot guide your initial listening.</p>
<p>State uncertainty or “cannot assess” when a language or sound is outside your knowledge. The current PDF builder expects English/Roman-script notes. Do not invent a judgment to complete a field.</p>
<label for="reviewer">Reviewer name / identifier</label><input id="reviewer" type="text" maxlength="1400" autocomplete="off" placeholder="" aria-describedby="reviewer-help">
<p id="reviewer-help" class="small">This name is attached only when you explicitly confirm a clip and provide all four notes.</p>
<p class="small">This page makes no external requests and stores nothing in browser storage. Reloading loses entries; download a partial copy to keep them. Downloads do not change source JSON or the existing report. Existing objective prose is preserved and may describe the pre-review state; importing notes and reconciling the report narrative is a separate step.</p></div>
<form id="review-form">''' + ''.join(cards)+'''
<div class="toolbar"><strong id="progress" aria-live="polite">0 of 10 reviews complete</strong><div class="buttons"><button id="partial" type="button">Download partial notes · not finalized</button><button id="complete" class="primary" type="button" disabled>Download complete observations</button></div><p id="download-message" aria-live="polite"></p></div></form>
<script id="review-data" type="application/json">'''+script_json(payload)+'''</script>
<script>
"use strict";
const payload = JSON.parse(document.getElementById("review-data").textContent);
const reviewedAt = Object.create(null);
const reviewer = document.getElementById("reviewer");
const byId = id => document.querySelector('[data-clip="' + id + '"]');
function readClip(clip) {
  const section=byId(clip.id), notes={};
  payload.fields.forEach(field => { notes[field]=section.querySelector('[data-note="'+field+'"]').value.trim(); });
  const listened=section.querySelector('[data-listened]').checked;
  return {listened,notes,complete: listened && reviewer.value.trim().length>0 && payload.fields.every(field=>notes[field].length>0)};
}
function refresh() {
  let count=0;
  payload.clips.forEach(clip => {
    const state=readClip(clip); if(state.complete) count++;
    document.getElementById('status-'+clip.id).textContent=state.complete ? 'Review complete' : state.listened ? 'Notes or reviewer incomplete' : 'Not yet confirmed';
  });
  document.getElementById('progress').textContent=count+' of '+payload.clips.length+' reviews complete';
  document.getElementById('complete').disabled=count!==payload.clips.length;
  return count;
}
function exportObservations(finalize) {
  const completed=refresh();
  if(finalize && completed!==payload.clips.length) throw new Error('All ten explicit listening confirmations, all notes, and the reviewer are required.');
  const output=JSON.parse(JSON.stringify(payload.baseline));
  const now=new Date().toISOString(), drafts={};
  payload.clips.forEach(clip => {
    const state=readClip(clip), item=output.samples[clip.id];
    drafts[clip.id]={listened_to_final_wav:state.listened,notes:state.notes,reviewed_at:reviewedAt[clip.id] || null};
    // A checkbox alone cannot turn a blank or partial record into an assessment.
    item.assessment_mode=state.complete ? 'listening' : 'objective_only';
    item.listening=state.complete ? Object.assign({reviewer:reviewer.value.trim(),reviewed_at:reviewedAt[clip.id] || now,listened_to_final_wav:true},state.notes) : null;
  });
  output.finalized=Boolean(finalize && completed===payload.clips.length);
  output.assessment_scope=output.finalized ? 'A named reviewer explicitly confirmed listening to all ten hash-identified WAVs and supplied the recorded notes. Existing objective observations are retained; listener statements are not automated assessments.' : 'Partial listening-review draft: '+completed+' of '+payload.clips.length+' records complete. This export is not finalized.';
  output.listening_review={schema_version:1,status:output.finalized?'complete':'partial',exported_at_utc:now,reviewer:reviewer.value.trim(),completed_clips:completed,total_clips:payload.clips.length,source_hashes:payload.sources,baseline_assessment_scope:payload.baseline.assessment_scope,drafts,
    note:'Existing objective entries and WAV hashes are unchanged. Browser export does not modify source files or regenerate the report; review its overall narrative after importing notes.'};
  return output;
}
function download(finalize) {
  try {
    const output=exportObservations(finalize), blob=new Blob([JSON.stringify(output,null,2)+'\\n'],{type:'application/json;charset=utf-8'});
    const url=URL.createObjectURL(blob), link=document.createElement('a');
    link.href=url;link.download=finalize?'report_observations.reviewed.json':'report_observations.partial.json';
    document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
    document.getElementById('download-message').textContent=finalize?'Complete review JSON downloaded. The existing report and source JSON are unchanged.':'Partial JSON downloaded with finalized: false. The existing report and source JSON are unchanged.';
  } catch(error) { document.getElementById('download-message').textContent=error.message; }
}
document.getElementById('review-form').addEventListener('submit',event=>event.preventDefault());
document.querySelectorAll('[data-listened]').forEach(box=>box.addEventListener('change',()=>{reviewedAt[box.dataset.listened]=box.checked?new Date().toISOString():null;refresh();}));
document.querySelectorAll('textarea,input[type="text"]').forEach(input=>input.addEventListener('input',refresh));
document.getElementById('partial').addEventListener('click',()=>download(false));
document.getElementById('complete').addEventListener('click',()=>download(true));
refresh();
</script></main></body></html>
'''
    destination=root/'Report/Listening_Review.html';destination.write_text(document,encoding='utf-8')
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args()
    print(build(args.root))

if __name__=='__main__':main()
