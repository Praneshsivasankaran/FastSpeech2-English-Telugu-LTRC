# Independent early English alignment QA

Read-only audit during full corpus preprocessing. Selected shortest, median-duration and longest among the first 147 completed English alignments. Source WAVs were resolved through `work/datasets/en/candidates.jsonl`; saved overlays show their actual waveform and already-computed acoustic boundaries. No frontend/preprocessing code or cached feature was changed.

| Clip suffix | Seconds | Assessment |
|---|---:|---|
| shard 1, row 18 | 1.038 | Transcript `2003`; mean speech-phone confidence .188, 82% below .2. No grossly long phone, but too little acoustic support for the expected expanded numeral. Reject under the proposed confidence heuristic. |
| shard 2, row 133 | 5.324 | Sentence starts “Apart from this, the English language…”; max speech phone .275s, mean confidence .458. Overall progression and leading silence look plausible. Several brief phones have low confidence, consistent with accent/pronunciation uncertainty. No structural failure observed. |
| shard 2, row 1214 | 27.178 | Long passage about Gujarat cricket. Most phrase pauses align to SIL, but one /æ/ spans 11.235–12.627s (1.393s) and includes ~36% low-energy frames. This is an obvious undesirable duration target; reject rather than silently repair. |

Structural checks passed on all three: monotonic phone order, matching phone/feature counts, finite mel/pitch/energy, and sum(duration)==mel-frame count. These structural checks do not establish acoustic correctness.

Additional worst outlier: shard 2 row 164, a list of local festival names. /m/ spans 6.413–8.173s (1.76s); ~52% of that span is low energy relative to max(.001, 2×10th percentile utterance 10ms RMS). /ɑːɹ/ also spans .87s at low confidence .138. The unusual names likely disagree with dictionary pronunciation, and acoustic alignment is not trustworthy.

## Recommended acceptance gate

Continue extracting/caching acoustics, then reject candidates whose maximum **non-SIL** phone duration exceeds 1.0s or whose mean **non-SIL** confidence is below .2. This is an explicit conservative heuristic, not a validated accuracy metric. Do not fabricate or uniformly redistribute their phone times. In a later snapshot of 231 clips (0.378h), this gate flagged 11 clips and about10.6% of audio; counts will change as the corpus completes. The existing extra candidate audio should absorb this loss, but verify actual retained hours before starting training.

The gate may reject true long vowels and retain some incorrect boundaries. A full human alignment audit and a Telugu-trained aligner remain improvements. The trained model's duration predictor may otherwise reproduce extreme timing targets.
