# Technical report status

The final report candidate and hash-bound observations are complete locally for
the frozen recipe below. The strict full-workspace evidence check passes.
Confirm the latest PDF rendering and integrity checks before publication.
Status audit: 6 November 2026.

Actual local evidence is available:

- Full prepared splits: approximately 5 h training and 1 h validation per language,
  with production overlap and target-fingerprint audits.
- Production runs stopped gracefully at English step 8,842 and Telugu step 7,451.
  Configured targets were not completed and convergence is not guaranteed.
- Final inference uses English `selected.pt` at step 8,842 with sentence context
  and ordinary energy (SIL zeroing OFF), and Telugu `selected.pt` at step 2,000
  with its original frontend. Minimum-total baselines remain preserved as `best.pt`
  (English 1,500; Telugu 2,000), and `latest.pt` retains each final optimizer state.
- All ten prescribed WAVs, thirty variance plots, ten raw prediction archives and
  four production loss plots are present. WAV signal, sample rate and manifest
  identity checks passed; these are not perceptual-quality checks.
- Objective measurements, final ASR diagnostics and all ten hash-bound
  objective-only observations match the frozen recipe. English final assignment
  WER/CER is 27.59%/11.38%; Telugu remains 86.49%/24.89%.
  `build_report.py --check-only` passed in the full workspace containing the
  original processed feature targets.
  The cache-free submission intentionally omits those arrays; its corresponding
  check fails on missing processed targets. Use `validate_submission.py` and
  `download_checkpoints.py --verify-only` for the packaged artifact checks.

The report must preserve the actual limitations. English is a post-hoc preference
despite failure of the predeclared duration gate (minimum generated/source ratio
0.4897 < 0.5). The frozen comparison remains inconclusive. SIL zeroing improved
three validation probes but worsened assignment ASR, so it was rejected. Final
English assignment WER/CER is 27.59%/11.38%, versus original step-7,000 results
24.14%/8.98%; the final choice is not uniformly better. Context changes vowel
targets but does not encode lexical stress. Earlier selection records are retained.
Telugu retains the baseline after later WER worsened while CER improved. Earlier
assignment-ASR concerns motivated this post-hoc three-probe/two-candidate review;
it is not a pristine blind test or a native-listener quality result.

Native listening has not been performed. Objective-only observations must leave
pronunciation, intelligibility and naturalness explicitly unassessed. Finite
tensors, nonzero WAVs and ASR text similarity must not be described as proving
good speech. The report includes actual per-sample measurements,
selection rationale, source-ASR controls, failures, and language-specific limits.

The final completeness/integrity checks must be rerun after any change. Public hosting and assignment submission remain separate approval-dependent actions.
