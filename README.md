# FastSpeech 2: English and Telugu

Two independent language models adapted from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), pinned at `d4e79eb52e8b01d24703b2dfc0385544092958f3`, for the IIIT Hyderabad LTRC assignment. **Bounded production training has stopped. The final local recipe is English step 8,842 with sentence-context processing and ordinary energy conditioning, plus Telugu step 2,000 with its original frontend. All ten WAVs, thirty variance plots, four loss plots and the final report candidate are complete locally; the strict report evidence check passes.** These are real generated artifacts; their existence does not establish satisfactory speech quality. Native-listener review remains unperformed.

The reference encoder, variance adaptor, length regulator, decoder, postnet and loss are retained. The adapters in `tts/` provide IPA vocabularies, device allocation, audited preparation and training/inference entry points. English and Telugu have separate configurations, vocabularies, statistics and weights. No pretrained acoustic TTS checkpoint is used. The separately attributed pretrained universal HiFi-GAN converts mel predictions to waveforms.

## Local artifacts and checkpoint decisions

| Language | Last completed training step | Minimum-total-loss `best.pt` | Inference `selected.pt` |
|---|---:|---:|---:|
| English | 8,842 | 1,500 | 8,842 |
| Telugu | 7,451 | 2,000 | 2,000 |

Training stopped gracefully and retained final validation, logs, optimizer state and recovery history. The saved 50,000-step configuration was not completed; the last bounded continuation targeted 10,000 steps and stopped at the counts above. `latest.pt` records the last saved step, `best.pt` remains the minimum-total-validation-loss baseline, and `selected.pt` identifies the explicitly documented inference artifact. Each `checkpoints/selection.json` preserves hashes, rationale and comparison evidence.

English step 8,842 is an explicitly **post-hoc artifact choice, not a validated-quality winner**. Its ordinary-energy fixed-validation WER/CER was 15.38%/11.19%, compared with 38.46%/30.60% for the earlier step-7,000 candidate. The original frozen duration gate remains failed and the original comparison inconclusive. Sentence-context inference changes the `conduct` vowel target by context for E2a/E2b, but lexical stress is not encoded. E1/E2a/E2b use the contextual frontend; E3/E4 retain the trained word-level fallback to preserve the supplied spacing and expansion contract. The training frontend and checkpoint loss history are unchanged.

The zero-SIL-energy intervention was tested and **rejected**. It improved the three fixed validation probes from WER/CER 15.38%/11.19% to 7.69%/2.99%, but worsened the five assignment prompts at the same step-8,842/context setting from 27.59%/11.38% to 37.93%/14.97%. The final recipe keeps `zero_sil_energy=false`, duration control 1.0 and pitch control 1.0. Raw predicted energy and actual conditioning energy therefore coincide. The original step-7,000 word-level artifact scored 24.14%/8.98% on the assignment prompts, so the final recipe is not uniformly better; it retains stronger fixed-validation recognition and the explicit vowel-target correction. All these choices were made after test diagnostics had been observed, not on a pristine blind test. `selection_v1.json`, `selection_v2.json` and `provenance/quality_audit/` preserve rejected variants and their measurements.

Telugu retains step 2,000 because the step-6,000 comparison was inconclusive: WER worsened from 73.53% to 79.41% while CER improved from 12.14% to 10.19%. Source-recording ASR controls on the same three prompts measured WER/CER of 7.69%/1.49% for English and 67.65%/8.74% for Telugu. These small-sample ASR diagnostics are not pronunciation, naturalness or native-listener quality scores. Earlier assignment-ASR concerns motivated the review, so this is not a pristine blind final test. Native listening has not been performed.

The ten local WAVs are checked for finite mono 22,050 Hz data, nonzero signal and matching selected-checkpoint/WAV hashes. Objective measurements and ASR diagnostics in each language's `analysis.json` and `asr_evaluation.json` match the final recipe. English final assignment WER/CER is 27.59%/11.38%; Telugu remains 86.49%/24.89%. Final objective-only observations and the strict full-workspace report check are complete. Native-listener assessment remains unperformed. Preserve the failures and limitations when evaluating or presenting these models.

Open the [technical report](Report/Technical_Report.pdf) or the [local listening worksheet](Report/Listening_Review.html). The worksheet links the exact final WAVs and records only assessments a reviewer explicitly supplies; it does not create a completed listening assessment by itself.

| English audio | Telugu audio |
|---|---|
| [E1](Results/English/test_generated_audios/E1.wav) | [T1](Results/Telugu/test_generated_audios/T1.wav) |
| [E2a](Results/English/test_generated_audios/E2a.wav) | [T2a](Results/Telugu/test_generated_audios/T2a.wav) |
| [E2b](Results/English/test_generated_audios/E2b.wav) | [T2b](Results/Telugu/test_generated_audios/T2b.wav) |
| [E3](Results/English/test_generated_audios/E3.wav) | [T3](Results/Telugu/test_generated_audios/T3.wav) |
| [E4](Results/English/test_generated_audios/E4.wav) | [T4](Results/Telugu/test_generated_audios/T4.wav) |

Supporting artifacts: [English measurements](Results/English/analysis.json), [Telugu measurements](Results/Telugu/analysis.json), [English ASR diagnostics](Results/English/asr_evaluation.json), [Telugu ASR diagnostics](Results/Telugu/asr_evaluation.json), [English variance plots](Results/English/variance_adaptor_plots/), and [Telugu variance plots](Results/Telugu/variance_adaptor_plots/).

## Measured preparation and limits

Both full-corpus preparation runs passed the production duration and overlap checks:

| Language | Training | Validation | Accepted / rejected candidates |
|---|---:|---:|---:|
| English | 3,047 utterances; 18,000.389 s (5.00011 h) | 615 utterances; 3,603.751 s (1.00104 h) | 3,971 / 383 |
| Telugu | 1,177 utterances; 18,005.328 s (5.00148 h) | 241 utterances; 3,614.103 s (1.00392 h) | 1,541 / 46 |

These are prepared-data facts, not claims of converged models or speech quality. Earlier ten-step smoke runs establish execution only; their evidence is labelled in `provenance/smoke_training_evidence.json`, and their audio is not an assignment result.

- English uses [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah), revision `ebbf7777fe771490696a3f7b007097606fa8c924`. Its three original ASR **test** shards contain 6,656 recordings, measured as 9.61444 h at 16 kHz. This project creates new TTS partitions; it is not a standard Svarah benchmark evaluation. The export lacks verified speaker IDs. Its 115 hashed demographic-profile groups are conditioning proxies, not 115 verified people; groups can merge voices.
- Telugu uses [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu), revision `7ab3c18491bef3f5d4e820c2701b5f78f54031d6`. The female voice in the first two training shards supplies 3,139 recordings in the inclusive 1-30 s range, measured as 13.35406 h at 16 kHz. These file measurements contradict the card's 48 kHz/8.74 h summary.
- Bournemouth Forced Aligner supplies acoustic phone timing. Telugu uses zero-shot CUPE-3-mls7; some Telugu distinctions share acoustic classes. Confidence/length screening does not establish linguist-verified boundaries. Original IPA labels remain model inputs.
- Training used word-level English eSpeak. Final contextual English inference distinguishes the `conduct` vowel target, while stress remains unencoded and the conservative word-level fallback remains for E3/E4. Time suffixes AM/PM expand to letter names. The supplied `rhyth m`, `andaroo` and Telugu internal space are preserved; no silent spelling repair or verified pronunciation quality is claimed.

Details and license qualifications are in `provenance/DATASETS.md`, `provenance/FRONTEND_AUDIT.md`, `provenance/ALIGNMENT_QUALITY.md` and `THIRD_PARTY_NOTICES.md`.

## Environments

Run commands from this project directory. Feature extraction, local inference and analysis use Python 3.11. The tested macOS environment is pinned in `requirements.txt` and recorded in `requirements-lock-macos.txt`:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

CUDA training uses a **separate** environment: PyTorch `2.4.1+cu124`, NumPy `1.26.4`, and `requirements-training.txt`. That file installs training dependencies, not the full audio/preprocessing toolchain. The macOS extraction environment uses PyTorch `2.15.0` and NumPy `2.4.6`; do not treat its lock as the CUDA runtime. Linux audio preprocessing additionally needs the full audio dependencies and may need a compiler for `pyworld`. Hardware-specific numerical equality is not promised.

```sh
python3.11 -m venv .venv-train
. .venv-train/bin/activate
python -m pip install -r requirements-training.txt
```

The production CUDA settings are batch size 16, BF16, seed 20261105, peak learning rate 0.001, 4,000 warmup steps and length-bucket multiplier 64. Completed steps and checkpoint selection are recorded above and in the actual logs; configuration targets are not completion claims.

## Download, import and reproduce candidate selection

Obtain your own authorized dataset access. Svarah requires logged-in acceptance of its conditions on the official page. `hf auth login` uses your local credential store; credentials are not command arguments or repository files.

```sh
hf auth login
python scripts/download_data.py --language en te --output data/downloads
python scripts/import_parquet.py --language en --parquet "data/downloads/en/data/test-*-of-00003.parquet" --output data/en --allow-profile-speaker-ids
python scripts/import_parquet.py --language te --parquet data/downloads/te/data/train-00000-of-00004.parquet data/downloads/te/data/train-00001-of-00004.parquet --output data/te --gender 0 --min-seconds 1 --max-seconds 30
python scripts/select_candidates.py --language en --manifest data/en/metadata.jsonl --output data/en/candidates.jsonl
python scripts/select_candidates.py --language te --manifest data/te/metadata.jsonl --output data/te/candidates.jsonl
```

The downloader pins both revisions. The importer preserves embedded WAV bytes and records source-file checksums, audio hashes, measured durations, source rows, text and speaker/profile provenance in private `metadata.jsonl` and `import_audit.json` files. Expected source checksums are in `provenance/dataset_sources.json`. English's `--allow-profile-speaker-ids` explicitly selects the documented proxy-identity limitation; a later verified speaker mapping would be a different experiment.

`select_candidates.py` filters measured durations to 1-30 s, sorts by source ID, and shuffles with `random.Random(20261105)`. English takes whole recordings until cumulative candidate duration first reaches 7.5 h: 4,354 records, 27,003.187 s. Telugu retains the complete shuffled female pool: 3,139 records, 48,074.632438 s. The helper verifies ordered sample identities against recorded hashes before writing; `--check-only` verifies without writing. It refuses to overwrite an existing candidate manifest.

The accompanying `candidates.audit.json` records a portable identity hash covering ordered IDs, audio hashes, text, source revision and conditioning metadata. A raw JSONL hash also covers machine-specific audio paths and can differ after relocation. Additional importer bookkeeping can also change byte hashes. Do not replace a recorded production fingerprint to force a resume: reproduce in a fresh output directory, compare sample identity, and preserve original run evidence.

## Preprocess and split

Activate the full audio environment before preprocessing:

```sh
python English/preprocess.py --config English/config.yaml --manifest data/en/candidates.jsonl --device mps --accepted-hours 6.5 --train-hours 5 --val-hours 1
python Telugu/preprocess.py --config Telugu/config.yaml --manifest data/te/candidates.jsonl --device mps --accepted-hours 6.5 --train-hours 5 --val-hours 1
```

Use `--device cpu` or `--device cuda` for another supported host. These commands write language-specific features and split metadata; use a fresh project copy when reproducing an existing run. MPS/CPU boundary consistency was tested on four real recordings, not the whole corpus.

The ordered scan stops when accepted audio reaches 6.5 h or candidates are exhausted. English exhausted its 4,354 candidates, retaining 6.49805 h after 383 rejections; Telugu scanned 1,587 candidates and retained 6.50154 h after 46 rejections. Whole connected groups sharing normalized transcript or audio SHA256 are shuffled with the same seed. Validation is allocated first toward 3,600 s, training toward 18,000 s, and surplus groups remain `unused`. No audio is cropped with unsliced text. Final counts/durations are shown above.

Preparation rejects out-of-range audio, duplicate audio, invalid acoustic features, unknown phones, decoder-length violations, backend bad-alignment flags, non-SIL phone spans above 1.0 s and mean non-SIL alignment confidence below 0.2. The last two are heuristics. Pitch/energy statistics are fitted only on training data. Vocabulary includes corpus and prescribed test **text**, never test audio or targets. `preparation_summary.json` records versions, thresholds, candidate prefix, source-manifest hash and split policy.

The training audit checks IDs, audio hashes, normalized text and feature-path separation, requires validation conditioning IDs to occur in training, and fingerprints actual target bytes. It rejects duration outside 5 h/1 h +/-10%. A feature-content or vocabulary/statistics change invalidates checkpoint resume.

## Train separate models

Package the prepared features for a CUDA host without including source recordings. These commands create local archives; they do not rent, upload to or start a paid service:

```sh
python scripts/build_training_bundle.py --language English --output data/training-bundles --batch-size 16 --steps 50000
python scripts/build_training_bundle.py --language Telugu --output data/training-bundles --batch-size 16 --steps 50000
```

Use a fresh bundle-output directory. Each archive contains one independent `project/`, selected features, a CUDA configuration and `requirements-training.txt`. On a CUDA host with the training environment active, extract each archive into a separate directory, enter its `project/`, and run the corresponding command:

```sh
python English/train.py --config English/config.yaml
python Telugu/train.py --config Telugu/config.yaml
# Resume using unchanged features and optimizer settings:
python English/train.py --config English/config.yaml --resume English/checkpoints/latest.pt
```

Only run the entry point present in that single-language bundle. A complete public project contains both. `best.pt` selects the smallest recorded validation total; `latest.pt` saves the most recent validated step and optimizer state. They need not have the same step. An optional explicit `selection.json` resolves inference to `selected.pt` while retaining the original `best.pt`; without a declaration, analysis falls back to `best.pt`. `run_summary.json` records the final invocation's update count, which need not equal total updates after resume; logs/progress retain absolute step numbers. `--steps N --benchmark` is development evidence, and production inference refuses its checkpoints.

## Inference, analysis and report

These commands require actual selected production checkpoints in the indicated folders; they do not create a trained model. Use the full audio environment and override the inference device without changing the trained acoustic configuration:

```sh
python English/inference.py --config English/config.yaml --checkpoint English/checkpoints/selected.pt --samples English/test_samples.json --speaker-id 60 --device mps
python Telugu/inference.py --config Telugu/config.yaml --checkpoint Telugu/checkpoints/selected.pt --samples Telugu/test_samples.json --speaker-id 0 --device mps
python scripts/analyze_results.py
```

For CPU/CUDA inference, change `--device` accordingly. English profile 60 has the largest prepared training duration: 510.827375 s across 73 recordings. It is a metadata profile, not a verified personal identity. Telugu ID 0 is the selected female voice. These are deterministic conditioning choices, not claims of the best perceived voice.

Inference writes exact-source-text manifests, WAVs, raw prediction arrays and 15 plots per language. Duration is in seconds, F0 in Hz, and energy in the denormalized linear spectral-energy units used by preprocessing. Predicted phone F0/energy are distinct from waveform F0/RMS. Analysis verifies source/checkpoint/output hashes and measures real WAVs; it does not score pronunciation or perceived quality.

After final inference, copy `Report/report_observations.template.json` to `Report/report_observations.json` and complete all ten entries against final WAV hashes. Listening claims require actual listening. `objective_only` leaves pronunciation/intelligibility/naturalness unassessed, so it is not equivalent to the assignment's requested listening assessment. Report rendering additionally needs ReportLab. Generate only after all evidence is complete, then render and visually inspect every page:

```sh
# Full prepared-data workspace only: requires the original feature targets.
python scripts/build_report.py --check-only
python scripts/build_report.py
# Also supported in the cache-free submission copy:
python scripts/validate_submission.py
python scripts/download_checkpoints.py --verify-only
```

Changing a config after inference invalidates the analysis config hash; rerun `analyze_results.py`. File completeness alone does not establish speech quality. The final report must discuss actual loss trajectories, every test sentence, variance behavior, limitations and failures.

## Assignment layout and distribution

The assignment's E2/T2 rows each contain two sentences, yielding IDs `E1,E2a,E2b,E3,E4,T1,T2a,T2b,T3,T4`. Visually audited text is in each language's `test_samples.json`. Extraction-only Telugu glyph spacing was distinguished from visible word spacing; see the frontend audit.

Required final layout:

```text
English/{train.py,preprocess.py,inference.py,config.yaml,checkpoints/}
Telugu/{train.py,preprocess.py,inference.py,config.yaml,checkpoints/}
Results/<language>/{training_loss.png,validation_loss.png,test_generated_audios/,variance_adaptor_plots/{duration,pitch,energy}/}
Report/Technical_Report.pdf
README.md
requirements.txt
```

Raw datasets, Parquet shards, private audio, credentials, development output, feature/alignment caches and the private assignment PDF are excluded from public distribution. Keep `stats.json`, `vocab.json`, `speakers.json`, split manifests/audits and unchanged trained checkpoint bytes: inference and provenance need them. Rebuilding features requires authorized dataset access. Checkpoint-internal historical runtime paths are not download locations; the supplied portable configuration determines current paths.

Prepare a **new sibling copy** only after the source submission passes validation; never sanitize active training files in place:

```sh
python scripts/prepare_public_copy.py --destination ../public-project --check-only
python scripts/prepare_public_copy.py --destination ../public-project
```

The dry run is a privacy/portability plan, not a completeness verdict. The real copy refuses an incomplete submission, removes only machine locators from split metadata, checks that fingerprint-relevant metadata is unchanged, sanitizes private smoke-audio locators, and excludes caches, raw data, Git history and unused upstream dataset presets. It does not rewrite checkpoint/result bytes or publish anything. Review `provenance/public_copy_audit.json` and final artifacts before publication.

The cache-free submission intentionally has no `preprocessed/features/` arrays. `build_report.py --check-only` therefore reports missing processed targets there; this strict source-evidence check belongs in the full prepared-data workspace **before** export. It must not be bypassed or weakened. In the submission, use `validate_submission.py` for packaged artifact integrity and `download_checkpoints.py --verify-only` for checkpoint sizes/hashes. Regenerating an exact original target fingerprint is a separate reproduction task; fresh preprocessing or retraining must retain its own new provenance when bytes differ.

The assignment requires a public GitHub repository and accessible separate checkpoints. Large checkpoints need Git LFS or stable external download links; links are still pending and must be tested before submission. Selected artifact hashes are:

| Artifact | SHA256 |
|---|---|
| `English/checkpoints/selected.pt` (step 8,842) | `48a2bc893c5192a7559408d4843a1d2f99b5bfab6196691edb636c6803ea7bac` |
| `Telugu/checkpoints/selected.pt` (step 2,000) | `7f32328011ab47edda7eadc67dc4331ce4e1a9f11a74fdc771b79e638f72f0bf` |

Publication and submission are authorized, but release upload and public download verification are separate final checks. The target repository is [FastSpeech2-English-Telugu-LTRC](https://github.com/Praneshsivasankaran/FastSpeech2-English-Telugu-LTRC). Preserve code, model and dataset notices, including the IndicTTS EULA qualifications in `THIRD_PARTY_NOTICES.md`.

Checkpoint asset names, sizes, destinations and SHA256 values are recorded in `provenance/checkpoint_assets.json`. Verify the local files with `python scripts/download_checkpoints.py --verify-only`. The configured release base URL is recorded there; after its assets are uploaded, `python scripts/download_checkpoints.py` downloads and verifies the four unique assets into both language checkpoint folders. A configured URL alone is not evidence that every remote asset has been uploaded and verified.
