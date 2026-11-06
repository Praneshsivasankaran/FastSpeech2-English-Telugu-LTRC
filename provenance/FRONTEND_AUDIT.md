# Frontend and alignment evidence

Implemented `tts.frontend` / `tts.alignment`, tested locally November 5, 2026.

- Native eSpeak NG loaded with espeakng-loader 0.2.4; phonemizer 3.6.0.
- Bournemouth Forced Aligner (BFA) 2.0.1, acoustic CUPE-3-en and CUPE-3-mls7, pinned model revisions in code and `alignment_models.json`.
- BFA code is GPL-3.0; CUPE model cards specify AGPL-3.0. eSpeak NG library is GPL. Keep third-party notices and comply when distributing.
- Original IPA phones are retained, including Telugu aspiration/retroflexion/vowel length. Acoustic ph66 mappings collapse some distinctions; these are not language-specific Telugu acoustic models.
- **Telugu alignment is zero-shot** using a model trained on English followed by seven European languages. The backend warns explicitly. Successful decoding is not validated Telugu alignment accuracy.
- Frontend preserves original sample text independently. NFC/whitespace normalization plus English time-meridiem expansion (`10:45 AM` -> `10:45 A M`); eSpeak expands numbers. No silent spelling repair. The supplied `అడ విలో` / `rhyth m` stays separated.
- Optional SIL at every word boundary prevents most long pauses being assigned to speech phones; zero durations are allowed for skipped SIL. Leading/trailing silence is explicit. Predicted timing comes from an acoustic segmental decoder, not uniform allocation.
- Fixed BFA frontend handling: use Unicode word breaker also for English, otherwise `<SIL>` is read aloud as letters. Preserve eSpeak word boundaries during number expansions, otherwise adjacent phonemes get concatenated. Fold multi-class acoustic continuations into the original IPA phone.
- English front end is word-level eSpeak: both senses of **conduct** default to the same pronunciation; `AM` following a time is now expanded to `A M` to avoid *am*. Other lexical ambiguity remains. These are real limitations for rubric analysis, not fixed claims.

## Tests run

`tests/smoke_frontend.py` was run on real Svarah english_0000 (8.7595 s) and english_0001 (3.1000625 s), plus real IndicTTS Telugu telugu_0000 (2.162 s) and telugu_0004 (8.268875 s). All returned complete target coverage, monotonic phone boundaries, finite confidence, variable speech durations, matching inference phonemes, and exact duration/frame sums. JSON and waveform overlays are in this directory. These are preprocessing tests, not synthesized samples or trained TTS results.

Two warm CPU measurements with torch threads=4: English 3.100 s clip took 4.105 s (RTF 1.324); Telugu 8.269 s clip took 3.892 s (RTF .471). These are tiny-clip measurements, not corpus throughput estimates or GPU comparisons.

All ten assignment test texts phonemize without unknown/drop errors: E1 97, E2a 61, E2b 68, E3 45, E4 100, T1 73, T2a 38, T2b 44, T3 37, T4 133 phones incl SIL.

## PDF visual audit

`Telugu/test_samples.json` reviewed against rendered page 2. T1 visually has joined `న్యూరోఫార్మకోలాజికల్`; pypdf's internal spaces in `న్యూ రోఫార్మ కోలాజికల్` come from a single NirmalaUI font run and are not visible word spaces. T2b visually has joined `వచ్చింది` (pypdf `వచ్చ ింది`). T3 has a clear intentional gap in `అడ విలో`, but visually joined `నిశ్శబ్దంగా` (pypdf `నిశ్శ బ్దంగా`). Current test_samples preserves those visible forms. These are visual transcriptions, not byte-exact recovery of the original editable document; preserve raw pypdf extraction beside the rendered PDF as provenance. The PDF contains glyph positioning and ToUnicode mapping, not authoritative original Unicode strings.

## Sources

- https://github.com/tabahi/bournemouth-forced-aligner
- https://pypi.org/project/bournemouth-forced-aligner/2.0.1/
- https://huggingface.co/Tabahi/CUPE-3-en
- https://huggingface.co/Tabahi/CUPE-3-mls7
- https://github.com/espeak-ng/espeak-ng

## MPS acceleration validation

All four clips were rerun on MPS with identical phoneme sequences and maximum CPU/MPS boundary difference 3.052e-7 seconds. Warm wall times: English 8.7595 s clip .286 s; English 3.1001 s clip .240 s; Telugu 2.162 s clip .208 s; Telugu 8.2689 s clip .274 s. These measurements justify MPS preprocessing on this Mac. JSON evidence: `alignment_mps_benchmark.json`.

## Final normalization checks

English time meridiems after numbers expand to separate uppercase letter names: `10:45 AM` -> `10:45 A M`, yielding /eɪ ɛm/. The original text remains in source_text. Ordinary `I am ready` remains unchanged. All ten assignment texts passed again; E4 now has 100 phones. `12వ`, `10:45`, and `2026` produced nonempty Telugu phone sequences, including ordinal suffix /ʋa/ for 12వ; no digit dropped silently. These are computational G2P checks, not native-listener correctness judgments. Contextual conduct pronunciation override was deliberately not added: upstream eSpeak operates word by word, and an untested syntactic heuristic could introduce errors. Report the homograph limitation.

## Telugu inventory correction after full-data discovery

The initial two Telugu smoke clips did not contain every phone. In the first 1,775 candidate texts, 562 utterances contained BFA-rejected phones: standalone `ʲ`, aspirated velar `ɡʰ`, or aspirated palatal `ɟʰ`. Scanning the expanded 3,139 candidate set found 1,049 /ʲ/, 198 /ɡʰ/, and 2 /ɟʰ/ occurrences. Those are ordinary frontend outputs, not grounds to drop a third of Telugu data.

`configure_phonemizer` now scopes three mapping overrides to each **Telugu** instance: `ʲ→j`, `ɡʰ→g`, `ɟʰ→ʒ`. The latter follows the existing `ɟ→ʒ` map; loss of aspiration follows BFA's existing `kʰ→k`, `tʰ→t`, `dʰ→d`, `pʰ→p`, and `bʰ→b` conventions. The palatalization marker is approximated by the palatal glide acoustic class, and its original IPA is retained. These approximations are for timing only. They do not imply that the aligner recognizes aspiration or Telugu palatalization accurately.

The original IPA strings remain model input labels and are never silently deleted. English's mapper is unmodified. The overrides use an instance-local copy of BFA's function globals, rather than patching the installed package or its shared dictionaries. All 3,139 texts now phonemize without unknown-phone errors. All ten assignment text tests pass. All four original waveform alignment tests retain exactly the same phones and boundaries; the formerly failing 20.796-second Telugu sample now aligns 211 phones including /ʲ/. See `telugu_inventory_audit.json`, `telugu_inventory_regression.json`, and `telugu_inventory_fixed_alignment.{json,png}`. The frontend hash changes, so old preprocessing caches must be invalidated/recomputed.


Full-corpus quality screening additionally rejects backend bad_alignment flags, speech phones over one second, and mean speech confidence below 0.2. These are heuristics, not independently verified phone labels.
