# Dataset provenance and measured selection

Recorded 5 November 2026. These are data-import facts, not a claim of completed training. Original datasets, account credentials and the private assignment PDF are not included in this repository.

## English — Svarah

Source: [AI4Bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah), revision `ebbf7777fe771490696a3f7b007097606fa8c924`. [Original project](https://github.com/AI4Bharat/Svarah); [paper](https://arxiv.org/abs/2305.15760).

All three original `test` parquet shards were imported. Measured embedded audio: **6,656 utterances, 34,611.9976875 seconds (9.6144438 hours), mono 16 kHz**. This assignment constructs new training/validation subsets from the corpus's original test partition. It is not a standard Svarah benchmark evaluation.

The source describes 117 speakers, but the distributed parquet schema has no speaker ID. Its embedded filename prefixes identify recordings: the first shard alone contains over 1,000 such prefixes. They are not usable as verified speaker identities. All eight available demographic fields jointly produce **115 profile groups** across the full corpus. The importer hashes each profile and assigns dense conditioning IDs 0–114 when explicitly invoked with `--allow-profile-speaker-ids`. Every English row retains `speaker_identity_verified=false`. Raw demographic values are excluded from the emitted manifest.

These groups are metadata-conditioning proxies. Multiple people may share a profile, and one person's inconsistent metadata could produce separate groups. Neither 115 distinct real voices nor verified speaker-disjoint evaluation is claimed. Split disjointness must instead be checked on utterance IDs, audio hashes and normalized transcripts. A verified speaker map can be supplied later through `--speaker-map`.

All seven official GitHub commit trees were inspected; the `meta_speaker_stats.csv` mentioned in its README was not present. The earlier download address preserved in commit `074065ee3addaadfe07739a2bb150f61eb763a28`, `https://indic-asr-public.objectstore.e2enetworks.net/svarah.tar`, returned HTTP 403 during this audit. No mapping was inferred as fact.

The official dataset card states CC BY 4.0 and requires logged-in acceptance of its access conditions. Each reproducer must obtain their own authorized access. Attribution: Tahir Javed et al., *Svarah: Evaluating English ASR Systems on Indian Accents*, INTERSPEECH 2023, pp. 5087–5091. Changes: import, duration measurement, profile grouping, later filtering and deterministic splitting. No endorsement by the dataset authors is implied.

## Telugu — IndicTTS Telugu

Source: [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu), revision `7ab3c18491bef3f5d4e820c2701b5f78f54031d6`. Original corpus: [IIT Madras IndicTTS](https://www.iitm.ac.in/donlab/indictts/database).

The first two original `train` parquet shards contain 4,288 rows. The selected **female voice only** (`gender=0`), filtered by measured duration **1–30 seconds inclusive**, contains **3,139 utterances, 48,074.6324375 seconds (13.3540646 hours), mono 16 kHz**. This is the candidate pool, not the final 5h/1h split. Conditioning ID is 0 throughout. The source card identifies one female and one male voice; this is the basis for treating the female subset as one speaker. The remaining shards are unnecessary for the selected pool.

Seven normalized-text duplicate groups (15 extra rows) occur in this candidate pool using Unicode NFC and collapsed whitespace. Split preparation must remove duplicates or assign each complete duplicate group to one partition. The final import measures original WAV frames/sample rate rather than relying on claimed duration.

**Source-card discrepancy:** the card claims 8.74 total hours and 48 kHz. These values contradict the downloaded files: the first shard alone measures 11.649929 hours, and the selected embedded WAVs are 16 kHz. The card's final reference to a Hindi portion is also inconsistent with its Telugu title and transcripts. Measured facts above supersede those stale statements. Full-corpus duration was not measured.

The corpus is subject to the original IITM IndicTTS EULA; the Telugu card does not independently declare CC BY. The official [license URL](https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf) timed out during the audit. A copy of the primary-authored EULA was inspected from [this third-party mirror](https://huggingface.co/k-m-irfan/Fastspeech2_HS_Flask_API/blob/main/license.pdf). Its identity with the currently served official document remains unverified. Section 2.1 permits data copying, modification and free distribution of derivative works; sections 2.2 and 5 require notice preservation. Retain the EULA and required notice when distributing applicable materials; do not relicense the original corpus as MIT or CC BY.

Acknowledgement: Speech Technology Consortium, TDIL/MeitY; Hema A Murthy and S Umesh, IIT Madras. See `THIRD_PARTY_NOTICES.md` for the required notice.

## Reproduce the import

Install project requirements, accept Svarah's access conditions in your own Hugging Face account, and run `hf auth login`. The downloader uses the caller's normal cached authorization without embedding credentials.

```bash
python scripts/download_data.py --language en te --output /private/path/data
python scripts/import_parquet.py --language en --parquet '/private/path/data/en/data/*.parquet' --output /private/path/imported/en --allow-profile-speaker-ids
python scripts/import_parquet.py --language te --parquet '/private/path/data/te/data/*.parquet' --output /private/path/imported/te --gender 0 --min-seconds 1 --max-seconds 30
```

Each import writes `metadata.jsonl`, preserved WAV bytes and `import_audit.json`. Machine-readable source revisions, selected source-file checksums, measured pool sizes and limitations are in `dataset_sources.json`. None of these import facts establishes successful alignment, training, convergence or generated-speech quality.
