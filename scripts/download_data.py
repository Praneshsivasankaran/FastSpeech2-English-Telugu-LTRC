#!/usr/bin/env python3
"""Download pinned assignment data using the caller's Hugging Face login.

Run `hf auth login` and accept Svarah's conditions on its official dataset page
before downloading English. This script never requests, prints or stores tokens.
It uses huggingface_hub's normal cached credentials and never bypasses gates.
"""
from __future__ import annotations
import argparse
from pathlib import Path

SOURCES = {
    'en': ('ai4bharat/Svarah', 'ebbf7777fe771490696a3f7b007097606fa8c924',
           [f'data/test-{i:05d}-of-00003.parquet' for i in range(3)]),
    'te': ('SPRINGLab/IndicTTS_Telugu', '7ab3c18491bef3f5d4e820c2701b5f78f54031d6',
           [f'data/train-{i:05d}-of-00004.parquet' for i in range(2)]),
}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--language', required=True, nargs='+', choices=SOURCES)
    p.add_argument('--output', type=Path, required=True, help='Private local data directory; not a public repository')
    p.add_argument('--include-unused-te-shards', action='store_true', help='Also fetch Telugu shards 2 and 3; unnecessary for the selected female subset')
    p.add_argument('--local-files-only', action='store_true', help='Offline cache-only retrieval')
    args = p.parse_args(argv)
    from huggingface_hub import snapshot_download
    for language in dict.fromkeys(args.language):
        repo, revision, files = SOURCES[language]
        files = list(files)
        if language == 'te' and args.include_unused_te_shards:
            files += [f'data/train-{i:05d}-of-00004.parquet' for i in (2,3)]
        dest = args.output.expanduser().resolve() / language
        snapshot_download(repo_id=repo, repo_type='dataset', revision=revision,
                          allow_patterns=files, local_dir=dest,
                          local_files_only=args.local_files_only, max_workers=2)
        print(f'{language}: pinned data available in {dest / "data"}')

if __name__ == '__main__':
    main()
