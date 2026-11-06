#!/usr/bin/env python3
"""Verify or atomically download the release checkpoints using only Python's stdlib.

The checked-in manifest has no release URL until publication is approved. Use
--verify-only for a strictly local check, or --base-url HTTPS_RELEASE_DOWNLOAD_URL
once assets are published. Existing valid files are preserved without rewriting.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from urllib.parse import quote, urlparse
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
CHUNK=1024*1024


def validate_base_url(value):
    if not isinstance(value,str):raise ValueError('The release base URL must be an HTTPS string')
    parsed=urlparse(value)
    if parsed.scheme!='https' or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('The release base URL must be HTTPS without credentials, query, or fragment')
    return value.rstrip('/')


def destination(root, name):
    path=PurePosixPath(name)
    if ('\\' in name or path.is_absolute() or '..' in path.parts or len(path.parts)!=3
        or path.parts[0] not in ('English','Telugu') or path.parts[1]!='checkpoints'
        or path.parts[2] not in ('best.pt','selected.pt','latest.pt')):
        raise ValueError('Invalid checkpoint destination: '+name)
    result=root.joinpath(*path.parts)
    if result.resolve().parent!=(root/path.parts[0]/'checkpoints').resolve():
        raise ValueError('Checkpoint destination escapes its directory: '+name)
    if result.resolve().is_relative_to(root) is False:
        raise ValueError('Checkpoint destination escapes the project root: '+name)
    return result


def load_manifest(path, root):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,dict) or type(data.get('schema_version')) is not int or data['schema_version']!=1:
        raise ValueError('Checkpoint asset manifest requires schema_version: 1')
    if data.get('base_url') is not None:validate_base_url(data['base_url'])
    assets=data.get('assets')
    if not isinstance(assets,list) or not assets:raise ValueError('Manifest requires a nonempty assets list')
    names=set(); destinations=set()
    for asset in assets:
        if not isinstance(asset,dict):raise ValueError('Each checkpoint asset must be an object')
        name=asset.get('name')
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]+\.pt',name) or name in names:
            raise ValueError('Invalid or duplicate asset name: '+str(name))
        names.add(name)
        if type(asset.get('size_bytes')) is not int or asset['size_bytes']<=0:raise ValueError('Invalid asset size: '+name)
        if not isinstance(asset.get('sha256'),str) or not re.fullmatch(r'[a-f0-9]{64}',asset['sha256']):
            raise ValueError('Invalid asset SHA256: '+name)
        if not isinstance(asset.get('destinations'),list) or not asset['destinations']:raise ValueError('Asset has no destinations: '+name)
        for target in asset['destinations']:
            if not isinstance(target,str) or target in destinations:raise ValueError('Invalid or duplicate destination: '+str(target))
            destination(root,target);destinations.add(target)
    return data


def file_status(path, asset):
    if not path.is_file():return 'missing'
    if path.stat().st_size!=asset['size_bytes']:return 'size_mismatch'
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(CHUNK),b''):digest.update(chunk)
    return 'valid' if digest.hexdigest()==asset['sha256'] else 'hash_mismatch'


def write_verified(stream, output, asset):
    digest=hashlib.sha256(); size=0
    while True:
        chunk=stream.read(CHUNK)
        if not chunk:break
        size+=len(chunk)
        if size>asset['size_bytes']:raise ValueError('Downloaded bytes exceed declared size: '+asset['name'])
        digest.update(chunk);output.write(chunk)
    output.flush();os.fsync(output.fileno())
    if size!=asset['size_bytes'] or digest.hexdigest()!=asset['sha256']:
        raise ValueError('Checkpoint size/SHA256 verification failed: '+asset['name'])


def install_verified(source, target, asset):
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.'+target.name+'-',suffix='.tmp',delete=False) as output:
            tmp=Path(output.name)
            with Path(source).open('rb') as stream:write_verified(stream,output,asset)
        os.replace(tmp,target)
    finally:
        if tmp is not None:tmp.unlink(missing_ok=True)


def obtain_asset(asset, targets, base_url, timeout):
    """Return a verified local source or download to a temporary file."""
    for target in targets:
        if file_status(target,asset)=='valid':return target,False
    if base_url is None:
        raise ValueError('No release URL configured; supply --base-url only after the approved release is published')
    targets[0].parent.mkdir(parents=True,exist_ok=True);tmp=None
    try:
        with tempfile.NamedTemporaryFile(dir=targets[0].parent,prefix='.'+asset['name']+'-',suffix='.download',delete=False) as output:
            tmp=Path(output.name)
            url=base_url+'/'+quote(asset['name'],safe='')
            with urlopen(url,timeout=timeout) as stream:
                if urlparse(stream.geturl()).scheme!='https':raise ValueError('Refusing a non-HTTPS download redirect')
                write_verified(stream,output,asset)
        return tmp,True
    except BaseException:
        if tmp is not None:tmp.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--base-url',help='HTTPS release download base URL; overrides the manifest URL')
    parser.add_argument('--verify-only',action='store_true',help='Check local files only; never accesses the network or writes checkpoints')
    parser.add_argument('--timeout',type=float,default=120)
    args=parser.parse_args(argv);root=args.root.resolve();report={'complete':False,'verify_only':args.verify_only,'assets':[]}
    try:
        if args.timeout<=0:raise ValueError('timeout must be positive')
        manifest=load_manifest(args.manifest or root/'provenance/checkpoint_assets.json',root)
        base_url=validate_base_url(args.base_url) if args.base_url else manifest.get('base_url')
        for asset in manifest['assets']:
            targets=[destination(root,t) for t in asset['destinations']]
            record={'name':asset['name'],'destinations':{},'action':'verified' if args.verify_only else 'preserved'}
            source=None;temporary=False
            try:
                statuses=[file_status(t,asset) for t in targets]
                if not args.verify_only and any(s!='valid' for s in statuses):
                    source,temporary=obtain_asset(asset,targets,base_url,args.timeout)
                    for target,status in zip(targets,statuses):
                        if status!='valid':install_verified(source,target,asset)
                    record['action']='downloaded' if temporary else 'copied_valid_local_asset'
                    statuses=[file_status(t,asset) for t in targets]
                record['destinations']=dict(zip(asset['destinations'],statuses))
            except (OSError,ValueError) as exc:
                record['error']=str(exc)
                record['destinations']={name:file_status(target,asset) for name,target in zip(asset['destinations'],targets)}
            finally:
                if temporary and source is not None:source.unlink(missing_ok=True)
            report['assets'].append(record)
        report['complete']=all('error' not in r and all(s=='valid' for s in r['destinations'].values()) for r in report['assets'])
    except (OSError,ValueError,TypeError,KeyError) as exc:report['error']=str(exc)
    print(json.dumps(report,indent=2))
    return 0 if report['complete'] else 1

if __name__=='__main__':sys.exit(main())
