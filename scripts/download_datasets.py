#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gzip, hashlib, io, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/manifests/dataset_raw_manifest.csv"
COMPRESSION = {"Exchange": "gz", "Jena": "zip"}

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def fetch(url: str, compression: str|None) -> bytes:
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    raw=urllib.request.urlopen(req, timeout=180).read()
    if compression=='gz': return gzip.decompress(raw)
    if compression=='zip':
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            members=[n for n in zf.namelist() if n.lower().endswith('.csv')]
            if not members: raise RuntimeError(f'No CSV found in {url}')
            return zf.read(members[0])
    return raw

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT/'datasets')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--verify', action='store_true')
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(MANIFEST.open(encoding='utf-8')))
    for r in rows:
        name=r['dataset']; path=args.output/f'{name}.csv'
        if args.force or not path.exists() or path.stat().st_size < 128:
            print(f'Downloading {name} ...', flush=True)
            path.write_bytes(fetch(r['url'], COMPRESSION.get(name)))
        digest=sha256_file(path)
        print(f'{name}: {path.stat().st_size} bytes, sha256={digest}')
        if args.verify:
            if digest != r['sha256']:
                raise SystemExit(f'{name}: SHA-256 mismatch; expected {r["sha256"]}')
            if path.stat().st_size != int(r['bytes']):
                raise SystemExit(f'{name}: byte-size mismatch; expected {r["bytes"]}')
    print('Dataset download/verification complete.')
if __name__=='__main__': main()
